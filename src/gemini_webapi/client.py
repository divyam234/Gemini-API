import asyncio
import functools
import json
import re
import time
import os
from asyncio import Task
from pathlib import Path
from typing import Any, Optional

from httpx import AsyncClient, ReadTimeout,Cookies

from .constants import Endpoint, ErrorCode, Headers, Model
from .exceptions import (
    AuthError,
    APIError,
    ImageGenerationError,
    TimeoutError,
    GeminiError,
    UsageLimitExceeded,
    ModelInvalid,
    TemporarilyBlocked,
)
from .types import WebImage, GeneratedImage, Candidate, ModelOutput
from .utils import logger

rotate_tasks: dict[str, Task] = {}

def running(retry: int = 0) -> callable:

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(client: "GeminiClient", *args, retry=retry, **kwargs):
            try:
                if not client.running:
                    await client.init(
                        timeout=client.timeout,
                        auto_close=client.auto_close,
                        close_delay=client.close_delay,
                        auto_refresh=client.auto_refresh,
                        refresh_interval=client.refresh_interval,
                        verbose=False,
                    )
                    if client.running:
                        return await func(client, *args, **kwargs)

                    # Should not reach here
                    raise APIError(
                        f"Invalid function call: GeminiClient.{func.__name__}. Client initialization failed."
                    )
                else:
                    return await func(client, *args, **kwargs)
            except APIError as e:
                # Image generation takes too long, only retry once
                if isinstance(e, ImageGenerationError):
                    retry = min(1, retry)

                if retry > 0:
                    await asyncio.sleep(1)
                    return await wrapper(client, *args, retry=retry - 1, **kwargs)

                raise

        return wrapper

    return decorator


class GeminiClient:

    __slots__ = [
        "cookies",
        "cookie_file",
        "proxy",
        "running",
        "client",
        "access_token",
        "timeout",
        "auto_close",
        "close_delay",
        "close_task",
        "auto_refresh",
        "refresh_interval",
        "kwargs",
    ]

    def __init__(
        self,
        cookie_file: Path = None,
        proxy: str | None = None,
        **kwargs
    ):
        self.proxy = proxy
        self.running: bool = False
        self.client: AsyncClient | None = None
        self.access_token: str | None = None
        self.timeout: float = 30
        self.auto_close: bool = False
        self.close_delay: float = 300
        self.close_task: Task | None = None
        self.auto_refresh: bool = True
        self.refresh_interval: float = 540
        self.cookies = Cookies()
        self.cookie_file = cookie_file
        self.kwargs = kwargs

    async def init(
        self,
        timeout: float = 30,
        auto_close: bool = False,
        close_delay: float = 300,
        auto_refresh: bool = True,
        refresh_interval: float = 540,
        verbose: bool = True,
    ) -> None:
        
        try:
            if not self.cookie_file.is_file():
                 raise FileNotFoundError("cookies file not found: "+str(self.cookie_file))
            
            with open(self.cookie_file) as f: 
                data = json.load(f)
                for cookie in data:
                    self.cookies.set(cookie["name"],cookie["value"],cookie["domain"])

            self.client = AsyncClient(
                http2=True,
                timeout=timeout,
                proxy=self.proxy,
                follow_redirects=True,
                cookies=self.cookies,
                **self.kwargs,
            )

            self.access_token = await self.get_access_token()
            self.running = True
            self.timeout = timeout
            self.auto_close = auto_close
            self.close_delay = close_delay
            if self.auto_close:
                await self.reset_close_task()

            self.auto_refresh = auto_refresh
            self.refresh_interval = refresh_interval
            if task := rotate_tasks.get("refresh_cookies"):
                task.cancel()
            if self.auto_refresh:
                rotate_tasks["refresh_cookies"] = asyncio.create_task(self.start_auto_refresh())

            if verbose:
                logger.success("Gemini client initialized successfully.")
        except Exception:
            await self.close()
            raise
    

    async def close(self, delay: float = 0) -> None:
        if delay:
            await asyncio.sleep(delay)

        self.running = False

        if self.close_task:
            self.close_task.cancel()
            self.close_task = None

        if self.client:
            await self.client.aclose()

    async def reset_close_task(self) -> None:

        if self.close_task:
            self.close_task.cancel()
            self.close_task = None
        self.close_task = asyncio.create_task(self.close(self.close_delay))
    
    async def get_access_token(self) -> tuple[str, dict]:
        try:
            response = await self.client.get(url=Endpoint.INIT.value,headers=Headers.GEMINI.value)
            response.raise_for_status()
            match = re.search(r'"SNlM0e":"(.*?)"', response.text)
            if match: return match.group(1)
            else: raise Exception()
        except Exception:
            raise AuthError("Could not get access token check cookies")
    
    async def rotate_cookies(self) -> str:
        if not (self.cookie_file.is_file() and time.time() - os.path.getmtime(self.cookie_file) <= 60):
                response = await self.client.post(
                    url=Endpoint.ROTATE_COOKIES.value,
                    headers=Headers.ROTATE_COOKIES.value,
                    data='[000,"-0000000000000000000"]',
                )
                if response.status_code == 401: raise AuthError
                response.raise_for_status()
                response = await self.client.get(url=Endpoint.INIT.value)
                response.raise_for_status()
                with open(self.cookie_file, 'w') as f:
                    cookies = []
                    for cookie in self.client.cookies.jar:
                        cookies.append({"name":cookie.name ,"value":cookie.value ,"domain":cookie.domain})
                    json.dump(cookies,f)


    async def start_auto_refresh(self) -> None:
        """
        Start the background task to automatically refresh cookies.
        """

        while True:
            try:
               await self.rotate_cookies()
            except AuthError:
                if task := rotate_tasks.get("refresh_cookies"):
                    task.cancel()
                logger.warning(
                    "Failed to refresh cookies. Background auto refresh task canceled."
                )

            logger.debug(f"Cookies refreshed")
            await asyncio.sleep(self.refresh_interval)

    @running(retry=2)
    async def generate_content(
        self,
        prompt: str,
        files: list[str | Path] | None = None,
        model: Model | str = Model.UNSPECIFIED,
        chat: Optional["ChatSession"] = None,
        **kwargs,
    ) -> ModelOutput:
        
        assert prompt, "Prompt cannot be empty."

        if not isinstance(model, Model):
            model = Model.from_name(model)

        if self.auto_close:
            await self.reset_close_task()

        try:
            response = await self.client.post(
                Endpoint.GENERATE.value,
                headers=Headers.GEMINI.value | model.model_header,
                data={
                    "at": self.access_token,
                    "f.req": json.dumps(
                        [
                            None,
                            json.dumps(
                                [
                                    files
                                    and [
                                        prompt,
                                        0,
                                        None,
                                        [
                                            [
                                                [await self.upload_file(file)],
                                                parse_file_name(file),
                                            ]
                                            for file in files
                                        ],
                                    ]
                                    or [prompt],
                                    None,
                                    chat and chat.metadata,
                                ]
                            ),
                        ]
                    ),
                },
                **kwargs,
            )
        except ReadTimeout:
            raise TimeoutError(
                "Request timed out, please try again. If the problem persists, consider setting a higher `timeout` value when initializing GeminiClient."
            )

        if response.status_code != 200:
            await self.close()
            raise APIError(
                f"Failed to generate contents. Request failed with status code {response.status_code}"
            )
        else:
            try:
                response_json = json.loads(response.text.split("\n")[2])

                body = None
                body_index = 0
                for part_index, part in enumerate(response_json):
                    try:
                        main_part = json.loads(part[2])
                        if main_part[4]:
                            body_index, body = part_index, main_part
                            break
                    except (IndexError, TypeError, ValueError):
                        continue

                if not body:
                    raise Exception
            except Exception:
                await self.close()

                try:
                    match ErrorCode(response_json[0][5][2][0][1][0]):
                        case ErrorCode.USAGE_LIMIT_EXCEEDED:
                            raise UsageLimitExceeded(
                                f"Failed to generate contents. Usage limit of {model.model_name} model has exceeded. Please try switching to another model."
                            )
                        case ErrorCode.MODEL_HEADER_INVALID:
                            raise ModelInvalid(
                                "Failed to generate contents. The specified model is not available. Please update gemini_webapi to the latest version. "
                                "If the error persists and is caused by the package, please report it on GitHub."
                            )
                        case ErrorCode.IP_TEMPORARILY_BLOCKED:
                            raise TemporarilyBlocked(
                                "Failed to generate contents. Your IP address is temporarily blocked by Google. Please try using a proxy or waiting for a while."
                            )
                        case _:
                            raise Exception
                except GeminiError:
                    raise
                except Exception:
                    logger.debug(f"Invalid response: {response.text}")
                    raise APIError(
                        "Failed to generate contents. Invalid response data received. Client will try to re-initialize on next request."
                    )

            try:
                candidates = []
                for candidate_index, candidate in enumerate(body[4]):
                    text = candidate[1][0]
                    if re.match(
                        r"^http://googleusercontent\.com/card_content/\d+$", text
                    ):
                        text = candidate[22] and candidate[22][0] or text

                    try:
                        thoughts = candidate[37][0][0]
                    except (TypeError, IndexError):
                        thoughts = None

                    web_images = (
                        candidate[12]
                        and candidate[12][1]
                        and [
                            WebImage(
                                url=web_image[0][0][0],
                                title=web_image[7][0],
                                alt=web_image[0][4],
                                proxy=self.proxy,
                            )
                            for web_image in candidate[12][1]
                        ]
                        or []
                    )

                    generated_images = []
                    if candidate[12] and candidate[12][7] and candidate[12][7][0]:
                        img_body = None
                        for img_part_index, part in enumerate(response_json):
                            if img_part_index < body_index:
                                continue

                            try:
                                img_part = json.loads(part[2])
                                if img_part[4][candidate_index][12][7][0]:
                                    img_body = img_part
                                    break
                            except (IndexError, TypeError, ValueError):
                                continue

                        if not img_body:
                            raise ImageGenerationError(
                                "Failed to parse generated images. Please update gemini_webapi to the latest version. "
                                "If the error persists and is caused by the package, please report it on GitHub."
                            )

                        img_candidate = img_body[4][candidate_index]

                        text = re.sub(
                            r"http://googleusercontent\.com/image_generation_content/\d+$",
                            "",
                            img_candidate[1][0],
                        ).rstrip()

                        generated_images = [
                            GeneratedImage(
                                url=generated_image[0][3][3],
                                title=f"[Generated Image {generated_image[3][6]}]",
                                alt=len(generated_image[3][5]) > image_index
                                and generated_image[3][5][image_index]
                                or generated_image[3][5][0],
                                proxy=self.proxy,
                                cookies=self.cookies,
                            )
                            for image_index, generated_image in enumerate(
                                img_candidate[12][7][0]
                            )
                        ]

                    candidates.append(
                        Candidate(
                            rcid=candidate[0],
                            text=text,
                            thoughts=thoughts,
                            web_images=web_images,
                            generated_images=generated_images,
                        )
                    )
                if not candidates:
                    raise GeminiError(
                        "Failed to generate contents. No output data found in response."
                    )

                output = ModelOutput(metadata=body[1], candidates=candidates)
            except (TypeError, IndexError):
                logger.debug(f"Invalid response: {response.text}")
                raise APIError(
                    "Failed to parse response body. Data structure is invalid."
                )

            if isinstance(chat, ChatSession):
                chat.last_output = output

            return output

    def start_chat(self, **kwargs) -> "ChatSession":

        return ChatSession(geminiclient=self, **kwargs)
    
    async def delete_chat(self, cid: str) -> bool:
        res = await self.client.post(
               Endpoint.DELETE.value,
               headers=Headers.GEMINI.value,
                data={
                    "at": self.access_token,
                    "f.req": json.dumps([[["GzXR5e",json.dumps([cid]),None,"generic"]]])
                },
            )
        return res.status_code == 200
    
    async def upload_file(self,file: str | Path) -> str:
        with open(file, "rb") as f:
            file = f.read()
        response = await self.client.post(
            url=Endpoint.UPLOAD.value,
            headers=Headers.UPLOAD.value,
            files={"file": file},
            )
        response.raise_for_status()
        return response.text
        
class ChatSession:

    __slots__ = [
        "__metadata",
        "geminiclient",
        "last_output",
        "model",
    ]

    def __init__(
        self,
        geminiclient: GeminiClient,
        metadata: list[str | None] | None = None,
        cid: str | None = None,  # chat id
        rid: str | None = None,  # reply id
        rcid: str | None = None,  # reply candidate id
        model: Model | str = Model.UNSPECIFIED,
    ):
        self.__metadata: list[str | None] = [None, None, None]
        self.geminiclient: GeminiClient = geminiclient
        self.last_output: ModelOutput | None = None
        self.model = model

        if metadata:
            self.metadata = metadata
        if cid:
            self.cid = cid
        if rid:
            self.rid = rid
        if rcid:
            self.rcid = rcid

    def __str__(self):
        return f"ChatSession(cid='{self.cid}', rid='{self.rid}', rcid='{self.rcid}')"

    __repr__ = __str__

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # update conversation history when last output is updated
        if name == "last_output" and isinstance(value, ModelOutput):
            self.metadata = value.metadata
            self.rcid = value.rcid

    async def send_message(
        self,
        prompt: str,
        files: list[str | Path] | None = None,
        **kwargs,
    ) -> ModelOutput:

        return await self.geminiclient.generate_content(
            prompt=prompt, files=files, model=self.model, chat=self, **kwargs
        )

    def choose_candidate(self, index: int) -> ModelOutput:

        if not self.last_output:
            raise ValueError("No previous output data found in this chat session.")

        if index >= len(self.last_output.candidates):
            raise ValueError(
                f"Index {index} exceeds the number of candidates in last model output."
            )

        self.last_output.chosen = index
        self.rcid = self.last_output.rcid
        return self.last_output

    @property
    def metadata(self):
        return self.__metadata

    @metadata.setter
    def metadata(self, value: list[str]):
        if len(value) > 3:
            raise ValueError("metadata cannot exceed 3 elements")
        self.__metadata[: len(value)] = value

    @property
    def cid(self):
        return self.__metadata[0]

    @cid.setter
    def cid(self, value: str):
        self.__metadata[0] = value

    @property
    def rid(self):
        return self.__metadata[1]

    @rid.setter
    def rid(self, value: str):
        self.__metadata[1] = value

    @property
    def rcid(self):
        return self.__metadata[2]

    @rcid.setter
    def rcid(self, value: str):
        self.__metadata[2] = value

def parse_file_name(file: str | Path) -> str:
    file = Path(file)
    if not file.is_file():
        raise ValueError(f"{file} is not a valid file.")

    return file.name
