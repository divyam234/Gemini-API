from asyncio import Task

from .upload_file import upload_file, parse_file_name  # noqa: F401
from .logger import logger, set_log_level  # noqa: F401


rotate_tasks: dict[str, Task] = {}
