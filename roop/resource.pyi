from typing import Any, Tuple
from sys import platform

if platform != "win32":
    RLIMIT_DATA: int
    def setrlimit(resource: int, limits: Tuple[int, int]) -> None: ...