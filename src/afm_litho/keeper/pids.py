"""Whether a pid is a running process."""

from __future__ import annotations

import ctypes
import os
import sys


def pid_alive(pid: int) -> bool:
    """Whether a process with this pid is running. Never signals it."""
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        return bool(ok) and code.value == 259           # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


class ProcessHandle:
    """An open handle to one process, so its exit is seen even if its pid is reused.

    A bare pid lookup says "alive" for a different process that took the number; a handle
    names the one process it was opened on. Where handles are unavailable (POSIX) it falls
    back to the pid lookup."""

    def __init__(self, pid: int):
        self.pid = pid
        self._handle = None
        if sys.platform == "win32":
            kernel32 = ctypes.windll.kernel32
            kernel32.OpenProcess.restype = ctypes.c_void_p
            # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
            self._handle = kernel32.OpenProcess(0x100000 | 0x1000, False, pid)

    @property
    def opened(self) -> bool:
        return sys.platform != "win32" or bool(self._handle)

    def exited(self) -> bool:
        """True when the process is gone, or could not be opened (the safe direction)."""
        if sys.platform != "win32":
            return not pid_alive(self.pid)
        if not self._handle:
            return True
        kernel32 = ctypes.windll.kernel32
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        # Only a timeout means still running; WAIT_FAILED is not "alive".
        return kernel32.WaitForSingleObject(ctypes.c_void_p(self._handle), 0) != 258

    def exit_code(self) -> int | None:
        """The process's exit code, or None while it runs. Where it cannot be read (POSIX,
        or a handle that did not open) a process that is gone reports 1."""
        if not self.exited():
            return None
        if sys.platform == "win32" and self._handle:
            code = ctypes.c_ulong()
            kernel32 = ctypes.windll.kernel32
            kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            if kernel32.GetExitCodeProcess(ctypes.c_void_p(self._handle), ctypes.byref(code)):
                return int(code.value)
        return 1

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))
