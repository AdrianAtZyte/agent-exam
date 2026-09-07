from __future__ import annotations

import contextlib
import os
import signal
import subprocess

_DEAD_PG = (ProcessLookupError, PermissionError)
_WINDOWS = os.name == "nt"


def _kill_tree(process: subprocess.Popen) -> None:
    if _WINDOWS:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    with contextlib.suppress(_DEAD_PG):
        os.killpg(process.pid, signal.SIGKILL)


def terminate_tree(process: subprocess.Popen, sigterm_timeout: float = 5.0) -> None:
    """Terminate a process and its entire process group.

    Sends SIGTERM first, waits up to `sigterm_timeout` seconds, then
    escalates to SIGKILL. Handles races where the process group has
    already exited (ProcessLookupError, PermissionError on macOS).
    Windows has no process groups or SIGKILL, so the graceful step calls
    `Popen.terminate()` and the escalation shells out to `taskkill /T /F`
    to reach the whole tree.

    Pass `sigterm_timeout=0` to skip SIGTERM and kill immediately.
    """
    if sigterm_timeout > 0:
        if _WINDOWS:
            with contextlib.suppress(OSError):
                process.terminate()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except _DEAD_PG:
                return
        try:
            process.wait(timeout=sigterm_timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    _kill_tree(process)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


def wait_or_terminate(process: subprocess.Popen, timeout_seconds: float) -> bool:
    """Wait for `process`, terminating its tree on timeout or on interruption.

    Returns whether the wait timed out. `process` runs in its own session
    (`start_new_session=True`) so a per-attempt timeout can kill just this
    tree without touching sibling attempts — but that same isolation means
    nothing else will ever signal it if the wait here is abandoned instead
    of completing normally, e.g. by Ctrl+C. An interrupt kills the tree
    immediately (no SIGTERM grace period) before re-raising, so the run
    stops as fast as the signal itself rather than leaving the harness
    subprocess running unattended.
    """
    try:
        process.wait(timeout=timeout_seconds)
        return False
    except subprocess.TimeoutExpired:
        terminate_tree(process)
        return True
    except BaseException:
        terminate_tree(process, sigterm_timeout=0)
        raise
