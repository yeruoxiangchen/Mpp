"""Serialize CPU checkpoint staging until each model has moved to its GPU."""
from contextlib import contextmanager
import ctypes
import fcntl
import gc
import json
import os
from pathlib import Path
import time


def memory_snapshot():
    result = {}
    for filename, fields in (("/proc/meminfo", ("MemAvailable", "SwapFree")),
                             ("/proc/self/status", ("VmRSS",))):
        try:
            for line in Path(filename).read_text().splitlines():
                name, _, value = line.partition(":")
                if name in fields:
                    result[name + "_gib"] = round(int(value.split()[0]) / 1024**2, 3)
        except (OSError, ValueError):
            pass
    return result


def release_cpu_staging():
    gc.collect()
    # Return freed CPU allocations to the OS before the next worker loads.
    # This only releases unused allocator pages; live tensors are unchanged.
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    trim(0)


@contextmanager
def model_load_guard(lock_path):
    """Hold a process lock across construction AND .to(device), never inference.

    Kernel file locks are released automatically if a worker is terminated.
    An empty path preserves the caller's existing loading behavior.
    """
    if not lock_path:
        yield
        return
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(stage):
        print(json.dumps(dict(stage=stage, pid=os.getpid(), lock=str(path),
                              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                              updated_unix_seconds=time.time(), **memory_snapshot())), flush=True)

    with path.open("a+") as handle:
        emit("model_load_wait")
        fcntl.flock(handle, fcntl.LOCK_EX)
        succeeded = False
        try:
            emit("model_load_acquired")
            yield
            succeeded = True
        finally:
            try:
                release_cpu_staging()
                emit("model_load_released" if succeeded else "model_load_failed")
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
