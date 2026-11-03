import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest

from mpp_video_qa.model_loading import model_load_guard


def loading_worker(lock, entered, release):
    with model_load_guard(lock):
        Path(entered).touch()
        deadline = time.monotonic() + 20
        while not Path(release).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("test worker was not released")
            time.sleep(.02)


def wait_for_file(path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(.02)
    return path.exists()


class ModelLoadingTests(unittest.TestCase):
    def run_workers(self, terminate_first=False):
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            lock = str(Path(directory) / "loading.lock")
            first_entered, next_entered = (Path(directory) / name for name in ("entered1", "entered2"))
            first_release, next_release = (Path(directory) / name for name in ("release1", "release2"))
            workers = [ctx.Process(target=loading_worker, args=(lock, first_entered, first_release)),
                       ctx.Process(target=loading_worker, args=(lock, next_entered, next_release))]
            try:
                workers[0].start()
                self.assertTrue(wait_for_file(first_entered, 10))
                workers[1].start()
                self.assertFalse(wait_for_file(next_entered, .5), "two CPU staging regions overlapped")
                if terminate_first:
                    workers[0].terminate()
                else:
                    first_release.touch()
                workers[0].join(10)
                self.assertFalse(workers[0].is_alive())
                self.assertTrue(wait_for_file(next_entered, 10), "lock was not released")
                next_release.touch()
                workers[1].join(10)
                self.assertEqual(workers[1].exitcode, 0)
                self.assertEqual(workers[0].exitcode, -15 if terminate_first else 0)
            finally:
                first_release.touch()
                next_release.touch()
                for worker in workers:
                    if worker.pid is not None:
                        if worker.is_alive():
                            worker.terminate()
                        worker.join(5)

    def test_concurrent_workers_wait_until_cpu_staging_finishes(self):
        self.run_workers()

    def test_sigterm_does_not_leave_a_stale_lock(self):
        self.run_workers(terminate_first=True)

    def test_exception_releases_lock_for_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "loading.lock"
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                with model_load_guard(lock):
                    raise RuntimeError("load failed")
            with model_load_guard(lock):
                pass


if __name__ == "__main__":
    unittest.main()
