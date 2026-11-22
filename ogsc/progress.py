"""Consistent progress reporting for long OGSC stages."""

import sys
import threading
import time
from contextlib import contextmanager


def progress_line(label, completed, total, elapsed_seconds):
    total = max(int(total), 1)
    completed = min(max(int(completed), 0), total)
    elapsed_seconds = max(float(elapsed_seconds), 0.0)
    if completed == 0:
        eta = 'ETA warming up'
    else:
        eta = f'ETA {elapsed_seconds * (total - completed) / completed / 60:.1f} min'
    return (f'OGSC {label}: {completed}/{total} ({completed / total:.0%}) | '
            f'{elapsed_seconds / 60:.1f} min elapsed | {eta}')


@contextmanager
def stage_progress(label, interval_seconds=60):
    """Announce stage boundaries and periodically heartbeat while work runs."""
    started = time.perf_counter()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(interval_seconds):
            elapsed = time.perf_counter() - started
            print(f'OGSC {label}: still running | {elapsed / 60:.1f} min elapsed',
                  file=sys.stdout, flush=True)

    print(f'OGSC {label}: started.', flush=True)
    thread = threading.Thread(target=heartbeat, name='ogsc-progress', daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        stop.set()
        thread.join()
        elapsed = time.perf_counter() - started
        print(f'OGSC {label}: failed after {elapsed / 60:.1f} min.', flush=True)
        raise
    else:
        stop.set()
        thread.join()
        elapsed = time.perf_counter() - started
        print(f'OGSC {label}: completed in {elapsed / 60:.1f} min.', flush=True)
