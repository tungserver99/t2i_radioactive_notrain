"""Consistent progress reporting for long OGSC stages."""

import sys
import threading
import time
from contextlib import contextmanager
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path


@contextmanager
def tee_run_log(path, command):
    """Append stdout/stderr to a run log while keeping live console output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    lock = threading.Lock()

    class Tee:
        def __init__(self, console):
            self.console = console
            self.encoding = getattr(console, 'encoding', 'utf-8')

        def write(self, value):
            with lock:
                self.console.write(value)
                log.write(value)
                self.flush()
            return len(value)

        def flush(self):
            self.console.flush()
            log.flush()

        def isatty(self):
            return self.console.isatty()

        def fileno(self):
            return self.console.fileno()

    with path.open('a', encoding='utf-8', buffering=1) as log:
        started = datetime.now().astimezone().isoformat(timespec='seconds')
        header = f'\n===== OGSC {command} started {started} =====\n'
        original_stdout.write(header)
        original_stdout.flush()
        log.write(header)
        with redirect_stdout(Tee(original_stdout)), redirect_stderr(Tee(original_stderr)):
            try:
                yield path
            finally:
                ended = datetime.now().astimezone().isoformat(timespec='seconds')
                print(f'===== OGSC {command} log segment ended {ended} =====', flush=True)


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
