"""Atomic files and hash-checked, marker-last stage and record commits."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager


def sha256(path):
    with path.open('rb') as stream:
        if hasattr(hashlib, 'file_digest'):
            return hashlib.file_digest(stream, 'sha256').hexdigest()
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def digest(value):
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


@contextmanager
def atomic_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    try:
        from pathlib import Path
        yield Path(name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                             encoding='utf-8')


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f'Frozen file differs: {path}')
    else:
        write_json(path, value)


def verify(path, expected):
    if not path.is_file():
        raise RuntimeError(f'Missing {path}; expected SHA256 {expected}')
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f'SHA256 mismatch: {path}; expected {expected}, actual {actual}')


class Journal:
    def __init__(self, root, identity):
        self.root = root
        self.identity = identity

    def marker(self, name):
        return self.root / 'state' / 'markers' / f'{name}.json'

    def get(self, name):
        marker = self.marker(name)
        if not marker.exists():
            return None
        data = read_json(marker)
        if data['identity'] != self.identity:
            raise RuntimeError(f'Source/input SHA256 mismatch: {marker}')
        for filename, expected in data['files'].items():
            verify(self.root / filename, expected)
        return data['result']

    def finish(self, name, files, result):
        freeze(self.marker(name), {
            'identity': self.identity,
            'files': {path.relative_to(self.root).as_posix(): sha256(path) for path in files},
            'result': result,
        })
        return result

    def require(self, name):
        result = self.get(name)
        if result is None:
            raise RuntimeError(f'Finish stage {name} first')
        return result


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.ogsc.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
