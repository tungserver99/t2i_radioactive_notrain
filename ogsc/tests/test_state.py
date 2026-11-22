import hashlib
import tempfile
import unittest
from pathlib import Path


class StateHashTests(unittest.TestCase):
    def test_sha256_hashes_file_on_python_310(self):
        from ogsc.state import sha256

        payload = b'OGSC resume hash compatibility'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'payload.bin'
            path.write_bytes(payload)

            actual = sha256(path)

        self.assertEqual(actual, hashlib.sha256(payload).hexdigest())


if __name__ == '__main__':
    unittest.main()
