import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


class ImageIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.harness = root / 'BEST_TEACHER_32bits_refine/harness'
        sys.path.insert(0, str(cls.harness))
        utility_path = cls.harness / 'utility.py'
        spec = importlib.util.spec_from_file_location('_ogsc_test_utility', utility_path)
        cls.utility = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.utility)

    def test_utility_cache_identity_uses_file_metadata_not_content_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / 'sample.png'
            image.write_bytes(b'png-like-test-data')

            identity = self.utility.image_file_identity([image])

        self.assertEqual(identity[0]['name'], 'sample.png')
        self.assertEqual(identity[0]['size'], len(b'png-like-test-data'))
        self.assertIn('mtime_ns', identity[0])
        self.assertNotIn('sha256', identity[0])

    def test_utility_finds_four_digit_ogsc_png_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'sample_0001.png').write_bytes(b'b')
            (folder / 'sample_0000.png').write_bytes(b'a')

            images = self.utility.ordered_image_files(folder, expected_count=2)

        self.assertEqual([path.name for path in images], ['sample_0000.png', 'sample_0001.png'])


if __name__ == '__main__':
    unittest.main()
