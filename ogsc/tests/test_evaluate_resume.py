import tempfile
import unittest
from pathlib import Path

from ogsc.state import write_json


class EvaluationResumeTests(unittest.TestCase):
    def test_valid_score_rows_match_expected_image_order_and_coordinates(self):
        from ogsc.evaluate import valid_score_rows

        metadata = [{'file_name': 'sample_0000.png', 'text': 'a'},
                    {'file_name': 'sample_0001.png', 'text': 'b'}]
        scores = [{'file': row['file_name'], 'coordinates': [0.0] * 32} for row in metadata]

        self.assertTrue(valid_score_rows(scores, metadata))
        self.assertFalse(valid_score_rows(scores[:1], metadata))
        self.assertFalse(valid_score_rows([dict(scores[0], file='wrong.png'), scores[1]], metadata))

    def test_score_checkpoint_resumes_only_when_identity_matches(self):
        from ogsc.evaluate import load_score_checkpoint

        metadata = [{'file_name': 'sample_0000.png', 'text': 'a'},
                    {'file_name': 'sample_0001.png', 'text': 'b'}]
        scores = [{'file': metadata[0]['file_name'], 'coordinates': [0.1] * 32}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'partial.json'
            write_json(path, {'identity': {'generation': 'one'}, 'rows': scores})

            restored = load_score_checkpoint(path, {'generation': 'one'}, metadata)
            stale = load_score_checkpoint(path, {'generation': 'two'}, metadata)

        self.assertEqual(restored, scores)
        self.assertEqual(stale, [])


if __name__ == '__main__':
    unittest.main()
