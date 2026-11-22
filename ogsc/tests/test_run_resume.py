import tempfile
import unittest
from pathlib import Path

from ogsc.state import read_json, sha256, write_json


class FullRunResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.output = self.root / 'run'
        self.source.mkdir()
        self.output.mkdir()
        write_json(self.source / 'prompts.json', {'validation': [{'text': 'v'}]})
        self.paths = {'source_run': str(self.source), 'output_run': str(self.output)}

    def tearDown(self):
        self.temp.cleanup()

    def test_full_reuses_positive_gamma_for_same_validation_prompts(self):
        from ogsc.run import _gamma_is_reusable

        write_json(self.output / 'calibration/gamma.json', {
            'status': 'selected', 'gamma': .123,
            'validation_prompts_sha256': sha256(self.source / 'prompts.json')})
        self.assertEqual(_gamma_is_reusable(self.paths)['gamma'], .123)

    def test_full_rejects_changed_prompts_or_lost_gamma_after_generation_started(self):
        from ogsc.run import _gamma_is_reusable

        write_json(self.output / 'calibration/gamma.json', {
            'status': 'selected', 'gamma': .123, 'validation_prompts_sha256': 'stale'})
        with self.assertRaisesRegex(RuntimeError, 'validation prompts changed'):
            _gamma_is_reusable(self.paths)
        (self.output / 'calibration/gamma.json').unlink()
        image_dir = self.output / 'images/teacher_clean_train'
        image_dir.mkdir(parents=True)
        (image_dir / 'sample_0000.png').write_bytes(b'partial run marker')
        with self.assertRaisesRegex(RuntimeError, 'gamma artifact is missing'):
            _gamma_is_reusable(self.paths)

    def test_stage_updates_preserve_completed_work(self):
        from ogsc.run import _update_stage

        write_json(self.output / 'state/stages.json', {'prepare': {'status': 'complete'},
                                                        'tune_gamma': {'status': 'complete'}})
        _update_stage(self.paths, 'generate_train', {'samples': 3345})
        stages = read_json(self.output / 'state/stages.json')
        self.assertEqual(set(stages), {'prepare', 'tune_gamma', 'generate_train'})
        self.assertEqual(stages['generate_train']['result']['samples'], 3345)


if __name__ == '__main__':
    unittest.main()
