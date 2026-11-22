import unittest
import tempfile
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path

import torch
from PIL import Image


class _FakeVAE:
    def __init__(self):
        self.config = SimpleNamespace(scaling_factor=1.0)
        self.calls = 0

    def decode(self, latent):
        self.calls += 1
        return SimpleNamespace(sample=latent[:, :3])


class _FakePipe:
    def __init__(self):
        self.vae = _FakeVAE()
        self.image_processor = SimpleNamespace(
            postprocess=lambda _tensor, output_type: [Image.new('RGB', (2, 2), 'white')])


class _FakeSources:
    def __init__(self, model):
        self.model = model

    def signed_view(self, _extractor, _carrier, _prompt):
        return lambda latent: self.model(latent.flatten(1))


class DirectGenerationTests(unittest.TestCase):
    def test_validation_clean_latent_reuses_gamma_tuning_cache(self):
        from ogsc.generation import clean_record_path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_records = root / 'state' / 'validation'
            gamma_records = root / 'calibration' / 'validation' / 'latents'
            gamma_records.mkdir(parents=True)
            cached = gamma_records / '0000.pt'
            torch.save({'latent': torch.zeros(1), 'image_id': 3, 'text': 'prompt', 'seed': 9}, cached)

            selected = clean_record_path(state_records, gamma_records, 'validation', 0)
            train_selected = clean_record_path(state_records, gamma_records, 'train', 0)

        self.assertEqual(selected, cached)
        self.assertEqual(train_selected, state_records / '0000.pt')

    def test_validation_reuses_selected_gamma_trial_images(self):
        from ogsc.generation import reuse_selected_validation_images

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            trial = output / 'calibration' / 'gamma_trials' / '0.1250000000'
            target = output / 'images' / 'teacher_ogsc_validation'
            trial.mkdir(parents=True)
            target.mkdir(parents=True)
            for index in range(2):
                Image.new('RGB', (4, 4), ('red', 'green')[index]).save(
                    trial / f'sample_{index:04d}.png')

            copied = reuse_selected_validation_images(
                output, target, [{'text': 'one'}, {'text': 'two'}], gamma=.125)

            self.assertEqual(copied, 2)
            for index in range(2):
                self.assertEqual((target / f'sample_{index:04d}.png').read_bytes(),
                                 (trial / f'sample_{index:04d}.png').read_bytes())

    def test_validation_reuse_repairs_corrupt_existing_copy(self):
        from ogsc.generation import reuse_selected_validation_images

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            trial = output / 'calibration' / 'gamma_trials' / '0.1250000000'
            target = output / 'images' / 'teacher_ogsc_validation'
            trial.mkdir(parents=True)
            target.mkdir(parents=True)
            source = trial / 'sample_0000.png'
            destination = target / 'sample_0000.png'
            Image.new('RGB', (4, 4), 'red').save(source)
            valid_bytes = source.read_bytes()
            destination.write_bytes(bytes([valid_bytes[0] ^ 0xFF]) + valid_bytes[1:])

            copied = reuse_selected_validation_images(
                output, target, [{'text': 'one'}], gamma=.125)
            with Image.open(destination) as repaired:
                repaired.verify()

        self.assertEqual(copied, 1)

    def test_validation_reuse_reports_cache_scan_progress(self):
        from ogsc.generation import reuse_selected_validation_images

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            trial = output / 'calibration' / 'gamma_trials' / '0.1250000000'
            target = output / 'images' / 'teacher_ogsc_validation'
            trial.mkdir(parents=True)
            target.mkdir(parents=True)
            Image.new('RGB', (4, 4), 'blue').save(trial / 'sample_0000.png')
            (target / 'sample_0000.png').write_bytes((trial / 'sample_0000.png').read_bytes())
            stream = io.StringIO()

            with redirect_stdout(stream):
                copied = reuse_selected_validation_images(
                    output, target, [{'text': 'one'}], gamma=.125)

        self.assertEqual(copied, 0)
        self.assertIn('validation trial reuse scan', stream.getvalue())
        self.assertIn('1/1', stream.getvalue())


    def test_cached_control_latent_is_reused_only_for_same_sample_and_gamma(self):
        from ogsc.generation import validate_marked_record

        record = {'latent': torch.zeros(1, 4, 2, 2), 'image_id': 7,
                  'text': 'blue house', 'seed': 19, 'gamma': .05}
        validate_marked_record(record, {'image_id': 7, 'text': 'blue house'}, 19, .05)
        with self.assertRaisesRegex(RuntimeError, 'gamma mismatch'):
            validate_marked_record(record, {'image_id': 7, 'text': 'blue house'}, 19, .06)
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
            validate_marked_record(record, {'image_id': 8, 'text': 'blue house'}, 19, .05)

    def test_decode_progress_estimates_only_uncached_work(self):
        from ogsc.generation import generation_progress_message

        message = generation_progress_message('train', 2900, 5000, 600,
                                             work_completed=20, work_total=100)
        self.assertIn('2900/5000', message)
        self.assertIn('20/100 this run', message)
        self.assertIn('ETA 40.0 min', message)

    def test_generation_metadata_does_not_hash_png_files(self):
        from ogsc.generation import write_image_metadata

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'sample_0000.png').write_bytes(b'png-like-test-data')
            write_image_metadata(folder, [{'text': 'prompt'}])

            self.assertFalse((folder / 'image_hashes.json').exists())
            self.assertEqual((folder / 'metadata.jsonl').read_text(encoding='utf-8'),
                             '{"file_name": "sample_0000.png", "text": "prompt"}\n')

    def test_evaluation_reads_metadata_without_per_image_hashes(self):
        from ogsc.evaluate import load_image_metadata
        from ogsc.generation import write_image_metadata

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'sample_0000.png').write_bytes(b'png-like-test-data')
            write_image_metadata(folder, [{'text': 'prompt'}])

            rows = load_image_metadata(folder, [{'text': 'prompt'}])

        self.assertEqual(rows, [{'file_name': 'sample_0000.png', 'text': 'prompt'}])

    def test_final_latent_is_controlled_then_decoded_once_without_clean_image(self):
        from ogsc.generation import infer_one

        torch.manual_seed(13)
        model = torch.nn.Linear(12, 32, bias=False).eval().requires_grad_(False)
        pipe = _FakePipe()
        latent = torch.randn(1, 3, 2, 2)
        image, marked, info = infer_one(pipe, _FakeSources(model), object(), object(),
                                         'a test prompt', latent, gamma=1e-4)
        self.assertEqual(pipe.vae.calls, 1)
        self.assertIsInstance(image, Image.Image)
        self.assertFalse(torch.equal(marked, latent))
        self.assertIsNotNone(info['loss_before'])


if __name__ == '__main__':
    unittest.main()
