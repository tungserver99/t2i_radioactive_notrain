"""Load and reuse the frozen carrier, teacher, and oracle source modules."""

import importlib
import importlib.util
import sys
from pathlib import Path

from .paths import METRICS, WEIGHT, oracle_directory


def _load(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sources:
    def __init__(self, paths, pinned=None):
        weight = Path(paths['weight32_repo']) / WEIGHT
        self.harness = weight / 'harness'
        self.oracle = oracle_directory(Path(paths['oracle_repo']))
        sys.path[:0] = [str(self.harness), str(self.oracle)]
        for name in ('semantic_carrier', 'semantic_teacher', 'semantic_sign_train'):
            module = importlib.import_module(name)
            expected = self.harness / f'{name}.py'
            if Path(module.__file__).resolve() != expected.resolve():
                raise RuntimeError(f'Import collision for {name}: {module.__file__}')
            setattr(self, name, module)
        self.oracle_math = _load('_ogsc_oracle_math', self.oracle / 'semantic_oracle_math.py')
        self.decoder = _load('_ogsc_oracle_decoder', self.oracle / 'semantic_sign_train.py')
        self.key_metrics = _load('_ogsc_original_key_metrics', weight / METRICS)

    def carrier(self, folder, device='cpu'):
        carrier = self.semantic_carrier.SemanticCarrier.load(folder, device)
        if (carrier.signs.shape != (32,) or carrier.carrier_version is not None
                or carrier.medians is not None):
            raise RuntimeError('Expected original threshold-based 32-bit carrier')
        return carrier

    def extractor(self, teacher_clean, device='cuda'):
        import torch
        from diffusers import StableDiffusionPipeline

        self.semantic_teacher.numerical_policy()
        pipe = StableDiffusionPipeline.from_pretrained(
            str(teacher_clean), torch_dtype=torch.float32,
            safety_checker=None, local_files_only=True)
        return self.semantic_carrier.SemanticExtractor(pipe, device=device)

    def signed_view(self, extractor, carrier, prompt):
        import torch

        hidden = extractor.text([prompt])

        def view(latent):
            image = self.decoder.decoded_image(extractor, latent)
            image = self.oracle_math.fake_quantize_rgb8_ste(image)
            coordinates = self.decoder.decoded_coordinates(extractor, carrier, image, hidden)
            return coordinates * carrier.signs.to(coordinates.device).view(1, 32)
        return view

    @staticmethod
    def verify_frozen(extractor):
        for module in (extractor.vae, extractor.unet, extractor.text_encoder):
            if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise RuntimeError('Frozen semantic model has trainable parameters or gradients')
