"""Semantic similarity reward between rendered and ground-truth figures.

TikZilla uses a DeTikZify image encoder with an Earth Mover's Distance over
patch embeddings. As a prototype stand-in we use SigLIP patch embeddings
(DeTikZify's encoder is SigLIP-derived) with a relaxed EMD: each patch is
matched to its best counterpart in the other image, averaged both ways.
Swap `SigLIPReward` for the DeTikZify-V2 encoder later without touching the
env — the interface is just (image, gt_image) -> float in [0, 1].

Requires the `reward` extra: uv sync --extra reward
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def relaxed_emd_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """x: (n, d), y: (m, d) L2-normalized patch embeddings -> [0, 1]-ish."""
    sim = x @ y.T  # cosine similarities, (n, m)
    forward = sim.max(axis=1).mean()   # each x-patch to best y-patch
    backward = sim.max(axis=0).mean()  # each y-patch to best x-patch
    return float((forward + backward) / 2)


class SigLIPReward:
    def __init__(self, model_id: str = "google/siglip-so400m-patch14-384",
                 device: str = "mps"):
        import torch
        from transformers import SiglipVisionModel, SiglipImageProcessor

        self.torch = torch
        self.device = device
        self.processor = SiglipImageProcessor.from_pretrained(model_id)
        self.model = (
            SiglipVisionModel.from_pretrained(model_id, torch_dtype=torch.float16)
            .to(device)
            .eval()
        )

    def _embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            patches = self.model(**inputs).last_hidden_state[0]  # (n_patches, d)
        patches = patches / patches.norm(dim=-1, keepdim=True)
        return patches.float().cpu().numpy()

    def __call__(self, image: Image.Image, gt_image: Image.Image) -> float:
        return relaxed_emd_similarity(self._embed(image), self._embed(gt_image))
