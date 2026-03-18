"""
CLOOMEEncoder: Pretrained CLOOME ResNet50 visual encoder for fine-tuning.

Loads the CLOOME checkpoint (pretrained on Cell Painting images + molecular
structures via contrastive learning), extracts the ResNet50 visual backbone,
and exposes encode_image() -> 512-D embeddings.

All backbone layers are unfrozen for full fine-tuning with a low learning rate.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add cloome to path so we can import CLIPGeneral
_CLOOME_DIR = str(Path(__file__).resolve().parent.parent.parent / "cloome" / "src")
if _CLOOME_DIR not in sys.path:
    sys.path.insert(0, _CLOOME_DIR)

from clip.model import CLIPGeneral  # noqa: E402


# CLOOME channel order: [Mito, ERSyto, ERSytoBleed, Ph_golgi, Hoechst]
# MFCP channel order:   [ERSyto, ERSytoBleed, Hoechst, Mito, Ph_golgi]
# Reorder indices: take MFCP[3, 0, 1, 4, 2] to get CLOOME order
MFCP_TO_CLOOME_CHANNEL_ORDER = [3, 0, 1, 4, 2]

# CLOOME normalization (mean == std, per-channel, in CLOOME channel order)
CLOOME_MEAN = (47.1314, 40.8138, 53.7692, 46.2656, 28.7243)
CLOOME_STD = (47.1314, 40.8138, 53.7692, 46.2656, 28.7243)


def illumination_threshold(arr: np.ndarray, perc: float = 0.01) -> float:
    """Return threshold to clip top perc% of brightest pixels (CLOOME-style)."""
    perc = perc / 100
    total_pixels = arr.shape[0] * arr.shape[1]
    n_pixels = max(int(np.around(total_pixels * perc)), 1)
    flat_inds = np.argpartition(arr.ravel(), -n_pixels)[-n_pixels:]
    threshold = arr.ravel()[flat_inds].min()
    return float(threshold)


def sixteen_to_eight_bit(arr: np.ndarray, display_max: float, display_min: float = 0) -> np.ndarray:
    """Convert 16-bit image to 8-bit using illumination thresholding (CLOOME-style)."""
    threshold_image = (arr.astype(np.float32) - display_min) * (arr > display_min)
    if display_max - display_min > 0:
        scaled_image = threshold_image * (255.0 / (display_max - display_min))
    else:
        scaled_image = threshold_image
    scaled_image = np.clip(scaled_image, 0, 255).astype(np.uint8)
    return scaled_image


class CLOOMEEncoder(nn.Module):
    """Pretrained CLOOME ResNet50 visual encoder for full fine-tuning.

    Produces (B, 512) embeddings from (B, 5, H, W) microscopy images.
    All backbone parameters are unfrozen for fine-tuning.
    """

    def __init__(
        self,
        config_path: str = "/data/huadi/cloome/src/training/model_configs/RN50.json",
        checkpoint_path: str | None = None,
    ):
        super().__init__()

        # Load config
        with open(config_path, "r") as f:
            model_info = json.load(f)

        # CLIPGeneral defaults backbone_architecture to ['ResNet', 'MLP']
        # We only need the visual (ResNet) part
        model = CLIPGeneral(**model_info)

        # Load pretrained checkpoint
        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download("anasanchezf/cloome", "cloome-bioactivity.pt")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["state_dict"]

        # Strip "module." prefix from DDP checkpoint keys
        clean_sd = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(clean_sd)

        # Extract only the visual backbone (ResNet50)
        self.visual = model.visual
        self.embed_dim = 512  # output dim from RN50.json

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 5, H, W) float tensor — preprocessed (CLOOME channel order,
                    illumination-thresholded, normalized).
        Returns:
            (B, 512) image embeddings.
        """
        return self.visual(images)

    def backbone_parameters(self):
        """Yield all backbone parameters (for optimizer group)."""
        return self.visual.parameters()

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_frozen_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
