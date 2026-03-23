"""
CellPaintingImageDataset: Loads 5-channel microscopy TIFF images.

Supports four modes:
  - Single FOV: one FOV per well → (5, 256, 256)
  - Multi FOV:  all FOVs per well → (max_sites, 5, 256, 256) + mask
  - Stacked FOV: all FOVs tiled into 2×3 grid → (5, 512, 768) with
    CLOOME-style preprocessing (illumination threshold + 16→8bit + channel reorder)
  - CLOOME Multi FOV: per-FOV at native resolution (e.g. 520×696) →
    (max_sites, 5, H, W) + mask, with CLOOME preprocessing (no crop/resize)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


# Channel column names in the split CSV, in order
CHANNEL_COLS = [
    "IMG_ERSyto",
    "IMG_ERSytoBleed",
    "IMG_Hoechst",
    "IMG_Mito",
    "IMG_Ph_golgi",
]
NUM_CHANNELS = len(CHANNEL_COLS)

# Per-channel stats computed over cpg0012 training set (CLOOME channel order:
# Mito, ERSyto, ERSytoBleed, Ph_golgi, Hoechst) after illumination
# thresholding + 16→8bit conversion.
CPG0012_MEAN = (47.6715, 37.9971, 49.1904, 50.7450, 26.2426)
CPG0012_STD = (25.7989, 24.7732, 29.0714, 25.7660, 30.6230)

# CLOOME inference resolution (center crop size matching pretraining)
CLOOME_CROP_SIZE = 224


def _load_tiff(path: str, target_size: int = 256) -> np.ndarray:
    """Load a single TIFF file and center-crop/pad to target_size x target_size.

    Returns uint8 numpy array of shape (H, W).
    """
    img = Image.open(path)
    arr = np.array(img)

    # Handle various dtypes — convert to float then scale to uint8 range
    if arr.dtype == np.uint16:
        arr = (arr.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
    elif arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = arr.clip(0, 255).astype(np.uint8)
    # else: assume uint8

    h, w = arr.shape[:2]
    if arr.ndim == 3:
        arr = arr[:, :, 0]  # take first channel if multi-channel

    # Center-crop or zero-pad to target_size
    if h >= target_size and w >= target_size:
        # Center crop
        top = (h - target_size) // 2
        left = (w - target_size) // 2
        arr = arr[top : top + target_size, left : left + target_size]
    else:
        # Pad
        out = np.zeros((target_size, target_size), dtype=arr.dtype)
        y_off = (target_size - h) // 2
        x_off = (target_size - w) // 2
        out[y_off : y_off + h, x_off : x_off + w] = arr[:target_size, :target_size]
        arr = out

    return arr


class CellPaintingImageDataset(Dataset):
    """Dataset that loads 5-channel Cell Painting TIFF images.

    Supports three modes:
      - multi_fov=False, stacked_fov=False (default): Returns one FOV per well,
        shape (5, 256, 256). Returns: (image, x_chem, y, compound_label)
      - multi_fov=True: Returns all FOVs per well, zero-padded to max_sites.
        Returns: (images, fov_mask, x_chem, y, compound_label)
      - stacked_fov=True: Returns all FOVs tiled into a 2×3 grid with
        CLOOME-style preprocessing. Shape (5, 512, 768).
        Returns: (stacked_image, x_chem, y, compound_label)

    Args:
        split_csv: Path to split CSV with columns PLATE_ID, WELL_POSITION, SITE,
                   INCHIKEY, IMG_ERSyto, IMG_ERSytoBleed, IMG_Hoechst, IMG_Mito,
                   IMG_Ph_golgi.
        chembert_dict: {inchikey: np.ndarray(768,)} mapping.
        label_dict: {inchikey: np.ndarray(num_tasks,)} mapping.
        image_root: Root directory for images (images at {image_root}/{PLATE_ID}/{filename}.tif).
        multi_fov: If True, load all FOVs per well; if False, load only site 1.
        stacked_fov: If True, stack all FOVs into a 2×3 grid with CLOOME preprocessing.
        max_sites: Maximum number of FOVs per well when multi_fov=True or stacked_fov=True.
        augment_chem: Whether to augment ChemBERT features.
        chem_noise_std: Gaussian noise std for ChemBERT augmentation.
        chem_dropout_prob: Feature dropout probability for ChemBERT.
    """

    def __init__(
        self,
        split_csv: str,
        chembert_dict: dict,
        label_dict: dict,
        image_root: str = "/data/huadi/cpg0012-wawer/images",
        multi_fov: bool = False,
        stacked_fov: bool = False,
        cloome_multi_fov: bool = False,
        max_sites: int = 6,
        image_size: int | None = None,
        augment_chem: bool = False,
        chem_noise_std: float = 0.1,
        chem_dropout_prob: float = 0.0,
    ):
        self.image_root = image_root
        self.multi_fov = multi_fov
        self.stacked_fov = stacked_fov
        self.cloome_multi_fov = cloome_multi_fov
        self.max_sites = max_sites
        self.image_size = image_size
        self.augment_chem = augment_chem
        self.chem_noise_std = chem_noise_std
        self.chem_dropout_prob = chem_dropout_prob

        # Load and filter the split CSV
        df = pd.read_csv(split_csv)

        # Check that all required image columns exist
        for col in CHANNEL_COLS:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in split CSV")

        # Filter to rows that have valid ChemBERT features and labels
        valid_mask = df["INCHIKEY"].apply(
            lambda ik: str(ik) in chembert_dict and str(ik) in label_dict
        )
        df = df[valid_mask].reset_index(drop=True)

        # Filter out rows whose plate directory doesn't exist on disk
        before = len(df)
        df = df[df["PLATE_ID"].apply(
            lambda pid: os.path.isdir(os.path.join(image_root, str(pid)))
        )].reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows (plate dir missing on disk)")

        if multi_fov or stacked_fov or cloome_multi_fov:
            # Sort by SITE so FOVs are ordered within each well
            if "SITE" in df.columns:
                df = df.sort_values("SITE", ascending=True)

            # Group rows by well (PLATE_ID, WELL_POSITION)
            # Each group contains 1–max_sites rows (one per FOV/site)
            self.well_groups = []
            for (plate_id, well_pos), group_df in df.groupby(
                ["PLATE_ID", "WELL_POSITION"], sort=False
            ):
                self.well_groups.append(group_df.reset_index(drop=True))

            self.chembert_dict = chembert_dict
            self.label_dict = label_dict

            # Build compound label mapping
            all_iks = [str(g.iloc[0]["INCHIKEY"]) for g in self.well_groups]
            unique_iks = sorted(set(all_iks))
            self.ik_to_label = {ik: i for i, ik in enumerate(unique_iks)}

            # Stats
            n_sites = [len(g) for g in self.well_groups]
            n_full = sum(1 for s in n_sites if s == max_sites)
            n_partial = len(n_sites) - n_full
            print(
                f"CellPaintingImageDataset: {len(self.well_groups)} wells "
                f"({n_full} with {max_sites} sites, {n_partial} with fewer), "
                f"{len(unique_iks)} unique compounds"
            )
        else:
            # Legacy mode: one FOV per well (take first site)
            if "SITE" in df.columns:
                df = df.sort_values("SITE", ascending=True)
                df = df.groupby(["PLATE_ID", "WELL_POSITION"]).first().reset_index()

            self.df = df
            self.chembert_dict = chembert_dict
            self.label_dict = label_dict

            # Build compound label mapping
            unique_iks = sorted(df["INCHIKEY"].astype(str).unique())
            self.ik_to_label = {ik: i for i, ik in enumerate(unique_iks)}

            print(f"CellPaintingImageDataset: {len(self.df)} samples, "
                  f"{len(unique_iks)} unique compounds")

    def __len__(self) -> int:
        if self.multi_fov or self.stacked_fov or self.cloome_multi_fov:
            return len(self.well_groups)
        return len(self.df)

    def _load_fov_raw(self, row) -> np.ndarray:
        """Load 5 channels for a single FOV row as raw uint16/uint8 arrays.

        Returns list of 5 np.ndarray of shape (256, 256), preserving original dtype.
        """
        plate_id = str(row["PLATE_ID"])
        channels = []
        for col in CHANNEL_COLS:
            filename = str(row[col])
            img_path = os.path.join(self.image_root, plate_id, filename)
            if not img_path.lower().endswith((".tif", ".tiff")):
                img_path = img_path + ".tif"
            img = Image.open(img_path)
            arr = np.array(img)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            # Center-crop or zero-pad to 256x256
            h, w = arr.shape
            target_size = 256
            if h >= target_size and w >= target_size:
                top = (h - target_size) // 2
                left = (w - target_size) // 2
                arr = arr[top : top + target_size, left : left + target_size]
            else:
                out = np.zeros((target_size, target_size), dtype=arr.dtype)
                y_off = (target_size - h) // 2
                x_off = (target_size - w) // 2
                out[y_off : y_off + h, x_off : x_off + w] = arr[:target_size, :target_size]
                arr = out
            channels.append(arr)
        return channels

    def _load_fov_cloome_fullres(self, row) -> np.ndarray:
        """Load 5 channels with CLOOME preprocessing at full resolution.

        Returns (5, H, W) float32 in CLOOME channel order, normalized.
        """
        from encoder.cloome_encoder import (
            MFCP_TO_CLOOME_CHANNEL_ORDER,
            illumination_threshold,
            sixteen_to_eight_bit,
        )

        plate_id = str(row["PLATE_ID"])
        channels = []
        for col in CHANNEL_COLS:
            filename = str(row[col])
            img_path = os.path.join(self.image_root, plate_id, filename)
            if not img_path.lower().endswith((".tif", ".tiff")):
                img_path = img_path + ".tif"
            img = Image.open(img_path)
            arr = np.array(img)
            if arr.ndim == 3:
                arr = arr[:, :, 0]

            # Apply CLOOME preprocessing
            if arr.dtype == np.uint16:
                thresh = illumination_threshold(arr)
                arr = sixteen_to_eight_bit(arr, thresh)
            elif arr.dtype != np.uint8:
                arr = arr.clip(0, 255).astype(np.uint8)

            channels.append(arr.astype(np.float32))

        # Reorder channels: MFCP → CLOOME order
        reordered = [channels[i] for i in MFCP_TO_CLOOME_CHANNEL_ORDER]
        stacked = np.stack(reordered, axis=0)  # (5, H, W)

        # Normalize with dataset-specific stats
        for c in range(NUM_CHANNELS):
            stacked[c] = (stacked[c] - CPG0012_MEAN[c]) / CPG0012_STD[c]

        return stacked

    def _load_fov(self, row) -> np.ndarray:
        """Load 5 channels for a single FOV row → (5, 256, 256) float32."""
        plate_id = str(row["PLATE_ID"])
        channels = []
        for col in CHANNEL_COLS:
            filename = str(row[col])
            img_path = os.path.join(self.image_root, plate_id, filename)
            if not img_path.lower().endswith((".tif", ".tiff")):
                img_path = img_path + ".tif"
            arr = _load_tiff(img_path)
            channels.append(arr)
        return np.stack(channels, axis=0).astype(np.float32)

    def _augment_chem(self, x_chem: torch.Tensor) -> torch.Tensor:
        if self.chem_noise_std > 0:
            x_chem = x_chem + torch.randn_like(x_chem) * self.chem_noise_std
        if self.chem_dropout_prob > 0:
            mask = torch.bernoulli(
                torch.ones_like(x_chem) * (1 - self.chem_dropout_prob)
            )
            x_chem = x_chem * mask
        return x_chem

    def __getitem__(self, idx: int):
        if self.cloome_multi_fov:
            return self._getitem_cloome_multi_fov(idx)
        if self.stacked_fov:
            return self._getitem_stacked_fov(idx)
        if self.multi_fov:
            return self._getitem_multi_fov(idx)
        return self._getitem_single_fov(idx)

    def _getitem_single_fov(self, idx: int):
        """Legacy: returns (image, x_chem, y, compound_label)."""
        row = self.df.iloc[idx]
        inchikey = str(row["INCHIKEY"])

        image = torch.from_numpy(self._load_fov(row))  # (5, 256, 256)
        x_chem = torch.from_numpy(self.chembert_dict[inchikey].astype(np.float32))
        y = torch.from_numpy(self.label_dict[inchikey].astype(np.float32))
        compound_label = torch.tensor(self.ik_to_label[inchikey], dtype=torch.long)

        if self.augment_chem:
            x_chem = self._augment_chem(x_chem)

        return image, x_chem, y, compound_label

    def _getitem_multi_fov(self, idx: int):
        """Multi-FOV: returns (images, fov_mask, x_chem, y, compound_label)."""
        group_df = self.well_groups[idx]
        inchikey = str(group_df.iloc[0]["INCHIKEY"])
        n_sites = len(group_df)

        # Load all FOVs for this well
        all_fov_images = []
        for site_idx in range(n_sites):
            fov_image = self._load_fov(group_df.iloc[site_idx])
            all_fov_images.append(fov_image)

        # Zero-pad to max_sites
        images = np.zeros(
            (self.max_sites, NUM_CHANNELS, 256, 256), dtype=np.float32
        )
        fov_mask = np.zeros(self.max_sites, dtype=bool)
        for i in range(n_sites):
            images[i] = all_fov_images[i]
            fov_mask[i] = True

        images = torch.from_numpy(images)          # (max_sites, 5, 256, 256)
        fov_mask = torch.from_numpy(fov_mask)       # (max_sites,)

        x_chem = torch.from_numpy(self.chembert_dict[inchikey].astype(np.float32))
        y = torch.from_numpy(self.label_dict[inchikey].astype(np.float32))
        compound_label = torch.tensor(self.ik_to_label[inchikey], dtype=torch.long)

        if self.augment_chem:
            x_chem = self._augment_chem(x_chem)

        return images, fov_mask, x_chem, y, compound_label

    def _getitem_cloome_multi_fov(self, idx: int):
        """CLOOME per-FOV: returns (images, fov_mask, x_chem, y, compound_label).

        Each FOV is loaded at full resolution with CLOOME preprocessing.
        Padded to (max_sites, 5, H, W) with a boolean mask.
        """
        group_df = self.well_groups[idx]
        inchikey = str(group_df.iloc[0]["INCHIKEY"])
        n_sites = len(group_df)

        # Load first FOV to get resolution
        first_fov = self._load_fov_cloome_fullres(group_df.iloc[0])
        _, H, W = first_fov.shape

        # Resize if image_size is specified
        if self.image_size is not None:
            H, W = self.image_size, self.image_size

        images = np.zeros(
            (self.max_sites, NUM_CHANNELS, H, W), dtype=np.float32
        )
        fov_mask = np.zeros(self.max_sites, dtype=bool)

        if self.image_size is not None:
            first_fov_t = torch.from_numpy(first_fov).unsqueeze(0)
            first_fov = F.interpolate(first_fov_t, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).numpy()

        images[0] = first_fov
        fov_mask[0] = True
        for i in range(1, min(n_sites, self.max_sites)):
            fov = self._load_fov_cloome_fullres(group_df.iloc[i])
            if self.image_size is not None:
                fov_t = torch.from_numpy(fov).unsqueeze(0)
                fov = F.interpolate(fov_t, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).numpy()
            images[i] = fov
            fov_mask[i] = True

        images = torch.from_numpy(images)      # (max_sites, 5, H, W)
        fov_mask = torch.from_numpy(fov_mask)   # (max_sites,)

        x_chem = torch.from_numpy(self.chembert_dict[inchikey].astype(np.float32))
        y = torch.from_numpy(self.label_dict[inchikey].astype(np.float32))
        compound_label = torch.tensor(self.ik_to_label[inchikey], dtype=torch.long)

        if self.augment_chem:
            x_chem = self._augment_chem(x_chem)

        return images, fov_mask, x_chem, y, compound_label

    def _getitem_stacked_fov(self, idx: int):
        """Stacked FOV mode for CLOOME: returns (stacked_image, x_chem, y, compound_label).

        Loads all FOVs for a well, applies CLOOME-style preprocessing per channel
        (illumination thresholding + 16→8bit conversion), reorders channels to
        CLOOME order [Mito, ERSyto, ERSytoBleed, Ph_golgi, Hoechst], and tiles
        FOVs into a 2×3 grid: 6 FOVs of 256×256 → single image of 512×768.

        Grid layout:
            Row 0: FOV1, FOV2, FOV3
            Row 1: FOV4, FOV5, FOV6
        Wells with <6 FOVs: zero-pad missing positions.
        """
        from encoder.cloome_encoder import (
            MFCP_TO_CLOOME_CHANNEL_ORDER,
            CLOOME_MEAN,
            CLOOME_STD,
            illumination_threshold,
            sixteen_to_eight_bit,
        )

        group_df = self.well_groups[idx]
        inchikey = str(group_df.iloc[0]["INCHIKEY"])
        n_sites = len(group_df)

        fov_size = 256
        grid_rows, grid_cols = 2, 3
        # Stacked image: (5, 512, 768) — zero-initialized for padding
        stacked = np.zeros(
            (NUM_CHANNELS, grid_rows * fov_size, grid_cols * fov_size),
            dtype=np.float32,
        )

        for site_idx in range(min(n_sites, self.max_sites)):
            # Load raw channels (preserving original dtype for thresholding)
            raw_channels = self._load_fov_raw(group_df.iloc[site_idx])

            # Apply CLOOME preprocessing per channel
            processed_channels = []
            for ch_arr in raw_channels:
                if ch_arr.dtype == np.uint16:
                    thresh = illumination_threshold(ch_arr)
                    ch_arr = sixteen_to_eight_bit(ch_arr, thresh)
                elif ch_arr.dtype != np.uint8:
                    ch_arr = ch_arr.clip(0, 255).astype(np.uint8)
                processed_channels.append(ch_arr.astype(np.float32))

            # Reorder channels: MFCP → CLOOME order
            reordered = [processed_channels[i] for i in MFCP_TO_CLOOME_CHANNEL_ORDER]

            # Place FOV in grid
            row = site_idx // grid_cols
            col = site_idx % grid_cols
            y0 = row * fov_size
            x0 = col * fov_size
            for c in range(NUM_CHANNELS):
                stacked[c, y0 : y0 + fov_size, x0 : x0 + fov_size] = reordered[c]

        # Apply CLOOME normalization
        for c in range(NUM_CHANNELS):
            stacked[c] = (stacked[c] - CLOOME_MEAN[c]) / CLOOME_STD[c]

        stacked_image = torch.from_numpy(stacked)  # (5, 512, 768)

        x_chem = torch.from_numpy(self.chembert_dict[inchikey].astype(np.float32))
        y = torch.from_numpy(self.label_dict[inchikey].astype(np.float32))
        compound_label = torch.tensor(self.ik_to_label[inchikey], dtype=torch.long)

        if self.augment_chem:
            x_chem = self._augment_chem(x_chem)

        return stacked_image, x_chem, y, compound_label


# Max FOVs per compound for aggregation (wells * sites). Lower = less GPU memory.
# ~4 wells * 6 sites = 24; use 24 to balance coverage vs memory.
MAX_FOVS_PER_COMPOUND = 24


class CellPaintingCompoundAggregatedDataset(Dataset):
    """Compound-level dataset that aggregates all wells per compound.

    Each sample is one compound. Images = all FOVs from all wells of that compound,
    concatenated. Returns (images, fov_mask, x_chem, y, compound_label) with images
    padded to MAX_FOVS_PER_COMPOUND. Use with collate that expects variable-length
    per compound (this dataset already pads internally for consistent __getitem__).
    """

    def __init__(
        self,
        split_csv: str,
        chembert_dict: dict,
        label_dict: dict,
        image_root: str = "/data/huadi/cpg0012-wawer/images",
        max_sites: int = 6,
        image_size: int | None = None,
        max_fovs_per_compound: int = MAX_FOVS_PER_COMPOUND,
        augment_chem: bool = False,
        chem_noise_std: float = 0.1,
        chem_dropout_prob: float = 0.0,
    ):
        self.image_root = image_root
        self.max_sites = max_sites
        self.image_size = image_size
        self.max_fovs_per_compound = max_fovs_per_compound
        self.augment_chem = augment_chem
        self.chem_noise_std = chem_noise_std
        self.chem_dropout_prob = chem_dropout_prob

        # Build well-level dataset to reuse loading logic
        self._well_ds = CellPaintingImageDataset(
            split_csv=split_csv,
            chembert_dict=chembert_dict,
            label_dict=label_dict,
            image_root=image_root,
            cloome_multi_fov=True,
            max_sites=max_sites,
            image_size=image_size,
            augment_chem=False,
        )

        # Group well indices by compound (INCHIKEY)
        ik_to_well_indices = {}
        for well_idx, group in enumerate(self._well_ds.well_groups):
            ik = str(group.iloc[0]["INCHIKEY"])
            if ik not in ik_to_well_indices:
                ik_to_well_indices[ik] = []
            ik_to_well_indices[ik].append(well_idx)

        self.compound_order = sorted(ik_to_well_indices.keys())
        self.ik_to_well_indices = ik_to_well_indices
        self.chembert_dict = chembert_dict
        self.label_dict = label_dict
        self.ik_to_label = self._well_ds.ik_to_label

        print(
            f"CellPaintingCompoundAggregatedDataset: {len(self.compound_order)} compounds "
            f"(aggregating all wells per compound)"
        )

    def __len__(self) -> int:
        return len(self.compound_order)

    def __getitem__(self, idx: int):
        inchikey = self.compound_order[idx]
        well_indices = self.ik_to_well_indices[inchikey]

        all_images = []
        all_masks = []
        for wi in well_indices:
            images, fov_mask, x_chem, y, compound_label = self._well_ds._getitem_cloome_multi_fov(wi)
            n = images.shape[0]
            for i in range(n):
                if fov_mask[i].item():
                    all_images.append(images[i])
                    all_masks.append(True)

        if len(all_images) == 0:
            # Fallback: use first well if something went wrong
            images, fov_mask, x_chem, y, compound_label = self._well_ds._getitem_cloome_multi_fov(well_indices[0])
            all_images = [images[i] for i in range(images.shape[0]) if fov_mask[i].item()]
            all_masks = [True] * len(all_images)

        n_valid = len(all_images)
        if n_valid > self.max_fovs_per_compound:
            all_images = all_images[: self.max_fovs_per_compound]
            all_masks = [True] * self.max_fovs_per_compound
            n_valid = self.max_fovs_per_compound

        C, H, W = all_images[0].shape
        images_tensor = torch.zeros((self.max_fovs_per_compound, C, H, W), dtype=torch.float32)
        fov_mask_tensor = torch.zeros(self.max_fovs_per_compound, dtype=torch.bool)
        for i, img in enumerate(all_images):
            images_tensor[i] = img
            fov_mask_tensor[i] = all_masks[i]

        x_chem = torch.from_numpy(self.chembert_dict[inchikey].astype(np.float32))
        y = torch.from_numpy(self.label_dict[inchikey].astype(np.float32))
        compound_label = torch.tensor(self.ik_to_label[inchikey], dtype=torch.long)

        if self.augment_chem:
            x_chem = self._well_ds._augment_chem(x_chem)

        return images_tensor, fov_mask_tensor, x_chem, y, compound_label
