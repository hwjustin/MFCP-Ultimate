"""
Train CLOOME + ChemBERT Contrastive-Only Model (Standard CLIP).

This script removes assay classification and CLUB disentanglement components.
It trains only image-chemistry contrastive alignment using CLIPLoss:
  - Image branch: CLOOME visual encoder -> projection head
  - Chem branch: ChemBERT feature -> projection head
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

# Distributed training: set by torchrun or torch.distributed.launch
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


def is_distributed() -> bool:
    return WORLD_SIZE > 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from clip_loss import CLIPLoss
from encoder.cloome_encoder import CLOOMEEncoder
from encoder.dataset import CellPaintingImageDataset


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_chembert_features(chembert_npz: str) -> dict[str, np.ndarray]:
    """Load ChemBERT features from preprocessed NPZ file."""
    data = np.load(chembert_npz, allow_pickle=True)
    x_chem = data["X"].astype(np.float32)
    inchikeys_chem = data["inchikeys"]

    chembert_dict = {}
    for i, ik in enumerate(inchikeys_chem):
        chembert_dict[str(ik)] = x_chem[i]

    if is_main_process():
        print(
            f"Loaded ChemBERT features for {len(chembert_dict)} compounds "
            f"with {x_chem.shape[1]} features"
        )
    return chembert_dict


def load_labels(labels_csv: str) -> dict[str, np.ndarray]:
    """
    Load labels dictionary only to filter valid compounds in dataset.

    Contrastive training does not use assay targets in the loss.
    """
    df = pd.read_csv(labels_csv)
    label_dict = {}
    for _, row in df.iterrows():
        ik = str(row["INCHIKEY"])
        label_dict[ik] = row.iloc[1:].values.astype(np.float32)

    if is_main_process():
        print(f"Loaded label map for {len(label_dict)} compounds (filtering only)")
    return label_dict


def create_compound_level_split(split_csv: str) -> str:
    """Create modified split CSV with one well per compound (keeping all sites)."""
    df = pd.read_csv(split_csv)
    if is_main_process():
        print(f"Original split: {len(df)} rows with {df['INCHIKEY'].nunique()} unique compounds")

    sort_cols = ["PLATE_ID", "WELL_POSITION", "SITE"] if "SITE" in df.columns else ["PLATE_ID", "WELL_POSITION"]
    df = df.sort_values(sort_cols)

    well_keys = df.groupby("INCHIKEY")[["PLATE_ID", "WELL_POSITION"]].first().reset_index()
    if is_main_process():
        print(f"Selected {len(well_keys)} compounds (one well per compound)")

    df_compound = df.merge(well_keys, on=["INCHIKEY", "PLATE_ID", "WELL_POSITION"], how="inner")
    if is_main_process():
        print(f"After compound-level selection: {len(df_compound)} rows (all sites for selected wells)")

    temp_csv = split_csv.replace(".csv", "_compound_temp.csv")
    df_compound.to_csv(temp_csv, index=False)
    return temp_csv


class CLOOMEChemContrastive(nn.Module):
    """CLOOME + ChemBERT contrastive-only model."""

    def __init__(
        self,
        chem_dim: int = 768,
        embed_dim: int = 256,
        clip_temperature: float = 0.07,
        cloome_config: str = "/data/huadi/cloome/src/training/model_configs/RN50.json",
        cloome_checkpoint: str | None = None,
    ):
        super().__init__()

        self.cloome_encoder = CLOOMEEncoder(
            config_path=cloome_config,
            checkpoint_path=cloome_checkpoint,
        )
        cloome_dim = self.cloome_encoder.embed_dim
        self.cloome_bn = nn.BatchNorm1d(cloome_dim)

        self.image_proj = nn.Sequential(
            nn.Linear(cloome_dim, cloome_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cloome_dim, embed_dim),
        )
        self.chem_proj = nn.Sequential(
            nn.Linear(chem_dim, chem_dim),
            nn.ReLU(inplace=True),
            nn.Linear(chem_dim, embed_dim),
        )

        self.clip_loss_fn = CLIPLoss(temperature=clip_temperature)

    def _encode_cloome(self, images: torch.Tensor, fov_mask: torch.Tensor | None) -> torch.Tensor:
        """Shared CLOOME encoding -> (B, 512)."""
        if fov_mask is not None:
            bsz, sites, channels, height, width = images.shape
            flat_images = images.reshape(bsz * sites, channels, height, width)
            flat_embeds = self.cloome_encoder(flat_images)
            per_fov = flat_embeds.reshape(bsz, sites, -1)
            mask_expanded = fov_mask.unsqueeze(-1).float()
            h_cloome = (per_fov * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h_cloome = self.cloome_encoder(images)
        return self.cloome_bn(h_cloome)

    def forward(
        self,
        images: torch.Tensor,
        x_chem: torch.Tensor,
        fov_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            clip_loss: scalar
            z_img: (B, D) image embeddings
            z_chem: (B, D) chem embeddings
        """
        h_img = self._encode_cloome(images, fov_mask)
        z_img = self.image_proj(h_img)
        z_chem = self.chem_proj(x_chem)
        clip_loss, _, _ = self.clip_loss_fn(z_img, z_chem)
        return clip_loss, z_img, z_chem


def _batch_pos_neg_from_embeddings(z_img: torch.Tensor, z_chem: torch.Tensor) -> tuple[float, float]:
    """Compute positive/negative cosine similarities for one batch."""
    z_img = F.normalize(z_img, dim=1)
    z_chem = F.normalize(z_chem, dim=1)
    sim = z_img @ z_chem.T
    diag = torch.diag(sim)
    pos = diag.mean().item()
    n = sim.shape[0]
    if n <= 1:
        neg = 0.0
    else:
        neg = ((sim.sum() - diag.sum()) / (n * n - n)).item()
    return pos, neg


def _reduce_metrics(running_clip: float, running_pos: float, running_neg: float, n_batches: int) -> tuple[float, float, float]:
    """All-reduce training metrics across ranks."""
    if not is_distributed():
        n = max(n_batches, 1)
        return running_clip / n, running_pos / n, running_neg / n

    t = torch.tensor([running_clip, running_pos, running_neg, float(n_batches)], dtype=torch.float32, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total_clip, total_pos, total_neg, total_batches = t.tolist()
    n = max(total_batches, 1)
    return total_clip / n, total_pos / n, total_neg / n


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int = 1,
    use_wandb: bool = False,
) -> dict[str, float]:
    """Train one epoch and return clip/similarity stats."""
    model.train()
    optimizer.zero_grad()

    running_clip = 0.0
    running_pos = 0.0
    running_neg = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False, disable=not is_main_process())
    use_autocast = use_amp and device.type == "cuda"

    for batch_idx, batch in enumerate(pbar):
        if len(batch) == 5:
            images_b, fov_mask_b, x_chem_b, _, _ = batch
            images_b = images_b.to(device, non_blocking=True)
            fov_mask_b = fov_mask_b.to(device, non_blocking=True)
            x_chem_b = x_chem_b.to(device, non_blocking=True)
        else:
            stacked_images_b, x_chem_b, _, _ = batch
            images_b = stacked_images_b.to(device, non_blocking=True)
            fov_mask_b = None
            x_chem_b = x_chem_b.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
            clip_loss_val, z_img, z_chem = model(images_b, x_chem_b, fov_mask_b)
            loss = clip_loss_val / grad_accum_steps

        if use_autocast and scaler is not None:
            scaler.scale(loss).backward()
            if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(loader)):
                optimizer.step()
                optimizer.zero_grad()

        pos, neg = _batch_pos_neg_from_embeddings(z_img.detach(), z_chem.detach())
        running_clip += clip_loss_val.item()
        running_pos += pos
        running_neg += neg
        n_batches += 1

        avg_clip = running_clip / max(n_batches, 1)
        avg_pos = running_pos / max(n_batches, 1)
        avg_neg = running_neg / max(n_batches, 1)
        pbar.set_postfix(clip=f"{avg_clip:.4f}", pos=f"{avg_pos:.4f}", neg=f"{avg_neg:.4f}")

        if use_wandb and is_main_process():
            wandb.log(
                {
                    "train/batch_clip_loss": avg_clip,
                    "train/batch_pos_sim": avg_pos,
                    "train/batch_neg_sim": avg_neg,
                }
            )

    clip_avg, pos_avg, neg_avg = _reduce_metrics(running_clip, running_pos, running_neg, n_batches)
    return {
        "clip_loss": clip_avg,
        "pos_sim": pos_avg,
        "neg_sim": neg_avg,
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Val",
) -> dict[str, float]:
    """Evaluate one epoch and return clip/similarity stats."""
    model.eval()

    running_clip = 0.0
    running_pos = 0.0
    running_neg = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=desc, leave=False, disable=not is_main_process())
    for batch in pbar:
        if len(batch) == 5:
            images_b, fov_mask_b, x_chem_b, _, _ = batch
            images_b = images_b.to(device, non_blocking=True)
            fov_mask_b = fov_mask_b.to(device, non_blocking=True)
            x_chem_b = x_chem_b.to(device, non_blocking=True)
        else:
            stacked_images_b, x_chem_b, _, _ = batch
            images_b = stacked_images_b.to(device, non_blocking=True)
            fov_mask_b = None
            x_chem_b = x_chem_b.to(device, non_blocking=True)

        clip_loss_val, z_img, z_chem = model(images_b, x_chem_b, fov_mask_b)
        pos, neg = _batch_pos_neg_from_embeddings(z_img, z_chem)

        running_clip += clip_loss_val.item()
        running_pos += pos
        running_neg += neg
        n_batches += 1

        avg_clip = running_clip / max(n_batches, 1)
        avg_pos = running_pos / max(n_batches, 1)
        avg_neg = running_neg / max(n_batches, 1)
        pbar.set_postfix(clip=f"{avg_clip:.4f}", pos=f"{avg_pos:.4f}", neg=f"{avg_neg:.4f}")

    clip_avg, pos_avg, neg_avg = _reduce_metrics(running_clip, running_pos, running_neg, n_batches)
    return {
        "clip_loss": clip_avg,
        "pos_sim": pos_avg,
        "neg_sim": neg_avg,
    }


def _gather_embeddings_for_retrieval(z_img: torch.Tensor, z_chem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather embeddings from all ranks and reconstruct full tensors in dataset order."""
    if not is_distributed():
        return z_img, z_chem

    z_img = z_img.contiguous().cuda()
    z_chem = z_chem.contiguous().cuda()
    n_local = z_img.shape[0]
    dim = z_img.shape[1]

    # Gather sizes
    size_tensor = torch.tensor([n_local], dtype=torch.long, device="cuda")
    size_list = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, size_tensor)
    sizes = [s.item() for s in size_list]
    max_len = max(sizes)
    total = sum(sizes)

    # Pad to max_len
    if n_local < max_len:
        pad_img = torch.zeros(max_len - n_local, dim, dtype=z_img.dtype, device="cuda")
        pad_chem = torch.zeros(max_len - n_local, dim, dtype=z_chem.dtype, device="cuda")
        z_img = torch.cat([z_img, pad_img], dim=0)
        z_chem = torch.cat([z_chem, pad_chem], dim=0)

    # All-gather
    img_list = [torch.zeros_like(z_img) for _ in range(dist.get_world_size())]
    chem_list = [torch.zeros_like(z_chem) for _ in range(dist.get_world_size())]
    dist.all_gather(img_list, z_img)
    dist.all_gather(chem_list, z_chem)

    # Reconstruct in dataset order (DistributedSampler: rank r has indices r, r+W, r+2W, ...)
    full_img = torch.zeros(total, dim, dtype=z_img.dtype, device="cuda")
    full_chem = torch.zeros(total, dim, dtype=z_chem.dtype, device="cuda")
    for r, n_r in enumerate(sizes):
        for j in range(n_r):
            idx = r + j * dist.get_world_size()
            full_img[idx] = img_list[r][j]
            full_chem[idx] = chem_list[r][j]

    return full_img, full_chem


@torch.no_grad()
def compute_retrieval_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Retrieval",
) -> dict[str, float]:
    """Compute image->chem and chem->image retrieval recall@k."""
    model.eval()
    all_img = []
    all_chem = []

    pbar = tqdm(loader, desc=desc, leave=False, disable=not is_main_process())
    for batch in pbar:
        if len(batch) == 5:
            images_b, fov_mask_b, x_chem_b, _, _ = batch
            images_b = images_b.to(device, non_blocking=True)
            fov_mask_b = fov_mask_b.to(device, non_blocking=True)
            x_chem_b = x_chem_b.to(device, non_blocking=True)
        else:
            stacked_images_b, x_chem_b, _, _ = batch
            images_b = stacked_images_b.to(device, non_blocking=True)
            fov_mask_b = None
            x_chem_b = x_chem_b.to(device, non_blocking=True)

        _, z_img, z_chem = model(images_b, x_chem_b, fov_mask_b)
        all_img.append(F.normalize(z_img, dim=1))
        all_chem.append(F.normalize(z_chem, dim=1))

    z_img = torch.cat(all_img, dim=0)
    z_chem = torch.cat(all_chem, dim=0)
    z_img, z_chem = _gather_embeddings_for_retrieval(z_img, z_chem)
    sim = z_img @ z_chem.T
    n = sim.shape[0]

    diag = torch.diag(sim)
    pos_sim = diag.mean().item()
    if n <= 1:
        neg_sim = 0.0
    else:
        neg_sim = ((sim.sum() - diag.sum()) / (n * n - n)).item()

    def _recall_at_k(sim_matrix: torch.Tensor, k: int) -> float:
        k = min(k, sim_matrix.shape[1])
        topk = torch.topk(sim_matrix, k=k, dim=1).indices
        gt = torch.arange(sim_matrix.shape[0], device=sim_matrix.device).unsqueeze(1)
        hits = (topk == gt).any(dim=1).float()
        return hits.mean().item()

    metrics = {
        "num_samples": int(n),
        "pos_sim": pos_sim,
        "neg_sim": neg_sim,
    }
    for k in (1, 5, 10):
        metrics[f"img_to_chem_r@{k}"] = _recall_at_k(sim, k)
        metrics[f"chem_to_img_r@{k}"] = _recall_at_k(sim.T, k)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLOOME + ChemBERT Contrastive-Only Model")

    # Data paths
    parser.add_argument("--train-split", type=str, required=True)
    parser.add_argument("--val-split", type=str, required=True)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument(
        "--chembert-npz",
        type=str,
        default="/data/huadi/cellpainting_data/cpg0012/chem_features/preprocessed_chem_assays.npz",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default="/data/huadi/cellpainting_data/cpg0012/labels/compound_assay_activity.csv",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/data/huadi/cpg0012-wawer/images",
    )
    parser.add_argument(
        "--aggregation-level",
        type=str,
        choices=["well", "compound"],
        default="well",
    )

    # Model
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--clip-temperature", type=float, default=0.07)
    parser.add_argument(
        "--cloome-config",
        type=str,
        default="/data/huadi/cloome/src/training/model_configs/RN50.json",
    )
    parser.add_argument("--cloome-checkpoint", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)

    # Training
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--downstream-lr", type=float, default=2e-3)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--backbone-warmup-epochs", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # Chem augmentation
    parser.add_argument("--augment-chem", action="store_true")
    parser.add_argument("--chem-noise-std", type=float, default=0.05)
    parser.add_argument("--chem-dropout-prob", type=float, default=0.0)

    # Output
    parser.add_argument(
        "--output-model",
        type=str,
        default="results/checkpoints/cloome_contrastive.pt",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="results/metrics/cloome_contrastive_metrics.json",
    )

    # WandB
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )

    args = parser.parse_args()

    # Initialize distributed training (torchrun sets LOCAL_RANK, WORLD_SIZE)
    if is_distributed():
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(backend="nccl")
        local_rank = LOCAL_RANK
        world_size = dist.get_world_size()
        per_gpu_batch = max(1, args.batch_size // world_size)
        if local_rank == 0:
            print(f"Distributed training: {world_size} GPUs, batch_size={args.batch_size} total ({per_gpu_batch} per GPU)")
    else:
        local_rank = 0
        world_size = 1
        per_gpu_batch = args.batch_size

    set_seed(args.seed)
    if is_main_process():
        print(f"Random seed set to {args.seed}")

    use_wandb = args.wandb_project is not None and WANDB_AVAILABLE
    if use_wandb and is_main_process():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "model": "cloome_contrastive",
                "embed_dim": args.embed_dim,
                "clip_temperature": args.clip_temperature,
                "backbone_lr": args.backbone_lr,
                "downstream_lr": args.downstream_lr,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "effective_batch_size": args.batch_size * args.grad_accum_steps,
                "max_iter": args.max_iter,
                "seed": args.seed,
                "aggregation_level": args.aggregation_level,
                "use_amp": args.use_amp,
                "augment_chem": args.augment_chem,
                "chem_noise_std": args.chem_noise_std,
                "chem_dropout_prob": args.chem_dropout_prob,
            },
        )
        print(f"WandB initialized: project={args.wandb_project}, run={wandb.run.name}")
    elif args.wandb_project is not None and not WANDB_AVAILABLE and is_main_process():
        print("Warning: WandB project specified but wandb is not installed")

    if is_main_process():
        print("\n=== Loading Inputs ===")
    chembert_dict = load_chembert_features(args.chembert_npz)
    label_dict = load_labels(args.labels_csv)

    train_split = args.train_split
    val_split = args.val_split
    test_split = args.test_split
    temp_files = []
    if args.aggregation_level == "compound":
        print("\n=== Creating Compound-Level Splits ===")
        train_split = create_compound_level_split(args.train_split)
        val_split = create_compound_level_split(args.val_split)
        temp_files.extend([train_split, val_split])
        if args.test_split:
            test_split = create_compound_level_split(args.test_split)
            temp_files.append(test_split)

    if is_main_process():
        print(f"\n=== Creating Datasets (cloome_multi_fov=True, aggregation_level={args.aggregation_level}) ===")
        print("\nTrain split:")
    train_ds = CellPaintingImageDataset(
        split_csv=train_split,
        chembert_dict=chembert_dict,
        label_dict=label_dict,
        image_root=args.image_root,
        cloome_multi_fov=True,
        image_size=args.image_size,
        augment_chem=args.augment_chem,
        chem_noise_std=args.chem_noise_std,
        chem_dropout_prob=args.chem_dropout_prob,
    )
    if is_main_process():
        print("\nVal split:")
    val_ds = CellPaintingImageDataset(
        split_csv=val_split,
        chembert_dict=chembert_dict,
        label_dict=label_dict,
        image_root=args.image_root,
        cloome_multi_fov=True,
        image_size=args.image_size,
        augment_chem=False,
    )
    test_ds = None
    if test_split:
        if is_main_process():
            print("\nTest split:")
        test_ds = CellPaintingImageDataset(
            split_csv=test_split,
            chembert_dict=chembert_dict,
            label_dict=label_dict,
            image_root=args.image_root,
            cloome_multi_fov=True,
            image_size=args.image_size,
            augment_chem=False,
        )

    if is_main_process():
        print("\n=== Computing ChemBERT Normalization Statistics ===")
    train_inchikeys = list(set(str(g.iloc[0]["INCHIKEY"]) for g in train_ds.well_groups))
    all_chem_features = [chembert_dict[str(ik)] for ik in train_inchikeys if str(ik) in chembert_dict]
    x_chem_train = np.stack(all_chem_features, axis=0)
    x_chem_train = np.clip(x_chem_train, -1e3, 1e3)
    chem_mean = x_chem_train.mean(axis=0, keepdims=True).astype(np.float32)
    chem_std = x_chem_train.std(axis=0, keepdims=True).astype(np.float32)
    chem_std[chem_std == 0] = 1.0
    if is_main_process():
        print(f"Computing normalization from {len(train_inchikeys)} unique compounds")
        print(f"ChemBERT mean: {chem_mean.mean():.4f}, std: {chem_std.mean():.4f}")

    def normalize_batch(batch):
        if len(batch[0]) == 5:
            images, fov_mask, x_chem, y, compound_label = zip(*batch)
            images = torch.stack(images, dim=0)
            fov_mask = torch.stack(fov_mask, dim=0)
            x_chem = torch.stack(x_chem, dim=0)
            y = torch.stack(y, dim=0)
            compound_label = torch.stack(compound_label, dim=0)
            x_chem = (x_chem - torch.from_numpy(chem_mean[0])) / torch.from_numpy(chem_std[0])
            return images, fov_mask, x_chem, y, compound_label

        stacked_images, x_chem, y, compound_label = zip(*batch)
        stacked_images = torch.stack(stacked_images, dim=0)
        x_chem = torch.stack(x_chem, dim=0)
        y = torch.stack(y, dim=0)
        compound_label = torch.stack(compound_label, dim=0)
        x_chem = (x_chem - torch.from_numpy(chem_mean[0])) / torch.from_numpy(chem_std[0])
        return stacked_images, x_chem, y, compound_label

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if is_distributed() and test_ds else None

    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu_batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=normalize_batch,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=per_gpu_batch,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=normalize_batch,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = None
    if test_ds:
        test_loader = DataLoader(
            test_ds,
            batch_size=per_gpu_batch,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=normalize_batch,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=args.num_workers > 0,
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_distributed():
        dist.barrier()
    if is_main_process():
        print(f"\n=== Initializing Model on {device} ===")
    # Build model on CPU first so both ranks load CLOOME checkpoint before any collective
    model = CLOOMEChemContrastive(
        chem_dim=768,
        embed_dim=args.embed_dim,
        clip_temperature=args.clip_temperature,
        cloome_config=args.cloome_config,
        cloome_checkpoint=args.cloome_checkpoint,
    )
    if is_distributed():
        dist.barrier()
    model = model.to(device)
    if is_distributed():
        dist.barrier()
        model = DDP(model, device_ids=[local_rank])

    _model = model.module if is_distributed() else model
    n_backbone_params = sum(p.numel() for p in _model.cloome_encoder.parameters())
    n_downstream_params = sum(
        p.numel() for n, p in _model.named_parameters() if p.requires_grad and not n.startswith("cloome_encoder")
    )
    n_total_params = sum(p.numel() for p in _model.parameters())
    n_trainable_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    if is_main_process():
        print("\nModel Statistics:")
        print(f"  CLOOME backbone params: {n_backbone_params:,}")
        print(f"  Downstream params (projection heads): {n_downstream_params:,}")
        print(f"  Total params: {n_total_params:,}")
        print(f"  Trainable params: {n_trainable_params:,}")

    if is_main_process():
        print("\n=== Setting Up Optimizer ===")
    if args.freeze_epochs > 0:
        for p in _model.cloome_encoder.parameters():
            p.requires_grad = False
        if is_main_process():
            print(f"Backbone FROZEN for first {args.freeze_epochs} epochs")

    backbone_params = list(_model.cloome_encoder.parameters())
    downstream_params = [p for n, p in _model.named_parameters() if not n.startswith("cloome_encoder")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 0.0 if args.freeze_epochs > 0 else args.backbone_lr, "weight_decay": 0.01},
            {"params": downstream_params, "lr": args.downstream_lr, "weight_decay": 0.0},
        ]
    )
    if is_main_process():
        print("Optimizer: AdamW")
        print(f"  Backbone lr: {0.0 if args.freeze_epochs > 0 else args.backbone_lr}, weight_decay: 0.01")
        print(f"  Downstream lr: {args.downstream_lr}, weight_decay: 0.0")

    scaler = None
    if args.use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        if is_main_process():
            print("Using mixed precision training (bfloat16)")
    if is_main_process():
        print(f"Gradient accumulation steps: {args.grad_accum_steps}")
        print(f"Effective batch size: {per_gpu_batch * world_size * args.grad_accum_steps}")
        print(f"CLIP temperature: {args.clip_temperature}")

    best_val_clip = float("inf")
    best_epoch = 0
    history = []

    if is_main_process():
        print("\n=== Training (Contrastive Only) ===")
    for epoch in tqdm(range(1, args.max_iter + 1), desc="Training epochs", disable=not is_main_process()):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            for p in _model.cloome_encoder.parameters():
                p.requires_grad = True
            if is_main_process():
                print(f"\n>>> Epoch {epoch}: Backbone UNFROZEN")

        if epoch > args.freeze_epochs:
            epochs_since_unfreeze = epoch - args.freeze_epochs
            if args.backbone_warmup_epochs > 0 and epochs_since_unfreeze <= args.backbone_warmup_epochs:
                warmup_factor = epochs_since_unfreeze / args.backbone_warmup_epochs
                backbone_lr = args.backbone_lr * warmup_factor
            else:
                backbone_lr = args.backbone_lr
            optimizer.param_groups[0]["lr"] = backbone_lr

        train_stats = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            use_amp=args.use_amp,
            scaler=scaler,
            epoch=epoch,
            use_wandb=use_wandb,
        )
        val_stats = evaluate_epoch(model, val_loader, device=device, desc=f"Epoch {epoch} [Val]")
        val_retrieval = compute_retrieval_metrics(
            model,
            val_loader,
            device=device,
            desc=f"Epoch {epoch} [Val Retrieval]",
        )

        if val_stats["clip_loss"] < best_val_clip:
            best_val_clip = val_stats["clip_loss"]
            best_epoch = epoch

        record = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "val_retrieval": val_retrieval,
        }
        history.append(record)

        if is_main_process():
            print(
                f"Epoch {epoch:3d}/{args.max_iter} | "
            f"train_clip={train_stats['clip_loss']:.4f}, "
            f"train_pos={train_stats['pos_sim']:.4f}, train_neg={train_stats['neg_sim']:.4f} | "
            f"val_clip={val_stats['clip_loss']:.4f}, "
            f"val_pos={val_stats['pos_sim']:.4f}, val_neg={val_stats['neg_sim']:.4f} | "
            f"val_i2c_r@1={val_retrieval['img_to_chem_r@1']:.4f}, "
            f"val_c2i_r@1={val_retrieval['chem_to_img_r@1']:.4f}"
            )

        if use_wandb and is_main_process():
            wandb.log(
                {
                    "epoch": epoch,
                    "train/clip_loss": train_stats["clip_loss"],
                    "train/pos_sim": train_stats["pos_sim"],
                    "train/neg_sim": train_stats["neg_sim"],
                    "val/clip_loss": val_stats["clip_loss"],
                    "val/pos_sim": val_stats["pos_sim"],
                    "val/neg_sim": val_stats["neg_sim"],
                    "val_retrieval/img_to_chem_r@1": val_retrieval["img_to_chem_r@1"],
                    "val_retrieval/img_to_chem_r@5": val_retrieval["img_to_chem_r@5"],
                    "val_retrieval/img_to_chem_r@10": val_retrieval["img_to_chem_r@10"],
                    "val_retrieval/chem_to_img_r@1": val_retrieval["chem_to_img_r@1"],
                    "val_retrieval/chem_to_img_r@5": val_retrieval["chem_to_img_r@5"],
                    "val_retrieval/chem_to_img_r@10": val_retrieval["chem_to_img_r@10"],
                },
                step=epoch,
            )

    if is_main_process():
        print("\n=== Final Retrieval Evaluation ===")
    val_retrieval_final = compute_retrieval_metrics(model, val_loader, device=device, desc="Final Val Retrieval")
    if is_main_process():
        print(
            "Val retrieval: "
        f"i2c R@1/5/10={val_retrieval_final['img_to_chem_r@1']:.4f}/"
        f"{val_retrieval_final['img_to_chem_r@5']:.4f}/"
        f"{val_retrieval_final['img_to_chem_r@10']:.4f}, "
        f"c2i R@1/5/10={val_retrieval_final['chem_to_img_r@1']:.4f}/"
        f"{val_retrieval_final['chem_to_img_r@5']:.4f}/"
        f"{val_retrieval_final['chem_to_img_r@10']:.4f}"
    )

    test_retrieval_final = None
    if test_loader is not None:
        test_retrieval_final = compute_retrieval_metrics(
            model,
            test_loader,
            device=device,
            desc="Final Test Retrieval",
        )
        if is_main_process():
            print(
                "Test retrieval: "
            f"i2c R@1/5/10={test_retrieval_final['img_to_chem_r@1']:.4f}/"
            f"{test_retrieval_final['img_to_chem_r@5']:.4f}/"
            f"{test_retrieval_final['img_to_chem_r@10']:.4f}, "
        f"c2i R@1/5/10={test_retrieval_final['chem_to_img_r@1']:.4f}/"
        f"{test_retrieval_final['chem_to_img_r@5']:.4f}/"
        f"{test_retrieval_final['chem_to_img_r@10']:.4f}"
            )

    if is_main_process():
        os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
        checkpoint = {
            "model_state_dict": _model.state_dict(),
            "chem_dim": 768,
            "embed_dim": args.embed_dim,
            "chem_mean": chem_mean,
            "chem_std": chem_std,
            "best_epoch": best_epoch,
            "best_val_clip_loss": float(best_val_clip),
            "cloome_config": args.cloome_config,
            "backbone_lr": args.backbone_lr,
            "downstream_lr": args.downstream_lr,
            "clip_temperature": args.clip_temperature,
            "aggregation_level": args.aggregation_level,
        }
        torch.save(checkpoint, args.output_model)
        print(f"\nSaved model checkpoint to {args.output_model}")

        metrics_dict = {
            "hyperparameters": {
                "embed_dim": args.embed_dim,
                "clip_temperature": args.clip_temperature,
                "backbone_lr": args.backbone_lr,
                "downstream_lr": args.downstream_lr,
                "batch_size": per_gpu_batch * world_size,
                "grad_accum_steps": args.grad_accum_steps,
                "effective_batch_size": per_gpu_batch * world_size * args.grad_accum_steps,
                "max_iter": args.max_iter,
                "seed": args.seed,
                "aggregation_level": args.aggregation_level,
                "use_amp": args.use_amp,
            },
            "model_statistics": {
                "cloome_backbone_params": n_backbone_params,
                "downstream_params": n_downstream_params,
                "total_params": n_total_params,
                "trainable_params": n_trainable_params,
            },
            "dataset_info": {
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "test_samples": len(test_ds) if test_ds else None,
                "aggregation_level": args.aggregation_level,
            },
            "history": history,
            "best_epoch": best_epoch,
            "best_val_clip_loss": float(best_val_clip),
            "final_validation_retrieval": val_retrieval_final,
        }
        if test_retrieval_final is not None:
            metrics_dict["final_test_retrieval"] = test_retrieval_final

        os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
        with open(args.output_metrics, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Saved metrics to {args.output_metrics}")

        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")

        if use_wandb:
            wandb.finish()
            print("WandB run finished")

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
