"""
Train CLOOME + ChemBERT Three-Branch Model (Concat + CLIP)

Fine-tunes the CLOOME ResNet50 visual encoder while keeping ChemBERT features frozen.
Three branches are concatenated for classification:
  Branch 1 (CLOOME): 512 → 256-D
  Branch 2 (ChemBERT): 768 → 256-D
  Branch 3 (CLIP): CLOOME → 256-D + ChemBERT → 256-D (L2-normalized, contrastive loss),
                    concat → 512 → 256-D
Final: 256 × 3 = 768-D → MLP classifier (209 tasks)

The CLIP branch uses SupervisedCLIPLoss to handle replicates (multiple wells per compound).

Each FOV is encoded separately at full resolution (5, 520, 696) through CLOOME,
then per-FOV embeddings are mean-pooled (with masking for variable FOV counts).
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from clip_loss import SupervisedCLIPLoss
from encoder.cloome_encoder import CLOOMEEncoder
from encoder.dataset import CellPaintingImageDataset


class CLUBMean(nn.Module):
    """CLUB with fixed unit variance (logvar=0). Estimates MI upper bound."""

    def __init__(self, x_dim, y_dim, hidden_size=512):
        super().__init__()
        if hidden_size is None:
            self.p_mu = nn.Linear(x_dim, y_dim)
        else:
            self.p_mu = nn.Sequential(
                nn.Linear(x_dim, int(hidden_size)),
                nn.ReLU(),
                nn.Linear(int(hidden_size), y_dim),
            )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        return mu, 0

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        positive = -(mu - y_samples) ** 2 / 2.0
        prediction_1 = mu.unsqueeze(1)       # (N, 1, D)
        y_samples_1 = y_samples.unsqueeze(0)  # (1, N, D)
        negative = -((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.0
        positive = positive.sum(dim=-1)
        negative = negative.sum(dim=-1)
        return (positive - negative).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)


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


def load_chembert_features(chembert_npz: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load ChemBERT features from preprocessed NPZ file."""
    data = np.load(chembert_npz, allow_pickle=True)

    X_chem = data["X"].astype(np.float32)
    inchikeys_chem = data["inchikeys"]
    label_names = data["label_names"].tolist()

    chembert_dict = {}
    for i, ik in enumerate(inchikeys_chem):
        chembert_dict[str(ik)] = X_chem[i]

    print(f"Loaded ChemBERT features for {len(chembert_dict)} compounds with {X_chem.shape[1]} features")

    return chembert_dict, label_names


def load_labels(labels_csv: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load compound-assay activity labels."""
    df = pd.read_csv(labels_csv)

    label_cols = list(df.columns[1:])

    label_dict = {}
    for _, row in df.iterrows():
        ik = str(row["INCHIKEY"])
        label_dict[ik] = row.iloc[1:].values.astype(np.float32)

    print(f"Loaded labels for {len(label_dict)} compounds across {len(label_cols)} assays")

    return label_dict, label_cols


def create_compound_level_split(split_csv: str) -> str:
    """Create modified split CSV with one well per compound (keeping all sites)."""
    df = pd.read_csv(split_csv)

    print(f"Original split: {len(df)} rows with {df['INCHIKEY'].nunique()} unique compounds")

    df = df.sort_values(["PLATE_ID", "WELL_POSITION", "SITE"] if "SITE" in df.columns else ["PLATE_ID", "WELL_POSITION"])

    well_keys = (
        df.groupby("INCHIKEY")[["PLATE_ID", "WELL_POSITION"]]
        .first()
        .reset_index()
    )
    print(f"Selected {len(well_keys)} compounds (one well per compound)")

    df_compound = df.merge(well_keys, on=["INCHIKEY", "PLATE_ID", "WELL_POSITION"], how="inner")
    print(f"After compound-level selection: {len(df_compound)} rows (all sites for selected wells)")

    temp_csv = split_csv.replace(".csv", "_compound_temp.csv")
    df_compound.to_csv(temp_csv, index=False)

    return temp_csv


class CLOOMEConcatMLP(nn.Module):
    """CLOOME + ChemBERT three-branch model (concat + CLIP)."""

    def __init__(
        self,
        chem_dim: int = 768,
        num_tasks: int = 209,
        embed_dim: int = 256,
        clip_temperature: float = 0.07,
        club_hidden_size: int = 512,
        cloome_config: str = "/data/huadi/cloome/src/training/model_configs/RN50.json",
        cloome_checkpoint: str | None = None,
    ):
        super().__init__()

        # CLOOME visual encoder (full fine-tuning)
        self.cloome_encoder = CLOOMEEncoder(
            config_path=cloome_config,
            checkpoint_path=cloome_checkpoint,
        )
        cloome_dim = self.cloome_encoder.embed_dim  # 512

        # Normalize CLOOME embeddings before projection
        self.cloome_bn = nn.BatchNorm1d(cloome_dim)

        # Branch 1 & 2: concat projection layers (2-layer MLP, STiL-style)
        self.cloome_proj = nn.Sequential(
            nn.Linear(cloome_dim, cloome_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cloome_dim, embed_dim),
        )
        self.chem_proj = nn.Sequential(
            nn.Linear(chem_dim, chem_dim),
            nn.ReLU(inplace=True),
            nn.Linear(chem_dim, embed_dim),
        )

        # Branch 3: CLIP projection layers (2-layer MLP, STiL-style)
        self.clip_cloome_proj = nn.Sequential(
            nn.Linear(cloome_dim, cloome_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cloome_dim, embed_dim),
        )
        self.clip_chem_proj = nn.Sequential(
            nn.Linear(chem_dim, chem_dim),
            nn.ReLU(inplace=True),
            nn.Linear(chem_dim, embed_dim),
        )
        # Fuse the two shared embeddings: concat(256, 256) = 512 → 256 (STiL-style reduce)
        self.clip_fusion = nn.Linear(2 * embed_dim, embed_dim)

        # CLIP loss
        self.clip_loss_fn = SupervisedCLIPLoss(temperature=clip_temperature)

        # CLUB mutual information estimators for disentanglement
        self.club_cloome = CLUBMean(x_dim=embed_dim, y_dim=embed_dim, hidden_size=club_hidden_size)
        self.club_chem = CLUBMean(x_dim=embed_dim, y_dim=embed_dim, hidden_size=club_hidden_size)

        # MLP classifier: 3 branches × 256 = 768 input
        width = 1024
        self.classifier = nn.Sequential(
            nn.Linear(3 * embed_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(width, num_tasks),
        )

    def _encode_cloome(self, images, fov_mask):
        """Shared CLOOME encoding → (B, 512)."""
        if fov_mask is not None:
            B, S, C, H, W = images.shape
            flat_images = images.reshape(B * S, C, H, W)
            flat_embeds = self.cloome_encoder(flat_images)
            per_fov = flat_embeds.reshape(B, S, -1)
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
        compound_labels: torch.Tensor | None = None,
    ):
        """
        Args:
            images: (B, S, 5, H, W) per-FOV images, or (B, 5, H, W) single image
            x_chem: (B, 768) ChemBERT features
            fov_mask: (B, S) bool mask indicating valid FOVs (None if single image)
            compound_labels: (B,) integer compound IDs for supervised CLIP loss

        Returns:
            logits: (B, num_tasks) bioassay predictions
            clip_loss: scalar CLIP contrastive loss (0 if compound_labels is None)
        """
        h_cloome = self._encode_cloome(images, fov_mask)  # (B, 512)

        # Branch 1: CLOOME projection
        h_cloome_proj = self.cloome_proj(h_cloome)   # (B, 256)

        # Branch 2: ChemBERT projection
        h_chem_proj = self.chem_proj(x_chem)          # (B, 256)

        # Branch 3: CLIP contrastive branch
        clip_cloome = self.clip_cloome_proj(h_cloome)  # (B, 256)
        clip_chem = self.clip_chem_proj(x_chem)        # (B, 256)

        # Compute CLIP loss on L2-normalized embeddings
        if compound_labels is not None:
            clip_loss, _, _ = self.clip_loss_fn(clip_cloome, clip_chem, compound_labels)
        else:
            clip_loss = torch.tensor(0.0, device=images.device)

        # CLUB disentanglement: minimize MI between shared (CLIP) and specific (concat) features
        club_cloome_mi = self.club_cloome(clip_cloome, h_cloome_proj)
        club_cloome_est = self.club_cloome.learning_loss(clip_cloome, h_cloome_proj)
        club_chem_mi = self.club_chem(clip_chem, h_chem_proj)
        club_chem_est = self.club_chem.learning_loss(clip_chem, h_chem_proj)
        club_loss = club_cloome_mi + club_cloome_est + club_chem_mi + club_chem_est

        # Fuse shared embeddings: concat → linear (STiL-style reduce, no normalization)
        h_clip = self.clip_fusion(torch.cat([clip_cloome, clip_chem], dim=-1))  # (B, 256)

        # Concatenate all 3 branches: 256 × 3 = 768
        h_concat = torch.cat([h_cloome_proj, h_chem_proj, h_clip], dim=-1)  # (B, 768)

        logits = self.classifier(h_concat)  # (B, num_tasks)

        return logits, clip_loss, club_loss


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.7,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Compute focal loss for binary classification."""
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * focal_weight * bce_loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    scaler: torch.cuda.amp.GradScaler = None,
    epoch: int = 0,
    use_wandb: bool = False,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.7,
    focal_gamma: float = 1.0,
    clip_loss_weight: float = 0.1,
    club_loss_weight: float = 0.1,
) -> tuple[float, float, float]:
    """Train for one epoch. Returns (avg_task_loss, avg_clip_loss, avg_club_loss)."""
    model.train()
    running_loss = 0.0
    running_clip_loss = 0.0
    running_club_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Support both 4-tuple (stacked) and 5-tuple (per-FOV) formats
        if len(batch) == 5:
            images_b, fov_mask_b, x_chem_b, yb, compound_labels_b = batch
            fov_mask_b = fov_mask_b.to(device)
        else:
            images_b, x_chem_b, yb, compound_labels_b = batch
            fov_mask_b = None

        images_b = images_b.to(device)
        x_chem_b = x_chem_b.to(device)
        yb = yb.to(device)
        compound_labels_b = compound_labels_b.to(device)

        mask = (yb != -1)
        if not mask.any():
            continue

        targets = (yb == 1).float()

        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, clip_loss_val, club_loss_val = model(images_b, x_chem_b, fov_mask_b, compound_labels_b)
                if use_focal_loss:
                    per_elem_loss = focal_loss(logits, targets, alpha=focal_alpha, gamma=focal_gamma)
                else:
                    per_elem_loss = F.binary_cross_entropy_with_logits(
                        logits, targets, reduction="none"
                    )
                task_loss = (per_elem_loss * mask.float()).sum() / mask.float().sum()
                loss = (task_loss + clip_loss_weight * clip_loss_val + club_loss_weight * club_loss_val) / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits, clip_loss_val, club_loss_val = model(images_b, x_chem_b, fov_mask_b, compound_labels_b)
            if use_focal_loss:
                per_elem_loss = focal_loss(logits, targets, alpha=focal_alpha, gamma=focal_gamma)
            else:
                per_elem_loss = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
            task_loss = (per_elem_loss * mask.float()).sum() / mask.float().sum()
            loss = (task_loss + clip_loss_weight * clip_loss_val + club_loss_weight * club_loss_val) / grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        running_loss += task_loss.item()
        running_clip_loss += clip_loss_val.item()
        running_club_loss += club_loss_val.item()
        n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        avg_clip = running_clip_loss / max(n_batches, 1)
        avg_club = running_club_loss / max(n_batches, 1)
        pbar.set_postfix({"task": f"{avg_loss:.4f}", "clip": f"{avg_clip:.4f}", "club": f"{avg_club:.4f}"})

        if use_wandb and (batch_idx + 1) % 50 == 0:
            try:
                wandb.log({
                    "train/batch_task_loss": avg_loss,
                    "train/batch_clip_loss": avg_clip,
                    "train/batch_club_loss": avg_club,
                    "train/batch": epoch * len(loader) + batch_idx,
                })
            except:
                pass

    pbar.close()

    # Final update if there are remaining gradients
    if n_batches % grad_accum_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return running_loss / max(n_batches, 1), running_clip_loss / max(n_batches, 1), running_club_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Eval",
    use_focal_loss: bool = True,
    focal_alpha: float = 0.7,
    focal_gamma: float = 1.0,
) -> tuple[float, float, float, float]:
    """Evaluate the model. Returns (avg_task_loss, accuracy, avg_clip_loss, avg_club_loss)."""
    model.eval()
    running_loss = 0.0
    running_clip_loss = 0.0
    running_club_loss = 0.0
    n_batches = 0
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for batch in pbar:
            if len(batch) == 5:
                images_b, fov_mask_b, x_chem_b, yb, compound_labels_b = batch
                fov_mask_b = fov_mask_b.to(device)
            else:
                images_b, x_chem_b, yb, compound_labels_b = batch
                fov_mask_b = None

            images_b = images_b.to(device)
            x_chem_b = x_chem_b.to(device)
            yb = yb.to(device)
            compound_labels_b = compound_labels_b.to(device)

            mask = (yb != -1)
            if not mask.any():
                continue

            targets = (yb == 1).float()

            logits, clip_loss_val, club_loss_val = model(images_b, x_chem_b, fov_mask_b, compound_labels_b)

            if use_focal_loss:
                per_elem_loss = focal_loss(logits, targets, alpha=focal_alpha, gamma=focal_gamma)
            else:
                per_elem_loss = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
            loss = (per_elem_loss * mask.float()).sum() / mask.float().sum()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds[mask] == targets[mask]).float().sum().item()
            total += float(mask.sum().item())

            running_loss += loss.item()
            running_clip_loss += clip_loss_val.item()
            running_club_loss += club_loss_val.item()
            n_batches += 1

            avg_loss = running_loss / max(n_batches, 1)
            avg_clip = running_clip_loss / max(n_batches, 1)
            avg_club = running_club_loss / max(n_batches, 1)
            accuracy = correct / max(total, 1.0)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "clip": f"{avg_clip:.4f}", "club": f"{avg_club:.4f}", "acc": f"{accuracy:.4f}"})

        pbar.close()

    avg_loss = running_loss / max(n_batches, 1)
    avg_clip = running_clip_loss / max(n_batches, 1)
    avg_club = running_club_loss / max(n_batches, 1)
    accuracy = correct / max(total, 1.0)

    return avg_loss, accuracy, avg_clip, avg_club


def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_names: list[str],
    desc: str = "Computing metrics",
) -> dict:
    """Compute per-assay metrics including ROC-AUC, AP, and F1."""
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for batch in pbar:
            if len(batch) == 5:
                images_b, fov_mask_b, x_chem_b, yb, _ = batch
                fov_mask_b = fov_mask_b.to(device)
            else:
                images_b, x_chem_b, yb, _ = batch
                fov_mask_b = None

            images_b = images_b.to(device)
            x_chem_b = x_chem_b.to(device)

            logits, _, _ = model(images_b, x_chem_b, fov_mask_b)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

        pbar.close()

    logits_val = np.concatenate(all_logits, axis=0)
    Y_val = np.concatenate(all_targets, axis=0)
    probs_val = 1.0 / (1.0 + np.exp(-logits_val))

    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    aucs = []
    aps = []
    f1s = []
    valid_assays = []

    n_labels = Y_val.shape[1]
    for j in range(n_labels):
        valid_mask = Y_val[:, j] != -1
        if valid_mask.sum() == 0:
            continue

        y_true = Y_val[valid_mask, j]
        y_score = probs_val[valid_mask, j]

        if len(np.unique(y_true)) < 2:
            continue

        try:
            auc = roc_auc_score(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            y_pred = (y_score >= 0.5).astype(int)
            f1 = f1_score(y_true, y_pred)

            aucs.append(auc)
            aps.append(ap)
            f1s.append(f1)
            valid_assays.append(label_names[j])
        except:
            continue

    return {
        "aucs": np.array(aucs),
        "aps": np.array(aps),
        "f1s": np.array(f1s),
        "assay_names": valid_assays,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CLOOME + ChemBERT Concatenation Model",
    )

    # Data paths
    parser.add_argument("--train-split", type=str, required=True)
    parser.add_argument("--val-split", type=str, required=True)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument(
        "--chembert-npz", type=str,
        default="/data/huadi/cellpainting_data/cpg0012/chem_features/preprocessed_chem_assays.npz",
    )
    parser.add_argument(
        "--labels-csv", type=str,
        default="/data/huadi/cellpainting_data/cpg0012/labels/compound_assay_activity.csv",
    )
    parser.add_argument(
        "--image-root", type=str,
        default="/data/huadi/cpg0012-wawer/images",
    )

    # Model architecture
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument(
        "--cloome-config", type=str,
        default="/data/huadi/cloome/src/training/model_configs/RN50.json",
    )
    parser.add_argument("--cloome-checkpoint", type=str, default=None,
                        help="Path to CLOOME checkpoint (auto-downloads from HuggingFace if not provided)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Resize FOV images to this size (e.g. 128). Full resolution if not set.")

    # Training
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backbone-lr", type=float, default=5e-5,
                        help="Learning rate for CLOOME backbone")
    parser.add_argument("--downstream-lr", type=float, default=2e-3,
                        help="Learning rate for projection + classifier")
    parser.add_argument("--freeze-epochs", type=int, default=0,
                        help="Number of epochs to freeze backbone (default: 0, no freezing)")
    parser.add_argument("--backbone-warmup-epochs", type=int, default=0,
                        help="Number of epochs to linearly warm up backbone LR after unfreezing")
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")

    # Aggregation level
    parser.add_argument(
        "--aggregation-level", type=str, choices=["well", "compound"], default="well",
    )

    # ChemBERT augmentation
    parser.add_argument("--augment-chem", action="store_true")
    parser.add_argument("--chem-noise-std", type=float, default=0.05)
    parser.add_argument("--chem-dropout-prob", type=float, default=0.0)

    # Output
    parser.add_argument("--output-model", type=str, default="results/checkpoints/cloome_concat.pt")
    parser.add_argument("--output-metrics", type=str, default="results/metrics/cloome_concat_metrics.json")
    parser.add_argument("--seed", type=int, default=42)

    # Loss function
    parser.add_argument("--use-focal-loss", action="store_true", default=True)
    parser.add_argument("--no-focal-loss", dest="use_focal_loss", action="store_false")
    parser.add_argument("--focal-alpha", type=float, default=0.7)
    parser.add_argument("--focal-gamma", type=float, default=1.0)

    # CLIP loss
    parser.add_argument("--clip-loss-weight", type=float, default=0.1,
                        help="Weight for CLIP contrastive loss")
    parser.add_argument("--clip-temperature", type=float, default=0.07,
                        help="Temperature for CLIP loss")

    # CLUB disentanglement loss
    parser.add_argument("--club-loss-weight", type=float, default=0.1,
                        help="Weight for CLUB disentanglement loss")
    parser.add_argument("--club-hidden-size", type=int, default=512,
                        help="Hidden size for CLUB estimator networks")

    # WandB logging
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")

    # Initialize WandB
    use_wandb = args.wandb_project is not None and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "model": "cloome_concat",
                "embed_dim": args.embed_dim,
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
                "clip_loss_weight": args.clip_loss_weight,
                "clip_temperature": args.clip_temperature,
                "club_loss_weight": args.club_loss_weight,
                "club_hidden_size": args.club_hidden_size,
            },
        )
        print(f"WandB initialized: project={args.wandb_project}, run={wandb.run.name}")
    elif args.wandb_project is not None and not WANDB_AVAILABLE:
        print("Warning: WandB project specified but wandb is not installed")

    # Load ChemBERT features
    print("\n=== Loading ChemBERT Features ===")
    chembert_dict, _ = load_chembert_features(args.chembert_npz)

    # Load labels
    print("\n=== Loading Labels ===")
    label_dict, label_names = load_labels(args.labels_csv)

    # Handle compound-level aggregation if requested
    train_split = args.train_split
    val_split = args.val_split
    test_split = args.test_split
    temp_files = []

    if args.aggregation_level == "compound":
        print("\n=== Creating Compound-Level Splits ===")
        print("Selecting one representative well per compound...")

        train_split = create_compound_level_split(args.train_split)
        temp_files.append(train_split)

        val_split = create_compound_level_split(args.val_split)
        temp_files.append(val_split)

        if args.test_split:
            test_split = create_compound_level_split(args.test_split)
            temp_files.append(test_split)

    # Create datasets (per-FOV full-res mode for CLOOME)
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

    # Compute normalization statistics for ChemBERT features from training set
    print("\n=== Computing ChemBERT Normalization Statistics ===")
    train_inchikeys = list(set(
        str(g.iloc[0]["INCHIKEY"]) for g in train_ds.well_groups
    ))
    all_chem_features = [chembert_dict[str(ik)] for ik in train_inchikeys if str(ik) in chembert_dict]

    X_chem_train = np.stack(all_chem_features, axis=0)
    X_chem_train = np.clip(X_chem_train, -1e3, 1e3)

    print(f"Computing normalization from {len(train_inchikeys)} unique compounds in training set")

    chem_mean = X_chem_train.mean(axis=0, keepdims=True).astype(np.float32)
    chem_std = X_chem_train.std(axis=0, keepdims=True).astype(np.float32)
    chem_std[chem_std == 0] = 1.0

    print(f"ChemBERT mean: {chem_mean.mean():.4f}, std: {chem_std.mean():.4f}")

    def normalize_batch(batch):
        """Custom collate function to normalize ChemBERT features."""
        if len(batch[0]) == 5:
            # Per-FOV format: (images, fov_mask, x_chem, y, compound_label)
            images, fov_mask, x_chem, y, compound_label = zip(*batch)
            images = torch.stack(images, dim=0)        # (B, S, 5, H, W)
            fov_mask = torch.stack(fov_mask, dim=0)    # (B, S)
            x_chem = torch.stack(x_chem, dim=0)
            y = torch.stack(y, dim=0)
            compound_label = torch.stack(compound_label, dim=0)

            x_chem = (x_chem - torch.from_numpy(chem_mean[0])) / torch.from_numpy(chem_std[0])

            return images, fov_mask, x_chem, y, compound_label
        else:
            # Stacked format: (stacked_images, x_chem, y, compound_label)
            stacked_images, x_chem, y, compound_label = zip(*batch)
            stacked_images = torch.stack(stacked_images, dim=0)
            x_chem = torch.stack(x_chem, dim=0)
            y = torch.stack(y, dim=0)
            compound_label = torch.stack(compound_label, dim=0)

            x_chem = (x_chem - torch.from_numpy(chem_mean[0])) / torch.from_numpy(chem_std[0])

            return stacked_images, x_chem, y, compound_label

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=normalize_batch,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=normalize_batch,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    test_loader = None
    if test_ds:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=normalize_batch,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Initializing Model on {device} ===")

    model = CLOOMEConcatMLP(
        chem_dim=768,
        num_tasks=len(label_names),
        embed_dim=args.embed_dim,
        clip_temperature=args.clip_temperature,
        club_hidden_size=args.club_hidden_size,
        cloome_config=args.cloome_config,
        cloome_checkpoint=args.cloome_checkpoint,
    ).to(device)

    # Print model statistics
    n_backbone_params = sum(p.numel() for p in model.cloome_encoder.parameters())
    n_downstream_params = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("cloome_encoder")
    )
    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  CLOOME backbone params: {n_backbone_params:,}")
    print(f"  Downstream params (proj + classifier): {n_downstream_params:,}")
    print(f"  Total params: {n_total_params:,}")
    print(f"  Trainable params: {n_trainable_params:,}")

    # Setup optimizer with differential learning rates
    print("\n=== Setting Up Optimizer ===")

    # Freeze backbone if freeze_epochs > 0
    if args.freeze_epochs > 0:
        for p in model.cloome_encoder.parameters():
            p.requires_grad = False
        print(f"Backbone FROZEN for first {args.freeze_epochs} epochs")

    backbone_params = list(model.cloome_encoder.parameters())
    downstream_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("cloome_encoder")
    ]

    param_groups = [
        {"params": backbone_params, "lr": 0.0 if args.freeze_epochs > 0 else args.backbone_lr, "weight_decay": 0.01},
        {"params": downstream_params, "lr": args.downstream_lr, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups)

    print(f"Optimizer: AdamW")
    print(f"  Backbone lr: {0.0 if args.freeze_epochs > 0 else args.backbone_lr}, weight_decay: 0.01")
    print(f"  Downstream lr: {args.downstream_lr}, weight_decay: 0.0")
    if args.backbone_warmup_epochs > 0:
        print(f"  Backbone warmup: {args.backbone_warmup_epochs} epochs after unfreeze")

    # Setup gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    if args.use_amp:
        print(f"Using mixed precision training (bfloat16)")

    print(f"Gradient accumulation steps: {args.grad_accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")

    # Training loop
    print("\n=== Training ===")
    best_val_loss = float('inf')
    best_epoch = 0

    if args.use_focal_loss:
        print(f"\nUsing Focal Loss: alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    else:
        print(f"\nUsing Binary Cross-Entropy Loss")

    if args.augment_chem:
        print(f"\nChemBERT augmentation enabled:")
        print(f"  Gaussian noise std: {args.chem_noise_std}")
        print(f"  Feature dropout prob: {args.chem_dropout_prob}")

    print(f"\nCLIP loss weight: {args.clip_loss_weight}, temperature: {args.clip_temperature}")
    print(f"CLUB loss weight: {args.club_loss_weight}, hidden size: {args.club_hidden_size}")

    for epoch in tqdm(range(1, args.max_iter + 1), desc="Training epochs"):
        # Handle backbone freeze/unfreeze and warmup
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            # Unfreeze backbone
            for p in model.cloome_encoder.parameters():
                p.requires_grad = True
            print(f"\n>>> Epoch {epoch}: Backbone UNFROZEN")

        if epoch > args.freeze_epochs:
            # Set backbone LR (with warmup if configured)
            epochs_since_unfreeze = epoch - args.freeze_epochs
            if args.backbone_warmup_epochs > 0 and epochs_since_unfreeze <= args.backbone_warmup_epochs:
                warmup_factor = epochs_since_unfreeze / args.backbone_warmup_epochs
                backbone_lr = args.backbone_lr * warmup_factor
                print(f"  Backbone LR warmup: {backbone_lr:.2e} ({epochs_since_unfreeze}/{args.backbone_warmup_epochs})")
            else:
                backbone_lr = args.backbone_lr
            optimizer.param_groups[0]["lr"] = backbone_lr

        train_loss, train_clip_loss, train_club_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_accum_steps=args.grad_accum_steps,
            use_amp=args.use_amp,
            scaler=scaler,
            epoch=epoch,
            use_wandb=use_wandb,
            use_focal_loss=args.use_focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            clip_loss_weight=args.clip_loss_weight,
            club_loss_weight=args.club_loss_weight,
        )
        val_loss, val_acc, val_clip_loss, val_club_loss = evaluate(
            model, val_loader, device, desc=f"Epoch {epoch} [Val]",
            use_focal_loss=args.use_focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
        )

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        # Compute full metrics every epoch
        epoch_metrics = compute_metrics(model, val_loader, device, label_names, desc=f"Epoch {epoch} [Metrics]")
        mean_auroc = epoch_metrics["aucs"].mean() if len(epoch_metrics["aucs"]) > 0 else 0.0
        mean_ap = epoch_metrics["aps"].mean() if len(epoch_metrics["aps"]) > 0 else 0.0
        mean_f1 = epoch_metrics["f1s"].mean() if len(epoch_metrics["f1s"]) > 0 else 0.0
        n_auc_gt_09 = (epoch_metrics["aucs"] > 0.9).sum() if len(epoch_metrics["aucs"]) > 0 else 0
        n_auc_gt_08 = (epoch_metrics["aucs"] > 0.8).sum() if len(epoch_metrics["aucs"]) > 0 else 0
        n_auc_gt_07 = (epoch_metrics["aucs"] > 0.7).sum() if len(epoch_metrics["aucs"]) > 0 else 0

        print(
            f"Epoch {epoch:3d}/{args.max_iter} | "
            f"train_loss={train_loss:.4f}, train_clip={train_clip_loss:.4f}, train_club={train_club_loss:.4f} | "
            f"val_loss={val_loss:.4f}, val_clip={val_clip_loss:.4f}, val_club={val_club_loss:.4f}, val_acc={val_acc:.4f} | "
            f"val_auroc={mean_auroc:.4f}, val_f1={mean_f1:.4f}, #auc>0.7={n_auc_gt_07}"
        )

        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/clip_loss": train_clip_loss,
                "train/club_loss": train_club_loss,
                "val/loss": val_loss,
                "val/clip_loss": val_clip_loss,
                "val/club_loss": val_club_loss,
                "val/accuracy": val_acc,
                "val/mean_roc_auc": mean_auroc,
                "val/mean_ap": mean_ap,
                "val/mean_f1": mean_f1,
                "val/n_assays_auc_gt_0.9": n_auc_gt_09,
                "val/n_assays_auc_gt_0.8": n_auc_gt_08,
                "val/n_assays_auc_gt_0.7": n_auc_gt_07,
            }, step=epoch)

    # Compute final validation metrics
    print("\n=== Final Evaluation ===")
    print("Validation set metrics:")
    val_metrics = compute_metrics(model, val_loader, device, label_names, desc="Final Val Metrics")

    if len(val_metrics["aucs"]) > 0:
        mean_auc = val_metrics["aucs"].mean()
        std_auc = val_metrics["aucs"].std()
        mean_ap = val_metrics["aps"].mean()
        std_ap = val_metrics["aps"].std()

        print(f"  Mean ROC-AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"  Mean AP: {mean_ap:.4f} +/- {std_ap:.4f}")
        print(f"  #assays with ROC-AUC > 0.9: {(val_metrics['aucs'] > 0.9).sum()}")
        print(f"  #assays with ROC-AUC > 0.8: {(val_metrics['aucs'] > 0.8).sum()}")
        print(f"  #assays with ROC-AUC > 0.7: {(val_metrics['aucs'] > 0.7).sum()}")

        if use_wandb:
            wandb.log({
                "final/val_mean_roc_auc": mean_auc,
                "final/val_std_roc_auc": std_auc,
                "final/val_mean_ap": mean_ap,
                "final/val_std_ap": std_ap,
                "final/val_n_assays_auc_gt_0.9": int((val_metrics['aucs'] > 0.9).sum()),
                "final/val_n_assays_auc_gt_0.8": int((val_metrics['aucs'] > 0.8).sum()),
                "final/val_n_assays_auc_gt_0.7": int((val_metrics['aucs'] > 0.7).sum()),
            })

    # Compute test metrics if test set provided
    test_metrics = None
    if test_loader:
        print("\nTest set metrics:")
        test_metrics = compute_metrics(model, test_loader, device, label_names, desc="Final Test Metrics")

        if len(test_metrics["aucs"]) > 0:
            mean_auc = test_metrics["aucs"].mean()
            std_auc = test_metrics["aucs"].std()
            mean_ap = test_metrics["aps"].mean()
            std_ap = test_metrics["aps"].std()

            print(f"  Mean ROC-AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
            print(f"  Mean AP: {mean_ap:.4f} +/- {std_ap:.4f}")
            print(f"  #assays with ROC-AUC > 0.9: {(test_metrics['aucs'] > 0.9).sum()}")
            print(f"  #assays with ROC-AUC > 0.8: {(test_metrics['aucs'] > 0.8).sum()}")
            print(f"  #assays with ROC-AUC > 0.7: {(test_metrics['aucs'] > 0.7).sum()}")

            if use_wandb:
                wandb.log({
                    "final/test_mean_roc_auc": mean_auc,
                    "final/test_std_roc_auc": std_auc,
                    "final/test_mean_ap": mean_ap,
                    "final/test_std_ap": std_ap,
                    "final/test_n_assays_auc_gt_0.9": int((test_metrics['aucs'] > 0.9).sum()),
                    "final/test_n_assays_auc_gt_0.8": int((test_metrics['aucs'] > 0.8).sum()),
                    "final/test_n_assays_auc_gt_0.7": int((test_metrics['aucs'] > 0.7).sum()),
                })

    # Save model checkpoint
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "chem_dim": 768,
        "num_tasks": len(label_names),
        "embed_dim": args.embed_dim,
        "label_names": label_names,
        "chem_mean": chem_mean,
        "chem_std": chem_std,
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "cloome_config": args.cloome_config,
        "backbone_lr": args.backbone_lr,
        "downstream_lr": args.downstream_lr,
        "aggregation_level": args.aggregation_level,
        "clip_loss_weight": args.clip_loss_weight,
        "clip_temperature": args.clip_temperature,
        "club_loss_weight": args.club_loss_weight,
        "club_hidden_size": args.club_hidden_size,
    }
    torch.save(checkpoint, args.output_model)
    print(f"\nSaved model checkpoint to {args.output_model}")

    # Save metrics to JSON
    os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
    metrics_dict = {
        "hyperparameters": {
            "embed_dim": args.embed_dim,
            "backbone_lr": args.backbone_lr,
            "downstream_lr": args.downstream_lr,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
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
        "validation": {
            "mean_roc_auc": float(val_metrics["aucs"].mean()) if len(val_metrics["aucs"]) > 0 else None,
            "std_roc_auc": float(val_metrics["aucs"].std()) if len(val_metrics["aucs"]) > 0 else None,
            "mean_ap": float(val_metrics["aps"].mean()) if len(val_metrics["aps"]) > 0 else None,
            "num_assays_auc_gt_09": int((val_metrics["aucs"] > 0.9).sum()) if len(val_metrics["aucs"]) > 0 else 0,
            "num_assays_auc_gt_08": int((val_metrics["aucs"] > 0.8).sum()) if len(val_metrics["aucs"]) > 0 else 0,
            "num_assays_auc_gt_07": int((val_metrics["aucs"] > 0.7).sum()) if len(val_metrics["aucs"]) > 0 else 0,
        },
        "best_epoch": best_epoch,
        "val_loss": float(best_val_loss),
    }

    if test_metrics is not None and len(test_metrics["aucs"]) > 0:
        metrics_dict["test"] = {
            "mean_roc_auc": float(test_metrics["aucs"].mean()),
            "std_roc_auc": float(test_metrics["aucs"].std()),
            "mean_ap": float(test_metrics["aps"].mean()),
            "num_assays_auc_gt_09": int((test_metrics["aucs"] > 0.9).sum()),
            "num_assays_auc_gt_08": int((test_metrics["aucs"] > 0.8).sum()),
            "num_assays_auc_gt_07": int((test_metrics["aucs"] > 0.7).sum()),
        }

    with open(args.output_metrics, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved metrics to {args.output_metrics}")

    # Log model statistics to wandb
    if use_wandb:
        wandb.log({
            "model/cloome_backbone_params": n_backbone_params,
            "model/downstream_params": n_downstream_params,
            "model/total_params": n_total_params,
            "model/trainable_params": n_trainable_params,
            "best_epoch": best_epoch,
        })

    # Clean up temporary files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Cleaned up temporary file: {temp_file}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()
        print("WandB run finished")


if __name__ == "__main__":
    main()
