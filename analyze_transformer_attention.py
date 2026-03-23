"""
Analyze whether the Task-Query Transformer is doing useful work.

1. Cross-attention analysis: per-task attention to [cell_specific, shared, chem_specific]
2. Task embedding similarity: do related assays have similar embeddings?
3. Attention diversity: is attention uniform (bad) or task-specific (good)?

Run: python analyze_transformer_attention.py --checkpoint results/checkpoints/cloome_concat.pt
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_cloome_chembert_concat import (
    CLOOMEConcatMLP,
    load_chembert_features,
    load_labels,
)
from encoder.dataset import CellPaintingImageDataset


def normalize_batch_from_checkpoint(batch, chem_mean, chem_std):
    """Collate and normalize ChemBERT features using checkpoint stats."""
    if len(batch[0]) == 5:
        images, fov_mask, x_chem, y, compound_label = zip(*batch)
        images = torch.stack(images, dim=0)
        fov_mask = torch.stack(fov_mask, dim=0)
        x_chem = torch.stack(x_chem, dim=0)
        y = torch.stack(y, dim=0)
        compound_label = torch.stack(compound_label, dim=0)
        x_chem = (x_chem - torch.from_numpy(chem_mean[0])) / torch.from_numpy(chem_std[0])
        return images, fov_mask, x_chem, y, compound_label
    else:
        stacked_images, x_chem, y, compound_label = zip(*batch)
        stacked_images = torch.stack(stacked_images, dim=0)
        x_chem = torch.stack(x_chem, dim=0)
        y = torch.stack(y, dim=0)
        compound_label = torch.stack(compound_label, dim=0)
        x_chem = (x_chem - torch.from_numpy(chem_mean[0])) / torch.from_numpy(chem_std[0])
        return stacked_images, x_chem, y, compound_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/cloome_concat.pt")
    parser.add_argument("--val-split", type=str,
                        default="/data/huadi/cellpainting_data/cpg0012/splits/datasplit1-val.csv")
    parser.add_argument("--chembert-npz", type=str,
                        default="/data/huadi/cellpainting_data/cpg0012/chem_features/preprocessed_chem_assays.npz")
    parser.add_argument("--labels-csv", type=str,
                        default="/data/huadi/cellpainting_data/cpg0012/labels/compound_assay_activity.csv")
    parser.add_argument("--image-root", type=str, default="/data/huadi/cpg0012-wawer/images")
    parser.add_argument("--cloome-config", type=str,
                        default="/data/huadi/cloome/src/training/model_configs/RN50.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-batches", type=int, default=50,
                        help="Max val batches to run (for speed)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results/transformer_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Loading checkpoint ===")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    chem_mean = ckpt["chem_mean"]
    chem_std = ckpt["chem_std"]
    label_names = ckpt["label_names"]
    num_tasks = len(label_names)
    embed_dim = ckpt["embed_dim"]
    transformer_dim = ckpt.get("transformer_dim", embed_dim)
    transformer_heads = ckpt.get("transformer_heads", 8)
    transformer_layers = ckpt.get("transformer_layers", 2)
    transformer_mlp_ratio = ckpt.get("transformer_mlp_ratio", 4.0)
    transformer_dropout = ckpt.get("transformer_dropout", 0.1)

    print("=== Building model ===")
    model = CLOOMEConcatMLP(
        chem_dim=768,
        num_tasks=num_tasks,
        embed_dim=embed_dim,
        transformer_dim=transformer_dim,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        transformer_mlp_ratio=transformer_mlp_ratio,
        transformer_dropout=transformer_dropout,
        cloome_config=args.cloome_config,
        cloome_checkpoint=None,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print("=== Loading data ===")
    chembert_dict, _ = load_chembert_features(args.chembert_npz)
    label_dict, _ = load_labels(args.labels_csv)

    val_ds = CellPaintingImageDataset(
        split_csv=args.val_split,
        chembert_dict=chembert_dict,
        label_dict=label_dict,
        image_root=args.image_root,
        cloome_multi_fov=True,
        image_size=args.image_size,
        augment_chem=False,
    )

    def collate(batch):
        return normalize_batch_from_checkpoint(batch, chem_mean, chem_std)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- 1. Collect cross-attention weights ---
    attention_store = []  # list of (B, num_heads, 209, 3) per layer per batch

    def make_hook(layer_idx):
        def hook(module, inputs, outputs):
            # outputs: (attn_output, attn_output_weights) when need_weights=True
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                attn_weights = outputs[1].detach().cpu().numpy()  # (B, 209, 3)
                attention_store.append((layer_idx, attn_weights))
        return hook

    hooks = []
    for i, layer in enumerate(model.classifier.layers):
        h = layer.cross_attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    print("=== Extracting cross-attention on validation set ===")
    attention_per_layer = {i: [] for i in range(transformer_layers)}
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 5:
                images, fov_mask, x_chem, _, _ = batch
                fov_mask = fov_mask.to(device)
            else:
                images, x_chem, _, _ = batch
                fov_mask = None
            images = images.to(device)
            x_chem = x_chem.to(device)

            attention_store.clear()
            _ = model(images, x_chem, fov_mask)

            for layer_idx, attn in attention_store:
                attention_per_layer[layer_idx].append(attn)

            n_batches += 1
            if n_batches >= args.max_batches:
                break

    for h in hooks:
        h.remove()

    # Aggregate: (B, num_heads, 209, 3) -> mean over batches and heads -> (209, 3)
    feature_names = ["cell_specific", "shared", "chem_specific"]

    results = {
        "attention_per_task": {},  # task_idx -> [cell, shared, chem] per layer
        "attention_diversity": {},  # entropy / max entropy per layer
        "task_embedding_stats": {},
        "summary": [],
    }

    for layer_idx in range(transformer_layers):
        if not attention_per_layer[layer_idx]:
            continue
        arr = np.concatenate(attention_per_layer[layer_idx], axis=0)  # (N, 209, 3)
        # PyTorch MHA returns (B, L, S) = (B, 209, 3) - averaged over heads
        mean_attn = np.asarray(arr).mean(axis=0)  # (209, 3)
        if mean_attn.ndim == 1:
            mean_attn = mean_attn.reshape(1, -1)
        if mean_attn.shape[0] != num_tasks:
            mean_attn = mean_attn.T
        assert mean_attn.shape == (num_tasks, 3), f"Unexpected shape: {mean_attn.shape}"
        if mean_attn.ndim == 1:
            mean_attn = mean_attn.reshape(1, -1)  # handle edge case
        results["attention_per_task"][f"layer_{layer_idx}"] = mean_attn.tolist()

        # Diversity: if uniform, entropy is max (ln(3)); if peaked, lower
        # Per-task entropy (over 3 features)
        eps = 1e-8
        entropies = -(mean_attn * np.log(mean_attn + eps)).sum(axis=-1)
        max_entropy = np.log(3)
        normalized_entropy = entropies / max_entropy  # 1 = uniform, 0 = peaked
        results["attention_diversity"][f"layer_{layer_idx}"] = {
            "mean_normalized_entropy": float(normalized_entropy.mean()),
            "std": float(normalized_entropy.std()),
            "min": float(normalized_entropy.min()),
            "max": float(normalized_entropy.max()),
        }

        # Which feature dominates per task?
        argmax_feature = mean_attn.argmax(axis=-1)
        counts = [int((argmax_feature == i).sum()) for i in range(3)]
        results["summary"].append({
            "layer": layer_idx,
            "tasks_prefer_cell_specific": counts[0],
            "tasks_prefer_shared": counts[1],
            "tasks_prefer_chem_specific": counts[2],
            "mean_normalized_entropy": float(normalized_entropy.mean()),
        })

    # --- 2. Task embedding analysis ---
    task_emb = model.classifier.task_queries.detach().cpu().numpy()  # (209, D)
    # Cosine similarity matrix
    norm_emb = task_emb / (np.linalg.norm(task_emb, axis=1, keepdims=True) + 1e-8)
    sim_matrix = norm_emb @ norm_emb.T
    results["task_embedding_similarity_mean"] = float(np.mean(sim_matrix))
    results["task_embedding_similarity_std"] = float(np.std(sim_matrix))
    # Diagonal should be 1; off-diagonal tells us how similar tasks are
    off_diag = sim_matrix[~np.eye(209, dtype=bool)]
    results["task_embedding_off_diag_mean"] = float(off_diag.mean())
    results["task_embedding_off_diag_std"] = float(off_diag.std())

    # Save attention matrix for last layer and optional heatmap
    if transformer_layers > 0:
        last_layer_key = f"layer_{transformer_layers - 1}"
        attn_mat = np.array(results["attention_per_task"][last_layer_key])
        np.save(os.path.join(args.output_dir, "attention_209x3.npy"), attn_mat)
        np.save(os.path.join(args.output_dir, "task_embedding_similarity.npy"), sim_matrix)

        # Plot heatmap if matplotlib available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 12))
            im = ax.imshow(attn_mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_yticks([])
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(feature_names)
            ax.set_xlabel("Feature")
            ax.set_ylabel("Task (0-208)")
            ax.set_title("Cross-attention: tasks -> [cell_specific, shared, chem_specific]")
            plt.colorbar(im, ax=ax, label="Attention weight")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "attention_heatmap.png"), dpi=120)
            plt.close()
            print(f"Saved heatmap to {args.output_dir}/attention_heatmap.png")
        except ImportError:
            pass

    # --- Save report ---
    report_path = os.path.join(args.output_dir, "transformer_analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved report to {report_path}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("TRANSFORMER ANALYSIS SUMMARY")
    print("=" * 60)
    for s in results["summary"]:
        print(f"\nLayer {s['layer']}:")
        print(f"  Tasks preferring cell_specific: {s['tasks_prefer_cell_specific']}")
        print(f"  Tasks preferring shared:       {s['tasks_prefer_shared']}")
        print(f"  Tasks preferring chem_specific:{s['tasks_prefer_chem_specific']}")
        print(f"  Attention diversity (1=uniform, 0=peaked): {s['mean_normalized_entropy']:.4f}")

    print(f"\nTask embedding similarity (off-diagonal): mean={results['task_embedding_off_diag_mean']:.4f}, std={results['task_embedding_off_diag_std']:.4f}")
    print("\nInterpretation:")
    if results["summary"]:
        ent = results["summary"][-1]["mean_normalized_entropy"]
        if ent > 0.95:
            print("  - Attention is nearly UNIFORM across features -> transformer may NOT be task-adaptive")
        elif ent < 0.5:
            print("  - Attention is PEAKED (task-specific) -> transformer IS learning different patterns per task")
        else:
            print("  - Attention shows MODERATE task specificity -> transformer has some effect")
    print("=" * 60)


if __name__ == "__main__":
    main()
