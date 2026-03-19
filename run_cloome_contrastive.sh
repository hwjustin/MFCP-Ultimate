#!/bin/bash
# Multi-GPU training on GPUs 0 and 2; batch_size=256 total (128 per GPU)
# NCCL_P2P_DISABLE=1 helps when GPUs 0 and 2 are not directly connected (NVLink)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cpg0012
CUDA_VISIBLE_DEVICES=0,2 NCCL_P2P_DISABLE=1 python -m torch.distributed.run --nproc_per_node=2 train_cloome_chembert_contrastive.py \
    --train-split /data/huadi/cellpainting_data/cpg0012/splits/datasplit1-train.csv \
    --val-split /data/huadi/cellpainting_data/cpg0012/splits/datasplit1-val.csv \
    --test-split /data/huadi/cellpainting_data/cpg0012/splits/datasplit1-test.csv \
    --image-root /data/huadi/cpg0012-wawer/images \
    --cloome-config /data/huadi/cloome/src/training/model_configs/RN50.json \
    --aggregation-level well \
    --image-size 224 \
    --backbone-lr 5e-5 \
    --downstream-lr 2e-4 \
    --embed-dim 256 \
    --clip-temperature 0.07 \
    --batch-size 256 \
    --grad-accum-steps 2 \
    --max-iter 50 \
    --use-amp \
    --wandb-mode disabled \
    --output-model results/checkpoints/cloome_contrastive.pt \
    --output-metrics results/metrics/cloome_contrastive_metrics.json
    # --wandb-project MFCP \
    # --wandb-run-name cloome-contrastive-only

