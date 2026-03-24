#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_cloome_chembert_concat_frozen.py \
    --train-split /data/huadi/cellpainting_data/cpg0012/splits/datasplit1-train.csv \
    --val-split /data/huadi/cellpainting_data/cpg0012/splits/datasplit1-val.csv \
    --test-split /data/huadi/cellpainting_data/cpg0012/splits/datasplit1-test.csv \
    --image-root /data/huadi/cpg0012-wawer/images \
    --cloome-config /data/huadi/cloome/src/training/model_configs/RN50.json \
    --aggregation-level well \
    --image-size 224 \
    --downstream-lr 2e-4 \
    --embed-dim 256 \
    --batch-size 64 \
    --grad-accum-steps 4 \
    --max-iter 50 \
    --clip-loss-weight 0.5 \
    --clip-temperature 0.07 \
    --club-loss-weight 0.1 \
    --club-hidden-size 512 \
    --no-focal-loss --use-amp \
    --wandb-mode disabled
    # --wandb-project MFCP \
    # --wandb-run-name cloome-3branch-clip-frozen

