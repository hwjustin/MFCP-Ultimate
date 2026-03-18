#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_cloome_chembert_concat.py \
    --train-split /data/huadi/cellpainting_data/cpg0012/splits/datasplit1-train.csv \
    --val-split /data/huadi/cellpainting_data/cpg0012/splits/datasplit1-val.csv \
    --test-split /data/huadi/cellpainting_data/cpg0012/splits/datasplit1-test.csv \
    --image-root /data/huadi/cpg0012-wawer/images \
    --cloome-config /data/huadi/cloome/src/training/model_configs/RN50.json \
    --aggregation-level well \
    --image-size 224 \
    --backbone-lr 5e-4 \
    --downstream-lr 2e-3 \
    --embed-dim 256 \
    --batch-size 64 \
    --grad-accum-steps 4 \
    --max-iter 50 \
    --clip-loss-weight 0.1 \
    --clip-temperature 0.07 \
    --use-focal-loss --use-amp \
    --wandb-project MFCP \
    --wandb-run-name cloome-3branch-clip
