# Knowledge Distillation for CAF-Net

This guide explains how to train a lightweight **student** CAF-Net (MiT-B0 backbone) using a pretrained **teacher** CAF-Net checkpoint.

---

## Overview

The distillation setup adds two extra loss terms on top of the standard segmentation loss:

| Loss term | Formula | Purpose |
|-----------|---------|---------|
| Pixel-wise logits KD | `KL(softmax(s/T) ‖ softmax(t/T)) × T²` | Transfer class probability distributions at every pixel |
| Boundary-aware KD | Same KL, weighted by teacher boundary map | Recover fine boundary details that lightweight backbones tend to lose |

Total loss:

```
L = L_seg  +  λ_kd × L_kd  +  λ_edge × L_edgeKD
```

---

## Configuration

The distillation knobs live under `training.distill` in `config.yaml`:

```yaml
training:
  distill:
    enable: false                 # set to true to activate distillation
    teacher_ckpt: ""              # absolute path to teacher checkpoint file
    teacher_backbone: "mit_b4"    # backbone used by the teacher model
    T: 4.0                        # distillation temperature
    lambda_kd: 1.0                # weight for pixel-wise KD loss
    lambda_edge: 1.0              # weight for boundary-aware KD loss
    edge_k: 3                     # morphological kernel size for boundary detection
    edge_boost: 1.0               # additional weight at boundary pixels (base = 1)
```

The student backbone is selected via `model.backbone` (default `mit_b4`; use `mit_b0` for the lightest student).

---

## Quick-start command

```bash
python main.py \
  training_dataset=Potsdam \
  model.backbone=mit_b0 \
  training.distill.enable=true \
  training.distill.teacher_ckpt=/home/wsj/FDMF-Net/Baseline_Potsdam_42-1/2026-03-17_17-36-21/results_Baseline_potsdam/best_model_potsdam \
  training.distill.teacher_backbone=mit_b4
```

> **Tip**: You can override any parameter through Hydra without editing `config.yaml`.

---

## Teacher / Student setup for paper experiments

| Role | Backbone | Notes |
|------|----------|-------|
| Teacher | MiT-B4 (default) | Pretrained CAF-Net checkpoint, kept frozen |
| Student | MiT-B0 | Trained from scratch with KD supervision |

Suggested ablation table for a paper:

| Model | mIoU | Params | FLOPs | FPS |
|-------|------|--------|-------|-----|
| Teacher (MiT-B4) | — | — | — | — |
| Student baseline (MiT-B0, no KD) | — | — | — | — |
| Student + Logits KD | — | — | — | — |
| Student + Logits KD + Boundary KD | — | — | — | — |

Fill in the numbers from your own runs.

---

## Key implementation files

| File | What changed |
|------|-------------|
| `config.yaml` | Added `training.distill` section |
| `train.py` | Added `pixel_kd_kl`, `edge_weight_from_predmask`; updated `train()` to accept `teacher_model` / `kd_cfg` |
| `main.py` | Instantiates teacher, loads checkpoint, wraps in `DataParallel`, passes to `train()` |
