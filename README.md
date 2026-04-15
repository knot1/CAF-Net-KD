## ⚙️ **Installation & Dependencies**

Before running the code, make sure you have the following dependencies installed:

```python
conda env create -f requirements.yml
```

## 🛰️ **Datasets**

Extensive experiments were conducted on four public datasets:

- ISPRS Vaihingen  &nbsp; &nbsp;          [Download Dataset](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)
- ISPRS Potsdam  &nbsp; &nbsp;            [Download Dataset](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)

## 🚀 **Usage: Training CAF-Net**


To train **CAF-Net** on the ISPRS Vaihingen dataset, use the following command:

```python
bash train_Vaihingen.sh 0 
```

<<<<<<< HEAD

# CAF-Net-KD
=======
## 🎓 **Knowledge Distillation**

CAF-Net supports optional knowledge distillation training: a pretrained teacher (e.g. MiT-B4 backbone) guides a lightweight student (e.g. MiT-B0) using pixel-wise logits KD and boundary-aware KD.

Example command:

```bash
python main.py \
  training_dataset=Potsdam \
  model.backbone=mit_b0 \
  training.distill.enable=true \
  training.distill.teacher_ckpt=/path/to/best_model_potsdam \
  training.distill.teacher_backbone=mit_b4
```

See [docs/distill.md](docs/distill.md) for full details and ablation guidance.
>>>>>>> m0蒸馏
