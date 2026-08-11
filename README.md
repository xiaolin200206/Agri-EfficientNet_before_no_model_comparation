# Agri-EfficientNet: Durian Disease Classification

A lightweight **Lesion Focus Attention (LFA)** module for EfficientNet-B0, designed for plant disease diagnosis under real field conditions.

LFA adds **1,281 parameters (+0.032%)** to the backbone with no measurable increase in model size or CPU inference latency. This repository contains the full implementation, evaluation pipeline, and figure generation code.

---

## Highlights

- **Field-collected dataset** — 560 images from commercial orchards, photographed under uncontrolled lighting, background and framing conditions, rather than in a laboratory setting.
- **Minimal-cost attention** — a single 1×1 convolution and sigmoid applied to the deepest feature map, learned end to end.
- **Controlled ablation** — LFA compared against no attention, SE and CBAM under identical seeds, schedules and augmentation.
- **Cross-country generalisation test** — models trained on Malaysian data evaluated zero-shot on an independent Vietnamese dataset, a test rarely reported in the plant disease literature.
- **Robustness analysis** — performance measured under five simulated field degradations.
- **Honest statistics** — five-fold cross-validation and McNemar significance tests, with the limitations of a 60-image test set stated rather than glossed over.

---

## The LFA module

Standard EfficientNet ends with global average pooling, which weights every spatial position equally. In a field photograph most of the frame is soil, sky and background foliage, so averaging dilutes the lesion signal. LFA learns a spatial mask before pooling:

```python
class LesionFocusAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid        = nn.Sigmoid()
        self.multiply       = FloatFunctional()

    def forward(self, x):
        return self.multiply.mul(x, self.sigmoid(self.attention_conv(x)))
```

The module is inserted after the deepest convolutional block, where features are semantically rich, rather than at multiple intermediate stages as in CBAM.

---

## Results

Two experiments report LFA's effect, under different protocols. They are not interchangeable, and the gain differs between them:

- **Ablation (Table 1)** — all variants trained under identical conditions. LFA gains **+2.5 pp Macro F1** over no attention.
- **Model comparison (Table 2)** — each architecture trained under its own standard protocol. Agri-EfficientNet gains **+7.8 pp Macro F1** over EfficientNet-B0 without LFA.

### Table 1 — Ablation study (identical training conditions)

| Model | Acc (%) | Prec (%) | Rec (%) | Macro F1 (%) | Size (MB) | CPU (ms) |
|---|---|---|---|---|---|---|
| EfficientNet-B0 (no attention) | 85.0 | 80.1 | 89.1 | 82.7 | 15.60 | 17.25 |
| EfficientNet-B0 + SE | 80.0 | 77.8 | 85.7 | 78.9 | 16.38 | 19.70 |
| EfficientNet-B0 + CBAM | 88.3 | 83.6 | 91.6 | 85.3 | 16.38 | 18.65 |
| **Agri-EfficientNet + LFA** | **88.3** | **83.5** | **91.6** | **85.2** | **15.60** | **18.63** |

SE regresses below the baseline, which suggests channel-only recalibration introduces optimisation instability on a dataset this small and imbalanced. CBAM edges out LFA by 0.1 pp F1 but costs 0.78 MB more. LFA is retained as the simpler and smaller option.

> **On the CPU timings.** The values above are measured inside the training loop and include batch processing overhead. Dedicated CPU-only profiling over 100 forward passes gives **18.55 ms** for the no-attention baseline and **18.39 ms** for LFA — that is, LFA's measured overhead is −0.16 ms, within noise.

### Table 2 — Model comparison (standard protocols, test set n = 60)

| Model | Acc (%) | Macro F1 (%) | Params (M) | Size (MB) | CPU (ms) |
|---|---|---|---|---|---|
| ConvNeXt-Tiny | 96.7 | 97.6 | 27.82 | 106.21 | 43.07 |
| ResNet-101 | 95.0 | 96.4 | 42.51 | 162.77 | 77.09 |
| MobileNetV2 | 90.0 | 92.7 | 2.23 | 8.74 | 13.23 |
| VGG-16 | 90.0 | 92.6 | 134.28 | 512.25 | 103.39 |
| ResNet-50 | 90.0 | 92.6 | 23.52 | 90.02 | 41.71 |
| MobileNetV3-Large | 91.7 | 87.8 | 4.21 | 16.25 | 13.00 |
| **Agri-EfficientNet (ours)** | **91.7** | **87.8** | **4.02** | **15.61** | **18.22** |
| EfficientNet-B0 (no LFA) | 85.0 | 80.0 | 4.01 | 15.60 | 18.14 |
| EfficientNetV2-S | 83.3 | 68.2 | 20.18 | 77.86 | 50.48 |
| ShuffleNetV2-1.0x | 16.7 | 20.8 | 1.26 | 4.97 | 13.18 |

This is a size-constrained result, not a leaderboard win. ConvNeXt-Tiny scores 9.8 pp higher but is 6.8× larger and 2.4× slower on CPU. Within the sub-20 MB bracket, Agri-EfficientNet ties MobileNetV3-Large on F1 at a slightly smaller size. EfficientNetV2-S is instructive in the other direction: five times the parameters and 19.6 pp *worse*, which shows parameter scale alone does not help on a small domain-specific dataset.

A supplementary experiment applying LFA to a MobileNetV2 backbone (`mobilenetv2_lfa.py`) reaches **96.4% Macro F1 at 8.75 MB**, a 3.7 pp gain over standard MobileNetV2, confirming LFA transfers across backbones. Its CPU latency is **56.03 ms**, however — roughly four times standard MobileNetV2 — likely reflecting how the attention operation interacts with depthwise-separable layers. Operator-level profiling would be needed to confirm this.

### Table 3 — Robustness under simulated field conditions (Macro F1, %)

| Perturbation | Agri-EfficientNet (LFA) | EfficientNet-B0 (no LFA) | ResNet-50 |
|---|---|---|---|
| Clean (baseline) | 87.8 | 80.0 | 92.6 |
| Gaussian noise (σ = 25) | 86.6 | 72.2 | 92.6 |
| Low brightness (×0.4) | 96.3 | 75.9 | 92.6 |
| High brightness (×1.6) | 91.3 | 66.1 | 92.6 |
| Motion blur (r = 3 px) | 85.3 | 77.2 | 92.6 |
| Occlusion (30% black) | 86.7 | 75.8 | 92.6 |

LFA's advantage over the no-attention baseline is largest under exposure problems: **+25.2 pp** under high brightness and **+20.4 pp** under low brightness. ResNet-50 is flat across all conditions, but its 90.02 MB size and 41.71 ms latency rule it out for edge deployment.

### Cross-country generalisation

Every model collapses when evaluated zero-shot on Vietnamese durian images (n = 320, four shared classes, no fine-tuning). VGG-16 scores highest at 29.1% Macro F1; Agri-EfficientNet reaches 23.7%, a 64.1 pp drop from its Malaysian score; ShuffleNetV2 falls to 0.6%. The mean drop across all ten models is **60.6 pp**.

This is a finding, not a bug. The domain gap is a property of the dataset distributions, not of any particular architecture — which means benchmark accuracy in one country should not be read as a claim about deployment in another.

### Stability and significance

- **Five-fold stratified cross-validation** on the full 560 images: mean Macro F1 **85.68% ± 5.80%**.
- **McNemar tests** comparing Agri-EfficientNet against each alternative are **non-significant (p > 0.05)**. With only 60 paired samples the test is underpowered; detecting a difference of this size at 80% power would need roughly 200–400 samples. The cross-validation figure is the more reliable estimate.
- Per-class results should be read with the support column in mind. Pink_disease has a single test image, so its 66.7% F1 carries essentially no statistical weight.

---

## Dataset

The Malaysia durian disease dataset (560 images, 5 classes) was collected from commercial orchards across Peninsular Malaysia between July 2025 and June 2026, with expert verification. It is **not publicly released**, due to confidentiality agreements with the collaborating farm operators.

Researchers who would like access for non-commercial academic use can open an issue on this repository or contact the maintainers; requests are handled case by case and may require a short data use agreement.

The external validation set is publicly available: Nguyen, V.T. et al. (2025), *A durian leaf image dataset of common diseases in Vietnam for agricultural diagnosis*, Data in Brief. https://doi.org/10.1016/j.dib.2025.111845

### Expected directory layout

```
data/
  malaysia/
    Algal/              # 162 images
    Leaf_rot/           # 153 images
    Phomopsis/          # 157 images
    Pink_disease/       #  10 images
    Root_disease/       #  78 images
  vietnam/              # external validation set
    Leaf_Algal/
    Leaf_Phomopsis/
    Leaf_Blight/
    Leaf_Colletotrichum/
    Leaf_Rhizoctonia/
```

`train.py` performs the 80/10/10 stratified split itself and writes it to `--split_dir`; the data does not need to be pre-split.

---

## Installation

```bash
git clone https://github.com/<user>/agri-efficientnet.git
cd agri-efficientnet
pip install -r requirements.txt
```

Tested on Python 3.10 and 3.12, PyTorch 2.5.1 with CUDA 12.4 (NVIDIA RTX 4050 Laptop GPU), on Windows 11 and Ubuntu 22.04. The code runs on CPU without modification, though full retraining will be slow.

---

## Usage

### Full pipeline

```bash
python train.py \
  --malaysia_data data/malaysia \
  --vietnam_data  data/vietnam \
  --save_dir      results/ \
  --ckpt_dir      checkpoints/ \
  --seed          42
```

Runs all eleven steps end to end: dataset split, dataset statistics, ablation study, ten-model comparison, per-class metrics and confusion matrix, Vietnam external validation, Grad-CAM, robustness analysis, McNemar tests, five-fold cross-validation, and LFA latency profiling. Figures and CSVs are written to `--save_dir`.

Existing checkpoints are reused if found. Pass `--retrain` to force training from scratch.

### Individual components

```bash
# Grad-CAM figure only
python regen_gradcam.py --split_dir data/malaysia_split --ckpt_dir checkpoints/

# Robustness analysis only
python fix_robustness.py --split_dir data/malaysia_split --ckpt_dir checkpoints/ --save_dir results/

# MobileNetV2 + LFA supplementary experiment (run after fix_robustness.py)
python mobilenetv2_lfa.py
```

---

## Repository structure

```
agri-efficientnet/
├── train.py             # Full training and evaluation pipeline (11 steps)
├── regen_gradcam.py     # Standalone Grad-CAM visualisation
├── fix_robustness.py    # Standalone robustness analysis
├── mobilenetv2_lfa.py   # MobileNetV2 + LFA supplementary experiment
├── requirements.txt     # Pinned dependency versions
└── README.md
```

Trained checkpoints for all ten comparison models and four ablation variants are published under [Releases](../../releases).

---

## Reproducibility

All experiments use a fixed random seed (42), a stratified split, and identical augmentation and training schedules across the ablation variants. Exact dependency versions are pinned in `requirements.txt`.

Results may still vary slightly across hardware because of non-deterministic CUDA kernels. The reported figures come from a single RTX 4050 Laptop GPU.

---

## Citation

If this code is useful in your work, please cite:

```bibtex
@misc{agriefficientnet2026,
  title  = {Agri-EfficientNet: A Lightweight Lesion Focus Attention Framework
            for Durian Disease Diagnosis Under Malaysian Field Conditions
            with Cross-Country Generalization Assessment},
  author = {<Author names>},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

This entry will be updated with full publication details once available.

---

## License

Code is released under the **MIT License** (see `LICENSE`). The dataset is not covered by this licence and is subject to separate agreements.
