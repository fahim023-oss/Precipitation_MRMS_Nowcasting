# Physics-Informed Diffusion Models for High-Resolution Spatio-Temporal Precipitation Nowcasting

This repository contains code for high-resolution precipitation nowcasting using Multi-Radar Multi-Sensor (MRMS) radar precipitation data. The project compares a deterministic ConvLSTM baseline with diffusion-based generative nowcasting models, including a physics-guided DDPM framework designed to improve storm structure, temporal coherence, and physical realism.

The associated project/report is titled:

**Physics-Informed Diffusion Models for High-Resolution Spatio-Temporal Precipitation Nowcasting**

Authors:

```text
Buddha Subedi, Md Syfullah Fahim, Varun Sethi, Esperanza Corral
University of Minnesota
Minneapolis, MN, USA
```

---

## Overview

Accurate precipitation nowcasting is important for flood forecasting, disaster management, aviation safety, and urban infrastructure planning. This project focuses on short-term high-resolution nowcasting using MRMS radar precipitation fields.

The default nowcasting task is:

```text
Input sequence  : 60 MRMS frames = previous 2 hours
Output sequence : 30 MRMS frames = next 1 hour
Temporal spacing: 2 minutes
Patch size      : 128 × 128 pixels
Spatial scale   : approximately 1 km resolution
```

This repository implements three model components:

1. **ConvLSTM baseline**  
   A deterministic encoder-decoder ConvLSTM model for sequence-to-sequence precipitation prediction.

2. **Conditional DDPM**  
   A denoising diffusion probabilistic model that generates stochastic future precipitation sequences conditioned on recent MRMS observations.

3. **Physics-guided DDPM inference**  
   A physics-guided realization-selection framework that generates multiple DDPM samples and selects the most physically plausible forecast using mass consistency, temporal smoothness, spatial smoothness, and persistence/advection-style constraints.

---

## Motivation

Deterministic deep learning models often produce overly smooth precipitation forecasts, especially at longer lead times. This is a common mean-regression behavior: when multiple future storm evolutions are possible, a deterministic model tends to predict an average solution, which blurs sharp convective cores and underrepresents localized extremes.

Diffusion models address this limitation by learning a distribution over possible future precipitation states. Instead of producing only one forecast, the DDPM can generate multiple plausible realizations. Physics-guided scoring is then used to select realizations that better satisfy physically meaningful constraints such as storm continuity, smooth temporal evolution, and approximate rainfall-volume consistency.

---

## Model Summary

### ConvLSTM Baseline

The ConvLSTM baseline maps the previous 2 hours of rainfall observations,

```text
X ∈ R^(B × 60 × 1 × 128 × 128)
```

to a 1-hour forecast,

```text
Y_hat ∈ R^(B × 30 × 1 × 128 × 128)
```

The model uses a stacked encoder-decoder ConvLSTM architecture with two hidden layers of size 64 and 96. The decoder is unrolled autoregressively for 30 forecast steps. A Softplus output layer enforces non-negative precipitation.

The training loss combines:

```text
L1 loss
MSE loss
heavy-rainfall penalty for pixels above 5 mm/hr
```

The ConvLSTM baseline is useful for comparison, but it tends to smooth precipitation fields and lose sharp storm structure at longer lead times.

---

### Conditional DDPM

The conditional DDPM learns to generate future precipitation fields by reversing a noise process. During training, Gaussian noise is added to the future rainfall sequence using a linear diffusion schedule. A conditional 3D U-Net is trained to predict the added noise from:

```text
1. noisy future precipitation sequence
2. historical MRMS conditioning sequence
3. diffusion timestep embedding
```

The DDPM framework generates stochastic precipitation realizations, allowing it to preserve sharper rainfall gradients and localized high-intensity structures compared with deterministic nowcasting.

---

### Physics-Guided DDPM

The physics-guided DDPM is not trained as a separate model. It is an inference and evaluation framework built on top of a trained DDPM checkpoint.

The workflow is:

```text
1. Train the conditional DDPM.
2. Load the trained DDPM checkpoint.
3. Generate multiple stochastic DDPM realizations.
4. Score each realization using physics-guided criteria.
5. Select the realization with the lowest physics-guided score.
```

The physics-guided score includes:

```text
mass consistency
temporal smoothness
spatial smoothness
persistence/advection-style consistency
```

This helps reduce excessive rainfall persistence, suppress unrealistic pixel-scale noise, and improve storm-evolution realism.

---

## Repository Structure

```text
Precipitation_MRMS_Nowcasting/
│
├── configs/
│   ├── convlstm.yaml
│   ├── ddpm.yaml
│   └── physics_ddpm.yaml
│
├── scripts/
│   ├── train_convlstm.py
│   ├── train_ddpm.py
│   └── evaluate_physics_ddpm.py
│
├── src/
│   └── mrms_nowcasting/
│       ├── __init__.py
│       ├── data.py
│       ├── metrics.py
│       ├── visualization.py
│       │
│       ├── convlstm/
│       │   ├── __init__.py
│       │   ├── model.py
│       │   ├── losses.py
│       │   └── trainer.py
│       │
│       ├── ddpm/
│       │   ├── __init__.py
│       │   ├── diffusion.py
│       │   ├── model.py
│       │   ├── losses.py
│       │   └── trainer.py
│       │
│       └── physics_ddpm/
│           ├── __init__.py
│           ├── scoring.py
│           └── inference.py
│
├── data/
│   └── mrms_3hr_cubes_128/
│       └── .gitkeep
│
├── outputs/
│   ├── checkpoints/
│   └── figures/
│
├── docs/
│
├── README.md
├── requirements.txt
├── environment.yml
└── .gitignore
```

---

## Data

The full MRMS dataset is not included in this repository because it contains thousands of large `.npz` files.

Each MRMS cube is expected to be a `.npz` file containing:

```text
precip: array with shape (90, 128, 128)
```

The expected local data folder is:

```text
data/mrms_3hr_cubes_128/
```

For the University of Minnesota MSI/HPC environment, the data may be stored at:

```text
/scratch.global/fahim023/mrms_3hr_cubes_128/
```

To use the HPC path, update the `data_folder` field in the config file:

```yaml
data:
  data_folder: "/scratch.global/fahim023/mrms_3hr_cubes_128"
```

For public use, the default config keeps the path as:

```yaml
data:
  data_folder: "data/mrms_3hr_cubes_128"
```

Large data files are ignored by Git through `.gitignore`.

---

## Installation

### Option 1: Using pip

```bash
git clone https://github.com/fahim023-oss/Precipitation_MRMS_Nowcasting.git
cd Precipitation_MRMS_Nowcasting

python -m pip install -r requirements.txt
```

### Option 2: Using conda

```bash
git clone https://github.com/fahim023-oss/Precipitation_MRMS_Nowcasting.git
cd Precipitation_MRMS_Nowcasting

conda env create -f environment.yml
conda activate mrms-nowcasting
```

---

## Running the Models

### 1. Train ConvLSTM Baseline

```bash
python scripts/train_convlstm.py --config configs/convlstm.yaml
```

Main outputs:

```text
outputs/checkpoints/best_convlstm.pt
outputs/convlstm_test_metrics.txt
outputs/figures/convlstm_training_curve.png
outputs/figures/convlstm_prediction_example.png
```

---

### 2. Train Conditional DDPM

```bash
python scripts/train_ddpm.py --config configs/ddpm.yaml
```

Main outputs:

```text
outputs/checkpoints/best_ddpm.pt
outputs/ddpm_test_metrics.txt
outputs/figures/ddpm_training_curve.png
```

---

### 3. Evaluate Physics-Guided DDPM

The physics-guided DDPM requires a trained DDPM checkpoint.

First train the DDPM:

```bash
python scripts/train_ddpm.py --config configs/ddpm.yaml
```

Then run physics-guided evaluation:

```bash
python scripts/evaluate_physics_ddpm.py --config configs/physics_ddpm.yaml
```

Main outputs:

```text
outputs/physics_ddpm_test_metrics.txt
outputs/figures/physics_ddpm_prediction_example.png
```

---

## Configuration Files

All major settings are controlled through YAML files in `configs/`.

### ConvLSTM Config

```text
configs/convlstm.yaml
```

Controls:

```text
input/output sequence length
hidden dimensions
learning rate
batch size
heavy-rain threshold
evaluation thresholds
output paths
```

### DDPM Config

```text
configs/ddpm.yaml
```

Controls:

```text
diffusion timesteps
beta schedule
3D U-Net channel size
training epochs
noise-prediction loss weight
checkpoint path
```

### Physics-Guided DDPM Config

```text
configs/physics_ddpm.yaml
```

Controls:

```text
number of stochastic DDPM realizations
mass consistency weight
temporal smoothness weight
spatial smoothness weight
persistence/advection-style weight
evaluation output path
```

---

## Evaluation Metrics

The repository supports both continuous and categorical nowcasting metrics.

Continuous metrics:

```text
RMSE
MAE
```

Categorical metrics:

```text
CSI = Critical Success Index
POD = Probability of Detection
FAR = False Alarm Ratio
```

Metrics are computed over precipitation thresholds such as:

```text
0.5 mm/hr
2.0 mm/hr
5.0 mm/hr
10.0 mm/hr
```

Metrics are reported over lead-time buckets:

```text
t01-10   = 0-20 minutes
t11-20   = 20-40 minutes
t21-30   = 40-60 minutes
overall  = 0-60 minutes
```

---

## Expected Outputs

After training or evaluation, generated files are written under `outputs/`.

```text
outputs/
├── checkpoints/
│   ├── best_convlstm.pt
│   └── best_ddpm.pt
│
├── figures/
│   ├── convlstm_training_curve.png
│   ├── convlstm_prediction_example.png
│   ├── ddpm_training_curve.png
│   └── physics_ddpm_prediction_example.png
│
├── convlstm_test_metrics.txt
├── ddpm_test_metrics.txt
└── physics_ddpm_test_metrics.txt
```

These output files are ignored by Git and should be regenerated as needed.

---

## Key Findings

The ConvLSTM baseline provides temporally smooth forecasts but suffers from deterministic mean-regression behavior. It tends to over-smooth convective cores and increasingly loses sharp storm structure at longer lead times.

The DDPM framework better preserves localized rainfall extremes and fine-scale storm morphology because it samples from a distribution of plausible future precipitation states rather than producing a single conditional mean.

The physics-guided DDPM further improves realism by selecting realizations that better satisfy physical consistency constraints. In particular, physics guidance reduces excessive rainfall persistence, improves mass consistency, and produces smoother storm evolution over longer lead times.

---

## Notes on Large Files

This repository does not track:

```text
MRMS .npz files
model checkpoints
large output figures
generated arrays
```

These files are intentionally ignored to keep the repository lightweight and reproducible.

Use:

```text
data/mrms_3hr_cubes_128/
```

for local data, and:

```text
outputs/checkpoints/
outputs/figures/
```

for generated model outputs.

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{subedi2026physicsnowcasting,
  title  = {Physics-Informed Diffusion Models for High-Resolution Spatio-Temporal Precipitation Nowcasting},
  author = {Subedi, Buddha and Fahim, Md Syfullah and Sethi, Varun and Corral, Esperanza},
  year   = {2026},
  url    = {https://github.com/fahim023-oss/Precipitation_MRMS_Nowcasting}
}
```

---

## Contributors

```text
Buddha Subedi
Md Syfullah Fahim
Varun Sethi
Esperanza Corral
```

University of Minnesota

---

## License

Add a license file before public release.

Recommended options:

```text
MIT License
BSD 3-Clause License
Apache License 2.0
```

For academic research code, MIT or BSD 3-Clause is usually simple and permissive.
