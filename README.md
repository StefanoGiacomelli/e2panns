# From Large Scale Audio Tagging to Real-Time Emergency Vehicle Sirens Detection

![Python 3.11](https://img.shields.io/badge/python-3.11.2-blue.svg) ![PyTorch](https://img.shields.io/badge/-PyTorch%202.x-333?style=flat&logo=pytorch) ![Lightning](https://img.shields.io/badge/-Lightning%202.x-792ee5?style=flat&logo=lightning) ![License](https://img.shields.io/badge/License-MIT-green)

*Work in Progress - Paper under review in IEEE TASLP journal*

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Installation & Setup](#installation--setup)
5. [Quick Start](#quick-start)
6. [Usage Guides](#usage-guides)
   - [Training & Testing](#61-training--testing-mainpy)
   - [Automated Benchmarking](#62-automated-benchmarking-main_benchmarkpy)
   - [Batch Experiment Runner](#63-batch-experiment-runner-run_all_experimentspy)
   - [Batch Benchmark Runner](#64-batch-benchmark-runner-run_all_benchmarkspy)
7. [Configuration](#configuration)
8. [Models](#models)
9. [Datasets](#datasets)
10. [Technical Profiling](#technical-profiling)
11. [Results & Outputs](#results--outputs)
12. [Requirements](#requirements)
13. [License](#license)
14. [Citation](#citation)
15. [Contact](#contact)

---

## Overview

**E2PANNs** (Emergency-to-PANNs) is a comprehensive PyTorch Lightning-based framework for **Emergency Vehicle (EV) siren detection and classification** using transfer learning from large-scale audio tagging models pre-trained on AudioSet.

The framework leverages **General-Purpose Audio Transformers (GP-AT)** and adapts them for specialized EV recognition tasks through fine-tuning on curated emergency vehicle datasets. It supports both **binary classification** (siren vs. non-siren) and **multi-class classification** (police, ambulance, fire truck, negative).

### Why E2PANNs?

- **Transfer Learning**: Leverages rich audio representations from AudioSet (527 classes, 2M+ samples)
- **Multi-Model Support**: Compare 3 state-of-the-art GP-AT architectures (E-PANNs, CED, CLAP)
- **Comprehensive Evaluation**: Automated benchmarking across 7 diverse audio datasets
- **Production-Ready**: Modular design with YAML configs, proper logging, and checkpoint management
- **Research-Oriented**: Cross-validation support, detailed metrics, and reproducible experiments

---

## Key Features

✅ **Multi-Model Support**: 3 GP-AT architectures (E-PANNs, CED, CLAP) with different sample rates and characteristics  
✅ **Dual-Task Framework**: Binary (siren detection) + Multi-class (siren type classification)  
✅ **7 Dataset Benchmark**: AudioSet-EV (v1 & v2), sireNNet, LSSiren, ESC-50, FSD50K, UrbanSound8K  
✅ **YAML Configuration**: Flexible experiment setup with validation and version control  
✅ **PyTorch Lightning**: Modern training pipeline with callbacks, logging, and distributed training support  
✅ **Automated Runners**: Batch execution for multiple experiments and benchmarks  
✅ **Cross-Validation**: Proper fold separation for CV datasets (ESC-50, UrbanSound8K, sireNNet)  
✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, F-beta, AUROC, Specificity  
✅ **Dual Checkpoint Format**: Lightning `.ckpt` (full state) + PyTorch `.pt` (model only)  
✅ **TensorBoard Integration**: Real-time training monitoring with loss curves and metrics  
✅ **Technical Profiling**: Pre-selection analysis of 18 GP-AT models with performance metrics  

---

## Project Structure

```text
E2PANNs/
├── main.py                          # Main training/testing pipeline
├── main_benchmark.py                # Automated benchmark evaluation
├── run_all_experiments.py           # Batch experiment runner
├── run_all_benchmarks.py            # Batch benchmark runner
│
├── models/                          # Model implementations
│   ├── lightning_models.py          # Lightning modules (BinaryEVClassifier, MultiClassSirenClassifier)
│   ├── callbacks.py                 # Custom callbacks (dual checkpoint saving)
│   ├── epanns/                      # E-PANNs (PANNs-based, 32kHz)
│   ├── ced/                         # CED (ConvNeXt-based, 16kHz)
│   └── clap/                        # CLAP (HTSAT-based, 48kHz)
│
├── datasets/                        # EV-Benchmark: 7 datasets w. PyTorch Datasets & Lightning DataModules
│   ├── AudioSet_EV_v1_2025/         # Primary EV dataset (2025 release)
│   ├── AudioSet_EV_v2PANNs_2020/    # Extended EV dataset (PANNs-aligned)
│   ├── sireNNet/                    # Urban EV sirens (multi-class)
│   ├── LSSiren/                     # Large-scale siren recordings
│   ├── ESC50/                       # Environmental sounds (50 classes, 5-fold CV)
│   ├── FSD50K/                      # Freesound dataset (200 classes)
│   ├── UrbanSound8K/                # Urban sounds (10 classes, 10-fold CV)
│   ├── datasets_mapping.json        # Label mappings
│   └── README.md                    # Datasets documentation
│
├── configs/                         # YAML configuration files
│   ├── binary_EV/                   # Binary task configurations
│   ├── multiclass_EV/               # Multi-class (EV types) task configurations
│   └── multiclass_siren/            # Multi-class (siren types) task configurations
│
├── benchmark_configs/               # Benchmark configuration files
│
├── preliminary_profiling_gp_at/     # Technical profiling of 18 GP-AT models
│   ├── results/                     # Profiling JSON files
│   ├── profile_main.py              # Profiling script
│   └── README.md                    # Profiling documentation
│
├── checkpoints/                     # Model checkpoints (.ckpt + .pt)
├── logs/                            # TensorBoard logs
├── results/                         # Test metrics and predictions
├── benchmark_results/               # Benchmark CSV reports
│
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/StefanoGiacomelli/e2panns.git
cd e2panns
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

Download our selected AudioSet-pretrained models:

- **E-PANNs**: [PANNs Checkpoint](https://zenodo.org/records/7939403) → `models/epanns/checkpoint_closeto_.44.pt`
- **CED**: [Efficient ViT Checkpoint](https://github.com/RicherMans/CED) → `models/ced/audiotransformer_base_mAP_4999.pt`
- **CLAP**: [HTSAT Checkpoint](https://github.com/LAION-AI/CLAP) → `models/clap/630k-audioset-fusion-best.pt`

### 5. Setup Datasets

Follow dataset-specific instructions in [`datasets/README.md`](datasets/README.md) to download and prepare each dataset.

---

## Quick Start

### Example 1: Train E-PANNs on AudioSet-EV v2 (Binary Task)

```bash
# 1. Modify or use an existing config
vim configs/binary_EV/epanns_finetune_fixedLR_AS-EV_v2.yaml

# 2. Run training + testing
python main.py --config configs/binary_EV/epanns_finetune_fixedLR_AS-EV_v2.yaml --mode fit+test

# 3. Monitor with TensorBoard (on another terminal)
tensorboard --logdir logs/
```

**Output:**

- Checkpoints: `checkpoints/epanns_finetune_fixedLR_AS-EV_v2/`
  - `epoch=XX_val_f1=0.XXXX.ckpt` (Lightning checkpoint)
  - `epoch=XX_val_f1=0.XXXX.pt`   (PyTorch state_dict only)
  - `last.ckpt` + `last.pt`
- Logs: `logs/epanns_finetune_fixedLR_AS-EV_v2/`
- Results: `results/epanns_finetune_fixedLR_AS-EV_v2/test/`
  - `test_metrics.json`
  - `test_predictions.npz`

### Example 2: Benchmark Pretrained Models on All Datasets

```bash
# Test E-PANNs pretrained on Audioset, against our 7 datasets EV Benchmark
python main_benchmark.py --config benchmark_configs/epanns_audioset_pretrained.yaml

# Or use default (backward compatible)
python main_benchmark.py
```

**Output:**

- `benchmark_results/epanns_binary_benchmark_YYYYMMDD_HHMMSS.csv`
- `benchmark_results/epanns_multiclass_benchmark_YYYYMMDD_HHMMSS.csv`

### Example 3: Batch Training (All Binary EV Configs)

```bash
# Run all experiments in configs/binary_EV/
python run_all_experiments.py --config_dir configs/binary_EV
```

---

## Usage Guides

### 6.1 Training & Testing (`main.py`)

The main script supports flexible experiment configuration via YAML files.

#### Execution Modes

```bash
# Training only
python main.py --config configs/binary_EV/my_config.yaml --mode fit

# Testing only (requires checkpoint)
python main.py --config configs/binary_EV/my_config.yaml --mode test

# Training + Testing (default)
python main.py --config configs/binary_EV/my_config.yaml --mode fit+test

# Benchmark mode (pre-defined CV splits)
python main.py --config configs/binary_EV/my_config.yaml --mode benchmark
```

#### Configuration Structure

See [`configs/binary_EV/epanns_finetune_fixedLR_AS-EV_v2.yaml`](configs/binary_EV/epanns_finetune_fixedLR_AS-EV_v2.yaml) for a complete example.

**Key sections:**

- **experiment**: Name, task (binary/multiclass), model selection, seed
- **paths**: Checkpoints, logs, results directories
- **data**: Dataset selection, mode (train/benchmark), batch size, augmentation
- **model**: Optimizer, scheduler, loss function, F-beta weight
- **training**: Epochs, early stopping, validation frequency
- **callbacks**: Checkpoint monitoring, save strategy

#### Mixed Dataset Training

Train on one dataset, test on another:

```yaml
data:
  train_dev_dataset: "AudioSet_EV_v2PANNs_2020"
  train_dev_mode: "train"
  test_dataset: "sireNNet"
  test_mode: "benchmark"
```

---

### 6.2 Automated Benchmarking (`main_benchmark.py`)

Evaluate a model across **all 7 EV datasets** with automatic task detection (binary/multiclass).

#### Configuration-Based Benchmark

```bash
python main_benchmark.py --config benchmark_configs/epanns_audioset_pretrained.yaml
```

**Config structure:**

```yaml
model:
  name: "epanns"
  checkpoint_type: "pretrained"  # or "finetuned"
  checkpoint_path: "./models/epanns/checkpoint_closeto_.44.pt"

benchmark:
  batch_size: 32
  num_workers: 0
  limit_test_batches: null  # null = full dataset, 0.1 = 10%
  datasets_to_test: []      # [] = all, or specify subset
  output_dir: "./benchmark_results"
```

**Output:**

- 2 CSV files (binary + multiclass) with comprehensive metrics per dataset
- Cross-validation: One metrics row per fold + aggregated results row

---

### 6.3 Batch Experiment Runner (`run_all_experiments.py`)

Execute multiple training experiments sequentially from a config directory.

```bash
# Run all configs in directory
python run_all_experiments.py --config_dir configs/binary_EV

# Continue even if one experiment fails
python run_all_experiments.py --config_dir configs/binary_EV --continue_on_error

# Use different Python interpreter
python run_all_experiments.py --config_dir configs/binary_EV --python /usr/bin/python3

# Different execution mode
python run_all_experiments.py --config_dir configs/binary_EV --mode test
```

**Features:**

- Sequential execution with proper isolation
- Real-time terminal output
- Progress tracking (Experiment N/M)
- Final summary with durations and success/failure status

---

### 6.4 Batch Benchmark Runner (`run_all_benchmarks.py`)

Execute multiple benchmark evaluations sequentially.

```bash
# Run all benchmark configs
python run_all_benchmarks.py --config_dir benchmark_configs

# With error tolerance
python run_all_benchmarks.py --config_dir benchmark_configs --continue_on_error
```

**Use cases:**

- Compare multiple model checkpoints
- Evaluate pretrained vs. finetuned models

---

## Models

E2PANNs supports 3 state-of-the-art General-Purpose Audio Transformer (GP-AT) architectures:

| Model | Architecture | Sample Rate | Params | GFLOPs | Min Input | Use Case |
|-------|-------------|-------------|--------|--------|-----------|----------|
| **E-PANNs** | CNN14 (PANNs) | 32 kHz | ~81M | ~6.5 | 320ms | Baseline, robust |
| **CED** | ConvNeXt-T | 16 kHz | ~28M | ~2.1 | 160ms | Efficiency, speed |
| **CLAP** | HTSAT-base | 48 kHz | ~85M | ~8.2 | 480ms | Zero-shot, multi-modal |

### Model Selection Criteria

All models were selected from a comprehensive profiling of **18 AudioSet-pretrained GP-AT models** across multiple dimensions:

- ✅ **CPU Performance**: Inference speed on consumer hardware
- ✅ **Computational Efficiency**: Low GFLOPs for real-time deployment
- ✅ **Model Compactness**: Small parameter count for edge devices
- ✅ **Input Flexibility**: Short minimum input length for variable-duration audio

For complete profiling results and methodology, see [`preliminary_profiling_gp_at/README.md`](preliminary_profiling_gp_at/README.md).

### Model Checkpoints

**Pre-trained (AudioSet):**
- Loaded automatically with `pretrained: true` in config
- Located in `models/{model_name}/`

**Fine-tuned (EV-specific):**
- Saved during training to `checkpoints/`
- Dual format: `.ckpt` (Lightning) + `.pt` (PyTorch model only)
- Best model selected via validation metric (e.g., `val_f1`)

---

## Datasets

E2PANNs provides a comprehensive benchmark across **7 diverse audio datasets**:

### Primary EV Datasets

| Dataset | Content | Labels | Duration | SR | Split |
|---------|---------|--------|----------|-----|-------|
| **AudioSet-EV v1** | 14K clips | Binary + Multi | 10s | 32k | Train/Dev/Test |
| **AudioSet-EV v2** | 28K clips | Binary + Multi | 10s | 32k | Train/Dev/Test |
| **sireNNet** | 1.6K clips | Multi-class | Variable | 44.1k | 10-fold CV |
| **LSSiren** | Large-scale | Binary | Variable | 16k | Test only |

### Auxiliary Benchmark Datasets

| Dataset | Content | EV Relevance | Split |
|---------|---------|--------------|-------|
| **ESC-50** | 2K clips, 50 classes | Urban sounds | 5-fold CV |
| **FSD50K** | 51K clips, 200 classes | General audio | Train/Val/Test |
| **UrbanSound8K** | 8.7K clips, 10 classes | Urban environment | 10-fold CV |

### Dataset Modes

- **`train` mode**: Random train/dev/test splits (80/10/10)
- **`benchmark` mode**: Pre-defined splits or cross-validation folds

For dataset setup instructions and detailed descriptions, see [`datasets/README.md`](datasets/README.md).

---

## Technical Profiling

The [`preliminary_profiling_gp_at/`](preliminary_profiling_gp_at/) directory contains a comprehensive analysis of **18 AudioSet-pretrained GP-AT models** conducted to identify the most suitable architectures for EV detection.

### Profiling Metrics

- **Performance**: CPU forward pass timing (mean, median, std, IQR)
- **Computational Cost**: FLOPs, MACs, GFLOPs
- **Model Size**: Parameters (total, trainable), memory footprint
- **Input Requirements**: Minimum input length (samples, seconds)
- **Architecture**: CNN, Transformer, Hybrid comparisons

### Selection Process

From the 18 profiled models, **3 were selected** based on:

1. **CPU Efficiency**: Fast inference for real-time applications
2. **Computational Cost**: Low GFLOPs for resource-constrained deployment
3. **Model Compactness**: Small parameter count for edge devices
4. **Input Flexibility**: Short minimum input for variable-length audio

**Result:**
- ✅ E-PANNs: Best overall performance and robustness
- ✅ CED: Most efficient (lowest GFLOPs, fastest CPU time)
- ✅ CLAP: Audio-language capabilities and zero-shot potential

For complete profiling results and visualizations, see [`preliminary_profiling_gp_at/README.md`](preliminary_profiling_gp_at/README.md).

---

## Results & Outputs

### Training Outputs

**Checkpoints** (`checkpoints/experiment_name/`):
```
epoch=010_val_f1=0.8523.ckpt    # Lightning checkpoint (full training state)
epoch=010_val_f1=0.8523.pt      # PyTorch model (inference only)
last.ckpt                        # Latest epoch
last.pt
```

**Logs** (`logs/experiment_name/`):
```
version_0/
├── events.out.tfevents...       # TensorBoard logs
└── hparams.yaml                 # Hyperparameters
```

**Test Results** (`results/experiment_name/test/`):
```
test_metrics.json                # Comprehensive metrics
test_predictions.npz             # Predictions, targets, probabilities
```

For cross-validation datasets:
```
fold_0_metrics.json
fold_1_metrics.json
...
cross_val_metrics.json           # Aggregated statistics
```

### Benchmark Outputs

**CSV Reports** (`benchmark_results/`):
```
epanns_binary_benchmark_20260222_164350.csv
epanns_multiclass_benchmark_20260222_164350.csv
```

CSV columns:

- Dataset, Task, Model
- Accuracy, Precision, Recall, F1, F-beta, AUROC, Specificity
- For CV datasets: Per-fold metrics + Mean ± Std

---

## Requirements

### Core Dependencies

```
Python >= 3.11.2
PyTorch >= 2.0.0
pytorch-lightning >= 2.6.0
torchmetrics >= 1.0.0
torchaudio >= 2.0.0
```

### Audio Processing

```
librosa >= 0.10.0
soundfile >= 0.12.0
audiomentations >= 0.30.0
```

### Machine Learning

```
scikit-learn >= 1.3.0
numpy >= 1.24.0
pandas >= 2.0.0
```

### Visualization & Logging

```
tensorboard >= 2.13.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
```

### Configuration

```
pyyaml >= 6.0
```

For the complete list, see [`requirements.txt`](requirements.txt).

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use E2PANNs in your research, please cite:

```bibtex
@article{giacomelli2026e2panns,
  title={TODO: Paper Title},
  author={Giacomelli, Stefano and TODO: Other Authors},
  journal={TODO: IEEE TASLP (under review)},
  year={2026},
  publisher={TODO: IEEE},
  doi={TODO: DOI},
  url={TODO: URL}
}
```

*Paper under review in IEEE Transactions on Audio, Speech, and Language Processing (TASLP)*

---

## Contact

<div align="center">

**Stefano Giacomelli**  
*Ph.D. Candidate in Information and Communication Technology*  
Department of Engineering, Information Science & Mathematics (DISIM dpt.)  
University of L'Aquila, Italy

![DISIM_logo](https://phdict.disim.univaq.it/wp-content/uploads/2024/06/logo-univaq-disim-2-2-768x283.png)

📧 Email: stefano.giacomelli@graduate.univaq.it  
🔗 GitHub: https://github.com/StefanoGiacomelli  
🆔 ORCID: https://orcid.org/0009-0009-0438-1748  
🎓 Scholar: https://scholar.google.com/citations?user=l-n0hl4AAAAJ&hl=it  
💼 LinkedIn: https://www.linkedin.com/in/stefano-giacomelli-811654135

---

*This project is funded under the Italian National Ministry of University and Research, for the Italian National Recovery and Resilience Plan (NRRP) "Methods of Computational Auditory Scene Analysis and Synthesis supporting eXtended and Immersive Reality Services"*

</div>
