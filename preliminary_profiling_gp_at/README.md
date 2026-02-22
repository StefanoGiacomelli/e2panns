# Preliminary GP-AT Models Profiling

This directory contains a comprehensive **technical profiling analysis** of 18 state-of-the-art **General-Purpose Audio Tagging (GP-AT)** models. All models are pre-trained on AudioSet (o related distributions) and support audio tagging across 527 classes. This preliminary analysis was conducted to identify the most suitable architectures for our Emergency Vehicle (EV) siren detection and classification experiments.

![Python 3.11](https://img.shields.io/badge/python-3.11.2-blue.svg) ![PyTorch](https://img.shields.io/badge/-PyTorch-333?style=flat&logo=pytorch) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

1. [Overview](#overview)
2. [Profiling Methodology](#profiling-methodology)
3. [Models Directory Note](#models-directory-note)
4. [Models Summary](#models-summary)
5. [Analysis Results](#analysis-results)
6. [Models Catalog](#models-catalog)
7. [Requirements](#requirements)
8. [License](#license)

---

## Overview

This preliminary profiling systematically evaluated AudioSet GP-AT models across multiple dimensions to assess their suitability for real-time Emergency Vehicle detection on resource-constrained devices (CPU-focused deployment). The analysis provides:

- **Performance Metrics**: CPU forward pass timing with comprehensive statistics
- **Computational Cost**: FLOPs and MACs measurements
- **Model Efficiency**: Parameter counts and memory footprint
- **Input Flexibility**: Minimum input length requirements
- **Architecture Insights**: Comparative analysis across CNN, Transformer, and Hybrid based architectures

**Key Findings**: The profiling results (`results/*.json`) and visualizations guide the selection of models for the E2PANNs framework.

---

## Profiling Methodology

### Technical Approach

The profiling pipeline (`profile_main.py` + `profile_utils.py`) provides commands for systematic evaluation of each model across the following dimensions:

#### 1. **Parameter Analysis**

- Total parameters count (millions)
- Trainable vs. frozen parameters
- Model memory footprint (MB)

#### 2. **Minimum Input Length Discovery**

- Binary search algorithm to find the shortest acceptable audio input
- Critical for real-time applications with variable-length audio and potential framing reduction for SED tasks
- Measured in samples and seconds (according to models sample rate)

#### 3. **Computational Cost**

- **FLOPs** (Floating Point Operations): Total operations per forward pass
- **MACs** (Multiply-Accumulate Operations): ~50% of FLOPs
- **GFLOPs**: FLOPs (in billions) for easier comparison
- Measured using `fvcore` library with dummy inputs shaping

#### 4. **CPU Performance Profiling**

- **Hardware**: consumer Apple Macbook Pro M4-series (ARM-based) CPU
- **Methodology**:
  - 3 warm-up iterations (discarded)
  - 10 timed forward passes with random audio inputs
  - 10-second audio clips at model-specific sample rates
- **Metrics Collected**:
  - Mean, median, min, max, standard deviation
  - Standard error, IQR (interquartile range), 75th percentile
  - Skewness and kurtosis (distribution shape)
  - Throughput (samples processed per second)

### Profiling Environment

- **Device**: CPU (Apple Silicon M4-series)
- **PyTorch**: Eager mode (no compilation)
- **Precision**: FP32 (single precision)
- **Batch Size**: 1 (single sample inference)

### Analysis Output

- **JSON Results**: 18 files in `results/` with complete profiling data
- **Analysis Script**: `results/analysis.py` extracts and analyze top 3 models (per category)

---

## Models Directory Note

⚠️ **Important**: The `../models/` directory contains **only the models actually selected and integrated** into the E2PANNs Emergency Vehicle detection framework, not all 18 models profiled here.

The models included in `../models/` were chosen based on the profiling results in this directory, prioritizing:

- CPU inference speed for real-time applications
- Computational efficiency (low GFLOPs)
- Model compactness (small parameter count)
- Input flexibility (short minimum input length)

For complete details on all 18 profiled models, refer to the [Models Catalog](#models-catalog) below and the JSON files in `results/`.

---

## Models Summary

| Model | Variant | Architecture | Year | Script | Sample Rate | Embed Dim | mAP |
|-------|---------|--------------|------|--------|-------------|-----------|-----|
| AST | AST | Transformer | 2021 | `ast/model.py` | 16000 | 768 | 0.459 |
| AudioCLIP | AudioCLIP | CLIP+ESResNeXt | 2021 | `audioclip/model.py` | 44100 | 1024 | — |
| AudioMAE | AudioMAE | Masked Autoencoder | 2022 | `audiomae/model.py` | 16000 | 768 | 0.473 |
| BEATs | BEATs | Transformer+Tokenizer | 2022 | `beats/model.py` | 16000 | 768 | 0.505 |
| CED | CED-Base | Ensemble Distillation | 2024 | `ced/model.py` | 16000 | 768 | 0.500 |
| CLAP | LAION-CLAP | HTSAT+RoBERTa | 2023 | `clap/model.py` | 48000 | 768 | — |
| PANNs | Wavegram-Logmel-CNN14 | CNN | 2020 | `panns/model_wavegram_logmel_cnn14.py` | 32000 | 2048 | 0.439 |
| ConvNeXt | ConvNeXt-Tiny | ConvNeXt | 2023 | `convnext/model.py` | 32000 | 768 | 0.471 |
| DyMN | DyMN-10 | Dynamic MobileNet | 2023 | `efficientat/model_dymn.py` | 32000 | 1920 | 0.477 |
| E-PANNs | E-PANNs | Pruned CNN14 | 2023 | `epanns/model.py` | 32000 | 2048 | 0.423 |
| HTS-AT | HTS-AT | Swin Transformer | 2022 | `htsat/model.py` | 32000 | 768 | 0.471 |
| M2D | M2D-CLAP | Masked Modeling Duo | 2024 | `m2d/model.py` | 16000 | 768 | 0.490 |
| MobileNetV3 | MN-10 | MobileNetV3 | 2022 | `efficientat/model_mn.py` | 32000 | 3840 | 0.471 |
| PaSST | PaSST-S | Patchout Transformer | 2022 | `passt/model.py` | 32000 | 768 | 0.476 |
| PSLA | EfficientNet-B2+Attn | CNN+Attention | 2021 | `psla/model.py` | 16000 | 1408 | 0.440 |
| PANNs | ResNet38 | ResNet | 2020 | `panns/model_resnet38.py` | 32000 | 2048 | 0.434 |
| VGGish | VGGish | VGG-style CNN | 2017 | `vggish/model.py` | 16000 | 128 | 0.310 |
| YAMNet | YAMNet | MobileNetV1 | 2018 | `yamnet/model.py` | 16000 | 1024 | 0.306 |

---

## Analysis Results

The profiling analysis identified optimal models across four critical dimensions for Emergency Vehicle detection applications. Complete results are available in `results/*.json` files.

### Running the Analysis

To reproduce the analysis and generate visualizations:

```bash
cd preliminary_profiling_gp_at/results/
python analysis.py
```

**Output:**

- Terminal output with top 3 models per category (ranked with 🥇🥈🥉)
- 5 high-resolution figures (SVG 600 DPI):
  1. `profiling_radar_comparison.svg` - Unified radar chart comparing all top 3 models across 4 metrics
  2. `profiling_performance_heatmap.svg` - Heatmap of all 18 models vs 4 metrics
  3. `profiling_pareto_speed_size.svg` - Pareto front (CPU speed vs model size)
  4. `profiling_architecture_distribution.svg` - Violin plots by architecture type
  5. `profiling_throughput_comparison.svg` - CPU throughput for all models

### Key Metrics Evaluated

1. **CPU Forward Time**: Inference latency (lower is better)
2. **Minimum Input Length**: Shortest acceptable audio input (lower = more flexible)
3. **GFLOPs**: Computational cost (lower = more efficient)
4. **Parameters**: Model size in MB (lower = more compact)

### Architecture Trends

Models are classified into three architectural families:

- **CNN-based**: PANNs, E-PANNs, VGGish, ConvNeXt, EfficientAT (MN/DyMN)
- **Transformer-based**: AST, BEATs, PaSST, AudioMAE, HTS-AT, CED
- **Hybrid/Other**: CLAP, AudioCLIP, M2D, PSLA

**Observations from profiling:**

- CNNs generally offer faster CPU inference and lower GFLOPs
- Transformers provide better accuracy (higher mAP) but higher computational cost
- E-PANNs stands out as the most efficient CNN variant (pruned architecture)
- Input flexibility varies significantly (0.025s to 10s minimum)

---

## Models Catalog

### AST - Audio Spectrogram Transformer

- **Paper**: Y. Gong, Y.-A. Chung, and J. Glass, "AST: Audio Spectrogram Transformer," in *Proc. Interspeech*, 2021, pp. 571–575.
- **Repository**: [https://github.com/YuanGongND/ast](https://github.com/YuanGongND/ast)
- **Checkpoint**: [audioset_10_10_0.4593.pth](https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1)
- **Architecture**: Vision Transformer (ViT) adapted for audio spectrograms.

### AudioCLIP

- **Paper**: A. Guzhov, F. Raue, J. Hees, and A. Dengel, "AudioCLIP: Extending CLIP to Image, Text and Audio," arXiv:2106.13043, 2021.
- **Repository**: [https://github.com/AndreyGuzhov/AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP)
- **Checkpoint**: [AudioCLIP-Full-Training.pt](https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt)
- **Architecture**: CLIP model extended with ESResNeXt audio encoder.

### AudioMAE - Audio Masked Autoencoder

- **Paper**: P.-Y. Huang, H. Xu, J. Li, A. Baevski, M. Auli, W. Galuba, F. Metze, and C. Feichtenhofer, "Masked Autoencoders that Listen," in *Proc. NeurIPS*, 2022.
- **Repository**: [https://github.com/facebookresearch/AudioMAE](https://github.com/facebookresearch/AudioMAE)
- **Checkpoint**: [finetuned.pth](https://drive.google.com/file/d/18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq)
- **Architecture**: Masked autoencoder with ViT backbone for audio.

### BEATs

- **Paper**: S. Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers," in *Proc. ICML*, 2023, pp. 5178–5193.
- **Repository**: [https://github.com/microsoft/unilm/tree/master/beats](https://github.com/microsoft/unilm/tree/master/beats)
- **Checkpoint**: [BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt](https://msranlcmtteamdrive.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt)
- **Architecture**: Transformer with acoustic tokenizers for self-supervised pre-training.

### CED - Consistent Ensemble Distillation

- **Paper**: H. Dinkel, Y. Wang, Z. Yan, J. Zhang, and Y. Wang, "CED: Consistent Ensemble Distillation for Audio Tagging," in *Proc. ICASSP*, 2024.
- **Repository**: [https://github.com/RicherMans/CED](https://github.com/RicherMans/CED)
- **Checkpoint**: [ced-base.pth](https://zenodo.org/record/8275347)
- **Architecture**: Efficient transformer trained via ensemble knowledge distillation.

### CLAP - Contrastive Language-Audio Pretraining

- **Paper**: Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, "Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation," in *Proc. ICASSP*, 2023.
- **Repository**: [https://github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)
- **Checkpoint**: [630k-audioset-fusion-best.pt](https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-fusion-best.pt)
- **Architecture**: Contrastive audio-text model with HTSAT audio encoder.

### ConvNeXt-Audio

- **Paper**: T. Pellegrini, I. Khalfaoui-Hassani, E. Labbé, and T. Masquelier, "Adapting a ConvNeXt Model to Audio Classification on AudioSet," in *Proc. Interspeech*, 2023.
- **Repository**: [https://github.com/topel/audioset-convnext-inf](https://github.com/topel/audioset-convnext-inf)
- **Checkpoint**: [convnext_tiny_471mAP.pth](https://zenodo.org/record/8020843)
- **Architecture**: ConvNeXt-Tiny adapted for audio spectrograms.

### EfficientAT (DyMN / MobileNetV3)

- **Paper**: F. Schmid, S. Koutini, and G. Widmer, "Efficient Large-Scale Audio Tagging via Transformer-To-CNN Knowledge Distillation," in *Proc. ICASSP*, 2023.
- **Paper (DyMN)**: F. Schmid, S. Koutini, and G. Widmer, "Dynamic Convolutional Neural Networks as Efficient Pre-trained Audio Models," *IEEE/ACM Trans. Audio, Speech, Language Process.*, submitted.
- **Repository**: [https://github.com/fschmid56/EfficientAT](https://github.com/fschmid56/EfficientAT)
- **Checkpoints**: Available via GitHub Releases.
- **Architecture**: MobileNetV3 and Dynamic MobileNet trained with knowledge distillation from transformers.

### E-PANNs - Efficient PANNs

- **Paper**: A. Singh, H. Liu, and M. D. Plumbley, "E-PANNs: Sound Recognition Using Efficient Pre-trained Audio Neural Networks," in *Proc. Inter-Noise*, 2023, pp. 7220–7228.
- **Paper**: A. Singh and M. D. Plumbley, "Efficient CNNs via Passive Filter Pruning," *IEEE/ACM Trans. Audio, Speech, Language Process.*, vol. 33, pp. 1763–1774, 2025.
- **Repository**: [https://github.com/Arshdeep-Singh-Boparai/E-PANNs](https://github.com/Arshdeep-Singh-Boparai/E-PANNs)
- **Checkpoint**: [efficient_cnn14.pth](https://doi.org/10.5281/zenodo.7939403)
- **Architecture**: Pruned CNN14 with ~70% parameter reduction.

### HTS-AT - Hierarchical Token-Semantic Audio Transformer

- **Paper**: K. Chen, X. Du, B. Zhu, Z. Ma, T. Berg-Kirkpatrick, and S. Dubnov, "HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection," in *Proc. ICASSP*, 2022.
- **Repository**: [https://github.com/RetroCirce/HTS-Audio-Transformer](https://github.com/RetroCirce/HTS-Audio-Transformer)
- **Checkpoint**: [HTSAT_AudioSet_Saved_1.ckpt](https://drive.google.com/drive/folders/1f5VYMk0uos_YnuBshgmaTVioXbs7Kmz6)
- **Architecture**: Swin Transformer with token-semantic module.

### M2D - Masked Modeling Duo

- **Paper**: D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, "Masked Modeling Duo: Towards a Universal Audio Pre-training Framework," *IEEE/ACM Trans. Audio, Speech, Language Process.*, vol. 32, pp. 2391–2406, 2024.
- **Paper (M2D-CLAP)**: D. Niizumi et al., "M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP," *IEEE Access*, vol. 13, pp. 163313–163330, 2025.
- **Repository**: [https://github.com/nttcslab/m2d](https://github.com/nttcslab/m2d)
- **Checkpoint**: Available via GitHub Releases.
- **Architecture**: Self-supervised learning with dual masked prediction.

### PANNs (CNN14 / ResNet38)

- **Paper**: Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, and M. D. Plumbley, "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition," *IEEE/ACM Trans. Audio, Speech, Language Process.*, vol. 28, pp. 2880–2894, 2020.
- **Repository**: [https://github.com/qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- **Checkpoints**: [Zenodo](https://zenodo.org/record/3987831)
- **Architecture**: ResNet38, Wavegram-Logmel-CNN14 variants.

### PaSST - Patchout Audio Spectrogram Transformer

- **Paper**: K. Koutini, J. Schlüter, H. Eghbal-zadeh, and G. Widmer, "Efficient Training of Audio Transformers with Patchout," in *Proc. Interspeech*, 2022, pp. 2753–2757.
- **Repository**: [https://github.com/kkoutini/PaSST](https://github.com/kkoutini/PaSST)
- **Checkpoint**: Available via GitHub Releases.
- **Architecture**: Vision Transformer with Patchout regularization.

### PSLA

- **Paper**: Y. Gong, Y.-A. Chung, and J. Glass, "PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation," *IEEE/ACM Trans. Audio, Speech, Language Process.*, vol. 29, pp. 3292–3306, 2021.
- **Repository**: [https://github.com/YuanGongND/psla](https://github.com/YuanGongND/psla)
- **Checkpoint**: [Dropbox](https://www.dropbox.com/sh/ihfbxcemxamihz9/AAD9zqnUptZzyZlquqpWllDya)
- **Architecture**: EfficientNet-B2 with 4-headed attention.

### VGGish

- **Paper**: S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification," in *Proc. ICASSP*, 2017, pp. 131–135.
- **Repository (Original)**: [https://github.com/tensorflow/models/tree/master/research/audioset](https://github.com/tensorflow/models/tree/master/research/audioset)
- **Repository (PyTorch)**: [https://github.com/w-hc/torch_audioset](https://github.com/w-hc/torch_audioset)
- **Architecture**: VGG-style CNN for audio embedding extraction.

### YAMNet

- **Paper**: (Based on MobileNetV1) A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861, 2017.
- **Repository (Original)**: [https://github.com/tensorflow/models/tree/master/research/audioset/yamnet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
- **Repository (PyTorch)**: [https://github.com/w-hc/torch_audioset](https://github.com/w-hc/torch_audioset)
- **Architecture**: MobileNetV1-based audio classifier.

---

## Requirements

Common dependencies for all models:

```python
torch>=1.9.0
torchaudio>=0.9.0
numpy>=1.19.0
scipy>=1.5.0
librosa>=0.8.0
```

Some models may require additional dependencies. Check the `requirements.txt` file in each model subdirectory if present.

---

## License

Each model retains its original license as specified by the respective authors. Please refer to the original repositories for licensing details:

- **MIT License**: PANNs, VGGish, YAMNet, HTS-AT, E-PANNs, EfficientAT, ConvNeXt-Audio
- **Apache 2.0**: PaSST, AST, AudioMAE
- **BSD-3-Clause**: PSLA
- **CC0-1.0**: CLAP (LAION)
- **GPL-3.0**: CED
- **Custom**: BEATs (Microsoft), M2D (NTT)

---

## Citation

If you use this collection in your research, please cite the respective papers for each model used. For the vendorization framework itself, please cite:

```bibtex
@misc{vendorized_audio_models,
  author = {Giacomelli, Stefano},
  title = {Vendorized Audio Models Collection},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/StefanoGiacomelli/vendorized-audio-models}
}
```

---

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
