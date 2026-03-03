# XAI Framework for Emergency Vehicle Recognition

Comprehensive Explainable AI framework for comparative analysis of E-PANNs, CED, and CLAP models on the Emergency Vehicle Recognition task.

## 📁 Structure

```
xAI/
├── config.yaml                 # Main configuration file
├── main.py                     # Pipeline orchestrator
├── core/                       # Model-specific explainers
│   ├── base_explainer.py      # Abstract base class
│   ├── cnn_explainer.py       # EPANNs explainer
│   ├── transformer_explainer.py # CED explainer
│   └── clap_explainer.py      # CLAP explainer
├── methods/                    # XAI techniques
│   ├── gradients.py           # Guided/Vanilla Backprop
│   ├── cam.py                 # Score-CAM, Grad-CAM
│   ├── attention.py           # Attention Rollout, Patch Importance
│   └── spectral.py            # Filterbank Analysis, Spectrogram Extraction
├── metrics/                    # Quantitative evaluation
│   ├── sensitivity.py         # Deletion, Insertion, Avg Drop
│   └── localization.py        # Sparsity, Peak-to-Mean, Cross-Model Agreement
├── visualization/              # Plotting utilities
│   ├── plots.py               # Individual plots
│   └── comparison.py          # Multi-model comparisons
├── test_samps/                 # Test audio samples
│   ├── TP_*.wav               # True Positive sample
│   ├── TN_*.wav               # True Negative sample
│   ├── TP_metadata.json       # TP metadata with probabilities
│   └── TN_metadata.json       # TN metadata
└── results/                    # Output directory
    ├── *_side_by_side.svg     # Comparison visualizations
    ├── *_time_series.svg      # Temporal saliency
    ├── *_filterbank_*.svg     # Filterbank analysis
    └── xai_analysis_report.json # Summary metrics
```

## 🎯 Features

### XAI Methods Implemented

1. **Gradient-Based**
   - Guided Backpropagation
   - Vanilla Backpropagation

2. **Activation-Based**
   - Score-CAM (gradient-free)
   - Grad-CAM

3. **Architecture-Specific**
   - **Transformers (CED)**: Attention Rollout, Patch Importance
   - **CLAP**: Window Attention Maps, TSCAM
   - **EPANNs & CLAP**: Filterbank Analysis (learned vs standard)

### Quantitative Metrics

**Consistency:**
- Gini sparsity
- Peak-to-Mean ratio
- Top-k concentration

**Faithfulness:**
- Deletion curve & AUC
- Insertion curve & AUC
- Average Drop

**Cross-Model:**
- Pearson/Spearman correlation
- Consensus score
- Temporal IoU

### Visualizations

- Side-by-side model comparison
- Saliency overlay on spectrograms
- Temporal saliency profiles
- Difference maps (TP vs TN)
- Filterbank comparisons
- Metrics summary charts
- Comprehensive multi-panel analysis

All outputs in **SVG @ 600 DPI** for publication quality.

## 🚀 Usage

### 1. Configuration

Edit `config.yaml` to specify:
- Model checkpoints
- Test samples
- Enabled methods and metrics
- Visualization settings

### 2. Run Analysis

```bash
cd /home/user/Documenti/E2PANNs
source .venv/bin/activate

# Run full pipeline
python -m models.xAI.main

# Or with custom config
python -m models.xAI.main --config path/to/config.yaml
```

### 3. View Results

Results saved to `models/xAI/results/`:
- `xai_analysis_report.json` - All metrics
- `*_side_by_side.svg` - Visual comparisons
- `*_filterbank_*.svg` - Filterbank analysis

## 📊 Sample Selection

Test samples automatically selected with **model consensus**:
- **TP**: All models predict ~100% EV probability
- **TN**: All models predict ~0% EV probability

Metadata in `test_samps/*_metadata.json` includes:
- Original file path
- Ground truth label
- Per-model probabilities

## 🔬 Extending the Framework

### Add New XAI Method

1. Create method in `methods/` (inherit from appropriate base)
2. Add to explainer in `core/`
3. Enable in `config.yaml`
4. Update `main.py` to call new method

### Add New Metric

1. Implement in `metrics/`
2. Add computation in `main.py::analyze_sample()`
3. Enable in `config.yaml`

### Add New Visualization

1. Add plot function to `visualization/`
2. Call in `main.py::generate_visualizations()`
3. Configure in `config.yaml`

## 📝 Configuration Reference

### Model Settings

```yaml
models:
  epanns:
    checkpoint: "path/to/checkpoint.ckpt"
    sample_rate: 32000
    architecture: "cnn"
```

### Method Settings

```yaml
methods:
  guided_backprop:
    enabled: true
    target_layers:
      epanns: ["conv_block6"]
```

### Metric Settings

```yaml
metrics:
  deletion:
    enabled: true
    steps: 20
```

### Visualization Settings

```yaml
visualization:
  output_format: "svg"
  dpi: 600
  plots:
    side_by_side_comparison: true
```

## 🎨 Output Examples

### Side-by-Side Comparison
- Left: Input spectrogram
- Middle: Saliency map
- Right: Overlay

### Time Series
- Temporal saliency profiles for all models overlaid

### Filterbank Analysis
- Learned vs Standard filterbanks
- Difference maps
-Centroid frequency comparison

## 📦 Dependencies

Core dependencies (already in main requirements.txt):
- PyTorch
- torchaudio
- torchlibrosa
- numpy
- scipy
- matplotlib
- pyyaml
- tqdm

## 🎯 Metrics Interpretation

**Sparsity (0-1):** Higher = more focused
**Peak-to-Mean:** Higher = more concentrated
**Deletion AUC:** Lower = better faithfulness
**Insertion AUC:** Higher = better faithfulness
**Average Drop:** Lower = better localization
**Cross-Model Correlation:** Higher = more agreement

## 🔧 Troubleshooting

### Low Cross-Model Agreement
- Normal for different architectures
- Focus on faithfulness metrics

### Attention Extraction Fails
- Architecture-specific - may need model modification
- Fallback to gradient-based methods

### OOM Errors
- Reduce batch_size in config
- Process samples sequentially
- Use CPU if GPU memory limited

## 📚 References

- **Score-CAM**: Wang et al., 2020
- **Guided Backprop**: Springenberg et al., 2015
- **Attention Rollout**: Abnar & Zuidema, 2020
- **Deletion/Insertion**: Petsiuk et al., 2018

---

**Author**: Stefano Giacomelli  
**Date**: March 2026  
**License**: See main repository LICENSE
