# SED System - Real-Time Sound Event Detection for Emergency Vehicles

Modular real-time Sound Event Detection (SED) system for Emergency Vehicle siren detection with support for multiple deep learning models (EPANNs, CED, CLAP).

---

## 🎯 Overview

The **SED System** provides a production-ready framework for processing audio streams and detecting emergency vehicle sirens in real-time. It simulates real-time audio ingestion using a circular buffer architecture with adaptive frame sizing, enabling efficient inference on resource-constrained devices.

### Key Features

✅ **Real-time Simulation**: Circular buffer with frame-by-frame inference  
✅ **Adaptive Window Sizing**: Dynamic frame duration based on model confidence  
✅ **Multi-Model Support**: EPANNs (32kHz), CED (16kHz), CLAP (48kHz)  
✅ **Dual Operation Modes**: Single file evaluation + Full dataset processing  
✅ **Comprehensive SED Metrics**: DCASE-compliant segment-based & event-based evaluation  
✅ **Performance Monitoring**: Real-time CPU/RAM usage tracking  
✅ **Visualization**: Mel-spectrogram overlay with predictions and detected events  

---

## 📂 Project Structure

```text
sed_system/
├── core/                            # Core inference components
│   ├── __init__.py                  # Core exports (load_inference_model, etc.)
│   ├── audio_processor.py           # Audio loading, resampling, normalization
│   ├── buffer.py                    # Thread-safe CircularBuffer
│   ├── frame_provider.py            # InputFrameProvider (adaptive sizing)
│   ├── inference_engine.py          # Inference loop and frame processing
│   └── model_loader.py              # Model loading utilities
│
├── monitoring/                      # Performance & metrics tracking
│   ├── __init__.py                  # Monitoring exports
│   ├── performance_monitor.py       # CPU/RAM monitoring
│   ├── sed_metrics.py               # SED metrics (segment & event)
│   └── metrics_logger.py            # Logging utilities
│
├── visualization/                   # Result visualization
│   ├── __init__.py                  # Visualization exports
│   └── plotter.py                   # Mel-spectrogram + predictions plotting
│
├── pipeline.py                      # Main SED pipeline
├── config.yaml                      # Example configuration file
└── README.md                        # This file
```

---

## 🚀 Main Scripts

The SED system is accessed through two main scripts in the project root:

### **`main_sed_file.py`** - Single File Processing

Process a single audio file with SED inference and visualization.

```bash
python main_sed_file.py benchmark_configs/sed/epanns_finetunedBinaryEV_TPfile.yaml
```

**Features:**
- Process single audio file (WAV format)
- Generate mel-spectrogram visualization with predictions
- Compute SED metrics if ground truth available
- Save results and plots

**Example config:** [`benchmark_configs/sed/epanns_finetunedBinaryEV_TPfile.yaml`](../benchmark_configs/sed/)

---

### **`main_sed_dataset.py`** - Dataset Processing

Process entire AudioSet_EV_Strong dataset (positives only) with batch evaluation.

```bash
python main_sed_dataset.py benchmark_configs/sed/epanns_finetunedBinaryEV_AS-EV_Strong_v2.yaml
```

**Features:**
- Process all positive samples from AudioSet_EV_Strong dataset
- Aggregate SED metrics across samples (segment & event-based)
- Performance tracking (throughput, CPU, RAM)
- Save comprehensive results (CSV + JSON)
- Progress bar with clean console output (logs to file)

**Example config:** [`benchmark_configs/sed/epanns_finetunedBinaryEV_AS-EV_Strong_v2.yaml`](../benchmark_configs/sed/)

**Output:**
- `results/sed_AudioSet_EV_v2_epanns_finetuned/`
  - `summary.json` - Aggregated metrics
  - `results.csv` - Per-sample metrics
  - `sample_metrics.json` - Detailed metrics
  - `run.log` - Processing log

---

## 🛠️ Core Components

### **CircularBuffer** (`core/buffer.py`)

Thread-safe lock-free circular buffer for real-time audio streaming simulation.

**Features:**
- Fixed-size buffer with wrap-around indexing
- Thread-safe read/write operations
- Writing completion signaling

### **InputFrameProvider** (`core/frame_provider.py`)

Adaptive frame extraction from circular buffer.

**Features:**
- Dynamic frame sizing (min → max duration)
- Confidence-based adaptation
- Thread-safe frame retrieval

### **Inference Engine** (`core/inference_engine.py`)

Frame-by-frame inference with model-agnostic processing.

**Features:**
- Background inference thread
- Single frame inference function
- Full simulation wrapper (`run_inference_simulation`)

### **SED Metrics** (`monitoring/sed_metrics.py`)

DCASE-compliant Sound Event Detection metrics using `sed_eval`.

**Metrics:**
- **Segment-based**: Frame-level TP/FP/FN, Precision, Recall, F1, Accuracy
- **Event-based**: Event-level matching with collar tolerance, Error Rate

### **Performance Monitor** (`monitoring/performance_monitor.py`)

Real-time system resource monitoring (for Linux-based edge devices).

**Tracked:**
- CPU usage (%) - mean, min, max
- RAM usage (MB) - mean, min, max  
- Throughput

---

## ⚙️ Configuration

SED system configurations are stored in [`benchmark_configs/sed/`](../benchmark_configs/sed/).

### Configuration Structure

```yaml
# Audio input
audio_file: "path/to/audio.wav"

# Model configuration
model:
  name: "epanns"  # Options: epanns, ced, clap
  checkpoint: "path/to/checkpoint.pt"
  device: "cpu"   # Options: cpu, mps, cuda

# Inference settings
inference:
  threshold: 0.5                    # Detection threshold
  chunk_duration: 0.310             # Minimum frame duration (s)
  buffer_duration: 20.0             # Circular buffer size (s)
  
  adaptive_window:
    enabled: true                   # Enable adaptive sizing
    frame_duration_max: 1.0         # Maximum frame duration (s)
    adapt_coeff: 0.4                # Adaptation coefficient

# SED metrics
sed_metrics:
  segment_time_resolution: 0.310    # Frame-level resolution (s)
  event_tolerance: 0.500            # Event matching collar (s)
  ground_truth_file: null           # Optional GT file (DCASE format)

# Visualization
visualization:
  plot_predictions: true
  save_plot: "results/predictions.png"

# Logging
logging:
  level: "INFO"  # DEBUG for detailed frame-by-frame logs
```

---

## 📊 SED Metrics Explained

### Segment-based Metrics (Frame-level)

Evaluated at fixed time resolution (e.g., 310ms frames):

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Error Rate**: (FP + FN) / Total Frames

### Event-based Metrics (Event-level)

Evaluated on detected events with onset/offset matching:

- **Precision**: Correctly detected events / Total detected events
- **Recall**: Correctly detected events / Total ground truth events
- **F1-Score**: Harmonic mean of event-based Precision and Recall
- **Error Rate**: (Insertions + Deletions) / Total GT events
- **Collar**: Time tolerance for onset/offset matching (default: 500ms)

---

## 🔧 Requirements

(already inside main `requirements.txt`):

```
torch
torchaudio
numpy
soundfile
librosa
sed_eval
matplotlib
psutil
pyyaml
tqdm
```

---

## 👨‍💻 Authors

**Stefano Giacomelli, Marco Giordano** - Ph.D. candidates in ICT  
DISIM Department - University of L'Aquila (Italy) 
[GitHub Profile](https://github.com/StefanoGiacomelli)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

See also: [GitHub E2PANNs-RBPI5 Alternative repository](https://github.com/marco-giordano/e2panns-rbpi5)

and if you use this tools or find this work useful, please cite:

```bibtex
@INPROCEEDINGS{11284671,
  author={Giordano, Marco and Giacomelli, Stefano and Rinaldi, Claudia and Graziosi, Fabio},
  booktitle={2025 IEEE 6th International Symposium on the Internet of Sounds (IS2)}, 
  title={{Real-Time Emergency Vehicle Siren Detection with Efficient CNNs on Embedded Hardware}ß}, 
  year={2025},
  volume={},
  number={},
  pages={1-10},
  doi={10.1109/IS264627.2025.11284671}}
```
