# KineScaper Emergency Vehicles Dataset

## Overview

**TODO: Add full description after Zenodo publication**

The KineScaper Emergency Vehicles (KineScaper_EV) dataset is a synthetic dataset with real siren sources for emergency vehicle detection and classification. It contains 61,600 audio samples (40 seconds each) with realistic urban traffic scenarios and simulated emergency vehicle siren trajectories.

## Dataset Characteristics

- **Total samples**: 61,600 audio files
- **Duration**: 40 seconds per file
- **Sample rate**: 32 kHz
- **Channels**: Mono
- **Format**: WAV
- **Total size**: ~293 GB

### Siren Classes

The dataset contains 7 balanced siren types:
- **hi-lo**: 8,800 samples
- **two-tone**: 8,800 samples
- **wail**: 8,800 samples
- **phaser**: 8,800 samples
- **piercer**: 8,800 samples
- **rumbler**: 8,800 samples
- **yelp**: 8,800 samples

### Siren Characteristics

Each sample varies in:
- **Siren type**: electronic, mechanical, pneumatic
- **Waveform**: sine, square, triangular, trapezoid, sawtooth
- **Velocity**: 40-90 km/h
- **Background SPL**: 50-71 dB (quantized at 3 dB steps)
- **Foreground SPL**: 80-110 dB (quantized at 3 dB steps)
- **SNR average**: -54.98 to 17.25 dB (mean: -20.46 dB)

### Temporal Annotations

Each sample includes:
- **Onset**: Start time of siren event (3-10 seconds)
- **Offset**: End time of siren event (19.66-40 seconds)
- **Mean duration**: ~30.28 seconds

## Installation

### 1. Download Dataset

**TODO: Add Zenodo DOI and download link**

```bash
# Download from Zenodo
wget https://zenodo.org/record/XXXXX/files/KineScaper_EV_dataset.zip

# Extract dataset
unzip KineScaper_EV_dataset.zip -d /path/to/extract/
```

### 2. Setup in E2PANNs Project

Place the dataset in the appropriate location:

```bash
# If using external storage (recommended due to size)
# Mount point: /mnt/ssd/Kinescaper_EV/dataset/

# Or copy to project directory (requires ~293 GB)
cp -r /path/to/KineScaper_EV_dataset ./datasets/KineScaper_EV/
```

### 3. Verify Structure

Ensure the dataset has the following structure:

```
KineScaper_EV/dataset/
├── audio/                  # 61,600 WAV files
├── json/
│   └── metadata.json       # Complete metadata (JSON format)
├── csv/
│   └── metadata.tsv        # Complete metadata (TSV format)
├── config_siren.yaml       # Generation configuration
├── generation_log_*.txt    # Generation log
└── summary.txt             # Dataset summary
```

## Usage

### Basic Example

```python
from datasets.KineScaper_EV.dataloader import KineScaper_EV_DataModule

# Binary classification (train mode)
dm = KineScaper_EV_DataModule(
    mode="train",
    label_type="binary",
    dataset_root="/mnt/ssd/Kinescaper_EV/dataset/",
    batch_size=32,
    num_workers=4
)

dm.setup()
train_loader = dm.train_dataloader()
```

### Modes

1. **Train mode**: 80/10/10 split for training, validation, and testing
2. **Benchmark mode**: Entire dataset used for evaluation
3. **Detection mode**: Full 40s samples with temporal label tracks

### Label Types

1. **Binary**: 0=negative, 1=positive (any siren)
2. **Multiclass**: 0-6=siren types, 7=negative
   - 0: hi-lo
   - 1: two-tone
   - 2: wail
   - 3: phaser
   - 4: piercer
   - 5: rumbler
   - 6: yelp
   - 7: negative

## Metadata Format

Each sample includes the following metadata fields:

- `event_label`: Siren class name
- `filename`: Audio file name
- `onset`: Event start time (seconds)
- `offset`: Event end time (seconds)
- `snr_min`, `snr_max`, `snr_avg`, `snr_std`: SNR statistics (dB)
- `frame_size`: Analysis frame size (samples)
- `velocity_kmh`: Vehicle velocity (km/h)
- `closest_distance`: Minimum distance to microphone (meters)
- `siren_class`: Siren type
- `subset_index`: Subset index
- `iteration`: Generation iteration
- `bg_spl_target`: Target background SPL (dB)
- `fg_spl_target`: Target foreground SPL (dB)

## File Naming Convention

Files follow this pattern:
```
{siren_class}_{siren_type}_{waveform}_{iteration}_{onset}_{offset}_i0.wav
```

Example:
```
hi-lo_electronic_sawtooth_00_3.164_34.409_i0.wav
```

## Citation

**TODO: Add citation information after publication**

```bibtex
@dataset{kinescaper_ev_2026,
  author    = {TODO},
  title     = {KineScaper Emergency Vehicles Dataset},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {TODO},
  url       = {TODO}
}
```

## License

**TODO: Specify license**


## Contact

For questions or issues, please contact:
- **Stefano Giacomelli**
- Open a GitHub issue on the project repository for technical questions about integration and dataloaders.

---

**Note**: This README documents the current in-repository integration state. Publication identifiers and final archival links will be added after public release.
