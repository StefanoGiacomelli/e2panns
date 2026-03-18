# Emergency Vehicles Audio Benchmark

This directory contains the **Emergency Vehicles (EV) Benchmark**, a comprehensive collection of audio datasets for training, evaluating, and benchmarking audio deep learning models on Emergency Vehicle (EV) siren detection and classification tasks.

The benchmark supports both **binary** (siren vs. non-siren) and **multi-class** (police, ambulance, fire truck, and negative samples) label systems.

---

## 📂 Directory Structure

```text
datasets/
├── AudioSet_EV_v1_2025/          # AudioSet-derived EV dataset (2025 release)
├── AudioSet_EV_v2PANNs_2020/     # AudioSet-derived EV dataset (2020 PANNs release)
├── AudioSet_EV_Strong/           # AudioSet_EV-derived Strong annotations (AudioSet Strong parsing)
├── KineScaper_EV/                # Synthetic-realistic EV siren dataset with temporal metadata
├── sireNNet/                     # sireNNet dataset (urban EV sirens)
├── ESC50/                        # Environmental Sound Classification-50
├── FSD50K/                       # Freesound Dataset 50K
├── LSSiren/                      # Large-Scale Siren dataset
├── UrbanSound8K/                 # Urban Sound 8K dataset
├── datasets_mapping.json         # Label mappings for all datasets
├── multi_class_utils.py          # Utilities for 4-way classification
└── README.md                     # This file
```

---

## 🎯 Datasets Overview

### **Primary EV Datasets**

#### **AudioSet EV v1 (2025)**

- **Source**: Curated from Google's AudioSet (2017)
- **Content**: ~14,000 audio clips (7,324 positives, 6,702 negatives)
- **Labels**: Binary + Multi-class (Police, Ambulance, Fire-trucks)
- **Duration**: 10-second clips at 32 kHz
- **Features**: Multi-label annotation support, segment-type metadata

#### **AudioSet EV v2 (2020 PANNs)**

- **Source**: Extended from PANNs pre-training AudioSet
- **Content**: ~28,000 audio clips (7,900 positives, 20,916 negatives)
- **Labels**: Binary + Multi-class (same as AudioSet EV v1) with stratified negatives
- **Duration**: 10-second clips at 32 kHz
- **Features**: Stratified negative sampling across 39 sound categories

#### **AudioSet EV Strong (v1 & v2)**

- **Source**: AudioSet Strong Metadata parsing over AudioSet-EV v1 and v2
- **Content**: ~28,000 audio clips (1186 positives, 119,283 negatives)
- **Labels**: Temporally aligned strong annotations (onset-offset) with negatives balancing procedures
- **Duration**: 10-second clips at 32 kHz
- **Features**: Sound Event Detection annotations w. comprehensive contents analysis

#### **KineScaper EV**

- **Source**: Physics-based siren trajectory simulation with controlled acoustic conditions
- **Content**: 61,600 samples (7 classes × 8,800 each)
- **Labels**: Binary, multiclass siren types, temporal onset/offset metadata
- **Duration**: 40-second clips at 32 kHz
- **Features**: Rich SNR statistics (`snr_min`, `snr_max`, `snr_avg`, `snr_std`), SPL targets, motion metadata

#### **sireNNet**

- **Source**: Urban environment recordings (Mendeley Data)
- **Content**: ~1,600 audio clips across 4 classes
- **Labels**: Police, Ambulance, Fire-truck, Traffic (balanced)
- **Duration**: Variable-length clips at 44.1 kHz
- **Features**: High-quality real-world recordings

### **Auxiliary Benchmark Datasets**

These datasets complement the primary EV datasets for transfer learning, robustness testing, and cross-domain evaluation:

- **ESC-50**: Environmental sound classification (50 classes)
- **FSD50K**: Freesound dataset with 200 sound classes
- **LSSiren**: Large-scale siren recordings
- **UrbanSound8K**: Urban sound classification (10 classes)

All datasets are parsed to filter SIREN categories as *positives*, URBAN and "challenging" sound classes as *negatives*.

---

## 🛠️ Setup Instructions

Each dataset subdirectory contains a **`README.md`** or **`GUIDE.md`** file with detailed instructions for:

1. **Downloading** the dataset from official sources
2. **Extracting** and organizing files
3. **Verifying** data integrity and structure
4. **Configuring** for use with training/evaluation scripts

---

## 📊 Utility Files

### **`datasets_mapping.json`**

Central configuration file containing:

- Label mappings for binary classification
- Multi-class mappings (police=1, ambulance=2, fire=3, negative=0)
- AudioSet MID (Machine ID) to class label mappings
- Negative focus labels for stratified sampling

### **`multi_class_utils.py`**

Python utilities for 4-way multi-class classification:

- `FourWayBalancer`: Balances datasets across 4 classes
- `parse_audioset_multi_labels()`: Parses AudioSet CSV with multi-label MIDs
- `get_class_names_from_mapping()`: Extracts human-readable class names
- `print_balance_summary()`: Displays dataset balance statistics

### **`positive_salt_mapping.py`**

Mapping label utilities from/towards SALT (Standardized Audio events Label Taxonomy) integration.

---

## 🎓 Usage Examples

### Binary Classification

```python
from datasets.AudioSet_EV_v1_2025.dataloader import AudioSetEV_v1_DataModule

dm = AudioSetEV_v1_DataModule(pos_csv_path="datasets/AudioSet_EV_v1_2025/EV_Positives.csv",
                              pos_audio_folder="datasets/AudioSet_EV_v1_2025/Positive_files",
                              neg_csv_path="datasets/AudioSet_EV_v1_2025/EV_Negatives.csv",
                              neg_audio_folder="datasets/AudioSet_EV_v1_2025/Negative_files",
                              mode='train',
                              label_mode='binary',  # Binary classification
                              batch_size=32,
                              seed=42)
```

### Multi-Class Classification (4-way)

```python
dm = AudioSetEV_v1_DataModule(# ... same parameters ...
                              label_mode='multi_class',  # 4-way classification
                              # Classes: 0=negative, 1=police, 2=ambulance, 3=fire
                              )
```

---

## 📖 Citation

If you use this benchmark in your research, please cite the associated paper:

```bibtex
@article{TODO,
  title={TODO},
  author={TODO},
  journal={TODO},
  year={TODO}
}
```

Individual datasets should also be cited according to their respective licenses and attribution requirements (see each dataset's README).

---

## 🔗 Related Resources

This benchmark represents an updated and extended version of the **EV-Benchmark**, originally released in preliminary form at:

**<https://github.com/StefanoGiacomelli/audioset-tools/tree/main/EV-benchmark>**

The current release includes:

- ✅ Expanded datasets coverage
- ✅ Multi-class support (4-way)
- ✅ Stratified Negatives sampling
- ✅ Comprehensive PyTorch Lightning DataModules (w. data augmentation & pre-processing integrations)
- ✅ KineScaper-EV integration for unified training and SED robustness studies

---

## 📄 License

Each dataset is ruled by its respective license. Please refer to individual README files for specific license information and attribution requirements.
