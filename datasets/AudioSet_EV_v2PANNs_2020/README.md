# AudioSet Emergency Vehicles v2 PANNs (2020)

This file provides instructions on downloading, extracting, and formatting AudioSet-EV v2 dataset to be ready for use. Follow the steps below to set up the dataset properly.

Commands pipeline is tested on Unix-based systems.

## 1. Download

Move inside the appropriate folder and download the dataset (from Zenodo), using the following commands:

```bash
cd ./datasets/AudioSet_EV_v2PANNs_2020/
wget https://zenodo.org/records/18668076/files/AudioSet-EV_v2_main.zip?download=1
```

or directly download it from [Zenodo](https://zenodo.org/records/18668076) an move ```.zip``` file inside this folder.

## 2. UnZip contents

Unzip contents in the CWD with:

```bash
unzip AudioSet-EV_v2_main.zip -d ./
rm -rf AudioSet-EV_v2_amin.zip          # Optional (remove .zip archive)
```

Resulting size of AudioSet-EV must be around 16GiB (without ```.zip``` file).

## 3. Check Directory contents

Check CWD contents

```bash
ls                                   # Output: Negatives_files, Positives_files, EV_Negatives.csv, EV_Positives.csv
```

## License

License Here ToDo

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{giacomelli2025audiosetev_v2,
author = {Giacomelli, Stefano and Rinaldi, Claudia},
title = {AudioSet-EV v2: a refined AudioSet-derived distribution of Emergency Vehicle Siren sounds},
month = feb,
year = 2025,
publisher = {Zenodo},
version = {v2.0},
doi = {10.5281/zenodo.18668076},
url = {https://zenodo.org/uploads/18668076}
}
```
