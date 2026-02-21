# ESC-50

This file provides instructions on downloading, extracting, and formatting ESC-50 dataset to be ready for use. Follow the steps below to set up the dataset properly.

Commands pipeline is tested on Unix-based systems.

## 1. Download

Move inside the appropriate folder and download the dataset (from GitHub), using the following commands:

```bash
cd ./datasets/ESC50/
wget https://github.com/karoldvl/ESC-50/archive/master.zip
```

or directly download it from [GitHub](https://github.com/karolpiczak/ESC-50) an move ```master.zip``` file inside this folder.

## 2. UnZip contents

Unzip contents in the CWD with:

```bash
unzip master.zip -d ./
rm -rf master.zip          # Optional (remove .zip archive)
```

Check CWD contents

```bash
ls                         # Output: audio, meta, create_folds.py
```

## 3. Run the Python script

Run the Python script ```create_folds.py``` to create a copy of the ESC-50 dataset contents organized in 5x cross-validation folders

```bash
python3 create_folds.py
rm -rf /audio              # Optional (remove old audio archive)
```

## License

The dataset is available under the terms of the [Creative Commons Attribution Non-Commercial license](http://creativecommons.org/licenses/by-nc/3.0/).

A smaller subset (clips tagged as ESC-10) is distributed under CC BY (Attribution).

Attributions for each clip are available in the [LICENSE](https://github.com/karolpiczak/ESC-50/blob/master/LICENSE) file.

## Citation

If you find this dataset useful in an academic setting please cite:

K. J. Piczak. ESC: Dataset for Environmental Sound Classification. Proceedings of the 23rd Annual ACM Conference on Multimedia, Brisbane, Australia, 2015. [LINK Paper](http://karol.piczak.com/papers/Piczak2015-ESC-Dataset.pdf)

DOI: [http://dx.doi.org/10.1145/2733373.2806390](http://dx.doi.org/10.1145/2733373.2806390)

```bibtex
@inproceedings{piczak2015dataset,
    title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
    author = {Piczak, Karol J.},
    booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
    date = {2015-10-13},
    url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
    doi = {10.1145/2733373.2806390},
    location = {{Brisbane, Australia}},
    isbn = {978-1-4503-3459-4},
    publisher = {{ACM Press}},
    pages = {1015--1018}
}
```
