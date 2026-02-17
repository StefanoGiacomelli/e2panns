# UrbanSound8K

This file provides instructions on downloading, extracting, and formatting [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html) to be ready for use. Follow the steps below to set up the dataset properly.

Commands pipeline is tested on Unix-based systems.

## 1. Download

Download the dataset filling the official form (from the Official Site, link above reccomended) and move contents inside this folder, or run:

```bash
cd ./datasets/UrbanSound8K/

pip install soundata

python3
import soundata
dataset = soundata.initialize('urbansound8k')
dataset.download()              # download the dataset
dataset.validate()              # validate that all the expected files are there
```

## 2. UnZip contents (only if downloaded from the Official Site, reccomended)

Unzip contents in the CWD with:

```bash
unzip UrbanSound8K.tar.gz
ls                              # Output: audio, metadata
```

## License

Datasets compiled by Justin Salamon, Christopher Jacoby and Juan Pablo Bello. All files come from <www.freesound.org>.
Please see FREESOUNDCREDITS.txt (included in the dataset) for an attribution list.

The UrbanSound and UrbanSound8K datasets are offered free of charge for non-commercial use only under the terms of the Creative Commons Attribution Noncommercial License (by-nc), version 3.0: <http://creativecommons.org/licenses/by-nc/3.0/>

The datasets and their contents are made available on an "as is" basis and without warranties of any kind, including without limitation satisfactory quality and conformity, merchantability, fitness for a particular purpose, accuracy or completeness, or absence of errors. Subject to any liability that may not be excluded or limited by law, NYU is not liable for, and expressly excludes, all liability for loss or damage however and whenever caused to anyone by any use of the UrbanSound or UrbanSound8K datasets or any part of them.

## Citation

When UrbanSound or UrbanSound8K is used for academic research, we would highly appreciate it if scientific publications of works partly based on these datasets cite:

J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014. [ACM](http://dl.acm.org/citation.cfm?id=2655045) [PDF](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf)

```bibtex
@inproceedings{Salamon:UrbanSound:ACMMM:14,
    Address = {Orlando, FL, USA},
    Author = {Salamon, J. and Jacoby, C. and Bello, J. P.},
    Booktitle = {22nd {ACM} International Conference on Multimedia (ACM-MM'14)},
    Month = {Nov.},
    Pages = {1041--1044},
    Title = {A Dataset and Taxonomy for Urban Sound Research},
    Year = {2014}
}
```

The creation of the datasets was supported by a seed grant by NYU's Center for Urban Science and Progress (CUSP).
