# FSD50K

This file provides instructions on downloading, extracting, and formatting [FSD50K dataset](https://annotator.freesound.org/fsd/release/FSD50K/) to be ready for use. Follow the steps below to set up the dataset properly.

Commands pipeline is tested on Unix-based systems.

## 1. Download

Move inside the appropriate folder and download the dataset (from Zenodo), using the following commands:

```bash
cd ./datasets/FSD50K/
wget https://zenodo.org/api/records/4060432/files-archive
```

or directly download it from [Zenodo](https://zenodo.org/records/4060432) an move all files inside this folder.

## 2. UnZip contents

Unzip contents in the CWD with:

```bash
zip -s 0 FSD50K.dev_audio.zip --out unsplit.zip
unzip unsplit.zip
```

or

```bash
7z x FSD50K.dev_audio.zip
```

Contents will be around 24.7GiB (you can check the correct directory structure at Zenodo, link above)

## 3. Run the Python script

Run the Python script ```metadata_parser.py``` to create 2x ```.csv``` files containing parsed samples from the ```eval``` folder

```bash
python3 metadata_parser.py
```

## License

All audio clips in FSD50K are released under Creative Commons (CC) licenses. Each clip has its own license as defined by the clip uploader in Freesound, some of them requiring attribution to their original authors and some forbidding further commercial reuse. Specifically:

The development set consists of 40,966 clips with the following licenses:

- CC0: 14,959
- CC-BY: 20,017
- CC-BY-NC: 4616
- CC Sampling+: 1374

The evaluation set consists of 10,231 clips with the following licenses:

- CC0: 4914
- CC-BY: 3489
- CC-BY-NC: 1425
- CC Sampling+: 403

For attribution purposes and to facilitate attribution of these files to third parties, mapping from the audio clips to their corresponding licenses is included. The licenses are specified in the files dev_clips_info_FSD50K.json and eval_clips_info_FSD50K.json.

In addition, FSD50K as a whole is the result of a curation process and it has an additional license: FSD50K is released under CC-BY. This license is specified in the LICENSE-DATASET file downloaded with the FSD50K.doc zip file. We note that the choice of one license for the dataset as a whole is not straightforward as it comprises items with different licenses (such as audio clips, annotations, or data split). The choice of a global license in these cases may warrant further investigation (e.g., by someone with a background in copyright law).

Usage of FSD50K for commercial purposes:

If you'd like to use FSD50K for commercial purposes, please contact Eduardo Fonseca and Frederic Font at <efonseca@google.com> and <frederic.font@upf.edu>.

Also, if you are interested in using FSD50K for machine learning competitions, please contact Eduardo Fonseca and Frederic Font at <efonseca@google.com> and <frederic.font@upf.edu>.

## Citation

If you use the FSD50K dataset, or part of it, please cite the related TASLP paper (available from [arXiv](https://arxiv.org/abs/2010.00475) or [TASLP](https://ieeexplore.ieee.org/document/9645159)):

```bibtex
@article{fonseca2022FSD50K,
    title={{FSD50K}: an open dataset of human-labeled sound events},
    author={Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    volume={30},
    pages={829--852},
    year={2022},
    publisher={IEEE}
}
```
