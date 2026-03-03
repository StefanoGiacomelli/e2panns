#!/usr/bin/env python3
"""
extract_false_positives.py

This script identifies and extracts a subset of false true positives (FTPs) from the
AudioSet_EV_Strong dataset based on user listening and manual selection.

It works by:
- Scanning the directory `files/AudioSet_EV_Strong_FTP/` for WAV files selected as false true positives.
- Extracting the YouTube IDs (ytid) from the filenames (e.g., ytid_Original.wav).
- Matching these ytid prefixes against the `segment_id` column in the annotations file
  `files/EV_Positives_merged.tsv`.
- Extracting all rows whose `segment_id` starts with one of the selected ytid values.
- Saving the matched entries into a new file: `files/EV_False_Positives.tsv`.

The output file:
- Has the same structure and header as the input TSV.
- Can be used for analysis, exclusion, or manual curation of strong-labeled datasets.

Usage:
-------
$ python extract_false_positives.py

Dependencies:
-------------
- pandas
- Python 3.7+

"""

import pandas as pd
from pathlib import Path

# Paths
FTP_FOLDER = Path("files/AudioSet_EV_Strong_FTP")
ANNOTATIONS_FILE = Path("files/EV_Positives_merged.tsv")
OUTPUT_FILE = Path("files/EV_False_Positives.tsv")

# Load annotation data
df_annotations = pd.read_csv(ANNOTATIONS_FILE, sep="\t")

# Extract ytid list from filenames (strip _Original.wav or _Reduced.wav)
ytids = {
    f.stem.split("_")[0]
    for f in FTP_FOLDER.glob("*.wav")
    if f.is_file()
}

# Extract ytid list from filenames (strip _Original.wav or _Reduced.wav)
ytids = {
    f.stem.split("_")[0]
    for f in FTP_FOLDER.glob("*.wav")
    if f.is_file()
}

# Extract ytid prefix from segment_id
df_annotations["ytid"] = df_annotations["segment_id"].apply(lambda x: x.split("_")[0])

# Filter rows where the ytid is in our FTP list
filtered_df = df_annotations[df_annotations["ytid"].isin(ytids)].drop(columns="ytid")

# Save to output TSV
filtered_df.to_csv(OUTPUT_FILE, sep="\t", index=False)

print(f"Saved {len(filtered_df)} rows to: {OUTPUT_FILE}")
