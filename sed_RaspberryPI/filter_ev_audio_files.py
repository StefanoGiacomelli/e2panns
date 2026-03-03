# filter_ev_audio_files.py

import pandas as pd
from pathlib import Path
import shutil

# Paths
MERGED_TSV = Path("./files/EV_Positives_merged.tsv")
AUDIO_DIR = Path("./files/AudioSet_EV_Positives")
OUTPUT_DIR = Path("./files/AudioSet_EV_Positives_No_Strong")

# Create output folder if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load merged TSV and extract unique ytids
df = pd.read_csv(MERGED_TSV, sep='\t')
ytids = set(df['segment_id'].str.split('_').str[0])

# Move unrelated audio files to OUTPUT_DIR
for audio_file in AUDIO_DIR.glob("*.wav"):
    base = audio_file.stem.split('_')[0]  # extract ytid
    if base not in ytids:
        target = OUTPUT_DIR / audio_file.name
        shutil.move(str(audio_file), str(target))
        print(f"Moved: {audio_file.name} -> {target}")

print("Filtering complete.")
