"""
plot_metrics.py

This script generates diagnostic plots from inference metrics produced by the real-time simulation pipeline.

It visualizes:
- Inference probability over time
- Inference duration per frame
- Frame size (in seconds) over time

The script works with the structured output directory:
    outputs/<checkpoint_name>/inference_metrics/<audio_stem>.csv

Usage examples:
---------------
# Use defaults from globals.py
$ python plot_metrics.py

# Specify audio file and checkpoint manually
$ python plot_metrics.py --audio files/AudioSet_EV_Positives_Debug/foo.wav --checkpoint files/checkpoints/model.ckpt --save

Dependencies:
-------------
- matplotlib
- pandas
- Python 3.7+

Figures will be saved in:
    outputs/<checkpoint_name>/figures/
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from globals import AUDIO_FILE_PATH, AUDIO_FILE, CHECKPOINT_PATH, CHECKPOINT, SAVE_FIGURES, HKL_PATH

# Argument parser
parser = argparse.ArgumentParser(description="Plot inference metrics for a given audio file and checkpoint.")
parser.add_argument("--audio", type=str, default=str(Path(AUDIO_FILE_PATH) / AUDIO_FILE), help="Path to the audio file")
parser.add_argument("--checkpoint", type=str, default=str(Path(CHECKPOINT_PATH) / CHECKPOINT), help="Path to the checkpoint file")
parser.add_argument("--save", action="store_true", help="Save figures instead of just showing them")
args = parser.parse_args()

# Build paths
audio_path = Path(args.audio)
checkpoint_path = Path(args.checkpoint)
audio_stem = audio_path.stem
checkpoint_name = checkpoint_path.name

# Input paths
metrics_file = Path(HKL_PATH) / checkpoint_name / "inference_metrics" / f"{audio_stem}.csv"
frame_size_file = Path(HKL_PATH) / checkpoint_name / "frame_size" / f"{audio_stem}.csv"
figures_dir = Path(HKL_PATH) / checkpoint_name / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Load CSV files
inference_data = pd.read_csv(metrics_file)
frame_data = pd.read_csv(frame_size_file)

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Plot inference result
axes[0].plot(inference_data["Inference Start Time (s)"], inference_data["Inference Result"],
             label="Inference Result", marker="o", linestyle="-")
axes[0].set_ylabel("Probability")
axes[0].set_title("Inference Probability Over Time")
axes[0].grid()

# Plot inference duration
axes[1].plot(inference_data["Inference Start Time (s)"], inference_data["Inference Duration (s)"],
             label="Inference Duration", marker="o", linestyle="-")
axes[1].set_ylabel("Duration (s)")
axes[1].set_title("Inference Duration Per Frame")
axes[1].grid()

# Plot frame size
axes[2].plot(frame_data["Frame Request Time (s)"], frame_data["Frame Size (s)"],
             label="Frame Size", marker="o", linestyle="-")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Frame Size (s)")
axes[2].set_title("Frame Size Over Time")
axes[2].grid()

# Layout and save
plt.tight_layout()
if args.save or SAVE_FIGURES:
    plt.savefig(figures_dir / f"{audio_stem}_metrics_with_frames.png", format="png", dpi=300)
    plt.savefig(figures_dir / f"{audio_stem}_metrics_with_frames.svg", format="svg")

plt.show()
