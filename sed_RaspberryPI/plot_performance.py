"""
plot_performance.py

This script visualises system performance logs generated during real-time inference.
It produces plots for:
- Total CPU usage
- Per-core CPU usage
- Memory usage
- Available memory

It works with the structured output directory:
    outputs/<checkpoint_name>/perf/<audio_stem>.csv

Figures are saved to:
    outputs/<checkpoint_name>/figures/

Usage examples:
---------------
# Use defaults from globals.py
$ python plot_performance.py

# Override audio/checkpoint path
$ python plot_performance.py --audio files/AudioSet_EV_Positives_Debug/foo.wav --checkpoint files/checkpoints/model.ckpt --save

Dependencies:
-------------
- matplotlib
- pandas
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from globals import AUDIO_FILE_PATH, AUDIO_FILE, CHECKPOINT_PATH, CHECKPOINT, SAVE_FIGURES, HKL_PATH

# CLI parser
parser = argparse.ArgumentParser(description="Plot CPU and memory usage for a given audio/checkpoint.")
parser.add_argument("--audio", type=str, default=str(Path(AUDIO_FILE_PATH) / AUDIO_FILE), help="Path to the audio file")
parser.add_argument("--checkpoint", type=str, default=str(Path(CHECKPOINT_PATH) / CHECKPOINT), help="Path to the checkpoint")
parser.add_argument("--save", action="store_true", help="Save figures to file")
args = parser.parse_args()

# Paths
audio_path = Path(args.audio)
checkpoint_path = Path(args.checkpoint)

audio_stem = audio_path.stem
checkpoint_name = checkpoint_path.name

perf_csv = Path(HKL_PATH) / checkpoint_name / "perf" / f"{audio_stem}.csv"
figures_dir = Path(HKL_PATH) / checkpoint_name / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Load performance log
data = pd.read_csv(perf_csv)

# Detect CPU cores dynamically
core_columns = [col for col in data.columns if "Core" in col and "(%)" in col]

# Plot Total CPU + per-core
fig, axes = plt.subplots(nrows=1 + len(core_columns), ncols=1, figsize=(12, 2 + 2 * (1 + len(core_columns))), sharex=True)

axes[0].plot(data["Time"], data["Total_CPU(%)"], label="Total CPU (%)", linewidth=2, marker="o")
axes[0].set_ylabel("CPU Load (%)")
axes[0].set_title("Total CPU Load Over Time")
axes[0].legend()
axes[0].grid(True)

for i, core in enumerate(core_columns):
    axes[i + 1].plot(data["Time"], data[core], label=core, linestyle="--", alpha=0.8, marker="o")
    axes[i + 1].set_ylabel("CPU Load (%)")
    axes[i + 1].set_title(core)
    axes[i + 1].legend()
    axes[i + 1].grid(True)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()

if args.save or SAVE_FIGURES:
    plt.savefig(figures_dir / f"{audio_stem}_CPU_per_core.png", dpi=300)
    plt.savefig(figures_dir / f"{audio_stem}_CPU_per_core.svg")

plt.show()

# Plot CPU + memory
plt.figure(figsize=(10, 6))
plt.plot(data["Time"], data["Total_CPU(%)"], label="CPU Load (%)", marker="o")
plt.plot(data["Time"], data["Memory(%)"], label="Memory Load (%)", marker="o")
plt.xlabel("Time (s)")
plt.ylabel("Load (%)")
plt.title("System CPU and Memory Load")
plt.legend()
plt.grid()
if args.save or SAVE_FIGURES:
    plt.savefig(figures_dir / f"{audio_stem}_CPU_mem.png", dpi=300)
    plt.savefig(figures_dir / f"{audio_stem}_CPU_mem.svg")
plt.show()

# Plot available memory
plt.figure(figsize=(10, 6))
plt.plot(data["Time"], data["Available Memory(MB)"], label="Available Memory (MB)", marker="o")
plt.xlabel("Time (s)")
plt.ylabel("Memory (MB)")
plt.title("Available Memory Over Time")
plt.legend()
plt.grid()
if args.save or SAVE_FIGURES:
    plt.savefig(figures_dir / f"{audio_stem}_ava_mem.png", dpi=300)
    plt.savefig(figures_dir / f"{audio_stem}_ava_mem.svg")
plt.show()
