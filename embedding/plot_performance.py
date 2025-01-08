import pandas as pd
import matplotlib.pyplot as plt
from globals import MONITORING_FILE, HKL_PATH, audio_file_path, SAVE_FIGURES

# Load the log file
log_file = MONITORING_FILE[:16] + "_NOVSC_1core.csv"  # Adjusted for your filename
data = pd.read_csv(log_file)

# Identify per-core CPU columns dynamically
core_columns = [col for col in data.columns if "Core" in col and "(%)" in col]

# Create subplots: 5 subplots (1 for total CPU and 4 for each core)
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 10), sharex=True)

# Plot Total CPU Load in the first subplot
axes[0].plot(data["Time"], data["Total_CPU(%)"], label="Total CPU Load (%)", linewidth=2, marker="o")
axes[0].set_ylabel("CPU Load (%)")
axes[0].set_title("Total CPU Load Over Time")
axes[0].legend()
axes[0].grid(True)

# Plot Per-Core CPU Load in separate subplots
for i, core in enumerate(core_columns):
    axes[i + 1].plot(data["Time"], data[core], label=core, linestyle="--", alpha=0.7)
    axes[i + 1].set_ylabel("CPU Load (%)")
    axes[i + 1].set_title(f"{core} Over Time")
    axes[i + 1].legend()
    axes[i + 1].grid(True)

# Set the x-axis label for the bottom subplot
axes[4].set_xlabel("Time (s)")

# Adjust layout and show the plots
plt.tight_layout()

# Save the figure
if SAVE_FIGURES:
    plt.savefig(HKL_PATH + "figures/" + audio_file_path[:-4] + "_CPU_per_core_separate.png", format='png', dpi=300)
    plt.savefig(HKL_PATH + "figures/" + audio_file_path[:-4] + "_CPU_per_core_separate.svg", format='svg')

plt.show()

# Plot Memory and Available Memory in separate figures
# ----------------------------------------------------

# Plot CPU and memory load over time
plt.figure(figsize=(10, 6))
plt.plot(data["Time"], data["Total_CPU(%)"], label="CPU Load (%)", marker="o")
plt.plot(data["Time"], data["Memory(%)"], label="Memory Load (%)", marker="o")
plt.xlabel("Time (s)")
plt.ylabel("Load (%)")
plt.title("System Performance Over Time")
plt.legend()
plt.grid()
if SAVE_FIGURES:
    plt.savefig(HKL_PATH + "figures/" + audio_file_path[:-4] + "_CPU_mem.png", format='png', dpi=300)
    plt.savefig(HKL_PATH + "figures/" + audio_file_path[:-4] + "_CPU_mem.svg", format='svg')
plt.show()

# Plot available memory over time
plt.figure(figsize=(10, 6))
plt.plot(data["Time"], data["Available Memory(MB)"], label="Available Memory (MB)", marker="o")
plt.xlabel("Time (s)")
plt.ylabel("Memory (MB)")
plt.title("Available Memory Over Time")
plt.legend()
plt.grid()
if SAVE_FIGURES:
    plt.savefig(HKL_PATH + "figures/" + audio_file_path[:-4] + "_ava_mem.png", format='png', dpi=300)
    plt.savefig(HKL_PATH + "figures/" + audio_file_path[:-4] + "_ava_mem.svg", format='svg')
plt.show()

