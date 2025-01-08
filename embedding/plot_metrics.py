import matplotlib.pyplot as plt
import pandas as pd
from globals import HKL_PATH, audio_file_path, SAVE_FIGURES

# Load data
inference_data = pd.read_csv(HKL_PATH + audio_file_path[:-4] + "_inference_metrics.csv")

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column of subplots

# Plot Inference Results
axes[0].plot(inference_data["Inference Start Time (s)"], inference_data["Inference Result"], 
             label="Inference Result", marker="o", linestyle="-")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Inference Probability")
axes[0].set_title("Inference Results Over Time")
axes[0].grid()

# Plot Inference Duration
axes[1].plot(inference_data["Inference Start Time (s)"], inference_data["Inference Duration (s)"], 
             label="Inference Duration", marker="o", linestyle="-")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Inference Duration (s)")
axes[1].set_title("Inference Duration Over Time")
axes[1].grid()

if SAVE_FIGURES:    
    plt.savefig(HKL_PATH + "figures/" + audio_file_path[:-4] + "_metrics_NOVSC.png", format='png', dpi=300)
    plt.savefig(HKL_PATH + "figures/" + audio_file_path[:-4] + "_metrics_NOVSC.svg", format='svg')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
