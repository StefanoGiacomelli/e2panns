from pathlib import Path
from main import main

# Define input folders
AUDIO_DIR = Path("./files/AudioSet_EV_Positives")
CHECKPOINT_DIR = Path("./files/checkpoints")

# Collect .wav and .ckpt files
audio_files = sorted(AUDIO_DIR.glob("*.wav"))
checkpoints = sorted(CHECKPOINT_DIR.glob("*.ckpt"))

# Sanity check
if not audio_files:
    print("No audio files found.")
    exit(1)
if not checkpoints:
    print("No checkpoint files found.")
    exit(1)

total_runs = len(audio_files) * len(checkpoints)
run_counter = 0

for c_idx, checkpoint in enumerate(checkpoints, start=1):
    for a_idx, audio_file in enumerate(audio_files, start=1):
        run_counter += 1
        print(f"[{run_counter}/{total_runs}] "
              f"Checkpoint {c_idx}/{len(checkpoints)}: {checkpoint.name} | "
              f"Audio {a_idx}/{len(audio_files)}: {audio_file.name}")
        
        main(audio_file=str(audio_file), checkpoint=str(checkpoint))
