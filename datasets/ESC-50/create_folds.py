############################################################################################################
#
#  This script generates the 5 cross-validation folds for the ESC-50 dataset
#
############################################################################################################
import os
import shutil
import pandas as pd


# Paths
dataset_dir = './datasets/ESC-50'
csv_path = os.path.join(dataset_dir, 'esc50.csv')
audio_dir = os.path.join(dataset_dir, 'original_audio/')
output_dir = os.path.join(dataset_dir, 'cross_val_folds/')

# Create output folders if not already present
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(1, 6):
    fold_path = os.path.join(output_dir, f'fold_{i}')
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)

# Read the CSV file
df = pd.read_csv(csv_path)

# Organize files into fold folders
for _, row in df.iterrows():
    filename = row['filename']
    fold = row['fold']
    
    source_path = os.path.join(audio_dir, filename)
    destination_path = os.path.join(output_dir, f'fold_{fold}', filename)
    
    # Move or copy the file
    shutil.copy(source_path, destination_path)