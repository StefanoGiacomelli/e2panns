# merge_ev_annotations.py

import pandas as pd
from pathlib import Path

# Input TSV paths
EVAL_TSV = Path("./files/EV_Positives_eval.tsv")
TRAIN_TSV = Path("./files/EV_Positives_train.tsv")

# Output path
MERGED_TSV = Path("./files/EV_Positives_merged.tsv")

# Load both TSV files
df_eval = pd.read_csv(EVAL_TSV, sep='\t')
df_train = pd.read_csv(TRAIN_TSV, sep='\t')

# Concatenate them together
df_combined = pd.concat([df_eval, df_train], ignore_index=True)

# Drop duplicates based on the first three columns
unique_df = df_combined.drop_duplicates(subset=["segment_id", "start_time_seconds", "end_time_seconds"])

# Drop the label column (no longer needed)
unique_df = unique_df.drop(columns=["label"])

# Save the merged TSV
unique_df.to_csv(MERGED_TSV, sep='\t', index=False)
print(f"Merged annotations saved to: {MERGED_TSV}")
