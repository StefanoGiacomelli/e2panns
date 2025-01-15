import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torchaudio
import pytorch_lightning as pl


# Dataset ------------------------------------------------------------------------------------------------
class AudioDataset(Dataset):
    def __init__(self, file_path, folder_path, target_size=320000, binary_label=1):
        self.cwd = os.getcwd()
        self.file_path = os.path.abspath(os.path.join(self.cwd, file_path))
        self.folder_path = os.path.abspath(os.path.join(self.cwd, folder_path))
        self.filenames = self.get_filenames(self.folder_path)
        self.target_size = target_size
        self.skipped_files = []
        self.label = binary_label

    def __len__(self):
        return len(self.filenames)
    
    def get_filenames(self, path):
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]

    def __getitem__(self, idx):
        file_path = self.filenames[idx]
        try:
            waveform_tensor, _ = torchaudio.load(file_path)
        except Exception as e:
            self.skipped_files.append((idx, file_path))
            print(f"Skipping Error loading {file_path}: {e}")
            return None

        # Pad or truncate waveform_tensor to target_size
        current_size = waveform_tensor.size(1)
        if current_size < self.target_size:
            padding = self.target_size - current_size
            waveform_tensor = F.pad(waveform_tensor, (0, padding), "constant", 0)
        elif current_size > self.target_size:
            waveform_tensor = waveform_tensor[:, :self.target_size]

        return waveform_tensor, self.label

def test_collate_fn(batch):
    """
    Custom collate function to filter out None values from a torch.Dataset batch.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    
    return torch.utils.data.default_collate(batch)


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, TP_file, TP_folder, TN_file, TN_folder, batch_size=32, split_ratios=(0.8, 0.1, 0.1), shuffle=True):
        super().__init__()
        self.pos_folder = TP_folder
        self.neg_folder = TN_folder
        self.pos_file = TP_file
        self.neg_file = TN_file
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.train_shuffle = shuffle

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Load the full datasets for TP and TN
        pos_dataset = AudioDataset(self.pos_file, self.pos_folder, binary_label=1)
        neg_dataset = AudioDataset(self.neg_file, self.neg_folder, binary_label=0)

        # Combine datasets
        combined_dataset = ConcatDataset([pos_dataset, neg_dataset])

        # Compute split sizes
        total_size = len(combined_dataset)
        train_size = int(self.split_ratios[0] * total_size)
        dev_size = int(self.split_ratios[1] * total_size)
        test_size = total_size - train_size - dev_size

        # Train/Dev/Test split
        self.train_dataset, self.dev_dataset, self.test_dataset = random_split(combined_dataset,
                                                                               [train_size, dev_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          collate_fn=test_collate_fn,
                          shuffle=self.train_shuffle,
                          num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset,
                          batch_size=self.batch_size,
                          collate_fn=test_collate_fn,
                          shuffle=False,
                          num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=test_collate_fn,
                          shuffle=False,
                          num_workers=2)

    def save_skipped_files_to_csv(self, file_path="./reading_fails.csv"):
        if self.train_dataset is None or self.dev_dataset is None or self.test_dataset is None:
            raise ValueError("The datasets are not initialized. Call setup() first.")

        skipped_details = []
        datasets = [("train", self.train_dataset), ("dev", self.dev_dataset), ("test", self.test_dataset)]
        for split_name, dataset in datasets:
            for idx, subdataset in enumerate(dataset.datasets):
                if hasattr(subdataset, "skipped_files"):
                    for file_idx, filename in subdataset.skipped_files:
                        skipped_details.append({"split": split_name,
                                                "dataset": "TPs" if isinstance(subdataset, AudioDataset) and subdataset.label == 1 else "TNs",
                                                "batch_idx": idx // self.batch_size,
                                                "sample_idx": file_idx % self.batch_size,
                                                "filepath": filename})

        # Write details to CSV
        with open(file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["split", "dataset", "batch_idx", "sample_idx", "filename"])
            writer.writeheader()
            writer.writerows(skipped_details)

        print(f"Skipped files saved to {file_path}")