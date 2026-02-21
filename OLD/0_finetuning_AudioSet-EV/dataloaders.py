import os
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
import torchaudio
import pytorch_lightning as pl


# AudioSet_EV Dataset ------------------------------------------------------------------------------------------------
class AudioSetEV_Dataset(Dataset):
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


class AudioSetEV_Aug_Dataset(Dataset):
    def __init__(self, file_path, folder_path, target_size=320000, binary_label=1, aug_p=0.7):
        self.cwd = os.getcwd()
        self.file_path = os.path.abspath(os.path.join(self.cwd, file_path))
        self.folder_path = os.path.abspath(os.path.join(self.cwd, folder_path))
        self.filenames = self.get_filenames(self.folder_path)
        self.target_size = target_size
        self.skipped_files = []
        self.label = binary_label
        self.augment_p = aug_p
        self.augmentations = self.define_augmentations()
        self.applied_augmentations = []

    def get_filenames(self, path):
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]

    def define_augmentations(self):
        return {"add_noise": self.add_random_noise,
                "time_roll": self.time_roll,
                "polarity_inversion": self.polarity_inversion,
                "rand_amp_scaling": self.random_amplification,}

    def add_random_noise(self, waveform, scale=0.1):
        noise_type = random.choice(["white", "gaussian"])
        noise = torch.randn_like(waveform) if noise_type == "white" else torch.normal(0, 1, size=waveform.shape)
        return waveform + noise * scale / torch.max(torch.abs(waveform + noise * scale)) 

    def time_roll(self, waveform):
        shift = random.randint(1, waveform.size(1))
        return torch.roll(waveform, shifts=shift, dims=1)

    def polarity_inversion(self, waveform):
        return waveform * -1

    def random_amplification(self, waveform):
        if random.random() > 0.5:
            scalar = random.uniform(0.1, 1.0)
            return waveform * scalar
        else:
            vector = torch.rand(waveform.size(1))
            return waveform * vector.unsqueeze(0)

    def apply_augmentations(self, waveform):
        applied = []
        augment_order = list(self.augmentations.keys())
        
        # Randomize order & apply augmentations
        random.shuffle(augment_order)
        for augment_name in augment_order:
            if random.random() < self.augment_p:
                waveform = self.augmentations[augment_name](waveform)
                applied.append(augment_name)
        
        return waveform, applied

    def __len__(self):
        return len(self.filenames)
    
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

        # Apply augmentations
        waveform_tensor, applied = self.apply_augmentations(waveform_tensor)

        # Track augmentations applied
        self.applied_augmentations.append({"file_path": file_path,
                                           "augmentations": applied})

        return waveform_tensor, self.label


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    
    return torch.utils.data.default_collate(batch)


class AudioSetEV_DataModule(pl.LightningDataModule):
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
        pos_dataset = AudioSetEV_Dataset(self.pos_file, self.pos_folder, binary_label=1)
        neg_dataset = AudioSetEV_Dataset(self.neg_file, self.neg_folder, binary_label=0)

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
                          collate_fn=custom_collate_fn,
                          shuffle=self.train_shuffle,
                          num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset,
                          batch_size=self.batch_size,
                          collate_fn=custom_collate_fn,
                          shuffle=False,
                          num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=custom_collate_fn,
                          shuffle=False,
                          num_workers=2)


class AudioSetEV_Aug_DataModule(pl.LightningDataModule):
    def __init__(self, TP_file, TP_folder, TN_file, TN_folder, batch_size=32, split_ratios=(0.8, 0.1, 0.1), shuffle=True, aug_prob=0.7):
        super().__init__()
        self.pos_folder = TP_folder
        self.neg_folder = TN_folder
        self.pos_file = TP_file
        self.neg_file = TN_file
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.train_shuffle = shuffle
        self.aug_prob = aug_prob

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Load the full datasets for TP and TN
        pos_dataset = AudioSetEV_Aug_Dataset(self.pos_file, self.pos_folder, binary_label=1, aug_p=self.aug_prob)
        neg_dataset = AudioSetEV_Aug_Dataset(self.neg_file, self.neg_folder, binary_label=0, aug_p=self.aug_prob)

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
                          collate_fn=custom_collate_fn,
                          shuffle=self.train_shuffle,
                          num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset,
                          batch_size=self.batch_size,
                          collate_fn=custom_collate_fn,
                          shuffle=False,
                          num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=custom_collate_fn,
                          shuffle=False,
                          num_workers=2)
