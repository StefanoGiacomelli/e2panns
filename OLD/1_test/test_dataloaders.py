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


# ESC-50 Dataset ------------------------------------------------------------------------------------------------
class ESC50_TestDataset(Dataset):
    def __init__(self, file_path, folder_path, target_size=160000, target_sr=32000):
        self.cwd = os.getcwd()
        self.file_path = os.path.abspath(os.path.join(self.cwd, file_path))
        self.folder_path = os.path.abspath(os.path.join(self.cwd, folder_path))
        self.target_size = target_size
        self.target_sr = target_sr
        self.filenames, self.labels = self.filter_filenames()
        self.skipped_files = []

    def __len__(self):
        return len(self.filenames)

    def filter_filenames(self):
        df = pd.read_csv(self.file_path)

        # Filter relevant categories and assign binary labels
        relevant_labels = ["siren", "helicopter", "chainsaw", "car_horn", "engine", "train", "church_bells", "airplane", "clock_alarm"]
        siren_label = 1
        other_labels = 0
        filenames = []
        labels = []

        for _, row in df.iterrows():
            if row["category"] == "siren":
                filenames.append(os.path.join(self.folder_path, f"fold_{row['fold']}", row["filename"]))
                labels.append(siren_label)
            elif row["category"] in relevant_labels:
                filenames.append(os.path.join(self.folder_path, f"fold_{row['fold']}", row["filename"]))
                labels.append(other_labels)

        return filenames, labels

    def __getitem__(self, idx):
        file_path = self.filenames[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(file_path)

            # Resample to target sample rate if necessary
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)

            # Pad or truncate waveform to target_size
            current_size = waveform.size(1)
            if current_size < self.target_size:
                padding = self.target_size - current_size
                waveform = F.pad(waveform, (0, padding), "constant", 0)
            elif current_size > self.target_size:
                waveform = waveform[:, :self.target_size]
        except Exception as e:
            self.skipped_files.append((idx, file_path))
            print(f"Skipping Error loading {file_path}: {e}")
            return None

        return waveform, label


class ESC50_DataModule(pl.LightningDataModule):
    def __init__(self, file_path, folder_path, target_size=160000, target_sr=32000, batch_size=32):
        super().__init__()
        self.file_path = file_path
        self.folder_path = folder_path
        self.target_size = target_size
        self.target_sr = target_sr
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = ESC50_TestDataset(file_path=self.file_path,
                                         folder_path=self.folder_path,
                                         target_size=self.target_size,
                                         target_sr=self.target_sr)

        # Prepare dataloaders for all folds
        self.test_loaders = {}
        for fold in range(1, 6):
            fold_indices = [i for i, file in enumerate(self.dataset.filenames) if f"fold_{fold}" in file]
            fold_subset = torch.utils.data.Subset(self.dataset, fold_indices)
            self.test_loaders[f"fold_{fold}"] = DataLoader(fold_subset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return list(self.test_loaders.values())


# sireNNet Dataset ------------------------------------------------------------------------------------------------
class sireNNet_TestDataset(Dataset):
    def __init__(self, folder_path, target_size=96000, target_sr=32000):
        self.folder_path = os.path.abspath(folder_path)
        self.target_size = target_size
        self.target_sr = target_sr
        self.file_paths, self.labels = self._load_files()
        self.skipped_files = []

    def __len__(self):
        return len(self.file_paths)

    def _load_files(self):
        labels_map = {"ambulance": 1,
                      "firetruck": 1,
                      "police": 1,
                      "traffic": 0}
        file_paths = []
        labels = []

        for category, label in labels_map.items():
            category_path = os.path.join(self.folder_path, category)
            if os.path.exists(category_path):
                for file_name in os.listdir(category_path):
                    if file_name.endswith('.wav'):
                        file_paths.append(os.path.join(category_path, file_name))
                        labels.append(label)

        return file_paths, labels

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(file_path)

            # Resample to target_sr if necessary
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)
            
            # Stereo to Mono: Sum channels and normalize
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Pad or truncate waveform to target_size
            current_size = waveform.size(1)
            if current_size < self.target_size:
                padding = self.target_size - current_size
                waveform = F.pad(waveform, (0, padding), "constant", 0)
            elif current_size > self.target_size:
                waveform = waveform[:, :self.target_size]
        except Exception as e:
            self.skipped_files.append((idx, file_path))
            print(f"Skipping Error loading {file_path}: {e}")
            return None

        return waveform, label


class sireNNet_DataModule(pl.LightningDataModule):
    def __init__(self, folder_path, batch_size=32, target_size=96000, target_sr=32000):
        super().__init__()
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.target_sr = target_sr

        # Sizes for multiple random balanced subsets (fractions of the dataset)
        self.sizes = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]

    def setup(self, stage=None):
        self.dataset = sireNNet_TestDataset(folder_path=self.folder_path,
                                            target_size=self.target_size,
                                            target_sr=self.target_sr)

    def get_balanced_subset(self, fraction):
        # Find indices of positive (1) and negative (0) samples
        indices_1s = [i for i, label in enumerate(self.dataset.labels) if label == 1]
        indices_0s = [i for i, label in enumerate(self.dataset.labels) if label == 0]

        # Determine the number of samples per class for the given fraction
        num_samples_per_class = int(len(self.dataset) * fraction // 2)

        # Shuffle and select the required number of samples
        random.shuffle(indices_1s)
        random.shuffle(indices_0s)

        selected_indices_1s = indices_1s[:num_samples_per_class]
        selected_indices_0s = indices_0s[:num_samples_per_class]

        # Combine and shuffle the selected indices
        balanced_indices = selected_indices_1s + selected_indices_0s
        random.shuffle(balanced_indices)

        # Create a subset
        return Subset(self.dataset, balanced_indices)

    def test_dataloader(self):
        dataloaders = []
        for fraction in self.sizes:
            subset = self.get_balanced_subset(fraction)
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            dataloaders.append(loader)

        return dataloaders


# LSSiren Dataset ------------------------------------------------------------------------------------------------
class LSSiren_TestDataset(Dataset):
    def __init__(self, folder_path, target_sr=32000, min_length=32000):
        self.folder_path = os.path.abspath(folder_path)
        self.target_sr = target_sr
        self.min_length = min_length
        self.file_paths, self.labels = self._load_files()
        self.skipped_files = []

    def _load_files(self):
        labels_map = {"Ambulance_data": 1, "Road_Noises": 0}
        file_paths = []
        labels = []

        for category, label in labels_map.items():
            category_path = os.path.join(self.folder_path, category)
            if os.path.exists(category_path):
                for file_name in os.listdir(category_path):
                    if file_name.endswith('.wav'):
                        file_paths.append(os.path.join(category_path, file_name))
                        labels.append(label)

        return file_paths, labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(file_path)

            # Stereo 2 mono: sum channels and normalize
            if waveform.size(0) > 1:  # More than 1 channel (stereo)
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to target sample rate if necessary
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)

            # Zero-pad if waveform is shorter than 1 second
            current_size = waveform.size(1)
            if current_size < self.min_length:
                padding = self.min_length - current_size
                waveform = F.pad(waveform, (0, padding), "constant", 0)
        
        except Exception as e:
            # Log and skip problematic files
            self.skipped_files.append((idx, file_path))
            print(f"Skipping Error loading {file_path}: {e}")
            return None

        return waveform, label


def lssiren_custom_collate_fn(batch):
    # Filter out invalid samples
    batch = [item for item in batch if item is not None]

    if not batch:
        raise ValueError("All samples in the batch are invalid.")

    waveforms, labels = zip(*batch)

    # Find the maximum length in the batch
    max_length = max(waveform.size(1) for waveform in waveforms)

    # Pad all waveforms to the maximum length
    padded_waveforms = torch.stack([F.pad(waveform, (0, max_length - waveform.size(1)), "constant", 0) for waveform in waveforms])

    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_waveforms, labels


class LSSiren_DataModule(pl.LightningDataModule):
    def __init__(self, folder_path, batch_size=32, target_sr=32000, min_length=32000):
        super().__init__()
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.target_sr = target_sr
        self.min_length = min_length

    def setup(self, stage=None):
        self.dataset = LSSiren_TestDataset(folder_path=self.folder_path,
                                           target_sr=self.target_sr,
                                           min_length=self.min_length)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, collate_fn=lssiren_custom_collate_fn)


# UrbanSound8K Dataset ------------------------------------------------------------------------------------------------
class UrbanSound8K_TestDataset(Dataset):
    def __init__(self, folder_path, metadata_path, target_sr=32000, min_length=32000, fold=None):
        self.folder_path = os.path.abspath(folder_path)
        self.metadata_path = os.path.abspath(metadata_path)
        self.target_sr = target_sr
        self.min_length = min_length
        self.fold = fold
        self.file_paths, self.labels = self._load_files()
        self.skipped_files = []

    def _load_files(self):
        # Load metadata CSV
        metadata = pd.read_csv(self.metadata_path)

        # Filter by fold if specified
        if self.fold is not None:
            metadata = metadata[metadata["fold"] == self.fold]

        # Assign labels
        file_paths = []
        labels = []

        for _, row in metadata.iterrows():
            file_path = os.path.join(self.folder_path, f"fold{row['fold']}", row["slice_file_name"])
            label = 1 if row["class"] == "siren" else 0
            file_paths.append(file_path)
            labels.append(label)

        return file_paths, labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(file_path)

            # Stereo to mono: Sum channels and normalize
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to target sample rate if necessary
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)

            # Zero-pad if waveform is shorter than 1 second
            current_size = waveform.size(1)
            if current_size < self.min_length:
                padding = self.min_length - current_size
                waveform = F.pad(waveform, (0, padding), "constant", 0)

        except Exception as e:
            self.skipped_files.append((idx, file_path))
            print(f"Skipping Error loading {file_path}: {e}")
            return None

        return waveform, label


def urbansound8k_collate_fn(batch):
    batch = [item for item in batch if item is not None]

    if not batch:
        raise ValueError("All samples in the batch are invalid.")

    waveforms, labels = zip(*batch)

    # Find the maximum length in the batch
    max_length = max(waveform.size(1) for waveform in waveforms)

    # Pad all waveforms to the maximum length
    padded_waveforms = torch.stack([F.pad(waveform, (0, max_length - waveform.size(1)), "constant", 0) for waveform in waveforms])

    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_waveforms, labels


class UrbanSound8K_DataModule(pl.LightningDataModule):
    def __init__(self, folder_path, metadata_path, batch_size=32, target_sr=32000, min_length=32000):
        super().__init__()
        self.folder_path = folder_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.target_sr = target_sr
        self.min_length = min_length

    def setup(self):
        self.datasets = {fold: UrbanSound8K_TestDataset(folder_path=self.folder_path,
                                                        metadata_path=self.metadata_path,
                                                        target_sr=self.target_sr,
                                                        min_length=self.min_length,
                                                        fold=fold) for fold in range(1, 11)}
        
        self.test_loaders = {fold: DataLoader(dataset,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         num_workers=2,
                                         collate_fn=urbansound8k_collate_fn) for fold, dataset in self.datasets.items()}

    def test_dataloaders(self):
        return list(self.test_loaders.values())


# FreeSound-50K Dataset -----------------------------------------------------------------------------------------------
class FSD50K_TestDataset(Dataset):
    def __init__(self, csv_file, folder_path, target_sr=16000, label=1):
        self.folder_path = os.path.abspath(folder_path)
        self.data = pd.read_csv(csv_file)
        self.target_sr = target_sr
        self.label = label
        self.skipped_files = []
        self.resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=self.target_sr)  # Will set dynamically
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_name = str(self.data.iloc[idx, 0]) + ".wav"
        file_path = os.path.join(self.folder_path, file_name)
        
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            if sample_rate != self.target_sr:
                self.resampler.orig_freq = sample_rate
                waveform = self.resampler(waveform)
        except Exception as e:
            self.skipped_files.append((idx, file_path))
            print(f"Skipping file {file_path} due to error: {e}")
            return None
        
        return waveform, self.label


def fsd50k_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("All samples in the batch are invalid.")
    
    waveforms, labels = zip(*batch)
    max_length = max(waveform.size(1) for waveform in waveforms)
    padded_waveforms = torch.stack([F.pad(waveform, (0, max_length - waveform.size(1)), "constant", 0) for waveform in waveforms])
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_waveforms, labels


class FSD50K_DataModule(pl.LightningDataModule):
    def __init__(self, pos_file, neg_file, folder_path, batch_size=32, target_sr=16000):
        super().__init__()
        self.pos_csv = pos_file
        self.neg_csv = neg_file
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.target_sr = target_sr
        self.test_dataset = None
    
    def setup(self, stage=None):
        pos_dataset = FSD50K_TestDataset(self.pos_csv, self.folder_path, target_sr=self.target_sr, label=1)
        neg_dataset = FSD50K_TestDataset(self.neg_csv, self.folder_path, target_sr=self.target_sr, label=0)
        
        self.test_dataset = torch.utils.data.ConcatDataset([pos_dataset, neg_dataset])
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=fsd50k_collate_fn, shuffle=False, num_workers=2)
