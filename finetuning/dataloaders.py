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
class ESC50_Dataset(Dataset):
    def __init__(self,
                 file_path: str,
                 folder_path: str,
                 target_size: int = 160000,
                 target_sr: int = 32000):
        """
        :param file_path: Path to the ESC-50 CSV (or a modified CSV).
        :param folder_path: Root folder containing subfolders fold_1, fold_2, etc.
        :param target_size: Desired number of audio samples (e.g. 160k for ~5s at 32kHz).
        :param target_sr: Desired sample rate for audio (e.g. 32 kHz).
        """
        super().__init__()
        self.file_path = os.path.abspath(file_path)
        self.folder_path = os.path.abspath(folder_path)
        self.target_size = target_size
        self.target_sr = target_sr

        # The old approach singled out 'siren' as label=1, others as 0 (from relevant_labels).
        # We'll keep that logic here.
        self.filenames, self.labels = self._collect_filenames_and_labels()

        # Track any files that fail to load
        self.skipped_files = []

    def __len__(self):
        return len(self.filenames)

    def _collect_filenames_and_labels(self):
        """
        Reads the CSV (with columns like 'filename', 'fold', 'category'),
        merges all folds into one big list, then uses the old 'siren vs. others' logic.
        """
        df = pd.read_csv(self.file_path)

        # Categories we consider "other" (label 0), plus "siren" (label 1).
        relevant_labels = ["siren",        # label=1
                           "helicopter", "chainsaw", "car_horn", "engine", "train", "church_bells", "airplane", "clock_alarm"]
        siren_label = 1
        other_label = 0

        filenames = []
        labels = []

        for _, row in df.iterrows():
            cat = row["category"]
            if cat not in relevant_labels:
                continue  # skip categories not in our relevant set

            # Build full path: folder_path/fold_X/filename.wav
            audio_path = os.path.join(self.folder_path,
                                      f"fold_{row['fold']}",
                                      row["filename"])

            if not os.path.exists(audio_path):
                # Optionally log or skip
                continue

            # "siren" => label=1, everything else => label=0
            label = siren_label if cat == "siren" else other_label
            filenames.append(audio_path)
            labels.append(label)

        return filenames, labels

    def __getitem__(self, idx):
        file_path = self.filenames[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(file_path)

            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)

            # Pad or truncate to target_size
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
    def __init__(self,
                 file_path: str,
                 folder_path: str,
                 batch_size: int = 32,
                 split_ratios: tuple = (0.8, 0.1, 0.1),
                 shuffle: bool = True,
                 target_size: int = 160000,
                 target_sr: int = 32000):
        """
        A DataModule aligned with AudioSetEV_DataModule style,
        but merges all official ESC-50 folds into one dataset, then
        randomly splits them into train/dev/test sets.
        
        :param file_path: Path to the ESC-50 CSV (with columns like fold, category, filename).
        :param folder_path: Directory containing fold_1, fold_2, etc.
        :param batch_size: Batch size for DataLoaders.
        :param split_ratios: (train_ratio, dev_ratio, test_ratio).
        :param shuffle: Whether to shuffle in train_dataloader().
        :param target_size: Number of samples (e.g. 160k for ~5s at 32kHz).
        :param target_sr: Sample rate to resample audio to (e.g. 32kHz).
        """
        super().__init__()
        self.file_path = file_path
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.train_shuffle = shuffle
        self.target_size = target_size
        self.target_sr = target_sr

        self.train_dataset = None
        self.dev_dataset   = None
        self.test_dataset  = None

    def setup(self, stage=None):
        """
        Build the full ESC-50 dataset, then split by ratio for train/dev/test.
        """
        # 1) Create the combined dataset from all folds
        full_dataset = ESC50_Dataset(file_path=self.file_path,
                                     folder_path=self.folder_path,
                                     target_size=self.target_size,
                                     target_sr=self.target_sr)

        # 2) Compute the split sizes
        total_len = len(full_dataset)
        train_len = int(self.split_ratios[0] * total_len)
        dev_len   = int(self.split_ratios[1] * total_len)
        test_len  = total_len - train_len - dev_len

        # 3) Use random_split to get train/dev/test subsets
        self.train_dataset, self.dev_dataset, self.test_dataset = random_split(full_dataset, [train_len, dev_len, test_len])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.train_shuffle,
                          num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2)


# sireNNet Dataset ------------------------------------------------------------------------------------------------
class sireNNet_Dataset(Dataset):
    def __init__(self, folder_path, target_size=96000, target_sr=32000):
        super().__init__()
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

            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)
            
            # Stereo to mono and normalize
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Pad or truncate
            current_size = waveform.size(1)
            if current_size < self.target_size:
                padding = self.target_size - current_size
                waveform = F.pad(waveform, (0, padding), "constant", 0)
            elif current_size > self.target_size:
                waveform = waveform[:, :self.target_size]

        except Exception as e:
            self.skipped_files.append((idx, file_path))
            print(f"Skipping Error loading {file_path}: {e}")
            raise e

        return waveform, label


class sireNNet_DataModule(pl.LightningDataModule):
    def __init__(self,
                 folder_path: str,
                 batch_size: int = 32,
                 split_ratios: tuple = (0.8, 0.1, 0.1),
                 shuffle: bool = True,
                 target_size: int = 96000,
                 target_sr: int = 32000):
        """
        A DataModule that follows the structure of AudioSetEV_DataModule
        but uses sireNNet_Dataset for train/dev/test splits.

        :param folder_path: Root folder for subfolders: [ambulance, firetruck, police, traffic].
        :param batch_size: Batch size for each DataLoader.
        :param split_ratios: (train_ratio, dev_ratio, test_ratio).
        :param shuffle: Whether to shuffle in the train DataLoader.
        :param target_size: Target audio length (in samples).
        :param target_sr: Target sample rate.
        """
        super().__init__()
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.train_shuffle = shuffle
        self.target_size = target_size
        self.target_sr = target_sr

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Init sireNNet Dataset
        full_dataset = sireNNet_Dataset(folder_path=self.folder_path,
                                        target_size=self.target_size,
                                        target_sr=self.target_sr)

        # Compute split sizes
        total_size = len(full_dataset)
        train_size = int(self.split_ratios[0] * total_size)
        dev_size   = int(self.split_ratios[1] * total_size)
        test_size  = total_size - train_size - dev_size

        # Perform the splits
        self.train_dataset, self.dev_dataset, self.test_dataset = random_split(full_dataset, [train_size, dev_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.train_shuffle,
                          num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2)


# LSSiren Dataset ------------------------------------------------------------------------------------------------
class LSSiren_Dataset(Dataset):
    def __init__(self, folder_path: str, target_sr: int = 32000, min_length: int = 32000):
        super().__init__()
        self.folder_path = os.path.abspath(folder_path)
        self.target_sr = target_sr
        self.min_length = min_length
        self.file_paths, self.labels = self._load_files()
        self.skipped_files = []

    def _load_files(self):
        labels_map = {"Ambulance_data": 1,
                      "Road_Noises": 0}
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

            # Stereo -> Mono (sum channels)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to target_sr if necessary
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)

            # Ensure at least min_length samples (pad if shorter)
            current_size = waveform.size(1)
            if current_size < self.min_length:
                padding = self.min_length - current_size
                waveform = F.pad(waveform, (0, padding), "constant", 0)

        except Exception as e:
            self.skipped_files.append((idx, file_path))
            print(f"Skipping file {file_path} due to error: {e}")
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
    def __init__(self,
                 folder_path: str,
                 batch_size: int = 32,
                 split_ratios: tuple = (0.8, 0.1, 0.1),
                 shuffle: bool = True,
                 target_sr: int = 32000,
                 min_length: int = 32000):
        """
        DataModule mirroring AudioSetEV_DataModule structure,
        but for LSSiren with a custom collate function.

        :param folder_path: Root folder path (containing subfolders Ambulance_data, Road_Noises)
        :param batch_size: Batch size for DataLoaders
        :param split_ratios: (train_ratio, dev_ratio, test_ratio)
        :param shuffle: Whether to shuffle the training DataLoader
        :param target_sr: Target sampling rate
        :param min_length: Minimum length (in samples) to pad waveforms
        """
        super().__init__()
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.train_shuffle = shuffle
        self.target_sr = target_sr
        self.min_length = min_length

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # 1) Create the full dataset
        full_dataset = LSSiren_Dataset(folder_path=self.folder_path,
                                       target_sr=self.target_sr,
                                       min_length=self.min_length)

        # 2) Determine split sizes
        total_size = len(full_dataset)
        train_size = int(self.split_ratios[0] * total_size)
        dev_size   = int(self.split_ratios[1] * total_size)
        test_size  = total_size - train_size - dev_size

        # 3) Randomly split into train/dev/test subsets
        self.train_dataset, self.dev_dataset, self.test_dataset = random_split(full_dataset, [train_size, dev_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.train_shuffle,
                          num_workers=2,
                          collate_fn=lssiren_custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2,
                          collate_fn=lssiren_custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2,
                          collate_fn=lssiren_custom_collate_fn)


# UrbanSound8K Dataset ------------------------------------------------------------------------------------------------
class UrbanSound8K_Dataset(Dataset):
    def __init__(self,
                 folder_path: str,
                 metadata_path: str,
                 target_sr: int = 32000,
                 min_length: int = 32000):
        """
        :param folder_path: Root folder containing subfolders: fold1, fold2, ..., fold10
        :param metadata_path: CSV with columns (fold, slice_file_name, class, etc.)
        :param target_sr: Target sample rate for audio
        :param min_length: Minimum length (in samples) for zero-padding
        """
        super().__init__()
        self.folder_path = os.path.abspath(folder_path)
        self.metadata_path = os.path.abspath(metadata_path)
        self.target_sr = target_sr
        self.min_length = min_length

        # Build the list of file paths and labels from all folds
        self.file_paths, self.labels = self._load_all_files()

        # Track any files that fail to load
        self.skipped_files = []

    def _load_all_files(self):
        """
        Read the entire metadata CSV, gather file paths for fold1..fold10.
        If row["class"] == "siren", label=1, else 0.
        """
        metadata = pd.read_csv(self.metadata_path)

        file_paths = []
        labels = []

        for _, row in metadata.iterrows():
            fold_num = row["fold"]  # 1..10
            slice_name = row["slice_file_name"]
            category   = row["class"]

            audio_path = os.path.join(self.folder_path, f"fold{fold_num}", slice_name)
            label = 1 if category == "siren" else 0

            # Optionally check file exists
            if not os.path.exists(audio_path):
                # Could log or skip
                continue

            file_paths.append(audio_path)
            labels.append(label)

        return file_paths, labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(audio_path)

            # Stereo -> mono
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)

            # Zero-pad if shorter than min_length
            current_size = waveform.size(1)
            if current_size < self.min_length:
                padding = self.min_length - current_size
                waveform = F.pad(waveform, (0, padding), "constant", 0)

        except Exception as e:
            self.skipped_files.append((idx, audio_path))
            print(f"Skipping file {audio_path} due to error: {e}")
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
    def __init__(self,
                 folder_path: str,
                 metadata_path: str,
                 batch_size: int = 32,
                 split_ratios: tuple = (0.8, 0.1, 0.1),
                 shuffle: bool = True,
                 target_sr: int = 32000,
                 min_length: int = 32000):
        """
        DataModule aligned with AudioSetEV_DataModule, but merges all
        UrbanSound8K folds (1..10) into one dataset, then random-splits
        them into train/dev/test.

        :param folder_path: Path to the root UrbanSound8K folder,
                            containing subfolders: fold1, fold2, ..., fold10.
        :param metadata_path: Path to UrbanSound8K metadata CSV
        :param batch_size: Batch size for the DataLoader
        :param split_ratios: (train_ratio, dev_ratio, test_ratio)
        :param shuffle: Whether to shuffle the training DataLoader
        :param target_sr: Sample rate to resample (e.g., 32000)
        :param min_length: Minimum length in samples to pad waveforms
        """
        super().__init__()
        self.folder_path = folder_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.train_shuffle = shuffle
        self.target_sr = target_sr
        self.min_length = min_length

        self.train_dataset = None
        self.dev_dataset   = None
        self.test_dataset  = None

    def setup(self, stage=None):
        # 1) Build the full dataset (merging fold1..fold10)
        full_dataset = UrbanSound8K_Dataset(folder_path=self.folder_path,
                                            metadata_path=self.metadata_path,
                                            target_sr=self.target_sr,
                                            min_length=self.min_length)

        # 2) Compute sizes for each split
        total_len = len(full_dataset)
        train_len = int(self.split_ratios[0] * total_len)
        dev_len   = int(self.split_ratios[1] * total_len)
        test_len  = total_len - train_len - dev_len

        # 3) random_split
        self.train_dataset, self.dev_dataset, self.test_dataset = random_split(full_dataset, [train_len, dev_len, test_len])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.train_shuffle,
                          num_workers=2,
                          collate_fn=urbansound8k_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2,
                          collate_fn=urbansound8k_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2,
                          collate_fn=urbansound8k_collate_fn)


# FreeSound-50K Dataset -----------------------------------------------------------------------------------------------
class FSD50K_Dataset(Dataset):
    def __init__(self, csv_file, folder_path, target_sr=16000, label=1):
        super().__init__()
        self.csv_file = os.path.abspath(csv_file)
        self.folder_path = os.path.abspath(folder_path)
        self.target_sr = target_sr
        self.label = label

        # Read the CSV
        self.data = pd.read_csv(self.csv_file)

        # We keep a resampler object that we can adjust if the file sr differs
        self.resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=self.target_sr)
        self.skipped_files = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # CSV is assumed to have an ID column in row 0. We append ".wav".
        file_id = str(self.data.iloc[idx, 0])  # e.g. "12345"
        file_name = file_id + ".wav"
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
    def __init__(self,
                 pos_dev_csv: str,
                 neg_dev_csv: str,
                 dev_folder_path: str,
                 pos_eval_csv: str,
                 neg_eval_csv: str,
                 eval_folder_path: str,
                 batch_size: int = 32,
                 split_ratios: tuple = (0.9, 0.1),
                 target_sr: int = 16000,
                 shuffle: bool = True):
        """
        :param pos_dev_csv: CSV file for positive dev samples
        :param neg_dev_csv: CSV file for negative dev samples
        :param dev_folder_path: Folder with dev WAV files
        :param pos_eval_csv: CSV file for positive eval samples
        :param neg_eval_csv: CSV file for negative eval samples
        :param eval_folder_path: Folder with eval WAV files
        :param batch_size: Batch size for DataLoader
        :param split_ratios: (train_ratio, val_ratio), e.g. (0.9, 0.1)
        :param target_sr: Target sample rate for resampling
        :param num_workers: Number of worker processes for DataLoader
        :param shuffle: Whether to shuffle in the train DataLoader
        """
        super().__init__()
        self.pos_dev_csv = pos_dev_csv
        self.neg_dev_csv = neg_dev_csv
        self.dev_folder_path = dev_folder_path

        self.pos_eval_csv = pos_eval_csv
        self.neg_eval_csv = neg_eval_csv
        self.eval_folder_path = eval_folder_path

        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.target_sr = target_sr
        self.shuffle = shuffle

        # We'll create these in .setup()
        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

    def setup(self, stage=None):
        """
        Build dev (pos+neg), split into train/val, 
        then build eval (pos+neg) for test.
        """
        # 1) Dev set: Concat positive + negative
        pos_dev_dataset = FSD50K_Dataset(csv_file=self.pos_dev_csv,
                                         folder_path=self.dev_folder_path,
                                         target_sr=self.target_sr,
                                         label=1)
        
        neg_dev_dataset = FSD50K_Dataset(csv_file=self.neg_dev_csv,
                                         folder_path=self.dev_folder_path,
                                         target_sr=self.target_sr,
                                         label=0)
        
        full_dev_dataset = ConcatDataset([pos_dev_dataset, neg_dev_dataset])

        # 2) Split dev dataset into train/val
        total_len = len(full_dev_dataset)
        train_len = int(self.split_ratios[0] * total_len)
        val_len   = total_len - train_len

        self.train_dataset, self.val_dataset = random_split(full_dev_dataset, [train_len, val_len])

        # 3) Eval set: Concat positive + negative
        pos_eval_dataset = FSD50K_Dataset(csv_file=self.pos_eval_csv,
                                          folder_path=self.eval_folder_path,
                                          target_sr=self.target_sr,
                                          label=1)
        neg_eval_dataset = FSD50K_Dataset(csv_file=self.neg_eval_csv,
                                          folder_path=self.eval_folder_path,
                                          target_sr=self.target_sr,
                                          label=0)
        
        self.test_dataset = ConcatDataset([pos_eval_dataset, neg_eval_dataset])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=2,
                          collate_fn=fsd50k_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2,
                          collate_fn=fsd50k_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2,
                          collate_fn=fsd50k_collate_fn)
