import os
import csv
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import torch.nn.functional as F


class AudioSetEV_Strong_Dataset(Dataset):
    def __init__(self, data_file, labels_file, audio_folder, bin_size=0.1, target_size=320000, out_type="pt", verbose=True):
        """
        Dataset for AudioSet_EV_Strong that loads audio files and their corresponding label tracks.
        """
        self.data_file = os.path.abspath(data_file)
        self.labels_file = os.path.abspath(labels_file)
        self.audio_folder = os.path.abspath(audio_folder)
        self.bin_size = bin_size
        self.target_size = target_size
        self.out_type = out_type
        self.verbose = verbose

        self.L = int(10 / self.bin_size)  # 10s segments

        self.failed_matches = []  # <-- Define before calling mapping functions
        self.failed_loads = []

        self.label_mapping = self._load_label_mapping()
        self.event_tracks = self._generate_event_tracks()
        self.segment_to_filepath = self._map_segment_to_file()

    def _load_label_mapping(self):
        label_mapping = {}
        with open(self.labels_file, 'r', newline='') as lf:
            reader = csv.DictReader(lf)
            for row in reader:
                label_mapping[row['mid']] = row['display_name']
        return label_mapping

    def _generate_event_tracks(self):
        aggregated_tracks = {}
        with open(self.data_file, 'r', newline='') as df:
            reader = csv.DictReader(df, delimiter='\t')
            for row in reader:
                full_seg_id = row['segment_id']
                try:
                    rel_start = float(row['start_time_seconds'])
                    rel_end = float(row['end_time_seconds'])
                except ValueError:
                    continue

                true_seg_id = full_seg_id.split('_')[0]

                event_start = max(0.0, min(rel_start, 10.0))
                event_end = max(0.0, min(rel_end, 10.0))

                start_idx = int(event_start / self.bin_size)
                end_idx = int(math.ceil(event_end / self.bin_size))
                if end_idx > self.L:
                    end_idx = self.L

                event_track = np.zeros(self.L, dtype=int)
                event_track[start_idx:end_idx] = 1

                mid = row['label'].strip()
                display_label = self.label_mapping.get(mid, mid)

                key = true_seg_id

                if key in aggregated_tracks:
                    aggregated_tracks[key] = np.maximum(aggregated_tracks[key], event_track)
                else:
                    aggregated_tracks[key] = event_track.copy()

        return aggregated_tracks

    def _map_segment_to_file(self):
        files = [f for f in os.listdir(self.audio_folder) if f.endswith('.wav')]
        mapping = {}

        for true_seg_id in self.event_tracks.keys():
            matched_files = [f for f in files if true_seg_id in f]
            if matched_files:
                mapping[true_seg_id] = os.path.join(self.audio_folder, matched_files[0])
            else:
                if self.verbose:
                    print(f"WARNING: No audio file found for segment_id '{true_seg_id}'.")
                self.failed_matches.append(true_seg_id)

        return mapping

    def __len__(self):
        return len(self.segment_to_filepath)

    def __getitem__(self, idx):
        seg_id = list(self.segment_to_filepath.keys())[idx]
        file_path = self.segment_to_filepath[seg_id]

        try:
            waveform_tensor, _ = torchaudio.load(file_path)

            if waveform_tensor.size(0) > 1:
                waveform_tensor = torch.mean(waveform_tensor, dim=0, keepdim=True)

            current_size = waveform_tensor.size(1)
            if current_size < self.target_size:
                padding = self.target_size - current_size
                waveform_tensor = F.pad(waveform_tensor, (0, padding), "constant", 0)
            elif current_size > self.target_size:
                waveform_tensor = waveform_tensor[:, :self.target_size]

            label_track = self.event_tracks.get(seg_id, np.zeros(self.L, dtype=int))
            if self.out_type == "pt":
                label_track = torch.from_numpy(label_track).float()

            return waveform_tensor, label_track
        
        except Exception as e:
            if self.verbose:
                print(f"ERROR: Failed to load or process file '{file_path}': {e}")
            self.failed_loads.append(seg_id)
            return torch.zeros((1, self.target_size)), torch.zeros((self.L,))


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    waveforms, label_tracks = zip(*batch)
    return torch.stack(waveforms), torch.stack(label_tracks)


def get_audioset_ev_strong_dataloader(data_file, labels_file, audio_folder, batch_size=32, num_workers=2, verbose=True):
    dataset = AudioSetEV_Strong_Dataset(data_file=data_file,
                                        labels_file=labels_file,
                                        audio_folder=audio_folder,
                                        bin_size=0.3,
                                        target_size=320000,
                                        out_type="pt",
                                        verbose=verbose)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)



# --- CONFIGURATION VARIABLES ---------------------------------------------------------
data_file = "./files/EV_Positives_eval.tsv"              # "./EV_Positives_train.tsv"    "./EV_Positives_eval.tsv"
labels_file = "./files/class_labels_indices.csv"
audio_folder = "./files/AudioSet_EV_Positives"
    
batch_size = 32
num_workers = 1
verbose = False
max_batches = None
# -------------------------------------------------------------------------------------

dataloader = get_audioset_ev_strong_dataloader(data_file=data_file,
                                                labels_file=labels_file,
                                                audio_folder=audio_folder,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                verbose=verbose)

print("\n✅ Dataloader created successfully!")
print(f"Total batches: {len(dataloader)}\n")

for batch_idx, (waveforms, label_tracks) in enumerate(dataloader):
    if waveforms is None:
        print(f"⚠️ Batch {batch_idx} skipped due to all failed loads.")
        continue

    print(f"🔹 Batch {batch_idx}:")
    print(f"   Waveforms shape: {waveforms.shape}")   
    print(f"   Label tracks shape: {label_tracks.shape}")

    if (max_batches is not None) and (batch_idx >= max_batches - 1):
        break

dataset = dataloader.dataset
print("\n✅ Dataset Loading Summary:")
print(f"  Total samples: {len(dataset)}")
print(f"  Failed matches (no audio found): {len(dataset.failed_matches)}")
print(f"  Failed loads (errors during load): {len(dataset.failed_loads)}")
#if dataset.failed_matches:
#    print("  ⚠️ Failed matches:", dataset.failed_matches)
#if dataset.failed_loads:
#    print("  ⚠️ Failed loads:", dataset.failed_loads)