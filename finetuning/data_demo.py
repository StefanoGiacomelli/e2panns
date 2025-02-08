from torch.utils.data import DataLoader
from dataloaders import (AudioSetEV_DataModule,
                         ESC50_DataModule,
                         sireNNet_DataModule,
                         LSSiren_DataModule,
                         UrbanSound8K_DataModule,
                         FSD50K_DataModule)

def compute_stats_from_dataloader(dl: DataLoader, sample_rate: int):
    """
    Computes statistics from a given dataloader.
    """
    total_samples = 0
    total_duration_sec = 0.0
    batch_durations = []
    total_positives = 0
    total_negatives = 0
    total_batches = 0

    for batch in dl:
        if batch is None:
            continue
        waveforms, labels = batch
        # waveforms shape: (batch_size, channels, samples)
        batch_size = waveforms.size(0)
        total_samples += batch_size
        total_batches += 1
        # Compute duration per sample (in seconds) from the number of samples (assuming constant sample rate
        # and all waveforms in the batch have the same length).
        duration_sec = waveforms.size(2) / sample_rate
        total_duration_sec += batch_size * duration_sec
        batch_durations.append(duration_sec)
        total_positives += (labels == 1).sum().item()
        total_negatives += (labels == 0).sum().item()

    avg_duration_per_batch = (total_duration_sec / total_batches) if total_batches > 0 else 0
    total_duration_min = total_duration_sec / 60.0
    return total_samples, total_duration_min, avg_duration_per_batch, total_positives, total_negatives

def process_dataloader(dl, sample_rate: int, dataset_name: str):
    """
    Process a single dataloader or a list of dataloaders.
    
    :param dl: Either a DataLoader or a list of DataLoaders.
    :param sample_rate: The sample rate in Hz.
    :param dataset_name: Name of the dataset (for printing purposes).
    """
    # If dl is a list of dataloaders, process them all and sum the stats.
    if isinstance(dl, list):
        total_samples = 0
        total_duration = 0.0
        total_positives = 0
        total_negatives = 0
        
        for idx, sub_dl in enumerate(dl):
            s, dur, avg_dur, pos, neg = compute_stats_from_dataloader(sub_dl, sample_rate)
            total_samples = s
            total_duration = dur  # in min
            total_positives = pos
            total_negatives = neg
            
            print(f"Dataset: {dataset_name} loader {idx}")
            print(f"  Total samples: {total_samples}")
            print(f"  Total duration (min): {total_duration:.2f}")
            print(f"  Total positives: {total_positives}")
            print(f"  Total negatives: {total_negatives}")
            print('....................................................')
    else:
        s, dur, avg_dur, pos, neg = compute_stats_from_dataloader(dl, sample_rate)
        print(f"Dataset: {dataset_name}")
        print(f"  Total samples: {s}")
        print(f"  Total duration (min): {dur:.2f}")
        print(f"  Average duration per batch (sec): {avg_dur:.2f}")
        print(f"  Total positives: {pos}")
        print(f"  Total negatives: {neg}")


if __name__ == "__main__":
    # AudioSetEV DataModule (native SR = 32000, native/target samples duration = 10sec)
    audioset_ev_dm = AudioSetEV_DataModule(TP_file="./datasets/AudioSet_EV/EV_Positives.csv",
                                           TP_folder="./datasets/AudioSet_EV/Positive_files/",
                                           TN_file="./datasets/AudioSet_EV/EV_Negatives.csv",
                                           TN_folder="./datasets/AudioSet_EV/Negative_files/",
                                           batch_size=32,
                                           split_ratios=(0.8, 0.1, 0.1),
                                           shuffle=True)
    audioset_ev_dm.setup()
    test_dl = audioset_ev_dm.test_dataloader()
    process_dataloader(test_dl, sample_rate=32000, dataset_name="AudioSetEV")
    print("--------------------------------------------------------")

    # ESC-50 DataModule (native SR = 16000, native samples duration = 5sec)
    esc50_dm = ESC50_DataModule(file_path="./datasets/ESC-50/esc50.csv",
                                folder_path="./datasets/ESC-50/cross_val_folds/",
                                target_size=160000,
                                target_sr=32000,
                                batch_size=32)
    esc50_dm.setup()
    test_dls = esc50_dm.test_dataloader()  # list of DataLoaders (one per fold)
    process_dataloader(test_dls, sample_rate=32000, dataset_name="ESC-50")
    print("--------------------------------------------------------")

    # sireNNet DataModule (native SR = XXXX, native samples duration = 3sec?????)
    sirennet_dm = sireNNet_DataModule(folder_path="./datasets/sireNNet/",
                                      target_sr=32000,
                                      target_size=96000,
                                      batch_size=32)
    sirennet_dm.setup()
    test_dls = sirennet_dm.test_dataloader()  # list of DataLoaders (different subset partitions)
    process_dataloader(test_dls, sample_rate=32000, dataset_name="sireNNet")
    print("--------------------------------------------------------")

    # LSSiren DataModule (native SR = 48000?, native samples duration = [3, 15]sec.)
    lssiren_dm = LSSiren_DataModule(folder_path="./datasets/Large-Scale_Audio_Dataset_for_Emergency_Vehicle_Sirens_and_Road_Noises/",
                                    target_sr=32000,
                                    min_length=32000,
                                    batch_size=32)
    lssiren_dm.setup()
    test_dl = lssiren_dm.test_dataloader()
    process_dataloader(test_dl, sample_rate=32000, dataset_name="LSSiren")
    print("--------------------------------------------------------")

    # UrbanSound8K DataModule (native SR = variable, native samples duration = variable)
    urbansound_dm = UrbanSound8K_DataModule(folder_path="./datasets/UrbanSound8K/audio",
                                            metadata_path="./datasets/UrbanSound8K/metadata/UrbanSound8K.csv",
                                            target_sr=32000,
                                            min_length=32000,
                                            batch_size=32)
    urbansound_dm.setup()
    test_dls = urbansound_dm.test_dataloaders()  # list of DataLoaders (one per fold)
    process_dataloader(test_dls, sample_rate=32000, dataset_name="UrbanSound8K")
    print("--------------------------------------------------------")

    # FSD50K DataModule (native SR = 44100, native samples duration = variable)
    fsd50k_dm = FSD50K_DataModule(pos_file="./datasets/FSD50K/FSD-eval_positives.csv",
                                  neg_file="./datasets/FSD50K/FSD-eval_negatives.csv",
                                  folder_path="./datasets/FSD50K/FSD50K.eval_audio/",
                                  target_sr=32000,
                                  batch_size=32)
    fsd50k_dm.setup()
    test_dl = fsd50k_dm.test_dataloader()
    process_dataloader(test_dl, sample_rate=32000, dataset_name="FSD50K")
