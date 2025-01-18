from dataloaders import UrbanSound8K_DataModule

# Initialize the DataModule
data_module = UrbanSound8K_DataModule(folder_path="./datasets/UrbanSound8K/audio",
                                      metadata_path="./datasets/UrbanSound8K/metadata/UrbanSound8K.csv",
                                      batch_size=32,
                                      target_sr=32000,
                                      min_length=32000)

# Setup the DataModule
data_module.setup()

for loader in data_module.test_dataloaders():
    for batch_idx, (waveforms, labels) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Waveforms Shape: {waveforms.shape}")  
        print(f"  Labels: {labels}")
    print("\n")