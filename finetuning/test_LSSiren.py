from dataloaders import LSSiren_DataModule

# Initialize the DataModule
data_module = LSSiren_DataModule(folder_path="./datasets/Large-Scale_Audio_Dataset_for_Emergency_Vehicle_Sirens_and_Road_Noises/",
                                 batch_size=32,
                                 target_sr=32000,
                                 min_length=32000)
# Setup the DataModule
data_module.setup()

# Get the test DataLoader
test_loader = data_module.test_dataloader()

# Process the dataset
for batch_idx, (waveforms, labels) in enumerate(test_loader):
    print(f"Batch {batch_idx}: Wave Shape: {waveforms.shape}, Labels: {labels}")

# Strange behavior: printing an error unknown about the hardware nont compatible (but it's working good)
