import os
from dataloaders import ESC50_DataModule
import pytorch_lightning as pl

# Initialize DataModule
data_module = ESC50_DataModule(file_path="./datasets/ESC-50/esc50.csv",
                               folder_path="./datasets/ESC-50/cross_val_folds/",
                               target_size=160000,                                      # 5-second audio samples
                               target_sr=32000,
                               batch_size=32)

# Setup the DataModule
data_module.setup()

# Check and Print Selected Samples for Each Fold
print("\nSelected Samples for Each Fold:")
for fold_name, loader in data_module.test_loaders.items():
    print(f"\n{fold_name}:")
    
    # Get indices of the fold from the Subset
    fold_indices = loader.dataset.indices  # Subset dataset provides indices
    
    # Extract corresponding filenames and fold
    for idx in fold_indices:
        filename = data_module.dataset.filenames[idx]
        fold = filename.split('/')[-2]  # Extract fold name from the path
        print(f"Fold: {fold}, Filename: {os.path.basename(filename)}")

# Check Dataloaders
print("Testing Data Loaders:")
for fold_name, loader in data_module.test_loaders.items():
    print(f"{fold_name}: {len(loader.dataset)} samples")

# Updated DummyModel
class DummyModel(pl.LightningModule):
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        print(f"Dataloader {dataloader_idx}, Batch {batch_idx}: {inputs.shape}, {labels.shape}")

# Initialize Model and Trainer
model = DummyModel()
trainer = pl.Trainer(accelerator="cpu", devices=1)

# Run Test
trainer.test(model, dataloaders=data_module.test_dataloader())
