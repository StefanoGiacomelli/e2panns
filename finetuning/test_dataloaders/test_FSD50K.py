from dataloaders import FSD50K_DataModule

def test_dataloader():
    # Define test parameters
    test_pos_csv = "./datasets/FSD50K/FSD-eval_positives.csv"
    test_neg_csv = "./datasets/FSD50K/FSD-eval_negatives.csv"
    test_folder = "./datasets/FSD50K/FSD50K.eval_audio/"
    batch_size = 32
    target_sr = 32000
    
    # Initialize data module
    data_module = FSD50K_DataModule(pos_csv=test_pos_csv, 
                                    neg_csv=test_neg_csv, 
                                    folder_path=test_folder, 
                                    
                                    batch_size=batch_size, target_sr=target_sr)
    data_module.setup()
    
    # Get dataloader
    test_loader = data_module.test_dataloader()
    
    # Fetch a batch and print details
    for batch_idx, (waveforms, labels) in enumerate(test_loader):
        print(f"Batch {batch_idx}:")
        print(f"Waveforms Shape: {waveforms.shape}")
        print(f"Labels: {labels}")
        if batch_idx == 2:  # Limit to 3 printed batches
            break

if __name__ == "__main__":
    test_dataloader()
