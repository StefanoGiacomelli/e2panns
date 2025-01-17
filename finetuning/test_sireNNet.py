from dataloaders import sireNNet_DataModule

# Initialize the DataModule
data_module = sireNNet_DataModule(folder_path="./datasets/sireNNet/",
                                  batch_size=32,
                                  target_size=96000,            # 3-second audio samples
                                  target_sr=32000)

# Cross-Validation simulation (set seed before entering in the loop --> sequence repeatable but randomization intra-runs)
num_runs = 10
for run in range(num_runs):
    print(f"\nRun {run+1}:")
    data_module.setup()  # Reinitialize the dataset for each run
    for fraction, loader in zip(data_module.sizes, data_module.test_dataloader()):
        # Count labels in the current subset
        num_ones, num_zeros = 0, 0
        for _, labels in loader:
            num_ones += (labels == 1).sum().item()
            num_zeros += (labels == 0).sum().item()
        # Print label counts for the current fraction
        print(f"  Fraction: {fraction*100:.2f}% - 1s: {num_ones}, 0s: {num_zeros}")
        print(f"  TOT Samples: {len(loader.dataset)}")
    print('\n'*2)
