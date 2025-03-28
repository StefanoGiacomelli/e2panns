from pytorch_lightning import seed_everything
from dataloaders import AudioSetEV_DataModule, sireNNet_DataModule, LSSiren_DataModule, ESC50_DataModule, UrbanSound8K_DataModule, FSD50K_DataModule


seed_everything(42)


dm = AudioSetEV_DataModule(TP_file="./datasets/AudioSet_EV/EV_Positives.csv",
                           TP_folder="./datasets/AudioSet_EV/Positive_files/",
                           TN_file="./datasets/AudioSet_EV/EV_Negatives.csv",
                           TN_folder="./datasets/AudioSet_EV/Negative_files/",
                           batch_size=32,
                           split_ratios=(0.8, 0.1, 0.1),
                           shuffle=True)
dm.setup()

train_loader = dm.train_dataloader()
val_loader   = dm.val_dataloader()
test_loader  = dm.test_dataloader()
print("AudioSetEV Dataloaders")
print("Train set size:", len(train_loader.dataset))
print("Val set size:  ", len(val_loader.dataset))
print("Test set size: ", len(test_loader.dataset))
print("\n"*2)


dm = sireNNet_DataModule(folder_path="./datasets/sireNNet/",
                         batch_size=32,
                         split_ratios=(0.8, 0.1, 0.1),
                         shuffle=True,
                         target_size=96000,
                         target_sr=32000)
dm.setup()

train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
test_loader = dm.test_dataloader()
print("SireNNet Dataloaders")
print("Train samples:", len(train_loader.dataset))
print("Val samples:  ", len(val_loader.dataset))
print("Test samples: ", len(test_loader.dataset))
print("\n"*2)



dm = LSSiren_DataModule(folder_path="./datasets/Large-Scale_Audio_Dataset_for_Emergency_Vehicle_Sirens_and_Road_Noises/",
                        batch_size=32,
                        split_ratios=(0.8, 0.1, 0.1),
                        shuffle=True,
                        target_sr=32000,
                        min_length=32000)
dm.setup()

train_loader = dm.train_dataloader()
val_loader   = dm.val_dataloader()
test_loader  = dm.test_dataloader()
print("LSSiren Dataloaders")
print("Train samples:", len(train_loader.dataset))
print("Val samples:  ", len(val_loader.dataset))
print("Test samples: ", len(test_loader.dataset))
print("\n"*2)


dm = ESC50_DataModule(file_path="./datasets/ESC-50/esc50.csv",
                      folder_path="./datasets/ESC-50/cross_val_folds/",
                      batch_size=32,
                      split_ratios=(0.8, 0.1, 0.1),
                      shuffle=True,
                      target_size=160000,
                      target_sr=32000)
dm.setup()

train_loader = dm.train_dataloader()
val_loader   = dm.val_dataloader()
test_loader  = dm.test_dataloader()
print("ESC50 Dataloaders")
print("Train set size:", len(train_loader.dataset))
print("Val set size:  ", len(val_loader.dataset))
print("Test set size: ", len(test_loader.dataset))
print("\n"*2)


dm = UrbanSound8K_DataModule(folder_path="./datasets/UrbanSound8K/audio",
                             metadata_path="./datasets/UrbanSound8K/metadata/UrbanSound8K.csv",
                             batch_size=32,
                             split_ratios=(0.8, 0.1, 0.1),
                             shuffle=True,
                             target_sr=32000,
                             min_length=32000)
dm.setup()

train_loader = dm.train_dataloader()
val_loader   = dm.val_dataloader()
test_loader  = dm.test_dataloader()
print("UrbanSound8K Dataloaders")
print("Train set size:", len(train_loader.dataset))
print("Val set size:  ", len(val_loader.dataset))
print("Test set size: ", len(test_loader.dataset))
print("\n"*2)


dm = FSD50K_DataModule(pos_dev_csv="./datasets/FSD50K/FSD-dev_positives.csv",
                       neg_dev_csv="./datasets/FSD50K/FSD-dev_negatives.csv",
                       dev_folder_path="./datasets/FSD50K/FSD50K.dev_audio/",
                       pos_eval_csv="./datasets/FSD50K/FSD-eval_positives.csv",
                       neg_eval_csv="./datasets/FSD50K/FSD-eval_negatives.csv",
                       eval_folder_path="./datasets/FSD50K/FSD50K.eval_audio/",
                       batch_size=32,
                       split_ratios=(0.8, 0.2),
                       target_sr=32000,
                       shuffle=True)
dm.setup()

train_loader = dm.train_dataloader()
val_loader   = dm.val_dataloader()
test_loader  = dm.test_dataloader()
print("FSD50K Dataloaders")
print("Train set size:", len(train_loader.dataset))
print("Val set size:  ", len(val_loader.dataset))
print("Test set size: ", len(test_loader.dataset))
