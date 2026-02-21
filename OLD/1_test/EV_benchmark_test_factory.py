from typing import Union, List
from torch.utils.data import DataLoader
from test_dataloaders import (AudioSetEV_DataModule,
                              AudioSetEV_Aug_DataModule,
                              ESC50_DataModule,
                              sireNNet_DataModule,
                              LSSiren_DataModule,
                              UrbanSound8K_DataModule,
                              FSD50K_DataModule)


def get_audioset_ev_testloaders(TP_file: str,
                                TP_folder: str,
                                TN_file: str,
                                TN_folder: str,
                                batch_size: int = 32,
                                split_ratios=(0.8, 0.1, 0.1)) -> Union[DataLoader, List[DataLoader]]:
    dm = AudioSetEV_DataModule(TP_file=TP_file,
                               TP_folder=TP_folder,
                               TN_file=TN_file,
                               TN_folder=TN_folder,
                               batch_size=batch_size,
                               split_ratios=split_ratios,
                               shuffle=False)
    dm.setup("test")
    
    return dm.test_dataloader()


def get_audioset_ev_aug_testloaders(TP_file: str,
                                    TP_folder: str,
                                    TN_file: str,
                                    TN_folder: str,
                                    batch_size: int = 32,
                                    split_ratios=(0.8, 0.1, 0.1),
                                    aug_prob: float = 0.7) -> Union[DataLoader, List[DataLoader]]:
    dm = AudioSetEV_Aug_DataModule(TP_file=TP_file,
                                   TP_folder=TP_folder,
                                   TN_file=TN_file,
                                   TN_folder=TN_folder,
                                   batch_size=batch_size,
                                   split_ratios=split_ratios,
                                   shuffle=False,
                                   aug_prob=aug_prob)
    dm.setup("test")
    
    return dm.test_dataloader()


def get_esc50_testloaders(csv_path: str,
                          wavs_folder: str,
                          batch_size: int = 32,
                          target_size: int = 160000,
                          target_sr: int = 32000) -> List[DataLoader]:
    dm = ESC50_DataModule(file_path=csv_path,
                          folder_path=wavs_folder,
                          target_size=target_size,
                          target_sr=target_sr,
                          batch_size=batch_size)
    dm.setup("test")
    
    # ESC50_DataModule.test_dataloader() returns a list of fold loaders
    return dm.test_dataloader()


def get_sirennet_testloaders(folder_path: str,
                             batch_size: int = 32,
                             target_size: int = 96000,
                             target_sr: int = 32000) -> List[DataLoader]:
    dm = sireNNet_DataModule(folder_path=folder_path,
                             batch_size=batch_size,
                             target_size=target_size,
                             target_sr=target_sr)
    dm.setup("test")
    
    # sireNNet_DataModule.test_dataloader() returns a list of loaders w. improving fractions
    return dm.test_dataloader()


def get_lssiren_testloader(folder_path: str,
                           batch_size: int = 32,
                           target_sr: int = 32000,
                           min_length: int = 32000) -> DataLoader:
    dm = LSSiren_DataModule(folder_path=folder_path,
                            batch_size=batch_size,
                            target_sr=target_sr,
                            min_length=min_length)
    dm.setup("test")
    
    return dm.test_dataloader()


def get_urbansound8k_testloaders(folder_path: str,
                                 metadata_path: str,
                                 batch_size: int = 32,
                                 target_sr: int = 32000,
                                 min_length: int = 32000) -> List[DataLoader]:
    dm = UrbanSound8K_DataModule(folder_path=folder_path,
                                 metadata_path=metadata_path,
                                 batch_size=batch_size,
                                 target_sr=target_sr,
                                 min_length=min_length)
    dm.setup()
    
    # UrbanSound8k_DataModule..test_dataloaders() returns a list of fold-based loaders
    return dm.test_dataloaders()


def get_fsd50k_testloader(pos_csv: str,
                          neg_csv: str,
                          folder_path: str,
                          batch_size: int = 32,
                          target_sr: int = 16000) -> DataLoader:
    dm = FSD50K_DataModule(pos_file=pos_csv,
                           neg_file=neg_csv,
                           folder_path=folder_path,
                           batch_size=batch_size,
                           target_sr=target_sr)
    dm.setup("test")
    
    return dm.test_dataloader()
