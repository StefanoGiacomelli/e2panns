import warnings
warnings.filterwarnings("ignore", ".*Can't initialize NVML.*", category=UserWarning)
warnings.filterwarnings("ignore", ".*torch.load.*weights_only=False.*", category=FutureWarning)
warnings.filterwarnings("ignore", ".*Implicit dimension choice for softmax.*", category=UserWarning)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

from model import E2PANNs_Model_DatasetAware
from epanns_inference import models
from dataloaders import (AudioSetEV_DataModule, sireNNet_DataModule, LSSiren_DataModule, ESC50_DataModule,
                         UrbanSound8K_DataModule, FSD50K_DataModule)
from globals import *

def start_tensorboard(logdir):
    try:
        print(f"Starting TensorBoard at: {logdir}")
        os.system(f"tensorboard --logdir {logdir} --host 0.0.0.0 &")
    except Exception as e:
        print(f"Could not start TensorBoard: {e}")


if __name__ == "__main__":
    # -------------------------------
    # Basic Setup
    # -------------------------------
    seed_everything(42)
    torch.set_float32_matmul_precision('high')

    logger = TensorBoardLogger(save_dir="./experiments/tb_logs",
                               name="EPANNs_F1_Sequential_Training")
    print('Logs will be saved in:', logger.log_dir)
    start_tensorboard(logger.log_dir)
    print('------------------------------------------------\n')

    # -------------------------------
    # Create DataModules
    # -------------------------------
    dm_audioset = AudioSetEV_DataModule(
        TP_file="./datasets/AudioSet_EV/EV_Positives.csv",
        TP_folder="./datasets/AudioSet_EV/Positive_files/",
        TN_file="./datasets/AudioSet_EV/EV_Negatives.csv",
        TN_folder="./datasets/AudioSet_EV/Negative_files/",
        batch_size=32,
        split_ratios=(0.8, 0.1, 0.1),
        shuffle=True
    )
    dm_audioset.setup()

    dm_sirenet = sireNNet_DataModule(
        folder_path="./datasets/sireNNet/",
        batch_size=32,
        split_ratios=(0.8, 0.1, 0.1),
        shuffle=True,
        target_size=96000,
        target_sr=32000
    )
    dm_sirenet.setup()

    dm_lssiren = LSSiren_DataModule(
        folder_path="./datasets/Large-Scale_Audio_Dataset_for_Emergency_Vehicle_Sirens_and_Road_Noises/",
        batch_size=32,
        split_ratios=(0.8, 0.1, 0.1),
        shuffle=True,
        target_sr=32000,
        min_length=32000
    )
    dm_lssiren.setup()

    dm_esc50 = ESC50_DataModule(
        file_path="./datasets/ESC-50/esc50.csv",
        folder_path="./datasets/ESC-50/cross_val_folds/",
        batch_size=32,
        split_ratios=(0.8, 0.1, 0.1),
        shuffle=True,
        target_size=160000,
        target_sr=32000
    )
    dm_esc50.setup()

    dm_urbansound = UrbanSound8K_DataModule(
        folder_path="./datasets/UrbanSound8K/audio",
        metadata_path="./datasets/UrbanSound8K/metadata/UrbanSound8K.csv",
        batch_size=32,
        split_ratios=(0.8, 0.1, 0.1),
        shuffle=True,
        target_sr=32000,
        min_length=32000
    )
    dm_urbansound.setup()

    dm_fsd50k = FSD50K_DataModule(
        pos_dev_csv="./datasets/FSD50K/FSD-dev_positives.csv",
        neg_dev_csv="./datasets/FSD50K/FSD-dev_negatives.csv",
        dev_folder_path="./datasets/FSD50K/FSD50K.dev_audio/",
        pos_eval_csv="./datasets/FSD50K/FSD-eval_positives.csv",
        neg_eval_csv="./datasets/FSD50K/FSD-eval_negatives.csv",
        eval_folder_path="./datasets/FSD50K/FSD50K.eval_audio/",
        batch_size=32,
        split_ratios=(0.8, 0.2),
        target_sr=32000,
        shuffle=True
    )
    dm_fsd50k.setup()

    all_datamodules = [
        dm_audioset,
        dm_sirenet,
        dm_lssiren,
        dm_esc50,
        dm_urbansound,
        dm_fsd50k
    ]
    dataset_names = ["AudioSetEV", "sireNNet", "LSSiren", "ESC-50", "UrbanSound8K", "FSD50K"]
    num_datasets = len(all_datamodules)
    print(f"Initialized {num_datasets} DataModules.\n")

    # -------------------------------
    # Build E2PANNs_Model (no test usage here)
    # -------------------------------
    base_model = models.Cnn14_pruned(pre_trained=False)
    model = E2PANNs_Model_DatasetAware(
        model=base_model,
        threshold=threshold,
        output_mode=output_mode,
        overall_training=overall_training,
        eta_max=eta_max,
        eta_min=eta_min,
        decay_epochs=decay_epochs,
        restart_eta=restart_eta,
        restart_interval=restart_interval,
        warmup_epochs=warmup_epochs,
        warmup_eta=warmup_eta,
        weight_decay=weight_decay,
        f_beta=f_beta,
        num_datasets=num_datasets
    )

    model.load_trained_weights(checkpoint_path=CHECKPOINT_PATH, verbose=False, validate_updates=False)

    # Single EarlyStopping
    early_stopping = EarlyStopping(
        monitor="epoch_val_accuracy",
        mode="max",
        patience=PATIENCE,
        verbose=True
    )

    # -------------------------------
    #  SEQUENTIAL TRAINING
    # -------------------------------
    for round_idx in range(ROUNDS):
        print(f"\n========== ROUND {round_idx+1}/{ROUNDS} ==========\n")
        for i, dm_i in enumerate(all_datamodules):
            print(f"=== TRAIN on Dataset {i+1}/{num_datasets}: {dataset_names[i]} for {EPOCHS} epochs ===")

            # Create a ModelCheckpoint that keeps only the best checkpoint for dataset i
            checkpoint_callback = ModelCheckpoint(
                dirpath="./checkpoints/",
                filename=f"dataset{i+1}",
                monitor="epoch_val_accuracy",
                mode="max",
                save_top_k=1,
                verbose=True,
                auto_insert_metric_name=False
            )

            # Create a Trainer specifically for this dataset
            trainer = Trainer(
                max_epochs=EPOCHS,
                accelerator="auto",
                devices=1,
                precision=32,
                logger=logger,
                callbacks=[checkpoint_callback, early_stopping],
                log_every_n_steps=5
            )

            # Indicate which dataset is active
            model.set_current_dataset_idx(i)

            # Fit on this dataset's train & val splits
            trainer.fit(
                model,
                train_dataloaders=dm_i.train_dataloader(),
                val_dataloaders=dm_i.val_dataloader(),
            )

            # Validate on ALL datasets => update F1-based weighting
            print("Validating on all datasets to update F1-based weighting:")
            for j, dm_j in enumerate(all_datamodules):
                val_metrics_j = trainer.validate(model, datamodule=dm_j, verbose=False)
                val_f1_j = val_metrics_j[0]["epoch_val_f1"]
                model.update_dataset_weight_by_f1(j, val_f1_j, small_threshold=SMALL_THRESHOLD, large_threshold=LARGE_THRESHOLD)
                print(f"   -> Dataset {j+1} ({dataset_names[j]}): F1={val_f1_j:.4f}, weight={model.dataset_weights[j].item():.3f}")

    # -------------------------------
    #  PLOT DATASET WEIGHT EVOLUTION
    # -------------------------------
    print("\nPlotting dataset weights evolution...")
    model.plot_dataset_weights_history(
        save_dir=model.model_results_path,
        dataset_names=dataset_names
    )

    print("\nDone - EOF")