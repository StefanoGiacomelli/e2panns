import os
import torch
from epanns_inference import models
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
torch.set_float32_matmul_precision('high')

from globals import *
from dataloaders import AudioSetEV_Aug_DataModule, AudioSetEV_DataModule
from models import EPANNs_Binarized_Model
################################################################################


seed_everything(42)


# ---------------- Logger ----------------
def start_tensorboard(logdir):    
    try:
        print(f"Starting TensorBoard at log directory: {logdir}")
        os.system(f"tensorboard --logdir {logdir} --host 0.0.0.0 &")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

logger = TensorBoardLogger(save_dir="./experiments/tb_logs", name="EPANNs_Binarized")
print('Logs will be saved in: ', logger.log_dir)
start_tensorboard(logger.log_dir)


# -------------- Data Module -------------
if augmentation:
    data_module = AudioSetEV_Aug_DataModule(TP_file=Positives_csv,
                                            TP_folder=Positives,
                                            TN_file=Negatives_csv,
                                            TN_folder=Negatives,
                                            split_ratios=split_ratios,
                                            batch_size=batch_size,
                                            aug_prob=0.5)
else:
    data_module = AudioSetEV_DataModule(TP_file=Positives_csv,
                                        TP_folder=Positives,
                                        TN_file=Negatives_csv,
                                        TN_folder=Negatives,
                                        split_ratios=split_ratios,
                                        batch_size=batch_size)
data_module.setup()
print('--------------------------------------------------------------------------')
print("\n" * 2, end="")

# ---------------- Model -----------------
base_model = models.Cnn14_pruned(pre_trained=True)
model = EPANNs_Binarized_Model(base_model,
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
                               f_beta=f_beta)
print('--------------------------------------------------------------------------')
print("\n" * 2, end="")

# ---------------- Trainer & Callbacks ----------------
checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_DIR,
                                      filename="{epoch}_{epoch_val_accuracy:.4f}",
                                      monitor="epoch_val_accuracy",
                                      mode="max",
                                      save_top_k=1,
                                      save_weights_only=False,
                                      verbose=True)

early_stopping = EarlyStopping(monitor="epoch_val_accuracy",
                               mode="max",
                               patience=PATIENCE,
                               verbose=True)

trainer = Trainer(max_epochs=EPOCHS,
                  accelerator="gpu", 
                  devices=1, 
                  precision=32,
                  callbacks=[checkpoint_callback, early_stopping],
                  logger=logger,
                  log_every_n_steps=5,
                  default_root_dir=RESULTS_DIR,
                  #overfit_batches=1,
                  #fast_dev_run=True
                  )


################################################################################
if __name__ == '__main__':
    print("Training Model...")
    trainer.fit(model, datamodule=data_module)
    print('--------------------------------------------------------------------------')
    print("\n" * 2, end="")

    print("Testing Model...")
    trainer.test(model, datamodule=data_module)
    print('EOF')
