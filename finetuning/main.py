import os
import torch
from epanns_inference import models
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
torch.set_float32_matmul_precision('high')

from globals import *
from dataloaders import AudioDataModule
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
data_module = AudioDataModule(TP_file=Positives_csv,
                              TP_folder=Positives,
                              TN_file=Negatives_csv,
                              TN_folder=Negatives,
                              batch_size=batch_size)
data_module.setup()
print('--------------------------------------------------------------------------')
print("\n" * 2, end="")

# ---------------- Model -----------------
base_model = models.Cnn14_pruned(pre_trained=True)
model = EPANNs_Binarized_Model(model=base_model,
                               threshold=threshold,
                               output_mode=output_mode,
                               overall_training=overall_training,
                               learning_rate=learning_rate,
                               weight_decay=weight_decay,
                               t_max=t_max,
                               eta_min=eta_min)
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
