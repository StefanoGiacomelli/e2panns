# Checkpoints

This folder stores trained model checkpoints produced by the training pipelines (`main.py`, `main_unified_EV.py`) and reused by benchmarking / SED scripts.

## Purpose

Checkpoints are used to:

- resume training (`.ckpt` format, full Lightning state)
- run inference/benchmark (`.pt` or `.ckpt`)
- compare models across experiments (pretrained, fine-tuned, re-trained, unified)

## Current structure

```text
checkpoints/
├── binary_EV/                    # Binary task experiments
├── multiclass_EV/                # Multi-class EV experiments
├── re-training_EV/               # Re-training experiments
├── unified_training_EV_ced/      # Unified EV fine-tuning (CED)
├── unified_training_EV_clap/     # Unified EV fine-tuning (CLAP)
└── unified_training_EV_epanns/   # Unified EV fine-tuning (EPANNs)
```

## Checkpoint formats

- `epoch=XXX_val_f1=Y.YYYY.ckpt`: Lightning checkpoint (full training state, optimizer/scheduler/callbacks)
- `epoch=XXX_val_f1=Y.YYYY.pt`: model-only state (lightweight inference usage)

## External hosting (large artifacts)

To keep the GitHub repository lightweight, full checkpoint archives are distributed through external storage.

- Download link (placeholder): `<<ONEDRIVE_CHECKPOINTS_LINK>>`

After download:

1. extract the archive
2. copy the extracted content into this folder (`checkpoints/`)
3. preserve directory names exactly as provided in the archive

This reconstructs the expected local state for running training continuation, benchmarking, and SED experiments.
