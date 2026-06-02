# birdclef-2026

The goal of this repo is to develop machine learning frameworks capable of identifying understudied species within continuous audio data from Brazil's Pantanal wetlands.

## Competition Link

Download the dataset from the Kaggle competition page:

https://www.kaggle.com/competitions/birdclef-2026/data

## Dataset Folder

Put the downloaded dataset in this project root using this folder name:

```text
birdclef-2026/
```

The first notebook also checks for the typo `birdcle-2026/`, but the correct competition folder name is `birdclef-2026/`.

Expected dataset contents:

```text
birdclef-2026/
  taxonomy.csv
  train.csv
  train_soundscapes_labels.csv
  sample_submission.csv
  recording_location.txt
  train_audio/
  train_soundscapes/
  test_soundscapes/
```

## Local Python Environment

Recommended: create a new virtual environment inside this project instead of reusing an old environment.

From PowerShell, run these commands inside the project folder:

```powershell
cd D:\BirdCLEF+
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch first. For your NVIDIA GPU on Windows, use the official PyTorch selector if this command ever changes:

https://pytorch.org/get-started/locally/

For Windows + pip + CUDA 12.8, the command is usually:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install the rest of the first notebook dependencies:

```powershell
pip install ipykernel pandas numpy matplotlib scipy librosa soundfile
python -m ipykernel install --user --name birdclef-2026 --display-name "Python (birdclef-2026)"
```

Then open the notebook in VS Code and choose this kernel:

```text
Python (birdclef-2026)
```

To confirm the notebook is using the project environment, run:

```python
import sys
print(sys.executable)
```

It should point to:

```text
D:\BirdCLEF+\.venv\Scripts\python.exe
```

### GPU PyTorch Note

The commands above install a CUDA-enabled PyTorch build plus the data-preparation and audio-preprocessing dependencies.

After installing PyTorch, confirm GPU access with:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

If `torch.cuda.is_available()` prints `False`, the notebook is still using CPU.

## Notebook 01: Dataset Preparation and Audio Preprocessing

Open and run:

```text
01_dataset_preparation_audio_preprocessing.ipynb
```

This notebook handles the first two pipeline steps:

### 1. Dataset Preparation

The notebook:

- Finds the dataset folder.
- Checks that all required CSV files and audio folders exist.
- Loads `taxonomy.csv`, `train.csv`, `train_soundscapes_labels.csv`, and `sample_submission.csv`.
- Builds the final 234-class label map in the same order as `sample_submission.csv`.
- Creates a `train_audio` manifest with file paths, labels, and class indices.
- Creates a labeled soundscape-window manifest with one row per 5-second labeled window.

Outputs are saved to:

```text
birdclef_work/
  label_map.csv
  train_audio_manifest.csv
  train_soundscape_windows.csv
```

### 2. Audio Preprocessing

The notebook:

- Loads `.ogg` audio.
- Resamples audio to 32 kHz.
- Cuts or pads audio into 5-second windows.
- Converts each 5-second window into a 128-bin log-mel spectrogram.
- Creates small example `.npz` batches to confirm shapes before training.

Optional example outputs:

```text
birdclef_work/
  example_train_audio_logmel_batch.npz
  example_soundscape_logmel_batch.npz
```

## GPU / CPU Check

Near the top of the notebook there is a GPU detection cell. The important line is:

```python
torch.cuda.is_available()
```

If it prints `True`, the notebook kernel can use the GPU.

If it prints `False`, the notebook is using CPU, even if `nvidia-smi` can see an NVIDIA GPU. In that case, install a CUDA-enabled PyTorch build or select a GPU runtime/kernel.

## Audio Dependencies

To preprocess `.ogg` audio, the notebook needs one of these audio backends:

- `librosa` with `soundfile`
- `torchaudio`

The metadata preparation cells will still run without them, but the log-mel spectrogram cells need an audio backend.

## Current Pipeline Position

This repo is currently prepared for:

```text
Dataset files
  -> label map and manifests
  -> 5-second audio windows
  -> log-mel spectrograms
```

The next notebook should handle model training:

```text
log-mel spectrograms
  -> CNN / audio model
  -> validation
  -> saved model weights
```

## Notebook 02: Soundscape Adaptation and Pseudo-Labeling

Open and run:

```text
02_soundscape_adaptation_pseudolabel_training.ipynb
```

This notebook implements the next two training stages:

### 4. Stage 2 Training: Soundscape Adaptation

The notebook:

- Loads `birdclef_work/label_map.csv`.
- Loads `birdclef_work/train_soundscape_windows.csv`.
- Splits labeled soundscape windows by filename so the same soundscape file is not in both train and validation.
- Builds a small CNN classifier for 128-bin log-mel spectrograms.
- Looks for this optional Stage 1 checkpoint:

```text
birdclef_work/checkpoints/stage1_train_audio_best.pt
```

If that checkpoint exists, the notebook loads it and fine-tunes the same model on real 5-second soundscape windows. If it does not exist, the notebook trains the same architecture from initialized weights on the labeled soundscape windows.

Stage 2 outputs:

```text
birdclef_work/
  stage2_soundscape_split.csv
  stage2_soundscape_history.csv
  checkpoints/
    stage2_soundscape_best.pt
    stage2_soundscape_last.pt
```

### 5. Stage 3 Optional Improvement: Pseudo-Labeling

Stage 3 is implemented in the same notebook but is disabled by default:

```python
RUN_STAGE3 = False
```

After Stage 2 finishes successfully, set it to:

```python
RUN_STAGE3 = True
```

The notebook will:

- Scan unlabeled files in `birdclef-2026/train_soundscapes/`.
- Split them into 5-second windows.
- Use the Stage 2 model to predict probabilities.
- Keep high-confidence predictions as pseudo-labels.
- Combine real labeled windows with pseudo-labeled windows.
- Fine-tune again.

Start with:

```python
PSEUDO_MAX_FILES = 200
```

After confirming it works, increase the number or set it to `None` to process all unlabeled soundscapes.

Stage 3 outputs:

```text
birdclef_work/
  stage3_unlabeled_candidates.csv
  stage3_pseudo_labels.csv
  stage3_combined_train_windows.csv
  stage3_pseudolabel_history.csv
  checkpoints/
    stage3_pseudolabel_best.pt
    stage3_pseudolabel_last.pt
```

Before the long training cell, Notebook 02 includes a smoke test that confirms:

- the selected kernel can see GPU/CPU correctly,
- audio files load,
- log-mel spectrograms are generated,
- model output shape is `(batch_size, 234)`,
- the training loss can be computed.

## Notebook 03: Validation, Inference, and Submission

Open and run:

```text
03_validation_inference_submission.ipynb
```

This notebook implements the final two pipeline steps:

### 6. Validation

The notebook:

- Loads the best available checkpoint in this order:

```text
birdclef_work/checkpoints/stage3_pseudolabel_best.pt
birdclef_work/checkpoints/stage2_soundscape_best.pt
birdclef_work/checkpoints/stage2_soundscape_last.pt
```

- Uses the Stage 2 split from `birdclef_work/stage2_soundscape_split.csv` by default.
- Can also create a fresh grouped split by file or site by changing:

```python
VALIDATION_SPLIT_MODE = "file"
```

or:

```python
VALIDATION_SPLIT_MODE = "site"
```

- Measures macro ROC-AUC while skipping classes that have no positive or no negative labels in validation.

Validation outputs:

```text
birdclef_work/
  validation_predictions.csv
  validation_metrics.json
  validation_per_class_auc.csv
```

### 7. Inference and `submission.csv`

The notebook:

- Reads hidden `.ogg` files from `birdclef-2026/test_soundscapes/`.
- Splits each soundscape into 5-second windows.
- Predicts 234 species probabilities for each window.
- Writes:

```text
submission.csv
```

The local public dataset has no real test `.ogg` files, only `test_soundscapes/readme.txt`. In that local case, the notebook writes a dry-run copy of `sample_submission.csv`. During Kaggle's hidden rerun, `test_soundscapes/` will be populated and the notebook will write real model predictions.

## Notebook 04: Stronger Competition Pipeline

Open:

```text
04_competition_cv_ensemble_pipeline.ipynb
```

This notebook is the stronger pipeline for a serious leaderboard attempt. It implements:

- Stage 1 full `train_audio` training.
- Stage 2 soundscape fine-tuning.
- Stage 3 pseudo-labeling from unlabeled `train_soundscapes`.
- `torchvision` EfficientNet-B0 backbone.
- grouped CV by soundscape file or site.
- 5 folds.
- ensemble inference by averaging logits.
- CPU-aware Kaggle inference.
- `submission.csv` creation.

By default, the notebook runs only the smoke test and dry-run inference. This is intentional so it does not accidentally start a multi-hour training run.

To start the full training run, change:

```python
RUN_FULL_TRAINING = False
```

to:

```python
RUN_FULL_TRAINING = True
```

The default full run trains:

```python
TRAIN_FOLDS = [0, 1, 2, 3, 4]
STAGE1_EPOCHS = 8
STAGE2_EPOCHS = 20
STAGE3_EPOCHS = 6
```

Expected strong-pipeline outputs:

```text
birdclef_work/
  strong_pipeline/
    train_audio_folds.csv
    soundscape_folds.csv
    pseudo_candidates_fold*.csv
    pseudo_labels_fold*.csv
    stage3_combined_fold*.csv
    ensemble_validation_metrics.json
    checkpoints/
      fold0_stage1_train_audio_best.pt
      fold0_stage2_soundscape_best.pt
      fold0_stage3_pseudolabel_best.pt
      ...
```

### Pretrained Weights Note

Notebook 04 uses `torchvision.models.efficientnet_b0`. It tries to load ImageNet pretrained weights. If the weights are not cached and the environment has no internet, it falls back to random initialization.

For the best chance at a good score, run the full training in an environment where the EfficientNet-B0 weights can be downloaded once, or provide cached pretrained weights before training. Kaggle submission/inference does not need internet after the trained checkpoints are attached.

### Kaggle Submission With Notebook 04

After training, upload the strong-pipeline checkpoints as a Kaggle dataset/input. In a Kaggle submission notebook, attach:

- the official BirdCLEF 2026 competition data,
- the trained checkpoint dataset containing `fold*_stage*_*.pt`.

Notebook 04 can search `/kaggle/input` for fold checkpoints and ensemble them. On CPU-only Kaggle inference, it limits the number of ensemble models with:

```python
MAX_ENSEMBLE_MODELS_FOR_CPU = 3
```

Increase this only if the notebook still runs under the competition time limit.
