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
