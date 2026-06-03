# BirdCLEF 2026 Audio Classification Pipeline

This repository is a personal machine learning project for bird and wildlife audio classification using the BirdCLEF 2026 dataset structure. It demonstrates a complete bioacoustics workflow: audio preprocessing, log-mel spectrogram generation, soundscape-domain adaptation, pseudo-labeling, grouped cross-validation, and ensemble evaluation.

This project is maintained as a portfolio and research-learning project, not as an active final competition submission.

## Problem Statement

The task is multi-label species detection from continuous audio soundscapes. The main challenge is domain shift: many training examples are short, cleaner species clips, while evaluation soundscapes are long, noisy, overlapping, and segmented into 5-second windows.

## Original Contributions

- Implemented soundscape adaptation to reduce the mismatch between short training clips and long continuous soundscapes.
- Added pseudo-labeling experiments to use high-confidence predictions from unlabeled soundscape windows.
- Used grouped cross-validation by soundscape file/site to reduce leakage between train and validation splits.
- Built an EfficientNet-B0 based ensemble pipeline for stronger feature extraction and fold averaging.
- Added reproducible evaluation artifacts, training histories, and README-ready result summaries.

## Dataset

Dataset source: https://www.kaggle.com/competitions/birdclef-2026/data

Expected local dataset layout:

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

The dataset is not included in this repository because of size and licensing constraints.

## Methodology

Audio is converted into fixed 5-second windows and represented as 128-bin log-mel spectrograms. The project starts with dataset preparation, then trains soundscape-aware models, optionally expands training data with pseudo-labels, and finally evaluates fold-based ensembles using macro ROC-AUC.

## Model Architecture

- Input: 5-second audio windows resampled to 32 kHz.
- Feature representation: 128-bin log-mel spectrograms.
- Baseline pipeline: compact CNN classifier for soundscape adaptation.
- Strong pipeline: `torchvision` EfficientNet-B0 backbone with a multi-label classification head.
- Output: probability scores for 234 target classes.

## Training Strategy

1. Stage 1: train on curated `train_audio` clips.
2. Stage 2: fine-tune on labeled soundscape windows.
3. Stage 3: generate pseudo-labels from unlabeled soundscapes and fine-tune again.
4. Cross-validation: grouped folds by soundscape file/site.
5. Ensemble: average logits from selected fold checkpoints.

## Evaluation Metrics

- Main metric: macro ROC-AUC.
- Classes with only positive or only negative labels in a validation split are skipped for ROC-AUC calculation.
- Local validation metrics are used for project comparison.
- Leaderboard score is not available because this project was not submitted as a final competition entry.

## Results

### Final Validation Summary

| Evaluation | Value |
| --- | ---: |
| Local ensemble macro ROC-AUC | 0.9916 |
| Validation rows used | 888 |
| Classes used for AUC | 57 |
| Ensemble checkpoints | 3 |
| Leaderboard score | Not available |

Note: the ensemble score is an internal validation diagnostic from saved fold checkpoints, not a public leaderboard result.

### Stage Comparison

| Stage | Validation domain | Folds completed | Best fold AUC range | Notes |
| --- | --- | ---: | ---: | --- |
| Stage 1 | Train-audio split | 3 | 0.8943-0.9523 | Useful for feature learning, but not directly comparable to soundscape validation. |
| Stage 2 | Labeled soundscape windows | 3 | 0.8564-0.8852 | Most relevant supervised soundscape adaptation stage. |
| Stage 3 | Soundscape + pseudo-label training | 3 | 0.8361-0.8833 | Explores unlabeled soundscape data; results depend strongly on pseudo-label quality. |

## Figures and Screenshots

I do not include generated images automatically in this README. Add final selected images to:

```text
assets/readme_images/
```

Recommended image files to add:

| Image | Suggested filename | Purpose |
| --- | --- | --- |
| Log-mel spectrogram sample | `logmel_sample.png` | Shows the audio representation used by the model. |
| Training loss curves | `training_loss_curves.png` | Shows convergence and overfitting/underfitting behavior. |
| Validation AUC curves | `validation_auc_curves.png` | Shows validation performance over epochs. |
| Pseudo-label volume by fold | `pseudo_label_volume.png` | Shows how much unlabeled data was kept per fold. |
| Validation summary | `validation_summary.png` | Highlights the final local validation result. |
| Pipeline diagram | `pipeline_diagram.png` | Explains the full workflow visually. |

After you provide the final images, they can be embedded here.

## Notebook Overview

### Notebook 01: Dataset Preparation and Audio Preprocessing

`01_dataset_preparation_audio_preprocessing.ipynb`

Creates dataset manifests, builds the 234-class label map, prepares 5-second audio windows, and creates log-mel spectrogram examples.

### Notebook 02: Soundscape Adaptation and Pseudo-Labeling

`02_soundscape_adaptation_pseudolabel_training.ipynb`

Fine-tunes a classifier on labeled soundscape windows and includes optional pseudo-labeling for unlabeled soundscapes.

### Notebook 03: Validation and Inference

`03_validation_inference_submission.ipynb`

Loads trained checkpoints, evaluates macro ROC-AUC, saves validation predictions and per-class metrics, and supports local inference.

### Notebook 04: Cross-Validation Ensemble Pipeline

`04_competition_cv_ensemble_pipeline.ipynb`

Main experiment notebook with full training stages, EfficientNet-B0, grouped cross-validation, ensemble inference, and saved training/evaluation summaries.

## How to Run

Run the notebooks in this order:

```text
01_dataset_preparation_audio_preprocessing.ipynb
02_soundscape_adaptation_pseudolabel_training.ipynb
03_validation_inference_submission.ipynb
04_competition_cv_ensemble_pipeline.ipynb
```

For Notebook 04, keep this setting for a smoke test or portfolio review:

```python
RUN_FULL_TRAINING = False
```

Set it to `True` only when you are ready for a longer training run.

## Reproducibility

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the correct PyTorch build for your CUDA version from the official PyTorch selector:

https://pytorch.org/get-started/locally/

## Folder Structure

```text
.
+-- 01_dataset_preparation_audio_preprocessing.ipynb
+-- 02_soundscape_adaptation_pseudolabel_training.ipynb
+-- 03_validation_inference_submission.ipynb
+-- 04_competition_cv_ensemble_pipeline.ipynb
+-- assets/
|   +-- readme_images/
+-- birdclef-2026/              # local dataset, ignored by git
+-- birdclef_work/              # generated artifacts, ignored by git
+-- requirements.txt
+-- README.md
```

## Expected Outputs

Generated outputs are written mainly to `birdclef_work/`:

```text
birdclef_work/
  label_map.csv
  train_audio_manifest.csv
  train_soundscape_windows.csv
  validation_predictions.csv
  validation_metrics.json
  validation_per_class_auc.csv
  strong_pipeline/
    train_audio_folds.csv
    soundscape_folds.csv
    stage*_fold*_history.csv
    pseudo_candidates_fold*.csv
    pseudo_labels_fold*.csv
    ensemble_validation_metrics.json
    checkpoints/
```

## Limitations

- Local validation is not a substitute for a hidden test or public leaderboard score.
- Pseudo-labeling can help or hurt depending on confidence thresholds and label noise.
- Stage 1 and Stage 2/3 validation results are not perfectly comparable because they use different validation domains.
- Training and inference are hardware-dependent; full cross-validation training can be time-consuming.

## Future Work

- Add a final curated visual report with user-provided figures.
- Compare EfficientNet-B0 with other audio backbones.
- Tune pseudo-label thresholds per class instead of using global thresholds.
- Add calibration analysis for predicted probabilities.
- Add experiment tracking with a lightweight tool such as MLflow or Weights & Biases.
- Package reusable preprocessing and training code into Python scripts.

## GitHub Notes

The dataset, generated manifests, cached spectrograms, checkpoints, virtual environment, and generated `submission.csv` are ignored by git. This keeps the repository focused on the workflow, modeling approach, and evaluation summary.
