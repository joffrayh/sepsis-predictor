# FRAMES Model Training Pipeline

This document covers how to run the training pipeline, what's happening under the hood, and how to configure it. If you haven't run the data pipeline yet, do that first as the model training pipeline needs the processed Parquet file as its input. See [DATA_PIPELINE.md](DATA_PIPELINE.md).

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Data pipeline complete | `data/processed_files/your_processed_data_file.parquet` must exist. Run `uv run src/data_processing/main.py` first — see [DATA_PIPELINE.md](DATA_PIPELINE.md) |
| MLflow | Included in project dependencies. Run `uv run mlflow ui` after training to see results in your browser |

---

## Quick Start

Check that `data.path` in `src/model_training/config.yaml` points to your processed Parquet file, then:

```bash
uv run src/model_training/main.py
```

By default this trains the XGBoost model and logs everything to MLflow. To switch to LightGBM or LSTM, change `active_model` in `config.yaml` — no code changes needed. See [Configuration](#configuration) below.

To use a custom config file instead of the default:

```bash
uv run src/model_training/main.py --config path/to/your/config.yaml
```

Once the run finishes, open the MLflow UI to inspect results:

```bash
uv run mlflow ui
```

Then go to `http://localhost:5000` in your browser.

---

## How the Pipeline Works

Everything is orchestrated by `main.py`. You shouldn't need to touch it, just edit the config. Here's what happens when you start a run:

### 1. Data Loading and Splitting

`data/loader.py` loads the Parquet file and filters to rows where sepsis isn't already active (`sepsis == 0`). It then creates the prediction target: a binary flag indicating whether the patient develops sepsis within the next `target_window` timesteps, looking `target_horizon` steps ahead. This is a forward-looking label, so the model is predicting future sepsis, not flagging current sepsis.

The data is split into train, validation, and test sets (for splits see `data.split` in [config.yaml](src/model_training/config.yaml)). The split is stratified by outcome and grouped by `stay_id`, so every timestep for a given patient ends up in the same partition, ensuring no data leakage across splits.

Optionally, lag features (e.g. `heart_rate_lag1`) and rolling statistics (mean/std/min/max over given  windows) can be added at this stage. Both are off by default and can be adjusted in the config.

### 2. MLflow Setup

An MLflow experiment is created (if it doesn't already exist) and a new run is opened. The full config YAML is immediately saved as an artifact under `reproduction_configs/`, so every run has a self-contained record of exactly what settings it used. All hyperparameters are also logged as run parameters.

### 3. Model Training

The model is selected from the factory by name and trained. What this looks like depends on the model:

- **LightGBM / XGBoost**: Optuna searches over the hyperparameter ranges defined in the config, maximising AUPRC on the validation set. Each trial uses early stopping. Once the search finishes, the best trial's parameters are used to train the final model.
- **LSTM**: Sequences are padded to a uniform length and batched via PyTorch DataLoaders. The model is trained with `BCEWithLogitsLoss`, with a `pos_weight` to handle class imbalance. A causal attention mask prevents the model from attending to future timesteps within a sequence. Training stops early if validation AUPRC doesn't improve for `patience` consecutive epochs.

### 4. Evaluation

After training, the model predicts on the held-out test set and three metrics are computed:

| Metric | Why |
|---|---|
| **AUPRC** | Primary metric, handles class imbalance better than AUROC |
| **AUROC** | Standard metric used in most literature |
| **F1** | Threshold-dependent, so useful for comparing pure binary performance |

Calibration, PR, and ROC curve plots are generated and logged to MLflow as image artifacts.

### 5. Custom Functions and Saving

For tree models, a SHAP summary plot is generated using a random sample of 2,000 rows from the training data. The LSTM skips this step. The trained model is then saved as an MLflow artifact.

---

## Configuration

All configuration lives in [`src/model_training/config.yaml`](src/model_training/config.yaml), which is commented throughout.

---

## File Map

| File | What it does |
|------|-------------|
| `main.py` | Top-level orchestrator, runs the full pipeline end to end |
| `config.yaml` | Configuration |
| `data/loader.py` | Data loading, target creation, feature engineering, train/val/test split |
| `data/sequence_utils.py` | PyTorch Dataset and collation for variable-length sequences (LSTM only) |
| `models/factory.py` | Maps model name strings to their wrapper classes |
| `models/base_model.py` | Abstract base class shared by all models |
| `models/lightgbm_model.py` | LightGBM wrapper with Optuna HPO |
| `models/xgboost_model.py` | XGBoost wrapper with Optuna HPO |
| `models/lstm_model.py` | LSTM with causal attention |
| `utils/metrics.py` | Evaluation metrics and plots |
| `custom_funcs/custom_plots.py` | SHAP explanations |

---

## Adding a New Model

A step-by-step tutorial for adding your own model to the pipeline is in [tutorial.ipynb](src/model_training/tutorial.ipynb).

---

## Linting

After any code changes to the training pipeline:

```bash
uv run ruff check src/model_training --fix
uv run ruff format src/model_training
```
