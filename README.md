# FRAMES

> **F**air f**R**amework **A**ssessing **M**IMIC-IV for **E**arly **S**epsis-prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![FAIR](https://img.shields.io/badge/FAIR-compliant-green.svg)](https://www.go-fair.org/fair-principles/)

---

## Introduction

Sepsis prediction research has produced very encouraging results, yet the field is restricted by a fundamental problem: significant variability across studies. Differences include patient cohorts, sepsis definitions, feature sets, and evaluation metrics, making it nearly impossible to compare or reproduce published findings.

**FRAMES** addresses this by providing a fully open-source, end-to-end framework built on the [MIMIC-IV](https://physionet.org/content/mimiciv/) dataset. It standardises every stage of the research process — from raw data extraction through to model evaluation — so that results produced by the framework are meaningful, reproducible, and comparable.

### FAIR Principles

FRAMES is designed around the [FAIR principles](https://www.go-fair.org/fair-principles/) for scientific data and software:

| Principle | How FRAMES applies it |
|---|---|
| **Findable** | Code hosted on GitHub; data hosted on PhysioNet; permanent DOI issued via Zenodo |
| **Accessible** | Fully open-source code and data; extensive documentation; tutorials provided; installation via `uv`; no database server required |
| **Interoperable** | Standardised Sepsis-3 definition and cohort; open data formats (CSV, Parquet, JSON); compatible with common ML frameworks; config-driven allows execution with any configuration file |
| **Reusable** | Seeding for all stochastic operations; full `config.yaml` saved as every run; heavy logging of every experiment; fully configurable pipelines; MIT License |

### Supported Models

Out of the box, FRAMES supports:

- **LightGBM**
- **XGBoost**
- **LSTM**

### Use Cases

FRAMES is built for researchers who want to:

- **Train and evaluate a new model** — use your own model and evaluate it on a robust, standardised pipeline without building the data pipeline yourself.
- **Reproduce a published result** — clone the repository, use the same configuration, and obtain comparable results under the same experiment conditions.
- **Compare models** — by using the same configuration, all models are evaluated on an identical cohort, feature set, and evaluation metrics, making comparisons between models meaningful.

---

## Prerequisites

Before installing FRAMES, ensure you have the following:

- **Python 3.11** and [**`uv`**](https://docs.astral.sh/uv/) (used for environment and dependency management)
- **MIMIC-IV v3.1** — requires credentialed access via PhysioNet. [Apply for access here](https://physionet.org/content/mimiciv/). Once approved, download the compressed dataset and place it at `data/raw/mimic-iv-3.1/`, ensuring the `hosp/` and `icu/` subdirectories are present with their `.csv.gz` files.

---

## Installation

```bash
git clone https://github.com/joffrayh/FRAMES
cd FRAMES
uv sync
```

This installs all dependencies into a virtual environment managed by `uv`.

---

## Quickstart

### 1. Run the data pipeline

On the first run, point to the raw MIMIC-IV data:

```bash
uv run src/data_processing/main.py --raw-data-dir data/raw/mimic-iv-3.1
```

On subsequent runs, the flag is not required:

```bash
uv run src/data_processing/main.py
```

This produces the ML-ready dataset used by model training. See [DATA_PIPELINE.md](DATA_PIPELINE.md) for full details on the pipeline and configuration options.

### 2. Run the model training pipeline

```bash
uv run src/model_training/main.py
```

Edit `src/model_training/config.yaml` to select a model, set hyperparameter search space, or adjust the data path. All runs are tracked automatically in MLflow. See [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) for full details on the training pipeline.

### 3. Tutorial

A step-by-step tutorial on adding your own model to the training pipeline is available in [tutorial.ipynb](src/model_training/tutorial.ipynb).

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository and create a branch.
2. Make your changes, ensuring they pass linting (`uv run ruff check`) and formatting (`uv run ruff format`).
3. Open a pull request with a clear description of the change and its motivation.

Please follow the NumPy-style docstrings convention and ensure appropriate documentation is provided.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
