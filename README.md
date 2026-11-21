# Nigeria Weather Forecasting Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Domain](https://img.shields.io/badge/Domain-Climate%20%26%20Weather%20Forecasting-green.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An end-to-end **spatio-temporal data processing and deep learning project** developed to investigate and forecast daily minimum and maximum temperature patterns across Nigeria, with a focused regional implementation covering Southwest Nigeria (Lagos, Ogun, Oyo, and Osun).

The project integrates historical climate reanalysis data from **ERA5**, topographical elevation information derived from Digital Elevation Models (DEM), and Recurrent Neural Networks (LSTM) in PyTorch to evaluate spatio-temporal temperature dynamics.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Pipeline Architecture](#pipeline-architecture)
- [Tech Stack](#tech-stack)
- [Getting Started & Installation](#getting-started--installation)
- [Usage & Execution Guide](#usage--execution-guide)
- [Dataset and Study Area](#dataset-and-study-area)
- [Model Development](#model-development)
- [Model Performance](#model-performance)
- [Key Findings](#key-findings)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [License](#license)

---

## Overview

Temperature forecasting plays an important role in climate analysis, agricultural planning, environmental monitoring, and regional decision-making.

This project explores how deep learning can be applied to historical meteorological data to predict daily minimum ($T_{\text{min}}$) and maximum ($T_{\text{max}}$) temperatures across multiple geographical grid locations.

The core pipeline encompasses:

1. **Climate Data Acquisition:** Retrieval and preparation of historical ERA5 temperature data and geographical elevation information (DEM).
2. **Multi-Year Data Integration:** Combining yearly climate datasets into a continuous time series covering 2015–2024.
3. **Data Preprocessing:** Handling missing observations, preparing daily temperature variables, and normalizing inputs.
4. **Temporal Data Preparation:** Constructing chronological training sequences using sliding windows.
5. **Baseline Model Development:** Developing a temperature-only LSTM model using historical observations.
6. **Geospatial Model Development:** Extending temporal forecasting with cyclical seasonal representations and static spatial features (latitude, longitude, elevation).
7. **Model Evaluation:** Benchmarking models against persistence and climatology reference forecasts.

---

## Key Contributions

- **Multi-Year Climate Data Processing:** Pipeline for ingesting and harmonizing 10 years (2015–2024) of daily ERA5 climate reanalysis data.
- **Geospatial Data Integration:** Fused static elevation profiles alongside spatial coordinates (latitude/longitude) with dynamic meteorological inputs.
- **Temporal Feature Engineering:** Implemented sliding-window sequence generators and cyclical sine/cosine transformations for day-of-year seasonality.
- **Deep Learning Architectures:** Designed and trained single-layer and multi-layer LSTM networks in PyTorch for multi-output prediction ($T_{\text{min}}$ and $T_{\text{max}}$).
- **Ablation & Sensitivity Analysis:** Systematic testing of input window lengths (7, 14, 30 days) and static geographical feature ablation.
- **Forecast Benchmarking:** Quantitative evaluation against persistence and day-of-year climatology reference baselines using MAE, RMSE, and Forecast Skill Scores.
- **Spatial Error Analysis:** Spatial visualization and error distribution mapping across regional grid cells.

---

## Pipeline Architecture

```text
┌──────────────────────────────────────────────┐
│ ERA5 AND DEM DATA ACQUISITION                │
│ Historical temperature and elevation data   │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ MULTI-YEAR DATA INTEGRATION                  │
│ Combining yearly climate datasets           │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ DAILY TEMPERATURE PREPROCESSING              │
│ Cleaning, missing-data handling and scaling │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ MODEL DATA PREPARATION                       │
│ Sliding windows and feature engineering     │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ CHRONOLOGICAL DATA PARTITIONING              │
│ Training, validation and testing             │
└──────────────────────┬───────────────────────┘
                       ▼
             ┌─────────┴─────────┐
             ▼                   ▼
┌────────────────────┐  ┌──────────────────────┐
│ TEMPERATURE-ONLY   │  │ GEOSPATIAL LSTM      │
│ BASELINE LSTM      │  │                      │
│                    │  │ Seasonal + location │
│ Tmin / Tmax        │  │ information          │
└─────────┬──────────┘  └──────────┬───────────┘
          └────────────┬───────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ MODEL EVALUATION AND COMPARISON              │
│ MAE, RMSE, forecast skill and spatial errors │
└──────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technologies / Libraries |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Climate & Raster Data** | ERA5 Reanalysis, Copernicus CDS API (`cdsapi`), `xarray`, `netCDF4`, `rasterio` |
| **Data Manipulation & Math** | `pandas`, `numpy`, `scipy` |
| **Deep Learning** | PyTorch |
| **Machine Learning Utilities** | `scikit-learn` |
| **Geospatial Processing** | `geopandas`, `shapely` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Environment** | Jupyter Notebook, Google Colab (NVIDIA Tesla T4 GPU) |

---

## Getting Started & Installation

### Prerequisites

- **Python**: Version 3.8 or higher.
- **Copernicus CDS API Account**: Required only if you plan to execute `ERA5_And_DEM_Dataset_Download.ipynb` to re-download raw ERA5 data. Set up your API key in `~/.cdsapirc`.
- **GPU Acceleration**: Recommended (e.g., NVIDIA GPU with CUDA) for neural network training, though CPU execution is supported.

### Installation Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/fawaztanigbola/Nigeria-Weather-ForeCasting-Model.git
   cd Nigeria-Weather-ForeCasting-Model
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install torch pandas numpy scipy scikit-learn xarray netCDF4 cdsapi geopandas rasterio shapely matplotlib seaborn jupyter
   ```

4. **Configure Copernicus CDS API Key (Optional)**
   If executing raw climate data downloads, create `~/.cdsapirc` (`C:\Users\<Username>\.cdsapirc` on Windows):
   ```text
   url: https://cds.climate.copernicus.eu/api/v2
   key: <YOUR-UID>:<YOUR-API-KEY>
   ```

---

## Usage & Execution Guide

The pipeline is organized sequentially across Jupyter Notebooks. To reproduce the workflow or run experiments, execute the notebooks in the following order:

```text
├── 1. ERA5_And_DEM_Dataset_Download.ipynb       # Downloads raw ERA5 climate & DEM elevation data
├── 2. Combine_Yearly_Files.ipynb               # Merges multi-year NetCDF files into unified datasets
├── 3. Daily_Temperature_Preprocessing.ipynb    # Cleans, extracts daily Tmin/Tmax, handles missing values
├── 4. Models_Data_Preparation.ipynb            # Generates sliding-window sequences & engineered features
├── 5. Chronologically_Splitting_Data.ipynb     # Splits data into train, validation, and test splits
├── 6. Baseline_LSTM.ipynb                      # Trains and evaluates the Baseline LSTM model
├── 7. Geospatial_Model.ipynb                   # Trains and evaluates the Geospatial LSTM model
└── 8. Southwest_Nigeria_Temperature_Project_Documentation.ipynb # Consolidated results & project summary
```

### Quick Launch via Jupyter

Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
Open and run notebooks sequentially starting from data acquisition or data preprocessing.

---

## Dataset and Study Area

The initial project scope targeted country-wide temperature forecasting across Nigeria. To accommodate memory (RAM) and training time constraints in standard Google Colab environments while maintaining multi-location spatial coverage, the study region was configured for **Southwest Nigeria**.

### Covered States
- Lagos
- Ogun
- Oyo
- Osun

```text
├── Original Regional Grid : 672 cells
└── Usable Grid Locations  : 456 cells
```

### Dataset Summary

| Property | Description |
| :--- | :--- |
| Climate Dataset | ERA5 Reanalysis |
| Study Period | 2015–2024 (10 years) |
| Temporal Resolution | Daily |
| Total Daily Observations | 3,653 days per grid location |
| Target Variables | Minimum Temperature ($T_{\text{min}}$), Maximum Temperature ($T_{\text{max}}$) |
| Static Spatial Features | Latitude, Longitude, Elevation (DEM) |

---

## Model Development

Two LSTM architectures were developed to investigate daily temperature forecasting.

### 1. Temperature-only Baseline LSTM

Evaluates forecasting capacity relying exclusively on historical temperature observations.

- **Input Features:** 7-day historical window of $T_{\text{min}}$ and $T_{\text{max}}$.
- **Output Targets:** Next-day $T_{\text{min}}$ and $T_{\text{max}}$.
- **Architecture:**
  - 1-Layer LSTM (Hidden size: 64)
  - Dense Linear Prediction Head (Output dim: 2)
  - Loss Function: Mean Squared Error (MSE)
  - Optimizer: Adam
  - Checkpointing & Early Stopping on validation loss

### 2. Geospatial LSTM

Extends temporal modelling by integrating cyclical seasonality and static geographical features.

- **Dynamic Temporal Inputs:** $T_{\text{min}}$, $T_{\text{max}}$, $\sin(\text{Day of Year})$, $\cos(\text{Day of Year})$.
- **Static Geographical Inputs:** Latitude, Longitude, Elevation.
- **Architecture:** The LSTM processes temporal sequence vectors. The final hidden state is concatenated with normalized spatial vectors before passing through the output prediction head.

### Feature & Hyperparameter Experiments

| Experiment | Investigation | Result |
| :--- | :--- | :--- |
| **Lookback Window** | 7, 14, and 30 days | 7-day window yielded optimal validation performance |
| **Network Complexity** | LSTM layer count & dropout rates | Single-layer LSTM without dropout performed best |
| **Spatial Feature Ablation** | With vs. without Lat/Lon/Elevation | Static features did not improve test metric generalisation |

---

## Model Performance

Evaluation metrics are computed using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) in degrees Celsius (°C).

### Baseline LSTM vs. Geospatial LSTM Test Performance

| Model | Target Variable | MAE (°C) | RMSE (°C) |
| :--- | :--- | ---: | ---: |
| **Baseline LSTM** | Minimum Temperature ($T_{\text{min}}$) | **0.502** | **0.690** |
| | Maximum Temperature ($T_{\text{max}}$) | **0.988** | **1.324** |
| **Geospatial LSTM** | Minimum Temperature ($T_{\text{min}}$) | 0.535 | 0.707 |
| | Maximum Temperature ($T_{\text{max}}$) | 1.021 | 1.346 |

---

### Forecast Skill Analysis

Forecast Skill Score measures relative error reduction against baseline reference models:

$$\text{Skill Score} = \left(1 - \frac{\text{Model Error}}{\text{Reference Error}}\right) \times 100$$

#### Baseline LSTM vs. Persistence Reference

| Target | MAE Error Reduction | RMSE Error Reduction |
| :--- | ---: | ---: |
| $T_{\text{min}}$ | 12.85% | 11.31% |
| $T_{\text{max}}$ | 11.55% | 16.94% |

#### Geospatial LSTM vs. Reference Methods

| Reference Model | Target | MAE Error Reduction | RMSE Error Reduction |
| :--- | :--- | ---: | ---: |
| **Persistence** | $T_{\text{min}}$ | 8.23% | 9.24% |
| | $T_{\text{max}}$ | 10.75% | 17.73% |
| **Day-of-Year Climatology** | $T_{\text{min}}$ | 41.34% | 41.43% |
| | $T_{\text{max}}$ | 25.91% | 22.24% |

*Note: Baseline and Geospatial models were evaluated across distinct chronological partition configurations and scaling pipelines; metrics highlight individual relative gains over baselines rather than direct head-to-head performance.*

---

## Key Findings

1. **Recurrent Modeling Efficacy:** Both LSTM implementations achieved superior performance over naive persistence and long-term climatology baselines.
2. **Optimal Temporal History:** A 7-day lookback window captured sufficient short-term autocorrelation; longer historical windows (14, 30 days) increased compute without improving validation loss.
3. **Static Spatial Features:** Concatenating raw static spatial coordinates (latitude, longitude, elevation) directly into the dense layer did not yield test-set error reductions over temperature-only temporal sequences.
4. **Seasonal Encoding:** Day-of-year cyclical encoding enabled the geospatial model to outperform historical climatology by up to 41.4%.

---

## Limitations and Future Improvements

- **Resource & Compute Scope:** High-resolution nationwide modeling was constrained by Google Colab memory allocation. Future iterations can scale to country-wide coverage using distributed computing or spatial tiling.
- **Advanced Spatial Fusion:** Modern spatio-temporal architectures (e.g., Graph Convolutional Networks or ConvLSTM) could better capture inter-grid spatial dependencies than concatenated static scalars.
- **Exogenous Meteorological Variables:** Incorporating additional ERA5 meteorological fields—such as relative humidity, solar radiation, soil moisture, and wind velocity—could improve $T_{\text{max}}$ predictive precision.
- **Unified Benchmark Split:** Implementing a standardized cross-validation protocol across all notebook pipelines will facilitate unified head-to-head comparisons.

---

## License

This project is open-source and released under the [MIT License](LICENSE).