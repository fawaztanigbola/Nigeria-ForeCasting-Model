# Nigeria Weather Forecasting Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Domain](https://img.shields.io/badge/Domain-Climate%20%26%20Weather%20Forecasting-green.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An end-to-end **spatio-temporal data processing and deep learning pipeline** developed to investigate and forecast daily temperature patterns across Southwest Nigeria (Lagos, Ogun, Oyo, and Osun states).

The project combines historical climate reanalysis data from Copernicus **ERA5**, topographical elevation data derived from Digital Elevation Models (DEM), and Recurrent Neural Networks (LSTM) in PyTorch to model and predict daily minimum ($T_{\text{min}}$) and maximum ($T_{\text{max}}$) temperatures across multiple grid locations.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features & Contributions](#key-features--contributions)
- [Pipeline Architecture](#pipeline-architecture)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Dataset & Study Area](#dataset--study-area)
- [Model Development](#model-development)
- [Model Performance](#model-performance)
- [Getting Started & Installation](#getting-started--installation)
- [Execution & Usage Guide](#execution--usage-guide)
- [Key Findings](#key-findings)
- [Limitations & Future Improvements](#limitations--future-improvements)
- [License](#license)

---

## Project Overview

Accurate temperature forecasting is vital for climate analysis, agricultural decision-making, and environmental risk assessment. This repository provides a complete, modular workflow for processing high-dimensional meteorological reanalysis datasets, performing spatial and temporal feature engineering, and training PyTorch deep learning models to perform next-day temperature forecasting.

Beginning with a broader Nigeria-wide modelling objective, the study scope was optimized to cover Southwest Nigeria ($456$ active grid cells over 10 years, 2015–2024). This focus ensures computational efficiency on resource-constrained environments (such as Google Colab) while preserving spatial heterogeneity across urban, coastal, and inland terrain.

---

## Key Features & Contributions

- **Multi-Year Climate Pipeline:** Ingests and harmonizes 10 years (2015–2024) of daily ERA5 reanalysis data ($T_{\text{min}}$ and $T_{\text{max}}$) for $3,653$ continuous days.
- **Geospatial Feature Integration:** Combines static topographical elevation (DEM) with spatial coordinates ($\text{Latitude}, \text{Longitude}$) and dynamic climate observations.
- **Temporal & Seasonal Feature Engineering:** Uses sliding-window temporal sequences ($7$-day historical lookbacks) and cyclical transformations ($\sin / \cos$ encoding of the day-of-year) to model seasonality.
- **Deep Learning Architectures:** Implements and evaluates two PyTorch models:
  1. **Baseline LSTM:** Temperature-only sequence modeling.
  2. **Geospatial LSTM:** Merges temporal sequence representations with static geographic encodings.
- **Comprehensive Benchmarking:** Evaluates models against standard baseline reference methods including **Persistence** and **Day-of-Year Climatology**.
- **Spatial Error Diagnostics:** Analyzes forecasting performance across spatial grid points to highlight geographic performance patterns.

---

## Pipeline Architecture

The end-to-end processing and modeling workflow follows a clear, sequential pipeline:

```text
┌──────────────────────────────────────────────────┐
│ 1. DATA ACQUISITION & INTEGRATION               │
│    - ERA5 NetCDF reanalysis download (CDS API)   │
│    - Topographical elevation extract (DEM)        │
│    - Yearly climate dataset concatenation         │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│ 2. PREPROCESSING & CLEANING                      │
│    - Grid filtering & spatial cropping           │
│    - Handling missing observations & scaling     │
│    - Daily Tmin / Tmax aggregation               │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│ 3. FEATURE ENGINEERING & PARTITIONING            │
│    - Cyclical day-of-year encoding (Sin/Cos)     │
│    - 7-day sliding window sequence generation    │
│    - Chronological Train / Val / Test splitting  │
└────────────────────────┬─────────────────────────┘
                         ▼
             ┌───────────┴───────────┐
             ▼                       ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│ BASELINE LSTM MODEL     │  │ GEOSPATIAL LSTM MODEL   │
│ - Historical Tmin/Tmax  │  │ - Temp + Cyclical DOY   │
│ - PyTorch 1-Layer LSTM  │  │ - Concat Lat/Lon/Elev   │
└────────────┬────────────┘  └────────────┬────────────┘
             └───────────┬───────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│ 4. MODEL EVALUATION & BENCHMARKING               │
│    - Metrics: MAE, RMSE (°C)                     │
│    - Baselines: Persistence & DOY Climatology    │
│    - Forecast Skill Score calculation            │
└──────────────────────────────────────────────────┘
```

---

## Repository Structure

```text
.
├── ERA5_And_DEM_Dataset_Download.ipynb           # Notebook 1: Download ERA5 climate data via CDS API & DEM elevation
├── Combine_Yearly_Files.ipynb                    # Notebook 2: Merge multi-year NetCDF files into continuous datasets
├── Daily_Temperature_Preprocessing.ipynb         # Notebook 3: Preprocessing, missing value handling, grid filtering
├── Models_Data_Preparation.ipynb                 # Notebook 4: Sequence creation, sliding windows, feature scaling
├── Chronologically_Splitting_Data.ipynb          # Notebook 5: Chronological train/validation/test dataset splits
├── Baseline_LSTM.ipynb                           # Notebook 6: PyTorch Baseline (Temperature-only) LSTM model
├── Geospatial_Model.ipynb                        # Notebook 7: PyTorch Geospatial LSTM (Temp + Seasonality + Geography)
├── Southwest_Nigeria_Temperature_Project_Documentation.ipynb # Notebook 8: Consolidated technical report & visualizations
└── README.md                                     # Main repository documentation
```

---

## Tech Stack

| Domain | Technologies & Libraries |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch (`torch`, `torch.nn`, `DataLoader`) |
| **Climate & Geospatial Ingestion** | `cdsapi`, `xarray`, `netCDF4`, `h5netcdf` |
| **Geospatial Processing** | `geopandas`, `rasterio`, `shapely`, `pyproj` |
| **Data Manipulation & Math** | `pandas`, `numpy`, `scipy`, `scikit-learn` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Environment** | Jupyter Notebook, Google Colab (Nvidia T4 GPU) |

---

## Dataset & Study Area

The primary study region covers **Southwest Nigeria** (Lagos, Ogun, Oyo, and Osun states), providing a representative mix of coastal, forest, and savannah climate zones.

| Parameter | Specification |
| :--- | :--- |
| **Data Source** | ERA5 Reanalysis (ECMWF) + SRTM DEM Elevation |
| **Temporal Range** | January 1, 2015 – December 31, 2024 ($3,653$ days) |
| **Temporal Resolution** | Daily ($T_{\text{min}}$, $T_{\text{max}}$) |
| **Spatial Grid Resolution** | $0.25^\circ \times 0.25^\circ$ grid |
| **Usable Grid Cells** | $456$ locations across Southwest Nigeria |
| **Target Variables** | Next-day Minimum ($T_{\text{min}}$) & Maximum ($T_{\text{max}}$) Temperature (°C) |
| **Static Features** | Latitude, Longitude, Elevation (meters) |
| **Dynamic Features** | 7-day temperature history, Day-of-Year $\sin / \cos$ cycles |

---

## Model Development

Two neural network architectures were implemented using PyTorch to evaluate the impact of spatial and temporal features:

### 1. Temperature-only Baseline LSTM
- **Input Features:** 7-day lookback sequence of past daily $T_{\text{min}}$ and $T_{\text{max}}$ observations.
- **Architecture:** Single-layer LSTM ($64$ hidden units) followed by a fully connected dense output layer ($2$ targets: $T_{\text{min}}, T_{\text{max}}$).
- **Optimization:** Adam optimizer, Mean Squared Error (MSE) loss, early stopping on validation loss.

### 2. Geospatial LSTM
- **Temporal Inputs:** 7-day lookback sequence of $T_{\text{min}}$, $T_{\text{max}}$, plus cyclical Day-of-Year encodings ($\sin(\frac{2\pi \cdot DOY}{365})$, $\cos(\frac{2\pi \cdot DOY}{365})$).
- **Static Inputs:** Latitude, Longitude, and DEM Elevation, normalized and concatenated directly with the LSTM's final hidden state before passing to the final linear layer.
- **Architecture:** Single-layer LSTM ($64$ hidden units) + Feature Concatenation + Dense Prediction Head.

---

## Model Performance

Evaluation was conducted on out-of-sample chronological test splits using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** in degrees Celsius (°C).

### Test Metrics Summary

| Model Architecture | Target Variable | MAE (°C) | RMSE (°C) |
| :--- | :--- | ---: | ---: |
| **Baseline LSTM** | Minimum Temp ($T_{\text{min}}$) | **0.502** | **0.690** |
| | Maximum Temp ($T_{\text{max}}$) | **0.988** | **1.324** |
| **Geospatial LSTM** | Minimum Temp ($T_{\text{min}}$) | 0.535 | 0.707 |
| | Maximum Temp ($T_{\text{max}}$) | 1.021 | 1.346 |

### Skill Score vs. Reference Baselines

Forecast Skill Score measures relative error reduction:
$$\text{Skill Score} = \left(1 - \frac{\text{Error}_{\text{Model}}}{\text{Error}_{\text{Reference}}}\right) \times 100\%$$

- **Baseline LSTM vs. Persistence Forecast:**
  - $T_{\text{min}}$: **12.85%** MAE reduction, **11.31%** RMSE reduction.
  - $T_{\text{max}}$: **11.55%** MAE reduction, **16.94%** RMSE reduction.
- **Geospatial LSTM vs. Day-of-Year Climatology:**
  - $T_{\text{min}}$: **41.34%** MAE reduction, **41.43%** RMSE reduction.
  - $T_{\text{max}}$: **25.91%** MAE reduction, **22.24%** RMSE reduction.

---

## Getting Started & Installation

### Prerequisites

- **Python:** Version `3.8` or higher
- **Hardware:** CPU supported; NVIDIA GPU with CUDA recommended for fast model training.
- **CDS API Credentials:** Required if re-downloading raw ERA5 datasets from ECMWF. Set up your API key at [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fawaztanigbola/Nigeria-Weather-ForeCasting-Model.git
   cd Nigeria-Weather-ForeCasting-Model
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Linux/macOS
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install --upgrade pip
   pip install torch numpy pandas xarray netcdf4 cdsapi scikit-learn geopandas rasterio matplotlib seaborn jupyter
   ```

---

## Execution & Usage Guide

To replicate the data processing, feature engineering, and model training, run the Jupyter notebooks in the following order:

1. **`ERA5_And_DEM_Dataset_Download.ipynb`**  
   Configure your `~/.cdsapirc` file and execute to download historical ERA5 NetCDF climate files and SRTM DEM elevation maps.

2. **`Combine_Yearly_Files.ipynb`**  
   Concatenate yearly climate files into continuous multi-year time-series arrays using `xarray`.

3. **`Daily_Temperature_Preprocessing.ipynb`**  
   Extract minimum/maximum daily temperatures, align spatial grids, handle missing values, and crop spatial boundaries to Southwest Nigeria.

4. **`Models_Data_Preparation.ipynb`**  
   Perform min-max feature scaling, construct cyclical seasonal features, and generate 7-day sliding window temporal tensors.

5. **`Chronologically_Splitting_Data.ipynb`**  
   Split sequences chronologically into Training, Validation, and Holdout Testing sets to prevent temporal data leakage.

6. **`Baseline_LSTM.ipynb`**  
   Train and evaluate the baseline temperature-only PyTorch LSTM model.

7. **`Geospatial_Model.ipynb`**  
   Train and evaluate the Geospatial PyTorch LSTM model with dynamic seasonal encodings and static topographical features.

8. **`Southwest_Nigeria_Temperature_Project_Documentation.ipynb`**  
   Generate consolidated evaluation charts, spatial loss maps, and comparative performance figures.

---

## Key Findings

1. **LSTM Superiority over Naive Baselines:** Deep learning models consistently beat persistence and climatology reference models, capturing short-term thermal inertia effectively.
2. **Optimal Lookback Window:** Experiments with 7, 14, and 30-day temporal windows revealed that a **7-day lookback** achieved optimal validation performance without introducing unnecessary memory overhead.
3. **Impact of Static Geospatial Features:** In the current single-region setup, static latitude, longitude, and elevation features did not improve test set accuracy over the pure temporal baseline. This suggests that historical sequence data already implicitly encapsulates local microclimate baselines.

---

## Limitations & Future Improvements

- **Spatial Scope Extension:** Future iterations can scale model execution nationwide across all regions of Nigeria by leveraging distributed PyTorch training (`DistributedDataParallel`) and spatial tiling.
- **Advanced Spatial Architectures:** Replace static concatenated features with spatial-aware architectures such as **ConvLSTM** or **Graph Neural Networks (GNNs)** to explicit model spatial interactions between neighboring grid cells.
- **Exogenous Meteorological Inputs:** Incorporate additional ERA5 variables such as solar radiation, surface pressure, relative humidity, and wind speed components ($u, v$).

---

## License

This project is licensed under the [MIT License](LICENSE).