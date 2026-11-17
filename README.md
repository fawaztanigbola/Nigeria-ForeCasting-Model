# Nigeria Weather Forecasting Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Domain](https://img.shields.io/badge/Domain-Climate%20%26%20Weather%20Forecasting-green.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An end-to-end data processing, spatio-temporal analysis, and deep learning pipeline designed to model and forecast meteorological conditions—specifically temperature dynamics—across Nigeria (with localized focus on Southwest Nigeria).

The project integrates high-resolution climate reanalysis datasets (**ERA5**) and Digital Elevation Model (**DEM**) topographical metrics with Recurrent Neural Networks (**LSTM**) and geospatial modeling techniques to deliver accurate time-series weather forecasts.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [CDS API Setup](#cds-api-setup)
- [Usage Guide](#usage-guide)
- [License](#license)

---

## Overview

Accurate climate and weather forecasting is critical for regional planning, agricultural productivity, and disaster response across West Africa. This repository provides a modular, notebook-driven framework for ingest, processing, and predictive modeling of daily temperature patterns in Nigeria:

1. **Automated Data Acquisition**: Programmatic download of ERA5 historical climate reanalysis metrics and high-resolution DEM topographical data.
2. **Multi-Year Data Fusion**: Merging annual raster/netCDF files into a continuous, structured spatio-temporal dataset.
3. **Data Preprocessing & Normalization**: Temporal resampling, missing-data handling, outlier detection, and scaling.
4. **Chronological Data Splitting**: Strict temporal partitioning into training, validation, and test sets to prevent data leakage in time-series evaluation.
5. **Deep Learning & Geospatial Modeling**: Implementing baseline Long Short-Term Memory (LSTM) networks and geospatial spatial-temporal models.

---

## Key Features

- **ECMWF ERA5 Ingestion**: Automated retrieval of historical temperature variables via `cdsapi`.
- **Topographical Integration**: Elevation metrics derived from DEM datasets to account for microclimatic variations.
- **Data Integrity Safeguards**: Strict chronological train-validation-test split specifically designed for autocorrelated climate data.
- **Modular Notebook Pipeline**: Fully documented Jupyter notebooks allowing full reproducibility from raw data download to model evaluation.

---

## Repository Structure

```
├── Baseline_LSTM.ipynb                                    # Deep Learning baseline model (LSTM)
├── Chronologically_Splitting_Data.ipynb                   # Time-aware data partitioning
├── Combine_Yearly_Files.ipynb                            # Stitching annual netCDF/raster files
├── Daily_Temperature_Preprocessing.ipynb                  # Cleaning, resampling, and normalization
├── ERA5_And_DEM_Dataset_Download.ipynb                    # Automated ERA5 & DEM data retrieval
├── Geospatial_Model.ipynb                                 # Spatial analysis and geospatial modeling
├── Models_Data_Preparation.ipynb                          # Feature engineering & sliding window creation
├── Southwest_Nigeria_Temperature_Project_Documentation.ipynb # Detailed project documentation & report
└── README.md                                              # Project documentation
```

---

## Pipeline Architecture

The sequential execution flow across the repository notebooks is structured as follows:

```
┌─────────────────────────────────────────┐
│ 1. ERA5_And_DEM_Dataset_Download.ipynb  │  ──> Fetch raw climate (ERA5) & DEM elevation data
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 2. Combine_Yearly_Files.ipynb           │  ──> Merge annual datasets into unified time series
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 3. Daily_Temperature_Preprocessing.ipynb│  ──> Clean, resample, and normalize daily temperature
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 4. Models_Data_Preparation.ipynb        │  ──> Construct sliding windows & lag features
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 5. Chronologically_Splitting_Data.ipynb │  ──> Chronological Train / Val / Test split
└────────────────────┬────────────────────┘
                     ▼
             ┌───────┴───────┐
             ▼               ▼
┌──────────────────┐   ┌───────────────────────┐
│ Baseline_LSTM    │   │ Geospatial_Model.ipynb│ ──> Train & evaluate predictive models
└──────────────────┘   └───────────────────────┘
```

---

## Tech Stack

| Category | Technologies / Libraries |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Data Ingestion** | `cdsapi`, `xarray`, `netCDF4` |
| **Data Manipulation** | `pandas`, `numpy`, `scipy` |
| **Machine Learning / DL** | `torch` / `tensorflow`, `scikit-learn` |
| **Spatial Analysis** | `geopandas`, `rasterio`, `shapely` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Environment** | Jupyter Notebook |

---

## Getting Started

### Prerequisites

- Python `3.8` or higher
- `pip` package manager
- An active account on the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/) (required for downloading raw climate reanalysis data).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fawaztanigbola/Nigeria-Weather-ForeCasting-Model.git
   cd Nigeria-Weather-ForeCasting-Model
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install jupyter pandas numpy scikit-learn matplotlib seaborn xarray netCDF4 cdsapi torch geopandas rasterio
   ```

### CDS API Setup

To download raw ERA5 data via `ERA5_And_DEM_Dataset_Download.ipynb`:

1. Register at [Copernicus CDS](https://cds.climate.copernicus.eu/).
2. Obtain your User ID (UID) and API Key from your CDS profile page.
3. Save your API credentials in your user home directory:

   **Linux/macOS (`~/.cdsapirc`):**
   ```bash
   cat <<EOT > ~/.cdsapirc
   url: https://cds.climate.copernicus.eu/api/v2
   key: <YOUR-UID>:<YOUR-API-KEY>
   EOT
   ```

   **Windows (`C:\Users\<Username>\.cdsapirc`):**
   ```text
   url: https://cds.climate.copernicus.eu/api/v2
   key: <YOUR-UID>:<YOUR-API-KEY>
   ```

---

## Usage Guide

Run the pipeline sequentially using Jupyter Notebook:

```bash
jupyter notebook
```

### Notebook Execution Order

1. **`ERA5_And_DEM_Dataset_Download.ipynb`**
   Retrieves ERA5 temperature reanalysis grids and DEM topographical elevation datasets for the spatial bounds of Nigeria.

2. **`Combine_Yearly_Files.ipynb`**
   Aggregates multi-year climate data files into a single unified temporal record.

3. **`Daily_Temperature_Preprocessing.ipynb`**
   Performs missing value imputation, quality checks, spatial aggregation, and daily temperature extraction.

4. **`Models_Data_Preparation.ipynb`**
   Constructs feature vectors, temporal lag variables, and sliding window arrays suitable for sequence modeling.

5. **`Chronologically_Splitting_Data.ipynb`**
   Splits dataset chronologically to maintain time-series validation validity.

6. **`Baseline_LSTM.ipynb`**
   Trains the Long Short-Term Memory (LSTM) baseline model and outputs quantitative metrics (MAE, RMSE).

7. **`Geospatial_Model.ipynb`**
   Evaluates spatial dependencies and runs geospatial forecasting experiments incorporating elevation features.

8. **`Southwest_Nigeria_Temperature_Project_Documentation.ipynb`**
   Contains comprehensive regional documentation, exploratory figures, and detailed analysis results.

---

## License

This project is licensed under the [MIT License](LICENSE).