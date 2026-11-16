# Nigeria Forecasting Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Domain](https://img.shields.io/badge/Domain-Climate%20%26%20Weather%20Forecasting-green.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

A end-to-end data processing and deep learning pipeline designed to model and forecast meteorological conditions (specifically temperature trends) across Nigeria. 

The project leverages climate reanalysis datasets (**ERA5**) and topographical data (**DEM** - Digital Elevation Model) combined with Recurrent Neural Networks (**LSTM**) for temporal forecasting.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Repository Workflow](#repository-workflow)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage Guide](#usage-guide)
- [License](#license)

---

## Overview

Accurate weather and climate forecasting is critical for agriculture, disaster management, and resource allocation in West Africa. This repository provides a modular, notebook-driven pipeline for:
1. **Automated Data Acquisition**: Downloading spatial and temporal datasets (ERA5 reanalysis and DEM topographical metrics).
2. **Data Aggregation & Cleaning**: Merging multi-year netCDF/raster/tabular files and performing spatial/temporal cleaning.
3. **Data Splitting**: Applying strict chronological splitting to maintain time-series integrity and avoid data leakage.
4. **Machine Learning & Modeling**: Building and evaluating long short-term memory (LSTM) neural network baselines tailored for spatio-temporal data forecasting.

---

## Repository Workflow

The execution flow of the project follows a step-by-step pipeline across the Jupyter Notebooks in this repository:

```
┌─────────────────────────────────────────┐
│ 1. ERA5_And_DEM_Dataset_Download.ipynb  │ ──> Download raw climate & elevation datasets
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 2. Combine_Yearly_Files.ipynb           │ ──> Aggregate multi-year climate records
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 3. Daily_Temperature_Preprocessing.ipynb│ ──> Clean & normalize daily temperature series
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 4. Models_Data_Preparation.ipynb       │ ──> Scale features & construct sliding windows
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 5. Chronologically_Splitting_Data.ipynb │ ──> Train / Validation / Test temporal split
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 6. Baseline_LSTM.ipynb                  │ ──> Train baseline Deep Learning (LSTM) model
└─────────────────────────────────────────┘
```

---

## Tech Stack

* **Language**: Python 3.8+
* **Data Ingestion & Climate Datasets**: ECMWF Data API (`cdsapi`), `xarray`, `netCDF4`
* **Data Manipulation & Analysis**: `pandas`, `numpy`, `scipy`
* **Machine Learning / Deep Learning**: `tensorflow` / `keras` or `pytorch`, `scikit-learn`
* **Visualization**: `matplotlib`, `seaborn`
* **Environment**: Jupyter Notebook

---

## Getting Started

### Prerequisites

* Python `3.8` or higher
* `pip` package manager
* An account on the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/) (required for running the data download notebook via `cdsapi`).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fawaztanigbola/Nigeria-Forecasting-Model.git
   cd Nigeria-Forecasting-Model
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

3. **Install required dependencies:**
   ```bash
   pip install --upgrade pip
   pip install jupyter pandas numpy scikit-learn matplotlib seaborn xarray netCDF4 cdsapi torch
   ```

4. **Configure Climate Data Store (CDS) API Key (Optional):**
   If you intend to download raw ERA5 data, set up your `.cdsapirc` file in your home directory:
   ```bash
   cat <<EOT > ~/.cdsapirc
   url: https://cds.climate.copernicus.eu/api/v2
   key: <YOUR-UID>:<YOUR-API-KEY>
   EOT
   ```

---

## Usage Guide

To reproduce the data processing pipeline or train the baseline model:

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. **Run notebooks in sequential order**:
   * **Data Collection**: Execute `ERA5_And_DEM_Dataset_Download.ipynb` to download ERA5 temperature reanalysis data and Digital Elevation Model (DEM) data over the geographic bounds of Nigeria.
   * **Data Processing**: 
     * Run `Combine_Yearly_Files.ipynb` to stitch annual climate datasets into a unified time series.
     * Run `Daily_Temperature_Preprocessing.ipynb` to handle missing values, resample daily metrics, and clean outliers.
   * **Feature Engineering & Splitting**:
     * Execute `Models_Data_Preparation.ipynb` to generate rolling windows/lagged features.
     * Execute `Chronologically_Splitting_Data.ipynb` to split data into train, validation, and testing sets chronologically.
   * **Model Training**: Run `Baseline_LSTM.ipynb` to train the LSTM neural network and output evaluation metrics (e.g., RMSE, MAE).

---

## License

This project is open-source and available under the [MIT License](LICENSE).