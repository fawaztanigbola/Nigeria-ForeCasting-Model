# Nigeria Weather Forecasting Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Domain](https://img.shields.io/badge/Domain-Climate%20%26%20Weather%20Forecasting-green.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An end-to-end **spatio-temporal data processing and deep learning project** developed to investigate and forecast daily temperature patterns across Nigeria, with a focused regional implementation covering Southwest Nigeria.

The project combines historical climate reanalysis data from **ERA5**, topographical elevation information derived from Digital Elevation Models (DEM), and Recurrent Neural Networks (LSTM) to investigate temperature forecasting across multiple geographical locations.

Beginning with a broader Nigeria-wide modelling objective, the project evolved into a regional forecasting study covering Lagos, Ogun, Oyo and Osun. This adjustment was made to accommodate the memory and training-time constraints of the available Google Colab computing environment while retaining a meaningful multi-location forecasting problem.

---

## 📋 Table of Contents

- [Project Overview](#overview)
- [Key Contributions](#key-contributions)
- [Repository Structure](#repository-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Tech Stack](#tech-stack)
- [Dataset and Study Area](#dataset-and-study-area)
- [Model Development](#model-development)
- [Model Performance](#model-performance)
- [Key Findings](#key-findings)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [License](#license)

---

## Overview

Temperature forecasting plays an important role in climate analysis, agricultural planning, environmental monitoring and regional decision-making.

This project explores how deep learning can be applied to historical meteorological data to predict daily minimum and maximum temperatures across multiple geographical grid locations.

The development process involved several stages:

1. **Climate Data Acquisition:** Retrieval and preparation of historical ERA5 temperature data and geographical elevation information.
2. **Multi-Year Data Integration:** Combining yearly climate datasets into a continuous time series covering 2015–2024.
3. **Data Preprocessing:** Handling missing observations, preparing daily temperature variables and normalizing model inputs.
4. **Temporal Data Preparation:** Constructing chronological training sequences using sliding windows.
5. **Baseline Model Development:** Developing an LSTM model using historical temperature observations.
6. **Geospatial Model Development:** Extending the temporal forecasting approach with seasonal and geographical features.
7. **Model Evaluation:** Comparing forecasting performance against persistence and climatology reference methods.

The project provides an applied exploration of temporal forecasting and the potential contribution of geographical information to temperature prediction.

---

## Key Contributions

- **Multi-Year Climate Data Processing:** Prepared ten years of daily temperature observations from ERA5 reanalysis data.
- **Geospatial Data Integration:** Incorporated elevation information alongside latitude and longitude for geographical modelling experiments.
- **Temporal Feature Engineering:** Developed sliding-window sequences and cyclical day-of-year features to represent seasonal patterns.
- **Deep Learning Implementation:** Built two LSTM-based temperature forecasting models using PyTorch.
- **Geospatial Feature Investigation:** Evaluated whether static geographical information improved forecasting performance.
- **Forecast Benchmarking:** Compared model performance against persistence and day-of-year climatology reference forecasts.
- **Spatial Error Analysis:** Investigated forecasting errors across geographical grid locations.

---

## Repository Structure

```text
├── Baseline_LSTM.ipynb
│   └── Temperature-only LSTM forecasting model
│
├── Chronologically_Splitting_Data.ipynb
│   └── Chronological data partitioning
│
├── Combine_Yearly_Files.ipynb
│   └── Integration of yearly climate datasets
│
├── Daily_Temperature_Preprocessing.ipynb
│   └── Daily temperature cleaning and preparation
│
├── ERA5_And_DEM_Dataset_Download.ipynb
│   └── ERA5 climate and elevation data acquisition
│
├── Geospatial_Model.ipynb
│   └── Geospatial LSTM development and experiments
│
├── Models_Data_Preparation.ipynb
│   └── Feature engineering and sequence preparation
│
├── Southwest_Nigeria_Temperature_Project_Documentation.ipynb
│   └── Consolidated project findings and technical documentation
│
└── README.md
    └── Project overview and results
```

---

## Pipeline Architecture

The project follows a sequential data processing and model development workflow.

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
| **Programming Language** | Python |
| **Climate Data** | ERA5 Reanalysis |
| **Data Ingestion** | `cdsapi`, `xarray`, `netCDF4` |
| **Data Manipulation** | `pandas`, `numpy`, `scipy` |
| **Deep Learning** | PyTorch |
| **Machine Learning Utilities** | `scikit-learn` |
| **Spatial Analysis** | `geopandas`, `rasterio`, `shapely` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Development Environment** | Jupyter Notebook, Google Colab |
| **GPU Acceleration** | NVIDIA Tesla T4 |

---

## Dataset and Study Area

The project initially targeted a broader geographical coverage across Nigeria. However, the volume of data required for country-wide modelling placed considerable demands on the available Google Colab RAM and computational resources.

Training across the full geographical dataset would also have increased training time substantially, making repeated experimentation and model validation difficult within the available environment.

To make model development and experimentation feasible, the geographical scope was reduced to **Southwest Nigeria**, covering:

- Lagos
- Ogun
- Oyo
- Osun

This regional selection retained multiple geographical grid locations and allowed the project to investigate temperature forecasting across a meaningful spatial area.

### Dataset Summary

| Property | Description |
| :--- | :--- |
| Climate dataset | ERA5 |
| Study period | 2015–2024 |
| Temporal resolution | Daily |
| Daily observations | 3,653 |
| Original regional grid | 672 cells |
| Usable grid locations | 456 |
| Target variables | Minimum and maximum temperature |
| Geographical features | Latitude, longitude and elevation |

The prepared dataset provided a ten-year historical foundation for temporal and geospatial forecasting experiments.

---

## Model Development

Two LSTM-based approaches were developed to investigate next-day temperature forecasting.

### 1. Temperature-only Baseline LSTM

The baseline model was developed to establish how effectively historical temperature observations could predict future temperature values without additional geographical inputs.

**Input features:**
- Previous seven days of minimum temperature.
- Previous seven days of maximum temperature.

**Prediction targets:**
- Next-day minimum temperature.
- Next-day maximum temperature.

**Architecture:**
- Single-layer LSTM.
- Hidden size of 64.
- Two-output prediction layer.
- Adam optimizer.
- Mean Squared Error loss.
- Early stopping and validation-based checkpoint selection.

The baseline established a reference deep learning model against which the subsequent geospatial experiments could be investigated.

### 2. Geospatial LSTM

The second model extended the temporal forecasting architecture by incorporating seasonal and geographical information.

**Temporal inputs:**
- Minimum temperature.
- Maximum temperature.
- Day-of-year sine.
- Day-of-year cosine.

**Static geographical inputs:**
- Latitude.
- Longitude.
- Elevation.

The LSTM processed the temporal sequence, after which the geographical features were concatenated with its final hidden representation before the prediction head.

This architecture allowed the model to consider both recent temperature history and the geographical characteristics of each grid location.

### Geospatial Experiments

Three main investigations were conducted:

| Experiment | Investigation | Finding |
| :--- | :--- | :--- |
| Input-window comparison | 7, 14 and 30 days | 7-day window performed best on validation |
| Model configuration | LSTM layers and dropout | One layer without dropout performed best |
| Geographical feature ablation | With and without latitude, longitude and elevation | Static features did not improve test metrics |

These experiments helped identify a suitable model configuration and investigate the contribution of geographical information rather than assuming that adding more features would automatically improve forecasting.

---

## Model Performance

The models were evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

Both metrics are expressed in degrees Celsius. Lower values indicate smaller forecasting errors.

### Temperature-only Baseline LSTM

| Target | MAE | RMSE |
| :--- | ---: | ---: |
| Minimum temperature (Tmin) | 0.502°C | 0.690°C |
| Maximum temperature (Tmax) | 0.988°C | 1.324°C |

### Geospatial LSTM

| Target | MAE | RMSE |
| :--- | ---: | ---: |
| Minimum temperature (Tmin) | 0.535°C | 0.707°C |
| Maximum temperature (Tmax) | 1.021°C | 1.346°C |

### Forecast Skill Against Reference Models

To understand the practical forecasting improvement, model errors were compared with simpler reference forecasts.

Forecast skill was calculated as:

\[
\text{Skill Score} =
\left(1-\frac{\text{Model Error}}{\text{Reference Error}}\right)\times100
\]

A positive score indicates a percentage reduction in error relative to the reference method.

**Baseline LSTM compared with persistence:**

| Target | MAE Error Reduction | RMSE Error Reduction |
| :--- | ---: | ---: |
| Tmin | 12.85% | 11.31% |
| Tmax | 11.55% | 16.94% |

**Geospatial LSTM compared with persistence:**

| Target | MAE Error Reduction | RMSE Error Reduction |
| :--- | ---: | ---: |
| Tmin | 8.23% | 9.24% |
| Tmax | 10.75% | 17.73% |

**Geospatial LSTM compared with day-of-year climatology:**

| Target | MAE Error Reduction | RMSE Error Reduction |
| :--- | ---: | ---: |
| Tmin | 41.34% | 41.43% |
| Tmax | 25.91% | 22.24% |

These results demonstrate that both LSTM approaches learned useful forecasting patterns beyond persistence. The geospatial model also achieved substantial error reductions compared with the climatology reference on its test period.

*Note: These are relative error reductions, not conventional prediction accuracy percentages. The baseline and geospatial notebooks use different chronological splits and preprocessing, so their direct comparison is indicative rather than a controlled head-to-head evaluation.*

---

## Key Findings

The experiments produced several important findings:

1. **LSTM-based temperature forecasting was effective against simple references.** The temperature-only baseline reduced test errors compared with persistence for both minimum and maximum temperature.

2. **Seasonal information was useful to the geospatial modelling approach.** The geospatial model incorporated cyclical day-of-year features alongside recent temperature observations.

3. **The geospatial model outperformed climatology and persistence on its test period.** This demonstrates that the model learned patterns beyond these simple forecasting references.

4. **A seven-day historical window was sufficient among the tested options.** Extending the input history to 14 or 30 days did not improve validation performance in this experiment.

5. **Adding geographical features did not automatically improve generalisation.** Latitude, longitude and elevation did not improve the geospatial model's test metrics in the feature-ablation experiment.

6. **The temperature-only model achieved the lowest reported test errors.** The geospatial model's additional inputs did not translate into better numerical performance in the current implementation.

7. **The project established a foundation for richer spatial modelling.** The data preparation, temporal modelling and geographical integration workflows provide a starting point for future experiments.

---

## Limitations and Future Improvements

### 1. Geographical Coverage and Computing Resources

The initial objective was to investigate temperature forecasting across a much larger geographical area covering Nigeria.

However, processing and training on the full dataset would have placed substantial demands on the available Google Colab RAM and training time. This would have made repeated model training, validation and experimentation impractical.

The project therefore adopted a smaller regional scope covering Lagos, Ogun, Oyo and Osun.

Future work could revisit nationwide coverage using higher-memory computing environments, more efficient data pipelines, spatial tiling and distributed training.

### 2. Geographical Feature Representation

The current geospatial model incorporated latitude, longitude and elevation as static features.

Although these features provided geographical context, they did not improve test-set performance in the current experiment.

Future investigations could explore neighbouring-grid information, richer spatial representations and additional environmental variables.

### 3. Model Evaluation

The baseline and geospatial models were trained using different chronological data splits and scaling approaches.

A controlled comparison using identical data partitions and preprocessing would provide a stronger basis for determining the contribution of geographical features.

### 4. Spatial and Temporal Resolution

The current implementation uses ERA5 grid-level data and predicts next-day temperatures.

It does not yet provide street-level forecasting or capture every local environmental influence on temperature.

Future work could investigate finer spatial datasets and additional meteorological variables where data availability and computing resources permit.

### 5. Generalisation

The models were developed for a selected region in Southwest Nigeria. Their performance should not be assumed to transfer directly to other Nigerian regions without additional training and evaluation.

---

## License

This project is licensed under the [MIT License](LICENSE).
