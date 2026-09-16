<div align="center">

# ⚡ Electricity Demand Forecasting System

### 📊 Data Engineering • 🤖 Machine Learning • 🔮 Multi-Month Forecasting • 📈 Verification

**An end-to-end electricity demand forecasting framework for Australian electricity regions**

<br>

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Forecasting-orange)
![Forecast Horizon](https://img.shields.io/badge/Forecast%20Horizon-Up%20to%204%20Months-brightgreen)
![Regions](https://img.shields.io/badge/Regions-6-blueviolet)
![Status](https://img.shields.io/badge/Status-Development-success)

</div>

---

## 🌏 Project Overview

The **Electricity Demand Forecasting System** is an end-to-end data science and machine learning project designed to forecast electricity demand across major Australian electricity regions.

The system integrates historical electricity demand, weather information, calendar effects, public holidays, temporal patterns, and engineered demand features to produce reliable regional forecasts.

A key capability of the system is its support for forecasting electricity demand **up to four months ahead**, allowing both shorter-term and extended demand planning scenarios to be analysed.

### 🗺️ Regions Covered

| Region | Code | Coverage |
|---|---|---|
| 🟦 New South Wales | `NSW1` | NEM |
| 🟩 Queensland | `QLD1` | NEM |
| 🟪 Victoria | `VIC1` | NEM |
| 🟨 South Australia | `SA1` | NEM |
| 🟥 Tasmania | `TAS1` | NEM |
| 🟧 Western Australia | `WA` | WEM |

Historical information from **2022–2026** supports model development, backtesting, forecast generation, and independent verification.

---

# 🔮 Forecasting Capability

<div align="center">

## ⚡ Forecast Horizon: Up to 4 Months

| Forecast Level | Purpose |
|---|---|
| ⏱️ 30-Minute | Detailed electricity demand behaviour |
| 📅 Daily | Daily demand tracking and comparison |
| 📆 Monthly | Long-term demand trend analysis |
| 🔮 Up to 4 Months | Extended electricity demand forecasting |

</div>

The forecasting pipeline can generate projections extending **up to four months beyond the forecasting origin**, depending on the selected modelling configuration and available predictor information.

This makes it possible to examine:

- 📈 Future electricity demand patterns
- ⚡ Expected peak-demand periods
- 📅 Daily and monthly demand behaviour
- 🌦️ Weather and seasonal influences
- 🗺️ Regional differences
- 📊 Forecast uncertainty and error behaviour
- 🔍 Actual-versus-forecast performance when observations become available

Forecast horizons are kept separate from historical training data to maintain chronological integrity and reduce the risk of information leakage.

---

# 🎯 Project Objectives

The project aims to:

- 📥 Acquire reliable electricity demand data from authoritative Australian sources.
- 🗃️ Preserve original source datasets before transformation.
- 🧹 Validate and clean electricity demand information.
- ⏱️ Standardise regional datasets for modelling.
- 🌦️ Integrate weather information.
- 📅 Include calendar and public holiday effects.
- 🧠 Engineer lag, rolling, temporal, and demand features.
- 🤖 Train and evaluate machine learning forecasting models.
- 🧪 Perform historical time-aware backtesting.
- 🔮 Generate forecasts extending up to **four months ahead**.
- ✅ Compare frozen forecasts with unseen actual demand.
- 📊 Deliver dashboard-ready forecasting outputs.
- 🔁 Maintain a reproducible modelling and verification workflow.

---

# 🏗️ System Architecture

The system follows a **four-stage architecture** covering the full lifecycle from raw data acquisition to operational monitoring and governance.

<p align="center">
  <img src="docs/images/system_architecture.png"
       alt="Electricity Demand Forecasting System Architecture"
       width="900">
</p>

---

## 📥 Stage 1 — Data Acquisition

Electricity demand is collected from **AEMO and WEM**, together with weather and calendar information.

Automated ingestion supports scheduled data retrieval while original datasets are retained as immutable raw snapshots.

**Main components:**

`Data Sources` → `Automated Ingestion` → `Raw Data Lake`

This stage maintains source provenance and allows historical datasets, corrections, and forecast vintages to remain traceable.

---

## 🧹 Stage 2 — Data Processing

Raw datasets are transformed into reliable modelling inputs.

The processing stage performs:

- ✅ Quality validation
- 🔍 Missing-value and gap detection
- 📉 Outlier assessment
- ⏱️ Time alignment
- 🌦️ Weather integration
- 📅 Calendar integration
- 🧠 Feature engineering

Regional information is ultimately organised into a curated store containing:

> **Demand + Features + Forecasts + Actual Observations**

---

## 🤖 Stage 3 — Model Operations

The model operations layer manages model development, evaluation, forecasting, and verification.

```text
Model Registry
      ↓
Historical Backtesting
      ↓
Model Approval
      ↓
Forecast Engine
      ↓
Up-to-4-Month Forecast
      ↓
Actual vs Forecast Verification
