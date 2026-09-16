<div align="center">

# ⚡ ELECTRICITY DEMAND FORECASTING SYSTEM

### 📊 Data Engineering • 🤖 Machine Learning • 🔮 Multi-Month Forecasting • ✅ Verification • 📡 Monitoring

**An end-to-end electricity demand forecasting framework for Australian electricity regions**

<br>

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Forecast](https://img.shields.io/badge/Forecast-Up%20to%204%20Months-brightgreen)
![Regions](https://img.shields.io/badge/Regions-6-blueviolet)
![Data](https://img.shields.io/badge/Data-2022--2026-orange)
![Status](https://img.shields.io/badge/Status-Development-success)

<br>

### ⚡ FORECAST HORIZON: **UP TO 4 MONTHS**

</div>

---

# 🌏 PROJECT OVERVIEW

The **Electricity Demand Forecasting System** is an end-to-end forecasting framework developed to analyse and predict electricity demand across major Australian electricity regions.

The system integrates:

⚡ Historical electricity demand  
🌦️ Weather information  
📅 Calendar and public holiday effects  
🕐 Time-based features  
🧠 Demand lag and rolling features  
🤖 Machine learning models  
🧪 Historical backtesting  
🔮 Forecast generation  
✅ Forecast verification  
📊 Dashboard monitoring  
🛡️ Governance and reproducibility  

The forecasting framework supports demand forecasts extending **up to four months ahead**, allowing both detailed short-term analysis and broader multi-month electricity demand planning.

---

# 🗺️ REGIONS COVERED

| Region | Code | Market |
|---|---|---|
| 🟦 New South Wales | `NSW1` | NEM |
| 🟩 Queensland | `QLD1` | NEM |
| 🟪 Victoria | `VIC1` | NEM |
| 🟨 South Australia | `SA1` | NEM |
| 🟥 Tasmania | `TAS1` | NEM |
| 🟧 Western Australia | `WA` | WEM |

Historical information from **2022–2026** supports model development, backtesting, forecast generation and independent verification.

---

# 🔮 FORECASTING CAPABILITY

<div align="center">

## ⚡ UP TO FOUR-MONTH ELECTRICITY DEMAND FORECASTING

| Forecast Level | Application |
|---|---|
| ⏱️ **30-Minute** | Detailed electricity demand patterns |
| 📅 **Daily** | Daily demand behaviour and peak analysis |
| 📆 **Monthly** | Long-term trend analysis |
| 🔮 **Up to 4 Months** | Extended demand forecasting |

</div>

The forecasting engine is designed to generate electricity demand projections extending **up to four months beyond the forecast origin**.

This enables analysis of:

- ⚡ Expected electricity demand
- 📈 Peak-demand periods
- 📅 Daily demand patterns
- 📆 Monthly demand trends
- 🌦️ Weather-related demand behaviour
- 🗺️ Regional differences
- 📉 Forecast error behaviour
- ✅ Actual-versus-forecast performance

Forecast periods are separated chronologically from model training data to reduce the risk of **future-data leakage**.

---

# 🎯 PROJECT OBJECTIVES

The project aims to:

- 📥 Acquire electricity demand data from authoritative Australian sources.
- 🗃️ Preserve original raw datasets before transformation.
- ✅ Validate data quality and identify missing observations.
- 🧹 Clean and standardise regional datasets.
- ⏱️ Align datasets to a common modelling framework.
- 🌦️ Integrate weather information.
- 📅 Include calendar and public holiday effects.
- 🧠 Engineer lag, rolling and temporal features.
- 🤖 Develop machine learning forecasting models.
- 🧪 Perform historical time-aware backtesting.
- 🔮 Generate electricity forecasts up to **four months ahead**.
- ✅ Compare frozen forecasts against unseen actual demand.
- 📊 Produce dashboard-ready outputs.
- 📡 Monitor forecast performance.
- 🛡️ Maintain model lineage and reproducibility.

---

# 🏗️ SYSTEM ARCHITECTURE

The forecasting system follows a **four-stage architecture** covering the complete lifecycle from source data acquisition to operational monitoring and governance.

<p align="center">
  <img src="docs/images/system_architecture.png"
       alt="Electricity Demand Forecasting System Architecture"
       width="950">
</p>

---

# 📥 STAGE 1 — DATA ACQUISITION

The first stage collects electricity demand, weather and calendar information required by the forecasting system.

### Core Components

```text
📡 Data Sources
AEMO • WEM • Weather • Calendar
             ↓
⚙️ Automated Ingestion
Scheduled Pulls • Correction Listener
             ↓
🗃️ Raw Data Lake
Immutable Snapshots • Forecast Vintages
