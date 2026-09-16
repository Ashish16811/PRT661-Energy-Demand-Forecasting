<div align="center">

# ⚡ ELECTRICITY DEMAND FORECASTING SYSTEM

### Australian Regional Electricity Demand Forecasting & Verification

**📊 Data Analytics • 🤖 Machine Learning • 🔮 Multi-Month Forecasting • ✅ Verification**

<br>

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Forecast-Up%20to%204%20Months-2EA043?style=for-the-badge&logo=googleanalytics&logoColor=white" alt="Forecast">
<img src="https://img.shields.io/badge/Regions-6-7C3AED?style=for-the-badge&logo=googlemaps&logoColor=white" alt="Regions">
<img src="https://img.shields.io/badge/Data-2022--2026-EF6C00?style=for-the-badge&logo=databricks&logoColor=white" alt="Data">
<img src="https://img.shields.io/badge/Status-Development-00A86B?style=for-the-badge&logo=github&logoColor=white" alt="Status">

<br><br>

## 🔮 FORECAST HORIZON — UP TO 4 MONTHS

### ⚡ 30-Minute • 📅 Daily • 📆 Monthly • 🔮 Multi-Month Forecasting

</div>

---

# 🌏 PROJECT OVERVIEW

The **Electricity Demand Forecasting System** is an end-to-end data analytics and machine learning project developed to forecast electricity demand across major Australian electricity regions.

The system combines historical electricity demand with **weather, calendar, public holiday, temporal and historical demand features** to generate regional forecasts.

A major capability of the project is the ability to produce electricity demand forecasts extending **up to four months ahead**, while also evaluating model performance through historical backtesting and comparison with unseen actual demand.

### ⚡ Core Capabilities

- 📥 Electricity demand data acquisition
- 🧹 Data validation and preprocessing
- 🧠 Feature engineering
- 🤖 Machine learning forecasting
- 🧪 Historical backtesting
- 🔮 Forecasting up to four months ahead
- ✅ Actual-versus-forecast verification
- 📊 Regional performance analysis
- 📡 Forecast monitoring
- 🌏 Multi-region comparison

---

# 🗺️ REGIONS COVERED

| Region | Code | Electricity Market |
|---|---|---|
| 🟦 **New South Wales** | `NSW1` | NEM |
| 🟩 **Queensland** | `QLD1` | NEM |
| 🟪 **Victoria** | `VIC1` | NEM |
| 🟨 **South Australia** | `SA1` | NEM |
| 🟥 **Tasmania** | `TAS1` | NEM |
| 🟧 **Western Australia** | `WA` | WEM |

Historical electricity demand information from **2022–2026** supports model development, backtesting and forecast verification.

---

# 🔮 FORECASTING CAPABILITY

<div align="center">

## ⚡ UP TO FOUR-MONTH ELECTRICITY DEMAND FORECASTING

| Forecast Level | Application |
|---|---|
| ⏱️ **30-Minute** | Detailed electricity demand behaviour |
| 📅 **Daily** | Daily demand and peak analysis |
| 📆 **Monthly** | Longer-term demand trends |
| 🔮 **Up to 4 Months** | Extended regional demand forecasting |

</div>

The forecasting system supports projections extending **up to four months beyond the forecast origin**.

This allows the project to examine:

- ⚡ Future electricity demand
- 📈 Peak-demand periods
- 📅 Daily consumption patterns
- 📆 Monthly demand trends
- 🌦️ Weather-related changes
- 🗺️ Regional differences
- 📉 Forecast accuracy
- ⚖️ Forecast bias
- ✅ Actual-versus-forecast performance

Forecast periods remain chronologically separated from model training periods to support realistic evaluation.

---

# 🎯 PROJECT OBJECTIVES

The project aims to:

- 📥 Acquire reliable electricity demand data from Australian electricity-market sources.
- 🧹 Prepare clean and consistent regional datasets.
- 🌦️ Integrate weather information with electricity demand.
- 📅 Include calendar and public holiday effects.
- 🧠 Develop meaningful predictive features.
- 🤖 Build regional machine learning forecasting models.
- 🧪 Evaluate models through time-aware historical backtesting.
- 🔮 Generate forecasts extending up to four months ahead.
- ✅ Compare forecasts with unseen actual demand.
- 📊 Present forecasting results through clear regional visualisations and dashboards.
- 📡 Monitor forecasting performance as new actual observations become available.

---

# 🏗️ SYSTEM ARCHITECTURE

The forecasting system follows a **four-stage architecture** covering data acquisition, processing, model operations and forecast monitoring.

<p align="center">
  <img src="docs/images/system_architecture.png"
       alt="Electricity Demand Forecasting System Architecture"
       width="1000">
</p>

---

# 📥 STAGE 1 — DATA ACQUISITION

Stage 1 collects the information required by the forecasting system.

### Main Inputs

- ⚡ **AEMO electricity demand data**
- ⚡ **WEM electricity demand data**
- 🌦️ **Weather information**
- 📅 **Calendar information**
- 🎉 **Public holiday information**

Original source datasets are retained before processing so that the modelling workflow can always be traced back to the source information.

---

# 🧹 STAGE 2 — DATA PROCESSING

Stage 2 prepares the raw information for modelling.

The processing stage performs:

- ✅ Data quality validation
- ❓ Missing-value checking
- 🔁 Duplicate checking
- 📉 Outlier assessment
- ⏱️ Timestamp alignment
- 🌦️ Weather integration
- 📅 Calendar integration
- 🧠 Feature engineering

Where required, regional datasets are aligned to a consistent **30-minute modelling interval**.

The result is a clean and consistent modelling dataset for each electricity region.

---

# 🧠 FEATURE ENGINEERING

Feature engineering transforms the cleaned electricity demand data into predictive inputs for machine learning models.

### ⏱️ Time Features

`hour` • `day_of_week` • `month` • `week_of_year` • `weekend`

### 📅 Calendar Features

`public_holiday` • `working_day` • `season`

### ⚡ Demand Features

`lag demand` • `previous day demand` • `previous week demand`

### 📊 Rolling Features

`rolling mean` • `rolling minimum` • `rolling maximum` • `rolling standard deviation`

### 🌦️ Weather Features

`temperature` • `humidity` • `wind speed` • `rainfall` • `apparent temperature`

These features help the models capture short-term, weekly, seasonal and weather-sensitive electricity demand behaviour.

---

# 🤖 STAGE 3 — MODEL OPERATIONS

Stage 3 develops, evaluates and applies the forecasting models.

Current model experiments include:

```text
RandomForest_A
RandomForest_B
```

Models are evaluated region-by-region using historical time-aware validation rather than a random train-test split.

### Model Assessment Considers

- 📉 Forecast accuracy
- 📊 Performance stability
- ⚡ Peak-demand behaviour
- ⚖️ Forecast bias
- 🌏 Regional consistency
- 🔄 Performance across multiple historical periods

The selected model is then used to produce regional electricity demand forecasts extending up to **four months ahead**.

---

# 🧪 HISTORICAL BACKTESTING

Historical backtesting tests how forecasting models would have performed on previously unseen historical periods.

| Validation Level | Purpose |
|---|---|
| ⏱️ **30-Minute** | Detailed interval-level accuracy |
| 📅 **Daily** | Daily forecast stability |
| 📆 **Monthly** | Longer-term performance |
| 🔮 **Four-Month Window** | Extended forecast evaluation |

Backtesting provides stronger model evidence than relying on a single validation period.

It also allows the project to compare model behaviour under different seasonal and demand conditions.

---

# 📏 FORECAST EVALUATION

The forecasting models are evaluated using multiple complementary metrics.

| Metric | Purpose |
|---|---|
| 📉 **MAE** | Average absolute forecast error |
| 📐 **RMSE** | Emphasises larger forecasting errors |
| 📊 **sMAPE** | Percentage-based forecast error |
| ⚖️ **Bias** | Identifies systematic over or under prediction |

### Mean Absolute Error

```text
MAE = mean(|Actual - Forecast|)
```

### Forecast Bias

```text
Bias = mean(Forecast - Actual)
```

### Normalised MAE

```text
Normalised MAE = MAE / Mean Demand
```

Using several metrics provides a broader understanding of model performance across regions with different electricity-demand scales.

---

# ✅ FORECAST VERIFICATION

Forecast verification compares previously generated forecasts with actual electricity demand observations that were excluded from model training.

<div align="center">

### 🔮 Frozen Forecast + ⚡ Actual Demand → ✅ Forecast Verification

</div>

Verification evaluates:

- 📉 MAE
- 📐 RMSE
- 📊 sMAPE
- ⚖️ Forecast bias
- ⚡ Error relative to average regional demand
- 🔄 Difference between historical backtesting and unseen forecast performance

This provides an independent indication of how the forecasting models behave under real unseen demand conditions.

---

# 📡 STAGE 4 — MONITORING & FEEDBACK

Stage 4 monitors forecast performance as new electricity demand observations become available.

Monitoring focuses on:

- 📉 Forecast error changes
- ⚖️ Forecast bias
- ⚡ Peak-demand forecasting
- 📆 Monthly performance
- 🌏 Regional differences
- 🔍 Changes in model performance
- 🔄 Differences between backtesting and actual forecast results

Monitoring results can then be used to identify areas where models may require improvement or retraining.

---

# 📈 FORECAST VISUALISATION

The project generates regional visualisations to support interpretation of forecast performance.

Typical outputs include:

### 📈 Actual vs Forecast

Shows how closely predicted electricity demand follows observed demand.

### 📅 Daily Demand Comparison

Provides a clearer view of daily forecast behaviour.

### 📆 Monthly Forecast Performance

Shows how forecasting accuracy changes across longer periods.

### 📉 Rolling Forecast Error

Tracks whether model accuracy improves or deteriorates through time.

### 🔵 Actual vs Predicted Scatter

Shows the relationship between forecast and observed electricity demand.

### ⚖️ Error Analysis

Identifies systematic over-prediction or under-prediction.

---

# 📊 REGIONAL DASHBOARD

The final forecasting results are prepared for an interactive regional dashboard.

### Dashboard Capabilities

- 🌏 Region selection
- ⚡ Historical electricity demand
- 🔮 Future electricity demand forecasts
- 📆 Forecast horizon up to four months
- 📈 Actual-versus-forecast comparison
- 📅 Daily demand analysis
- 📆 Monthly demand analysis
- 📉 MAE
- 📐 RMSE
- 📊 sMAPE
- ⚖️ Forecast bias
- 🤖 Model comparison
- ✅ Forecast verification
- 📡 Performance monitoring

The same dashboard structure can be used across all supported electricity regions.

---

# 🛠️ TECHNOLOGIES

<div align="center">

| Technology | Use |
|---|---|
| 🐍 **Python** | Forecasting pipeline |
| 🐼 **Pandas** | Data processing |
| 🔢 **NumPy** | Numerical analysis |
| 🤖 **Scikit-learn** | Machine learning |
| 📈 **Matplotlib** | Data visualisation |
| 💾 **Joblib** | Model storage |
| 🌿 **Git** | Version control |
| 🐙 **GitHub** | Collaboration |
| 💻 **Visual Studio Code** | Development |

</div>

---

# 🎓 PROJECT OUTCOME

The project demonstrates the integration of:

<div align="center">

## 📥 DATA ACQUISITION  
### ↓
## 🧹 DATA PROCESSING  
### ↓
## 🧠 FEATURE ENGINEERING  
### ↓
## 🤖 MACHINE LEARNING  
### ↓
## 🔮 FORECASTING  
### ↓
## ✅ VERIFICATION  
### ↓
## 📡 MONITORING

</div>

The final system provides a structured approach for transforming historical Australian electricity demand data into **regional forecasts extending up to four months ahead**, while maintaining evidence through backtesting, forecast verification and performance monitoring.

---

# 👥 PROJECT COLLABORATION

The project is collaboratively developed using **Git and GitHub**.

Team contributions are demonstrated through:

- 🌿 Feature branches
- 💾 Git commits
- 🔀 Pull requests
- 👀 Code reviews
- 📥 Data acquisition
- 🧹 Data preparation
- 🧠 Feature engineering
- 🤖 Model development
- 🧪 Backtesting
- ✅ Forecast verification
- 📊 Dashboard development
- 📝 Documentation

---

# 👨‍💻 CONTRIBUTORS

| Contributor | Contribution |
|---|---|
| 👤 **Aashish Shrestha** | Data / Modelling / Documentation |
| 👤 **Bishal Dahal** | Project Development |
| 👤 **Group Member 3** | Project Development |
| 👤 **Group Member 4** | Project Development |

---

<div align="center">

# ⚡ ELECTRICITY DEMAND FORECASTING SYSTEM

## 🔮 FORECAST HORIZON — UP TO 4 MONTHS

<img src="https://img.shields.io/badge/Forecast-Up%20to%204%20Months-2EA043?style=for-the-badge&logo=googleanalytics&logoColor=white">
<img src="https://img.shields.io/badge/Australia-6%20Regions-0072B1?style=for-the-badge&logo=googlemaps&logoColor=white">

<br><br>

### 📥 Acquire → 🧹 Process → 🧠 Engineer → 🤖 Model → 🔮 Forecast → ✅ Verify → 📡 Monitor

**Australian Regional Electricity Demand Forecasting & Verification**

</div>
