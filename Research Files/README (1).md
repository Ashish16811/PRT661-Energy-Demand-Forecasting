# 📊 Energy Demand Forecasting System

A deep learning-based system to predict **next-day electricity demand** using a hybrid **CNN + LSTM** model. The project includes data preprocessing, model training with Optuna-based hyperparameter tuning, and deployment-ready packaging using **TorchServe**.

---

## 🚀 Project Overview

This project simulates a real-world energy forecasting pipeline with:
- CNN for **local feature extraction**
- LSTM for **temporal sequence modeling**
- Full **data preprocessing and cleaning**
- Hyperparameter tuning using **Optuna**
- Deployment-ready export using **TorchServe**

---

## 🧠 Problem Statement

Forecast electricity demand for the **next 24 hours** based on hourly consumption data from the **previous 24 hours**. This helps energy distribution systems prepare for demand fluctuations.

---

## 🏗️ Architecture

- **CNN (1D)**: Captures short-term patterns
- **LSTM**: Models long-term dependencies
- **Optuna**: Optimizes dropout, filters, learning rate, and hidden size
- **TorchServe**: For scalable API-based inference

---

## 🗂️ Project Structure
Energy_Demand_Forecasting_System/

├── data/

│   └── processed/                # Cleaned dataset

├── models/

│   ├── cnn_lstm_model.pth        # Trained model

│   ├── cnn_lstm_model_optimized.pt  # Traced model for TorchServe

├── model_store/

│   └── cnn_lstm_model.mar        # TorchServe-ready model archive

├── scripts/

│   ├── fetch_data.py             # EIA API integration

│   ├── preprocess_data.py        # Data cleaning + transformation

│   ├── train_model.py            # Model training + export

│   ├── tune_hyperparameters.py   # Optuna-based tuning

│   └── handler.py                # TorchServe custom handler

---

## 📈 Results

✅ Achieved ~96% accuracy on normalized input.

---

## 💻 Getting Started

### 1. Clone the Repository
git clone https://github.com/your-username/Energy_Demand_Forecasting_System.git
cd Energy_Demand_Forecasting_System

### 2. Setup Environment
python -m venv energy_venv
energy_venv\Scripts\activate  # On Windows
pip install -r requirements.txt

### 3. Fetch and Preprocess Data
python scripts/fetch_data.py

### 4. Train the Model
python scripts/train_model.py

### 5. Package the Model
torch-model-archiver \
  --model-name cnn_lstm_model \
  --version 1.0 \
  --serialized-file models/cnn_lstm_model_optimized.pt \
  --handler scripts/handler.py \
  --export-path model_store

### 6. Start TorchServe
torchserve --start --model-store model_store --models cnn_lstm_model=cnn_lstm_model.mar

## 📬 API Usage

### ➤ Endpoint
POST http://127.0.0.1:8080/predictions/cnn_lstm_model

### ➤ Headers
Content-Type: application/json

### ➤ Sample Input
[0.02, 0.04, 0.01, 0.03, 0.09, 0.1, 0.12, 0.2,
 0.15, 0.13, 0.07, 0.09, 0.06, 0.1, 0.18, 0.21,
 0.19, 0.23, 0.22, 0.25, 0.24, 0.29, 0.28, 0.3]

## 📦 Requirements

All required packages are listed in `requirements.txt`.

### 🔧 Key Dependencies
- `torch`
- `optuna`
- `pandas`
- `requests`
- `torchserve`
- `torch-model-archiver`

### 📥 Install with:
```bash
pip install -r requirements.txt


