# House Price Prediction MLOps Project

This project implements a machine learning pipeline for predicting house prices using DVC (Data Version Control) for MLOps.

## Project Structure

```
house-price-prediction/
│
├── data/
│   ├── raw/            ← Original dataset CSV(s)
│   └── processed/      ← Cleaned, feature-engineered data
│
├── src/
│   ├── data_prep.py    ← Scripts for loading & cleaning
│   ├── features.py     ← Feature engineering
│   ├── train.py        ← Training script
│   └── evaluate.py     ← Evaluation & metrics
│
├── params.yaml         ← Hyperparameters, config
├── dvc.yaml           ← Pipeline stages
├── dvc.lock           ← (auto-generated)
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize DVC:
```bash
dvc init
```

4. Add your dataset:
- Place your house prices dataset in `data/raw/house_prices.csv`
- Add it to DVC:
```bash
dvc add data/raw/house_prices.csv
```

## Running the Pipeline

Run the entire pipeline:
```bash
dvc repro
```

To run specific stages:
```bash
dvc repro <stage-name>
```

## Pipeline Stages

1. `data_prep`: Clean and preprocess raw data
2. `feature_engineering`: Create and transform features
3. `train`: Train the model with parameters from params.yaml
4. `evaluate`: Evaluate model performance and generate metrics

## Metrics

View the latest metrics:
```bash
dvc metrics show
```

## Parameters

Adjust model parameters in `params.yaml`.