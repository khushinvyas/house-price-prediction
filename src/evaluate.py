import pandas as pd
import yaml
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

MODEL_PATH = "model.pkl"
PROCESSED_DIR = "data/processed"
TEST_FEAT_PATH = os.path.join(PROCESSED_DIR, "test_features.csv")
METRICS_PATH = "metrics.json"

def main():
    # Load data
    df = pd.read_csv(TEST_FEAT_PATH)
    X_test = df.drop(columns=[params["base"]["target_col"]])
    y_test = df[params["base"]["target_col"]]

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    # Save metrics to JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation metrics:", metrics)

if __name__ == "__main__":
    main()
