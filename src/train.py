import pandas as pd
import yaml
import pickle
import os
from sklearn.ensemble import RandomForestRegressor

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

MODEL_DIR = "model.pkl"
PROCESSED_DIR = "data/processed"
TRAIN_FEAT_PATH = os.path.join(PROCESSED_DIR, "train_features.csv")

def main():
    df = pd.read_csv(TRAIN_FEAT_PATH)

    X = df.drop(columns=[params["base"]["target_col"]])
    y = df[params["base"]["target_col"]]

    model = RandomForestRegressor(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        random_state=params["base"]["random_state"],
    )

    model.fit(X, y)

    # Save model
    with open(MODEL_DIR, "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained and saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
