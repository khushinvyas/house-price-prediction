import pandas as pd
import yaml
import os
from sklearn.preprocessing import StandardScaler

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TARGET = params["base"]["target_col"]

PROCESSED_DIR = "data/processed"
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
TRAIN_FEAT_PATH = os.path.join(PROCESSED_DIR, "train_features.csv")
TEST_FEAT_PATH = os.path.join(PROCESSED_DIR, "test_features.csv")

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Separate features and target
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # Simple feature scaling (numeric only)
    numeric_cols = X_train.select_dtypes(include="number").columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Save processed features
    train_features = pd.concat([X_train, y_train], axis=1)
    test_features = pd.concat([X_test, y_test], axis=1)

    train_features.to_csv(TRAIN_FEAT_PATH, index=False)
    test_features.to_csv(TEST_FEAT_PATH, index=False)

    print("Feature engineering completed")

if __name__ == "__main__":
    main()
