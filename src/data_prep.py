import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TEST_SIZE = params["data"]["test_size"]
RANDOM_STATE = params["base"]["random_state"]

# Input & output paths
RAW_DATA_PATH = "data/raw/house_price_prediction.csv"
PROCESSED_DIR = "data/processed"
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

def main():
    df = pd.read_csv(RAW_DATA_PATH)

    # Basic cleaning: drop duplicates, handle missing values
    df = df.drop_duplicates()
    df = df.dropna()  # keep it simple

    # Train-test split
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Data split: {train_df.shape} train, {test_df.shape} test")

if __name__ == "__main__":
    main()
