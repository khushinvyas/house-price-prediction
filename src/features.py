import pandas as pd
import yaml
import os
import numpy as np
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

def create_features(df):
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Add new features
    df['age'] = 2023 - df['yr_built']
    df['renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    df['total_sqft'] = df['sqft_living'] + df['sqft_basement']
    df['price_per_sqft'] = df['total_sqft'] / df['sqft_lot']
    df['rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Log transform for skewed features
    df['sqft_living'] = np.log1p(df['sqft_living'])
    df['sqft_lot'] = np.log1p(df['sqft_lot'])
    df['sqft_above'] = np.log1p(df['sqft_above'])
    df['sqft_basement'] = np.log1p(df['sqft_basement'])
    
    return df

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Create new features
    train_df = create_features(train_df)
    test_df = create_features(test_df)

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
