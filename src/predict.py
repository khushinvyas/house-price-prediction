import pandas as pd
import pickle
import yaml
import sys
import joblib
from sklearn.preprocessing import StandardScaler

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TARGET = params["base"]["target_col"]
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"  # We'll need the scaler used during training

def prepare_features(data):
    """Prepare features in the same way as during training"""
    # Only apply scaling to numeric columns
    numeric_cols = data.select_dtypes(include="number").columns
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

def predict(input_csv, output_csv="predictions.csv"):
    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    
    # Load input data (no target column expected)
    data = pd.read_csv(input_csv)
    
    # Prepare features
    data = prepare_features(data)
    
    # Make predictions
    preds = model.predict(data)
    
    # Save predictions
    result = pd.DataFrame({"prediction": preds})
    result.to_csv(output_csv, index=False)
    
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <input_csv> [output_csv]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"

    predict(input_csv, output_csv)