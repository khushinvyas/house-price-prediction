import pandas as pd
import pickle
import yaml
import sys

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TARGET = params["base"]["target_col"]

MODEL_PATH = "model.pkl"

def predict(input_csv, output_csv="predictions.csv"):
    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load input data (no target column expected)
    data = pd.read_csv(input_csv)

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