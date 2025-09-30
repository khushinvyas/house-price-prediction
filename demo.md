# House Price Prediction MLOps Project Demo

## 1. Project Setup & Data Version Control
```bash
# Clone the repository
git clone https://github.com/khushinvyas/house-price-prediction.git
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Configure AWS and pull data/models
aws configure  # Enter AWS credentials
dvc pull      # Pull data and models from S3
```

## 2. Project Structure
Show how the project is organized:
- `data/` - Version controlled datasets
- `src/` - All Python scripts
- `params.yaml` - Model parameters
- `dvc.yaml` - Pipeline configuration

## 3. Data Version Control (DVC)
Show how data is tracked:
```bash
# View DVC pipeline stages
dvc dag

# Show tracked files
dvc status

# Show remote storage
dvc remote list
```

## 4. MLOps Pipeline Demonstration
```bash
# Run full pipeline
dvc repro

# Show metrics
dvc metrics show

# Show pipeline dependencies
dvc dag --dot | dot -Tpng > pipeline.png
```

## 5. Model Prediction
```bash
# Make predictions on sample data
python src/predict.py data/raw/sample_predict.csv data/processed/new_predictions.csv

# View predictions
cat data/processed/new_predictions.csv
```

## 6. Parameter Tuning
1. Modify parameters in `params.yaml`
2. Rerun pipeline: `dvc repro`
3. Compare metrics: `dvc metrics diff`

## 7. Key Features
- Data versioning with DVC
- Automated ML pipeline
- AWS S3 integration for data storage
- Parameter management
- Model performance tracking
- Prediction service

## 8. Reproducibility Demo
```bash
# Create new directory
cd ..
mkdir house-price-demo
cd house-price-demo

# Clone and reproduce
git clone https://github.com/khushinvyas/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
dvc pull
dvc repro
```

This demonstrates:
- Code version control (Git)
- Data version control (DVC)
- Pipeline reproducibility
- Model deployment readiness