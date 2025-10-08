import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.features import create_features

def test_create_features():
    # Create sample data
    sample_data = pd.DataFrame({
        'bedrooms': [3, 4],
        'bathrooms': [2, 3],
        'sqft_living': [2000, 3000],
        'sqft_lot': [8000, 10000],
        'yr_built': [1990, 2000],
        'yr_renovated': [0, 2010],
        'sqft_basement': [500, 0]
    })
    
    # Process features
    processed_data = create_features(sample_data)
    
    # Check if new features are created
    assert 'age' in processed_data.columns
    assert 'renovated' in processed_data.columns
    assert 'total_sqft' in processed_data.columns
    assert 'price_per_sqft' in processed_data.columns
    assert 'rooms' in processed_data.columns

    # Check transformations
    assert processed_data['renovated'].iloc[0] == 0  # No renovation
    assert processed_data['renovated'].iloc[1] == 1  # Has renovation
    assert processed_data['total_sqft'].iloc[0] == 2500  # 2000 + 500
    assert processed_data['total_sqft'].iloc[1] == 3000  # 3000 + 0