# Buy-From-Egypt Recommendation System Source Code

This directory contains the core functionality of the recommendation system.

## Structure

- **data_processing/**: Data preparation and transformation code
  - `prepare_data.py`: Processes raw data for use in the recommendation system
  - `transform.py`: Transformation functions for data normalization and encoding

- **models/**: Recommendation model implementation
  - `collaborative_filtering.py`: User-based and item-based collaborative filtering models
  - `content_based.py`: Content-based recommendation models
  - `hybrid_model.py`: Combined recommendation approach
  - `train_models.py`: Code to train recommendation models
  - `inference.py`: Model loading and inference functions

- **utils/**: Utility functions
  - `metrics.py`: Functions for evaluating recommendation quality
  - `data_loader.py`: Functions for loading data files
  - `egyptian_context.py`: Egyptian economic context processing

## Usage

The code in this directory is primarily accessed through the main entry points in the project root:

```python
# Import specific modules
from src.models.inference import load_recommendation_engine
from src.data_processing.prepare_data import preprocess_raw_data

# Create a recommendation engine
recommendation_engine = load_recommendation_engine()

# Get recommendations
recommendations = recommendation_engine.recommend_products_for_customer(
    customer_id="12345", 
    num_recommendations=10
)
```

## Implementation Details

### Hybrid Recommendation Approach

The recommendation system uses a hybrid approach that combines:

1. **Collaborative Filtering**: Analyzes past user behavior to find patterns in purchase history
2. **Content-Based Filtering**: Recommends products and business partners based on specific attributes
3. **Egyptian Economic Context**: Incorporates economic indicators to adjust recommendations

This hybrid approach overcomes the limitations of any single method and provides more relevant recommendations for the unique Egyptian business environment.

### Egyptian Context Features

Key Egyptian-specific features included in the system:

- **Seasonality Factors**: Adjusts recommendations based on tourism seasons and holidays
- **Regional Characteristics**: Considers regional business landscapes across Egypt
- **Economic Indicators**: Incorporates GDP growth, inflation, and other economic factors

## Development

When making changes, follow these guidelines:

1. Always run tests after making changes to ensure functionality is preserved
2. Document any new functions or classes with docstrings
3. When adding a new feature, update the relevant tests