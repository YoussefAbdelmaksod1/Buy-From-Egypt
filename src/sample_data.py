import pandas as pd
import os
from pathlib import Path
import pickle

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")

def get_sample_data():
    """
    Print sample customer IDs and business names for testing.
    """
    print("Retrieving sample data for testing recommendations...")
    
    # Get valid customer IDs
    try:
        with open(MODELS_DIR / "user_id_map.pkl", "rb") as f:
            user_id_map = pickle.load(f)
        
        sample_customers = list(user_id_map.keys())[:5]
        print(f"\nSample customer IDs for testing:")
        for customer in sample_customers:
            print(f"  - {customer}")
    except Exception as e:
        print(f"Error loading customer IDs: {e}")
    
    # Get valid business names
    try:
        with open(MODELS_DIR / "business_id_map.pkl", "rb") as f:
            business_id_map = pickle.load(f)
        
        sample_businesses = list(business_id_map.keys())[:5]
        print(f"\nSample business names for testing:")
        for business in sample_businesses:
            print(f"  - {business}")
    except Exception as e:
        print(f"Error loading business names: {e}")
    
    # Get business product affinity
    try:
        with open(MODELS_DIR / "business_product_affinity.pkl", "rb") as f:
            business_product_affinity = pickle.load(f)
        
        businesses_with_products = [b for b, p in business_product_affinity.items() if p]
        if businesses_with_products:
            print(f"\nBusinesses with product recommendations:")
            for business in businesses_with_products[:5]:
                print(f"  - {business}")
        else:
            print("\nNo businesses with product recommendations found.")
    except Exception as e:
        print(f"Error loading business product affinity: {e}")

if __name__ == "__main__":
    get_sample_data() 