#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_data():
    """Run the data processing pipeline."""
    from src.data_processing.prepare_data import main as prepare_data_main
    logger.info("Starting data processing...")
    prepare_data_main()
    logger.info("Data processing completed.")

def train_models():
    """Train the recommendation models."""
    from src.models.train_models import main as train_models_main
    logger.info("Starting model training...")
    train_models_main()
    logger.info("Model training completed.")

def run_api():
    """Run the API server."""
    import uvicorn
    logger.info("Starting API server...")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

def main():
    """Main entry point for the recommendation system."""
    parser = argparse.ArgumentParser(description='Egypt Business Recommendation System')
    parser.add_argument('--process', action='store_true', help='Process data')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--api', action='store_true', help='Run API server')
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline (process, train, API)')
    
    args = parser.parse_args()
    
    # Check if any action was specified, otherwise show help
    if not (args.process or args.train or args.api or args.all):
        parser.print_help()
        return
    
    try:
        # Create necessary directories
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Execute requested actions
        if args.all or args.process:
            process_data()
        
        if args.all or args.train:
            train_models()
        
        if args.all or args.api:
            run_api()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 