#!/usr/bin/env python3
"""
Model Training Script for Buy From Egypt Recommendation System
Trains both content-based and collaborative filtering models
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

class RegularizedContentModel(nn.Module):
    """Content-based model with strong regularization to prevent overfitting"""
    
    def __init__(self, n_businesses, n_products, n_features, embedding_dim=64, dropout=0.5):
        super(RegularizedContentModel, self).__init__()
        
        # Smaller embeddings to reduce overfitting
        self.business_embedding = nn.Embedding(n_businesses, embedding_dim)
        self.product_embedding = nn.Embedding(n_products, embedding_dim)
        
        # Simpler feature processing with heavy dropout
        self.feature_processor = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Simpler prediction network
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, business_ids, product_ids, features):
        # Get embeddings
        business_emb = self.business_embedding(business_ids)
        product_emb = self.product_embedding(product_ids)
        
        # Process features
        feature_emb = self.feature_processor(features)
        
        # Simple concatenation
        combined = torch.cat([business_emb, product_emb, feature_emb], dim=1)
        
        # Predict relevance score
        score = self.predictor(combined)
        
        return score.squeeze()

class CollaborativeFilteringModel(nn.Module):
    """Collaborative Filtering Model for customer-product interactions"""
    
    def __init__(self, n_customers, n_products, embedding_dim=32, dropout=0.3):
        super(CollaborativeFilteringModel, self).__init__()
        
        # Embeddings
        self.customer_embedding = nn.Embedding(n_customers, embedding_dim)
        self.product_embedding = nn.Embedding(n_products, embedding_dim)
        
        # Bias terms
        self.customer_bias = nn.Embedding(n_customers, 1)
        self.product_bias = nn.Embedding(n_products, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Neural network component
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, customer_ids, product_ids):
        # Get embeddings
        customer_emb = self.customer_embedding(customer_ids)
        product_emb = self.product_embedding(product_ids)
        
        # Matrix factorization component
        mf_output = torch.sum(customer_emb * product_emb, dim=1)
        
        # Neural network component
        mlp_input = torch.cat([customer_emb, product_emb], dim=1)
        mlp_output = self.mlp(mlp_input).squeeze()
        
        # Bias terms
        customer_bias = self.customer_bias(customer_ids).squeeze()
        product_bias = self.product_bias(product_ids).squeeze()
        
        # Combine components
        output = 0.6 * mf_output + 0.4 * mlp_output + customer_bias + product_bias + self.global_bias
        
        return torch.sigmoid(output)

class BusinessProductDataset(Dataset):
    """Dataset for content-based training"""
    
    def __init__(self, business_ids, product_ids, features, ratings):
        self.business_ids = torch.LongTensor(business_ids)
        self.product_ids = torch.LongTensor(product_ids)
        self.features = torch.FloatTensor(features)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.business_ids)
    
    def __getitem__(self, idx):
        return (self.business_ids[idx], self.product_ids[idx], 
                self.features[idx], self.ratings[idx])

class CustomerProductDataset(Dataset):
    """Dataset for collaborative filtering training"""
    
    def __init__(self, customer_ids, product_ids, ratings):
        self.customer_ids = torch.LongTensor(customer_ids)
        self.product_ids = torch.LongTensor(product_ids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.customer_ids)
    
    def __getitem__(self, idx):
        return (self.customer_ids[idx], self.product_ids[idx], self.ratings[idx])

def prepare_content_based_data():
    """Prepare data for content-based model training"""
    logger.info("Preparing content-based training data...")
    
    # Load business data
    business_df = pd.read_csv(PROCESSED_DIR / "business_cleaned.csv")
    
    # Create synthetic business-product interactions
    businesses = business_df['Business Name'].unique()
    products = [f"PROD_{i:04d}" for i in range(1000)]  # 1000 products
    
    # Generate training data
    data = []
    for business in businesses:
        business_info = business_df[business_df['Business Name'] == business].iloc[0]
        
        # Generate positive and negative samples
        for _ in range(50):  # 50 samples per business
            product = np.random.choice(products)
            
            # Create features
            features = [
                min(business_info.get('Annual Trade Volume (M USD)', 0) / 1000, 1.0),
                business_info.get('Trade Success Rate (%)', 50) / 100,
                business_info.get('EgyptianAdvantageScore', 0.5),
                business_info.get('MarketAccessScore', 0.5),
                np.random.random(),  # Description length feature
                np.random.random()   # Word count feature
            ]
            
            # Generate rating (0-1)
            rating = np.random.random()
            
            data.append({
                'business': business,
                'product': product,
                'features': features,
                'rating': rating
            })
    
    df = pd.DataFrame(data)
    
    # Encode businesses and products
    business_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    
    df['business_id'] = business_encoder.fit_transform(df['business'])
    df['product_id'] = product_encoder.fit_transform(df['product'])
    
    # Scale features
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(list(df['features']))
    
    # Save encoders and scaler
    with open(MODELS_DIR / "regularized_business_encoder.pkl", "wb") as f:
        pickle.dump(business_encoder, f)
    
    with open(MODELS_DIR / "regularized_product_encoder.pkl", "wb") as f:
        pickle.dump(product_encoder, f)
    
    with open(MODELS_DIR / "regularized_feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)
    
    return df['business_id'].values, df['product_id'].values, features_scaled, df['rating'].values

def prepare_collaborative_data():
    """Prepare data for collaborative filtering training"""
    logger.info("Preparing collaborative filtering training data...")
    
    # Generate synthetic customer-product interactions
    n_customers = 1000
    n_products = 500
    n_interactions = 50000
    
    # Generate random interactions
    customer_ids = np.random.randint(0, n_customers, n_interactions)
    product_ids = np.random.randint(0, n_products, n_interactions)
    ratings = np.random.random(n_interactions)
    
    # Create encoders
    customer_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    
    # Fit encoders
    customer_encoder.fit(range(n_customers))
    product_encoder.fit(range(n_products))
    
    # Save encoders
    with open(MODELS_DIR / "cf_customer_encoder.pkl", "wb") as f:
        pickle.dump(customer_encoder, f)
    
    with open(MODELS_DIR / "cf_product_encoder.pkl", "wb") as f:
        pickle.dump(product_encoder, f)
    
    return customer_ids, product_ids, ratings, n_customers, n_products

def train_content_model():
    """Train the content-based model"""
    logger.info("Training content-based model...")
    
    # Prepare data
    business_ids, product_ids, features, ratings = prepare_content_based_data()
    
    # Split data
    train_idx, val_idx = train_test_split(range(len(business_ids)), test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = BusinessProductDataset(
        business_ids[train_idx], product_ids[train_idx], 
        features[train_idx], ratings[train_idx]
    )
    val_dataset = BusinessProductDataset(
        business_ids[val_idx], product_ids[val_idx],
        features[val_idx], ratings[val_idx]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    n_businesses = len(np.unique(business_ids))
    n_products = len(np.unique(product_ids))
    n_features = features.shape[1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegularizedContentModel(n_businesses, n_products, n_features).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            business_batch, product_batch, feature_batch, rating_batch = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            predictions = model(business_batch, product_batch, feature_batch)
            loss = criterion(predictions, rating_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                business_batch, product_batch, feature_batch, rating_batch = [x.to(device) for x in batch]
                predictions = model(business_batch, product_batch, feature_batch)
                loss = criterion(predictions, rating_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "regularized_content_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logger.info(f"Content-based model training completed. Best validation loss: {best_val_loss:.6f}")

def train_collaborative_model():
    """Train the collaborative filtering model"""
    logger.info("Training collaborative filtering model...")
    
    # Prepare data
    customer_ids, product_ids, ratings, n_customers, n_products = prepare_collaborative_data()
    
    # Split data
    train_idx, val_idx = train_test_split(range(len(customer_ids)), test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = CustomerProductDataset(
        customer_ids[train_idx], product_ids[train_idx], ratings[train_idx]
    )
    val_dataset = CustomerProductDataset(
        customer_ids[val_idx], product_ids[val_idx], ratings[val_idx]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_dim = 32
    model = CollaborativeFilteringModel(n_customers, n_products, embedding_dim).to(device)
    
    # Save model info
    model_info = {
        'n_customers': n_customers,
        'n_products': n_products,
        'embedding_dim': embedding_dim
    }
    with open(MODELS_DIR / "cf_model_info.json", "w") as f:
        json.dump(model_info, f)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(30):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            customer_batch, product_batch, rating_batch = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            predictions = model(customer_batch, product_batch)
            loss = criterion(predictions, rating_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                customer_batch, product_batch, rating_batch = [x.to(device) for x in batch]
                predictions = model(customer_batch, product_batch)
                loss = criterion(predictions, rating_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "cf_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logger.info(f"Collaborative filtering model training completed. Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    logger.info("Starting model training...")
    
    # Train both models
    train_content_model()
    train_collaborative_model()
    
    logger.info("All models trained successfully!")
