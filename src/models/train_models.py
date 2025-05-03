import pandas as pd
import numpy as np
import os
import logging
import pickle
import joblib
from pathlib import Path
from datetime import datetime
import torch
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
try:
    from implicit.als import AlternatingLeastSquares
    has_implicit = True
except ImportError:
    has_implicit = False
    print("Warning: implicit package not available. Using simple matrix factorization instead.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)

# Check for CUDA availability
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    logger.info(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("CUDA is not available. Training on CPU.")

def load_data():
    """
    Load the processed data required for model training.
    """
    logger.info("Loading processed data...")
    
    try:
        # Since processed data may not exist yet, we'll work with the raw data
        # and perform simpler processing
        
        # Load business data
        business_features = pd.read_csv(DATA_DIR / "enhanced_egypt_import_export_v2.csv")
        
        # For retail data, we'll use a subset for demo purposes
        try:
            retail_data = pd.read_csv(DATA_DIR / "data.csv", encoding='ISO-8859-1', nrows=10000)
            retail_data = retail_data.dropna(subset=['CustomerID'])
            products = retail_data[['StockCode', 'Description']].drop_duplicates()
        except Exception as e:
            logger.warning(f"Error loading retail data: {e}")
            # Create dummy retail data if real data is not available
            retail_data = pd.DataFrame({
                'CustomerID': np.random.randint(1000, 2000, 1000),
                'StockCode': np.random.randint(10000, 20000, 1000),
                'Description': [f"Product {i}" for i in range(1000)],
                'Quantity': np.random.randint(1, 10, 1000),
                'UnitPrice': np.random.uniform(1, 100, 1000)
            })
            products = retail_data[['StockCode', 'Description']].drop_duplicates()
        
        # Create a simple binary user-item matrix
        user_item_df = pd.crosstab(
            index=retail_data['CustomerID'], 
            columns=retail_data['StockCode'],
            values=retail_data['Quantity'],
            aggfunc='sum'
        ).fillna(0)
        
        binary_matrix = (user_item_df > 0).astype(int)
        
        # Create a simple representation of economic data
        economic_data = pd.DataFrame({
            'gdp_growth_annual_pct': [4.35],
            'inflation_consumer_prices_annual_pct': [5.04],
            'population_growth_annual_pct': [1.73]
        })
        
        # Save processed data for future use
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        user_item_df.to_csv(PROCESSED_DIR / "user_item_matrix.csv")
        binary_matrix.to_csv(PROCESSED_DIR / "binary_interaction_matrix.csv")
        
        logger.info("Data loaded successfully.")
        
        return {
            'user_item_matrix': user_item_df,
            'binary_matrix': binary_matrix,
            'business_features': business_features,
            'products': products,
            'economic_data': economic_data,
            'retail_data': retail_data
        }
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_collaborative_filtering(data):
    """
    Train collaborative filtering models with PyTorch or using Implicit ALS if available.
    """
    logger.info("Training collaborative filtering model...")
    
    try:
        # Get user-item matrix
        user_item_df = data['user_item_matrix']
        
        # Create mappings between IDs and indices
        user_ids = user_item_df.index.tolist()
        item_ids = user_item_df.columns.tolist()
        
        user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
        item_id_map = {item_id: i for i, item_id in enumerate(item_ids)}
        
        reverse_user_map = {i: user_id for user_id, i in user_id_map.items()}
        reverse_item_map = {i: item_id for item_id, i in item_id_map.items()}
        
        # Train model
        if has_implicit:
            # Use Implicit ALS
            sparse_user_item = csr_matrix(user_item_df.values)
            
            # Try to train ALS model safely with and without GPU
            use_implicit_gpu = False
            if USE_CUDA:
                try:
                    logger.info("Attempting to train ALS model with GPU acceleration")
                    # Try with 'use_gpu' parameter
                    model = AlternatingLeastSquares(factors=64, regularization=0.1, iterations=50, use_gpu=True)
                    use_implicit_gpu = True
                except (TypeError, ValueError) as e:
                    logger.warning(f"GPU acceleration not available for Implicit ALS: {e}")
                    use_implicit_gpu = False
            
            if not use_implicit_gpu:
                logger.info("Training ALS model on CPU")
                model = AlternatingLeastSquares(factors=64, regularization=0.1, iterations=50)
                
            # Train the model
            model.fit(sparse_user_item)
        else:
            # Simple PyTorch matrix factorization
            logger.info("Using PyTorch matrix factorization")
            user_item_tensor = torch.tensor(user_item_df.values, dtype=torch.float32)
            
            # Define simple matrix factorization model
            n_users, n_items = user_item_tensor.shape
            n_factors = 64
            
            # Initialize user and item embeddings
            user_factors = torch.randn(n_users, n_factors, requires_grad=True)
            item_factors = torch.randn(n_factors, n_items, requires_grad=True)
            
            if USE_CUDA:
                logger.info("Moving tensors to GPU")
                try:
                    user_item_tensor = user_item_tensor.cuda()
                    user_factors = user_factors.cuda()
                    item_factors = item_factors.cuda()
                except Exception as e:
                    logger.warning(f"Error moving tensors to GPU: {e}")
                    # Fall back to CPU
                    user_factors = torch.randn(n_users, n_factors, requires_grad=True)
                    item_factors = torch.randn(n_factors, n_items, requires_grad=True)
            
            # Simple training loop
            optimizer = torch.optim.Adam([user_factors, item_factors], lr=0.01)
            
            for epoch in range(10):
                # Forward pass: compute predicted ratings
                predicted_ratings = torch.matmul(user_factors, item_factors)
                
                # Only consider non-zero entries for loss calculation
                mask = (user_item_tensor > 0).float()
                
                # Calculate loss (MSE on non-zero entries)
                loss = torch.sum(mask * (user_item_tensor - predicted_ratings) ** 2) / torch.sum(mask)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                logger.info(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
            
            # Store the learned factors as the model
            model = {
                'user_factors': user_factors.detach().cpu().numpy(),
                'item_factors': item_factors.detach().cpu().numpy()
            }
        
        # Save model and mappings
        with open(MODELS_DIR / "cf_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        with open(MODELS_DIR / "user_id_map.pkl", "wb") as f:
            pickle.dump(user_id_map, f)
        
        with open(MODELS_DIR / "item_id_map.pkl", "wb") as f:
            pickle.dump(item_id_map, f)
        
        with open(MODELS_DIR / "reverse_user_map.pkl", "wb") as f:
            pickle.dump(reverse_user_map, f)
        
        with open(MODELS_DIR / "reverse_item_map.pkl", "wb") as f:
            pickle.dump(reverse_item_map, f)
        
        logger.info("Collaborative filtering model trained and saved.")
        
        return model, user_id_map, item_id_map, reverse_user_map, reverse_item_map
    
    except Exception as e:
        logger.error(f"Error training collaborative filtering model: {e}")
        raise

def train_content_based_business(data):
    """
    Train content-based filtering model for business recommendations.
    Incorporate Egyptian-specific features for better recommendations.
    """
    logger.info("Training content-based business recommendation model with Egyptian context...")
    
    try:
        # Get business features
        business_df = data['business_features']
        
        # Remove any duplicate business entries
        business_df = business_df.drop_duplicates(subset=['Business Name'])
        
        # Create a unique business ID
        business_df['BusinessID'] = range(1, len(business_df) + 1)
        
        # Convert categorical variables to numeric
        trade_type_map = {'Importer': 0, 'Exporter': 1, 'Both': 2}
        business_size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
        
        business_df['Trade Type Encoded'] = business_df['Trade Type'].map(trade_type_map)
        business_df['Business Size Encoded'] = business_df['Business Size'].map(business_size_map)
        
        # Get unique categories and create mapping
        categories = business_df['Category'].unique()
        category_map = {cat: i for i, cat in enumerate(categories)}
        business_df['Category Encoded'] = business_df['Category'].map(category_map)
        
        # Get unique regions (if available) and create mapping
        if 'Region' in business_df.columns:
            regions = business_df['Region'].unique()
            region_map = {region: i for i, region in enumerate(regions)}
            business_df['Region Encoded'] = business_df['Region'].map(region_map)
        
        # Select numerical features for similarity calculation, including Egyptian-specific features
        features = [
            'Annual Trade Volume (M USD)', 
            'Trade Growth Rate (%)', 
            'Trade Success Rate (%)', 
            'Trade Frequency (per year)', 
            'Trade Type Encoded',
            'Business Size Encoded',
            'Category Encoded'
        ]
        
        # Add Egyptian-specific features if available
        egyptian_features = [
            'EgyptianAdvantageScore',
            'LogisticsAccess',
            'MarketAccessScore',
            'TraditionalIndustryScore',
            'Region Encoded'
        ]
        
        # Add available Egyptian features
        for feature in egyptian_features:
            if feature in business_df.columns:
                features.append(feature)
                logger.info(f"Using Egyptian-specific feature: {feature}")
        
        # Extract features
        X = business_df[features].fillna(0).values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate cosine similarity (use GPU if available)
        gpu_cosine_sim_success = False
        if USE_CUDA:
            try:
                logger.info("Computing cosine similarity on GPU")
                # Convert to PyTorch tensors and move to GPU
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).cuda()
                # Compute dot product normalized by magnitudes
                X_norm = torch.norm(X_tensor, dim=1, keepdim=True)
                X_normalized = X_tensor / X_norm
                similarity_matrix = torch.mm(X_normalized, X_normalized.t()).cpu().numpy()
                gpu_cosine_sim_success = True
            except Exception as e:
                logger.warning(f"Error computing cosine similarity on GPU: {e}. Falling back to CPU.")
                gpu_cosine_sim_success = False
        
        if not USE_CUDA or not gpu_cosine_sim_success:
            logger.info("Computing cosine similarity on CPU")
            similarity_matrix = cosine_similarity(X_scaled)
        
        # Create business ID mapping
        business_ids = business_df['BusinessID'].values
        business_names = business_df['Business Name'].values
        
        business_id_map = {name: id for name, id in zip(business_names, business_ids)}
        business_idx_map = {id: idx for idx, id in enumerate(business_ids)}
        
        # Save model components
        np.save(MODELS_DIR / "business_similarity_matrix.npy", similarity_matrix)
        
        with open(MODELS_DIR / "business_id_map.pkl", "wb") as f:
            pickle.dump(business_id_map, f)
        
        with open(MODELS_DIR / "business_idx_map.pkl", "wb") as f:
            pickle.dump(business_idx_map, f)
        
        joblib.dump(scaler, MODELS_DIR / "business_scaler.joblib")
        
        # Save feature list for future reference
        with open(MODELS_DIR / "business_features_list.pkl", "wb") as f:
            pickle.dump(features, f)
        
        # Save the business dataframe for API use
        business_df.to_csv(PROCESSED_DIR / "business_features.csv", index=False)
        
        logger.info("Content-based business recommendation model trained and saved with Egyptian context.")
        
        return similarity_matrix, business_id_map, business_idx_map
    
    except Exception as e:
        logger.error(f"Error training content-based business model: {e}")
        raise

def train_economic_context_model(data):
    """
    Create an economic context model focused on Egyptian economic indicators.
    """
    logger.info("Creating Egyptian economic context model...")
    
    try:
        # Get economic data
        economic_data = data['economic_data']
        
        # Extract relevant economic indicators for Egyptian context
        # Focus on indicators that matter most for Egyptian businesses
        
        # Core economic indicators
        core_indicators = {
            'gdp_growth': economic_data.get('gdp_growth_annual_pct', [4.2]).iloc[0],
            'inflation': economic_data.get('inflation_consumer_prices_annual_pct', [10.2]).iloc[0],
            'population_growth': economic_data.get('population_growth_annual_pct', [1.9]).iloc[0]
        }
        
        # Egypt-specific economic factors
        egypt_specific = {}
        
        # Tourism sensitivity (important for Egyptian economy)
        if 'tourism_sensitivity' in economic_data.columns:
            egypt_specific['tourism_sensitivity'] = economic_data['tourism_sensitivity'].iloc[0]
        else:
            egypt_specific['tourism_sensitivity'] = 0.12  # Default if not available
        
        # Trade balance
        if 'trade_balance' in economic_data.columns:
            egypt_specific['trade_balance'] = economic_data['trade_balance'].iloc[0]
        
        # Economic stability index
        if 'economic_stability_index' in economic_data.columns:
            egypt_specific['economic_stability_index'] = economic_data['economic_stability_index'].iloc[0]
        
        # Industry specific indicators
        industry_indicators = {}
        
        # Get manufacturing contribution to GDP
        if 'manufacturing_value_added_pct_of_gdp' in economic_data.columns:
            industry_indicators['manufacturing_pct_gdp'] = economic_data['manufacturing_value_added_pct_of_gdp'].iloc[0]
        
        # Get agriculture contribution to GDP
        if 'agriculture_forestry_and_fishing_value_added_pct_of_gdp' in economic_data.columns:
            industry_indicators['agriculture_pct_gdp'] = economic_data['agriculture_forestry_and_fishing_value_added_pct_of_gdp'].iloc[0]
        
        # Get services contribution to GDP
        if 'services_value_added_pct_of_gdp' in economic_data.columns:
            industry_indicators['services_pct_gdp'] = economic_data['services_value_added_pct_of_gdp'].iloc[0]
        
        # Combine all indicators
        context_features = {
            **core_indicators,
            **egypt_specific,
            **industry_indicators
        }
        
        # Add Egyptian seasonal factors (current date)
        today = datetime.now()
        month = today.month
        
        # Is winter tourism season (Oct-Mar)
        winter_months = [10, 11, 12, 1, 2, 3]
        context_features['is_winter_tourism_season'] = 1 if month in winter_months else 0
        
        # Add predicted Ramadan impact
        # This is a simplification - real implementation would use actual calendar
        ramadan_2023 = [3, 4]  # March-April 2023
        ramadan_2024 = [3]     # March 2024
        ramadan_2025 = [2, 3]  # February-March 2025
        
        context_features['is_ramadan_season'] = 1 if month in ramadan_2023 or month in ramadan_2024 or month in ramadan_2025 else 0
        
        # Add industry weights based on Egyptian economy
        # These weights determine how much each sector should influence recommendations
        context_features['industry_weights'] = {
            'Textiles': 0.15,
            'Agriculture': 0.18,
            'Spices': 0.12,
            'Fruits & Vegetables': 0.15,
            'Chemicals': 0.08,
            'Pharmaceuticals': 0.07,
            'Electronics': 0.06,
            'Machinery': 0.05,
            'Metals': 0.08,
            'Automobiles': 0.03,
            'Seafood': 0.06,
            'Manufacturing': 0.10
        }
        
        # Save economic context
        with open(MODELS_DIR / "economic_context.pkl", "wb") as f:
            pickle.dump(context_features, f)
        
        logger.info("Egyptian economic context model created and saved.")
        
        return context_features
    
    except Exception as e:
        logger.error(f"Error creating Egyptian economic context model: {e}")
        raise

def calculate_business_product_affinity(data):
    """
    Calculate affinity between businesses and products based on their attributes.
    Emphasize Egyptian context and traditional Egyptian products.
    """
    logger.info("Calculating Egyptian business-product affinity...")
    
    try:
        # This is a simplified approach - in a full implementation, we would:
        # 1. Extract features from businesses and products
        # 2. Calculate similarity or relevance scores
        # 3. Store these as recommendations
        
        # For now, we'll create a mapping based on business categories
        business_df = data['business_features']
        products = data['products']
        
        # Add Egyptian relevance score if available in product data
        # If not available, we'll create a score based on product descriptions
        if 'EgyptRelevance' not in products.columns:
            # These are keywords relevant to Egyptian markets
            egyptian_keywords = [
                'cotton', 'textile', 'spice', 'craft', 'ceramic', 'papyrus',
                'leather', 'copper', 'silver', 'gold', 'carpet', 'rug',
                'dates', 'olive', 'tea', 'coffee', 'lamp', 'glass', 'metal',
                'furniture', 'decoration', 'ornament', 'jewelry', 'herb'
            ]
            
            products['EgyptRelevance'] = products['Description'].str.lower().apply(
                lambda desc: sum(1 for keyword in egyptian_keywords if keyword in str(desc).lower()) / len(egyptian_keywords)
            )
        
        # Create mappings with Egyptian relevance in mind
        # Map business categories to relevant product categories (with Egyptian focus)
        category_product_map = {
            'Spices': ['SPICES', 'FOOD', 'KITCHEN', 'HERB', 'TEA'],
            'Agriculture': ['GARDEN', 'PLANTS', 'OUTDOOR', 'ORGANIC', 'COTTON', 'FLOWER'],
            'Metals': ['METAL', 'HARDWARE', 'TOOLS', 'COPPER', 'SILVER', 'GOLD'],
            'Electronics': ['ELECTRONICS', 'TECHNOLOGY', 'BATTERIES', 'PHONE', 'COMPUTER'],
            'Textiles': ['TEXTILES', 'FABRIC', 'CLOTHING', 'COTTON', 'LINEN', 'CARPET'],
            'Fruits & Vegetables': ['FOOD', 'KITCHEN', 'STORAGE', 'FRUIT', 'ORGANIC'],
            'Machinery': ['TOOLS', 'HARDWARE', 'EQUIPMENT', 'METAL'],
            'Seafood': ['FOOD', 'KITCHEN', 'STORAGE', 'FISH'],
            'Pharmaceuticals': ['HEALTH', 'WELLNESS', 'BATHROOM', 'HERBAL', 'MEDICINE'],
            'Manufacturing': ['TOOLS', 'EQUIPMENT', 'HARDWARE', 'FACTORY'],
            'Chemicals': ['CLEANING', 'HOUSEHOLD', 'GARDEN', 'LABORATORY'],
            'Automobiles': ['TRANSPORT', 'TRAVEL', 'OUTDOOR', 'VEHICLE', 'CAR']
        }
        
        # Enhanced with Egyptian region-specific products
        region_product_map = {
            'Greater Cairo': ['URBAN', 'MODERN', 'FURNITURE', 'DECOR', 'OFFICE'],
            'Mediterranean Coast': ['SEAFOOD', 'MARINE', 'BEACH', 'SUMMER', 'FISH'],
            'Upper Egypt': ['CRAFT', 'TRADITIONAL', 'HANDMADE', 'POTTERY', 'STATUE'],
            'Nile Delta': ['COTTON', 'TEXTILE', 'AGRICULTURE', 'FRUIT'],
            'Suez Canal': ['SHIPPING', 'LOGISTICS', 'TRADE', 'INTERNATIONAL'],
            'Red Sea': ['TOURISM', 'BEACH', 'SUMMER', 'CORAL', 'DIVING'],
            'Sinai': ['CRAFT', 'TRADITIONAL', 'BEDOUIN', 'HERBS', 'DESERT']
        }
        
        # Create a mapping structure to save
        business_product_affinity = {}
        
        for _, business in business_df.iterrows():
            business_name = business['Business Name']
            business_category = business['Category']
            
            # Initialize with category-based keywords
            keywords = []
            if business_category in category_product_map:
                keywords.extend(category_product_map[business_category])
            
            # Add region-based keywords if available
            if 'Region' in business.keys() and business['Region'] in region_product_map:
                keywords.extend(region_product_map[business['Region']])
            
            # Find products matching these keywords
            matched_products = []
            for _, product in products.iterrows():
                description = str(product['Description']).upper()
                
                # Calculate match score
                match_score = 0
                for keyword in keywords:
                    if keyword in description:
                        match_score += 1
                
                # Add Egyptian relevance boost if available
                if 'EgyptRelevance' in product:
                    match_score *= (1 + float(product['EgyptRelevance']))
                
                # If match score is above threshold, add to recommendations
                if match_score > 0:
                    # Normalize score to 0-1 range
                    normalized_score = min(0.9, match_score / len(keywords))
                    
                    matched_products.append({
                        'StockCode': str(product['StockCode']),
                        'Description': str(product['Description']),
                        'Score': normalized_score,
                        'EgyptRelevance': product.get('EgyptRelevance', 0)
                    })
            
            # Sort by score and keep top matches
            matched_products = sorted(matched_products, key=lambda x: x['Score'], reverse=True)
            business_product_affinity[business_name] = matched_products[:20] if matched_products else []
        
        # Save the mapping
        with open(MODELS_DIR / "business_product_affinity.pkl", "wb") as f:
            pickle.dump(business_product_affinity, f)
        
        logger.info("Egyptian business-product affinity calculated and saved.")
        
        return business_product_affinity
    
    except Exception as e:
        logger.error(f"Error calculating business-product affinity: {e}")
        raise

def evaluate_collaborative_filtering(cf_model, user_item_df, test_size=0.2, k=10):
    """
    Evaluate collaborative filtering model using precision@k, recall@k and RMSE.
    
    Args:
        cf_model: The trained collaborative filtering model
        user_item_df: User-item interaction dataframe
        test_size: Proportion of data to use for testing
        k: Number of recommendations to consider for precision/recall metrics
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating collaborative filtering model with k={k}...")
    
    # Create train/test split
    try:
        # Convert to numpy for easier manipulation
        user_item_matrix = user_item_df.values
        n_users, n_items = user_item_matrix.shape
        
        # Create mask for test data (20% of non-zero entries)
        test_mask = np.zeros_like(user_item_matrix, dtype=bool)
        for u in range(n_users):
            # Get indices of items that the user has interacted with
            interacted_items = np.where(user_item_matrix[u, :] > 0)[0]
            if len(interacted_items) > 5:  # Only test users with enough interactions
                # Sample 20% of the interactions for test
                n_test = max(1, int(test_size * len(interacted_items)))
                test_indices = np.random.choice(interacted_items, n_test, replace=False)
                test_mask[u, test_indices] = True
        
        # Create train matrix (zero out test entries)
        train_matrix = user_item_matrix.copy()
        train_matrix[test_mask] = 0
        
        # Compute recommendations using trained model
        if isinstance(cf_model, dict):  # Matrix factorization model
            user_factors = cf_model['user_factors']
            item_factors = cf_model['item_factors']
            
            # Generate predictions
            if item_factors.shape[0] == user_factors.shape[1]:  # Check if dimensions match
                predicted_ratings = np.dot(user_factors, item_factors)
            else:  # Need to transpose
                predicted_ratings = np.dot(user_factors, item_factors.T)
        else:
            # For other model types - use a simple fallback
            predicted_ratings = train_matrix
        
        # Calculate RMSE for non-zero entries in test set
        test_ratings = user_item_matrix[test_mask]
        predicted_test_ratings = predicted_ratings[test_mask]
        rmse = sqrt(mean_squared_error(test_ratings, predicted_test_ratings))
        
        # Calculate precision and recall at k
        precision_at_k = []
        recall_at_k = []
        
        for u in range(n_users):
            # Get indices of items in test set for this user
            test_items = np.where(test_mask[u, :])[0]
            if len(test_items) == 0:
                continue
                
            # Get top k predicted items excluding training items
            train_items = np.where(train_matrix[u, :] > 0)[0]
            pred_ratings = predicted_ratings[u, :]
            # Set ratings of already interacted items to -inf to exclude them
            pred_ratings[train_items] = -np.inf
            
            # Get top k items
            top_k_items = np.argsort(-pred_ratings)[:k]
            
            # Calculate precision and recall
            n_relevant_and_recommended = len(set(top_k_items) & set(test_items))
            precision = n_relevant_and_recommended / min(k, len(top_k_items)) if len(top_k_items) > 0 else 0
            recall = n_relevant_and_recommended / len(test_items) if len(test_items) > 0 else 0
            
            precision_at_k.append(precision)
            recall_at_k.append(recall)
        
        # Calculate average precision and recall
        avg_precision = np.mean(precision_at_k) if precision_at_k else 0
        avg_recall = np.mean(recall_at_k) if recall_at_k else 0
        
        # Calculate F1 score
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        metrics = {
            'rmse': rmse,
            f'precision@{k}': avg_precision,
            f'recall@{k}': avg_recall,
            f'f1@{k}': f1_score
        }
        
        logger.info(f"Collaborative filtering evaluation metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating collaborative filtering model: {e}")
        return {
            'rmse': 0.0,
            f'precision@{k}': 0.0,
            f'recall@{k}': 0.0,
            f'f1@{k}': 0.0
        }

def evaluate_business_recommendations(similarity_matrix, business_df, k=5):
    """
    Evaluate business recommendation model using industry similarity as ground truth.
    
    Args:
        similarity_matrix: The similarity matrix between businesses
        business_df: Business dataframe with features
        k: Number of recommendations to consider
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating business recommendation model with k={k}...")
    
    try:
        # Create industry similarity as a proxy for ground truth
        # Businesses in the same category should be recommended to each other
        categories = business_df['Category'].values
        n_businesses = len(categories)
        
        # Create ground truth matrix (1 if same category)
        ground_truth = np.zeros((n_businesses, n_businesses))
        for i in range(n_businesses):
            for j in range(n_businesses):
                if i != j and categories[i] == categories[j]:
                    ground_truth[i, j] = 1
        
        # Calculate precision and recall at k
        precision_at_k = []
        recall_at_k = []
        
        for i in range(n_businesses):
            # Get top k similar businesses based on similarity matrix
            # Exclude self
            sim_scores = similarity_matrix[i, :]
            sim_scores[i] = -np.inf  # Exclude self
            top_k_indices = np.argsort(-sim_scores)[:k]
            
            # Get relevant businesses (same category)
            relevant_indices = np.where(ground_truth[i, :] == 1)[0]
            
            if len(relevant_indices) == 0:
                continue
                
            # Calculate precision and recall
            n_relevant_and_recommended = len(set(top_k_indices) & set(relevant_indices))
            precision = n_relevant_and_recommended / min(k, len(top_k_indices)) if len(top_k_indices) > 0 else 0
            recall = n_relevant_and_recommended / len(relevant_indices) if len(relevant_indices) > 0 else 0
            
            precision_at_k.append(precision)
            recall_at_k.append(recall)
        
        # Calculate average precision and recall
        avg_precision = np.mean(precision_at_k) if precision_at_k else 0
        avg_recall = np.mean(recall_at_k) if recall_at_k else 0
        
        # Calculate F1 score
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        # Calculate Category Coverage (% of categories that have at least one recommendation)
        unique_categories = len(np.unique(categories))
        recommended_categories = set()
        for i in range(n_businesses):
            sim_scores = similarity_matrix[i, :]
            sim_scores[i] = -np.inf
            top_k_indices = np.argsort(-sim_scores)[:k]
            recommended_categories.update(categories[top_k_indices])
        
        category_coverage = len(recommended_categories) / unique_categories if unique_categories > 0 else 0
        
        metrics = {
            f'precision@{k}': avg_precision,
            f'recall@{k}': avg_recall,
            f'f1@{k}': f1_score,
            'category_coverage': category_coverage
        }
        
        logger.info(f"Business recommendation evaluation metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating business recommendation model: {e}")
        return {
            f'precision@{k}': 0.0,
            f'recall@{k}': 0.0,
            f'f1@{k}': 0.0,
            'category_coverage': 0.0
        }

def train_hybrid_recommendation_model(data):
    """
    Train a hybrid recommendation model that combines collaborative filtering, 
    content-based filtering, and Egyptian context.
    
    This approach leverages the strengths of multiple recommendation techniques:
    1. Collaborative filtering - captures user-item interaction patterns
    2. Content-based filtering - leverages business attributes
    3. Context-aware model - incorporates Egyptian economic indicators
    """
    logger.info("Training hybrid recommendation model with Egyptian context...")
    
    try:
        # Get required data components
        user_item_df = data['user_item_matrix']
        business_features = data['business_features']
        economic_data = data['economic_data']
        retail_data = data['retail_data']
        
        # 1. COLLABORATIVE FILTERING COMPONENT
        # Train collaborative filtering model on the entire dataset using either
        # Implicit ALS or PyTorch based on availability
        if has_implicit:
            # Use Implicit ALS
            sparse_user_item = csr_matrix(user_item_df.values)
            
            # Try with CUDA if available
            use_implicit_gpu = False
            if USE_CUDA:
                try:
                    logger.info("Attempting to train ALS model with GPU acceleration")
                    model_cf = AlternatingLeastSquares(
                        factors=128,  # Increased for better representation
                        regularization=0.05,
                        iterations=50,
                        use_gpu=True
                    )
                    use_implicit_gpu = True
                except (TypeError, ValueError) as e:
                    logger.warning(f"GPU acceleration not available for Implicit ALS: {e}")
                    use_implicit_gpu = False
            
            if not use_implicit_gpu:
                logger.info("Training ALS model on CPU")
                model_cf = AlternatingLeastSquares(
                    factors=128,
                    regularization=0.05,
                    iterations=50
                )
                
            # Train the model
            model_cf.fit(sparse_user_item)
            
            # Extract user and item factors
            user_factors = model_cf.user_factors
            item_factors = model_cf.item_factors
        else:
            # Use PyTorch matrix factorization with more sophisticated approach
            logger.info("Using PyTorch for collaborative filtering component")
            
            # Initialize user-item interaction tensor
            user_item_tensor = torch.tensor(user_item_df.values, dtype=torch.float32)
            
            # Define dimensions
            n_users, n_items = user_item_tensor.shape
            n_factors = 128  # Increased for better representation
            
            # Initialize embeddings with Xavier initialization for better convergence
            user_factors = torch.randn(n_users, n_factors) / np.sqrt(n_factors)
            item_factors = torch.randn(n_factors, n_items) / np.sqrt(n_factors)
            
            user_factors.requires_grad = True
            item_factors.requires_grad = True
            
            if USE_CUDA:
                logger.info("Moving tensors to GPU")
                try:
                    user_item_tensor = user_item_tensor.cuda()
                    user_factors = user_factors.cuda()
                    item_factors = item_factors.cuda()
                except Exception as e:
                    logger.warning(f"Error moving tensors to GPU: {e}")
                    # Fall back to CPU
                    user_factors = torch.randn(n_users, n_factors, requires_grad=True) / np.sqrt(n_factors)
                    item_factors = torch.randn(n_factors, n_items, requires_grad=True) / np.sqrt(n_factors)
            
            # Adam optimizer with learning rate scheduling
            optimizer = torch.optim.Adam([user_factors, item_factors], lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
            
            # Training loop with early stopping
            best_loss = float('inf')
            patience = 3
            patience_counter = 0
            
            for epoch in range(25):  # Increased epochs for better convergence
                # Forward pass
                predicted_ratings = torch.matmul(user_factors, item_factors)
                
                # Mask for non-zero entries
                mask = (user_item_tensor > 0).float()
                
                # Calculate loss with L2 regularization
                l2_reg = 0.001 * (torch.sum(user_factors**2) + torch.sum(item_factors**2))
                mse_loss = torch.sum(mask * (user_item_tensor - predicted_ratings)**2) / torch.sum(mask)
                loss = mse_loss + l2_reg
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Log progress
                logger.info(f"Epoch {epoch+1}/25, Loss: {loss.item():.4f}")
                
                # Early stopping
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Convert to numpy arrays for consistent interface
            user_factors = user_factors.detach().cpu().numpy()
            item_factors = item_factors.detach().cpu().numpy()
        
        # 2. CONTENT-BASED FILTERING COMPONENT
        # Process business features with Egyptian-specific enhancements
        logger.info("Processing content-based features with Egyptian context...")
        
        # Extract meaningful features for content-based filtering
        # Focus on Egyptian-relevant features
        if isinstance(business_features, pd.DataFrame):
            # Process available columns
            features_to_use = []
            
            # Check for Egyptian-specific features
            if 'Region' in business_features.columns:
                logger.info("Using Egyptian-specific feature: Region")
                features_to_use.append('Region')
                
            if 'Category' in business_features.columns:
                features_to_use.append('Category')
                
            if 'Location' in business_features.columns:
                features_to_use.append('Location')
                
            if 'Trade Type' in business_features.columns:
                features_to_use.append('Trade Type')
                
            if 'Subcategory' in business_features.columns:
                features_to_use.append('Subcategory')
                
            if 'Business Size' in business_features.columns:
                features_to_use.append('Business Size')
            
            # One-hot encode categorical features
            if features_to_use:
                business_encoded = pd.get_dummies(business_features[features_to_use])
                
                # Scale features
                scaler = StandardScaler()
                business_features_scaled = scaler.fit_transform(business_encoded)
                
                # Compute similarity matrix for business-to-business recommendations
                logger.info("Computing cosine similarity on GPU" if USE_CUDA else "Computing cosine similarity")
                if USE_CUDA:
                    try:
                        # Try to use GPU for similarity calculation
                        business_tensor = torch.tensor(business_features_scaled, dtype=torch.float32).cuda()
                        
                        # Normalize for cosine similarity
                        business_norm = torch.norm(business_tensor, dim=1, keepdim=True)
                        business_normalized = business_tensor / business_norm
                        
                        # Compute similarity matrix
                        similarity_matrix = torch.mm(business_normalized, business_normalized.t())
                        business_similarity = similarity_matrix.cpu().numpy()
                    except Exception as e:
                        logger.warning(f"Error using GPU for similarity: {e}")
                        business_similarity = cosine_similarity(business_features_scaled)
                else:
                    business_similarity = cosine_similarity(business_features_scaled)
                
                # Create mappings between business names and indices
                business_names = business_features['Business Name'].tolist()
                business_id_map = {name: i for i, name in enumerate(business_names)}
                business_idx_map = {i: name for i, name in enumerate(business_names)}
            else:
                # Fallback if no features found
                logger.warning("No features found for content-based filtering. Using dummy similarity.")
                business_similarity = np.eye(len(business_features))
                business_names = business_features['Business Name'].tolist() if 'Business Name' in business_features.columns else [f"Business {i}" for i in range(len(business_features))]
                business_id_map = {name: i for i, name in enumerate(business_names)}
                business_idx_map = {i: name for i, name in enumerate(business_names)}
        else:
            # Fallback for missing business features
            logger.warning("No business features found. Using dummy business similarity.")
            business_similarity = np.eye(10)
            business_id_map = {f"Business {i}": i for i in range(10)}
            business_idx_map = {i: f"Business {i}" for i in range(10)}
        
        # 3. CONTEXTUAL COMPONENT - EGYPTIAN ECONOMIC CONTEXT
        logger.info("Adding Egyptian economic context to recommendation model...")
        
        # Process economic indicators
        if isinstance(economic_data, pd.DataFrame) and not economic_data.empty:
            # Extract relevant indicators for Egyptian context
            economic_context = {
                'gdp_growth': economic_data['gdp_growth_annual_pct'].iloc[0] if 'gdp_growth_annual_pct' in economic_data.columns else 4.35,
                'inflation': economic_data['inflation_consumer_prices_annual_pct'].iloc[0] if 'inflation_consumer_prices_annual_pct' in economic_data.columns else 5.04,
                'population_growth': economic_data['population_growth_annual_pct'].iloc[0] if 'population_growth_annual_pct' in economic_data.columns else 1.73,
                'tourism_sensitivity': 0.85,  # High sensitivity to tourism (scale 0-1)
                'economic_stability_index': 0.65,  # Medium-high stability (scale 0-1)
                'trade_balance': -0.12,  # Trade deficit as proportion of GDP
            }
            
            # Add seasonality factors specific to Egypt
            current_month = datetime.now().month
            economic_context['is_winter_tourism_season'] = 1 if current_month in [10, 11, 12, 1, 2, 3] else 0
            economic_context['is_ramadan_season'] = 0  # Would need Islamic calendar calculation
        else:
            # Fallback for missing economic data
            economic_context = {
                'gdp_growth': 4.35,
                'inflation': 5.04,
                'population_growth': 1.73,
                'tourism_sensitivity': 0.85,
                'economic_stability_index': 0.65,
                'trade_balance': -0.12,
                'is_winter_tourism_season': 1 if datetime.now().month in [10, 11, 12, 1, 2, 3] else 0,
                'is_ramadan_season': 0
            }
        
        # 4. BUSINESS-PRODUCT AFFINITY
        logger.info("Calculating product affinities for Egyptian businesses...")
        
        # Create product mapping
        products = data['products']
        product_id_map = {product_id: i for i, product_id in enumerate(products['StockCode'].values)}
        
        # Calculate business-product affinity scores
        business_product_affinity = {}
        
        for business_name in business_names:
            # Find businesses similar to this one
            if business_name in business_id_map:
                business_idx = business_id_map[business_name]
                similar_businesses = np.argsort(-business_similarity[business_idx])[:10]  # Top 10 similar businesses
                
                # Get products commonly purchased by customers of similar businesses
                product_scores = {}
                
                # If we have transactional data, use it to find popular products
                if isinstance(retail_data, pd.DataFrame) and not retail_data.empty:
                    # Create a simplified score based on frequency
                    product_counts = retail_data['StockCode'].value_counts()
                    for product_id, count in product_counts.items():
                        if product_id in product_id_map:
                            # More sophisticated scoring that considers business similarity
                            similarity_boost = 1.0  # Base score
                            
                            # Check if this product is Egyptian-relevant
                            if 'Description' in retail_data.columns:
                                product_info = retail_data[retail_data['StockCode'] == product_id]['Description'].iloc[0]
                                egyptian_keywords = [
                                    'cotton', 'textile', 'spice', 'craft', 'ceramic', 'papyrus',
                                    'leather', 'copper', 'silver', 'gold', 'carpet', 'rug',
                                    'dates', 'olive', 'tea', 'coffee', 'lamp', 'glass', 'metal',
                                    'furniture', 'decoration', 'ornament', 'jewelry', 'herb'
                                ]
                                if any(keyword in str(product_info).lower() for keyword in egyptian_keywords):
                                    similarity_boost *= 1.5  # Boost Egyptian-relevant products
                            
                            # Adjust for current economic context
                            if economic_context['is_winter_tourism_season'] == 1:
                                # Boost products popular during tourism season
                                tourism_products = ['craft', 'souvenir', 'gift', 'decor', 'jewelry']
                                if 'Description' in retail_data.columns:
                                    product_info = retail_data[retail_data['StockCode'] == product_id]['Description'].iloc[0]
                                    if any(item in str(product_info).lower() for item in tourism_products):
                                        similarity_boost *= 1.3
                            
                            # Calculate final score
                            product_scores[product_id] = count * similarity_boost / retail_data.shape[0]
                else:
                    # Fallback when no retail data is available
                    for stock_code in products['StockCode'].values:
                        product_scores[stock_code] = np.random.random()  # Random scores as fallback
                
                # Get top products
                top_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:20]
                
                # Format for storage
                business_products = []
                for product_id, score in top_products:
                    if product_id in products['StockCode'].values:
                        product_info = products[products['StockCode'] == product_id]
                        if not product_info.empty:
                            description = product_info['Description'].values[0]
                            
                            business_products.append({
                                'StockCode': product_id,
                                'Description': description,
                                'Score': float(score)
                            })
                
                business_product_affinity[business_name] = business_products
        
        # 5. SAVE HYBRID MODEL COMPONENTS
        logger.info("Saving hybrid recommendation model components...")
        
        # Create hybrid model structure
        hybrid_model = {
            'collaborative_filtering': {
                'user_factors': user_factors,
                'item_factors': item_factors
            },
            'content_based': {
                'business_similarity': business_similarity,
                'business_id_map': business_id_map,
                'business_idx_map': business_idx_map
            },
            'economic_context': economic_context,
            'business_product_affinity': business_product_affinity
        }
        
        # Convert NumPy arrays to lists for JSON serialization
        business_similarity_list = business_similarity.tolist()
        
        # Save model components
        # CF model
        with open(MODELS_DIR / "cf_model.pkl", "wb") as f:
            pickle.dump({
                'user_factors': user_factors,
                'item_factors': item_factors
            }, f)
        
        # User and item ID mappings
        user_ids = user_item_df.index.tolist()
        item_ids = user_item_df.columns.tolist()
        
        user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
        item_id_map = {item_id: i for i, item_id in enumerate(item_ids)}
        
        reverse_user_map = {i: user_id for user_id, i in user_id_map.items()}
        reverse_item_map = {i: item_id for item_id, i in item_id_map.items()}
        
        with open(MODELS_DIR / "user_id_map.pkl", "wb") as f:
            pickle.dump(user_id_map, f)
        
        with open(MODELS_DIR / "item_id_map.pkl", "wb") as f:
            pickle.dump(item_id_map, f)
        
        with open(MODELS_DIR / "reverse_user_map.pkl", "wb") as f:
            pickle.dump(reverse_user_map, f)
        
        with open(MODELS_DIR / "reverse_item_map.pkl", "wb") as f:
            pickle.dump(reverse_item_map, f)
        
        # Content-based model
        np.save(MODELS_DIR / "business_similarity_matrix.npy", business_similarity)
        
        with open(MODELS_DIR / "business_id_map.pkl", "wb") as f:
            pickle.dump(business_id_map, f)
        
        with open(MODELS_DIR / "business_idx_map.pkl", "wb") as f:
            pickle.dump(business_idx_map, f)
        
        # Economic context
        with open(MODELS_DIR / "economic_context.pkl", "wb") as f:
            pickle.dump(economic_context, f)
        
        # Business-product affinity
        with open(MODELS_DIR / "business_product_affinity.pkl", "wb") as f:
            pickle.dump(business_product_affinity, f)
        
        logger.info("Hybrid recommendation model trained and saved successfully.")
        
        return hybrid_model
    
    except Exception as e:
        logger.error(f"Error training hybrid recommendation model: {e}")
        raise

def main():
    """
    Main entry point for model training.
    Trains the recommendation models using processed data.
    """
    logger.info("Starting model training with CUDA acceleration if available...")
    
    try:
        # Load data
        data = load_data()
        
        # Train hybrid recommendation model
        hybrid_model = train_hybrid_recommendation_model(data)
        
        # Evaluate the model on all available data
        logger.info("Evaluating hybrid recommendation model...")
        
        # Evaluate collaborative filtering component
        cf_metrics = evaluate_collaborative_filtering(
            hybrid_model['collaborative_filtering'],
            data['user_item_matrix'],
            test_size=0.2,
            k=10
        )
        logger.info(f"Collaborative filtering evaluation metrics: {cf_metrics}")
        
        # Evaluate business recommendation component
        business_metrics = evaluate_business_recommendations(
            hybrid_model['content_based']['business_similarity'],
            data['business_features'],
            k=5
        )
        logger.info(f"Business recommendation evaluation metrics: {business_metrics}")
        
        # Evaluate hybrid recommendations
        # This is a custom metric combining collaborative and content-based performance
        hybrid_score = (
            cf_metrics.get('f1@10', 0) * 0.5 +
            business_metrics.get('f1@5', 0) * 0.3 +
            (1 - min(cf_metrics.get('rmse', 100) / 100, 1)) * 0.2  # Normalize RMSE to 0-1 range
        )
        logger.info(f"Hybrid recommendation score: {hybrid_score:.4f}")
        
        # Save metrics for monitoring
        all_metrics = {
            'collaborative_filtering': cf_metrics,
            'business_recommendation': business_metrics,
            'hybrid_score': hybrid_score,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create metrics directory if it doesn't exist
        metrics_dir = Path("models/metrics")
        metrics_dir.mkdir(exist_ok=True, parents=True)
        
        # Save metrics as JSON
        with open(metrics_dir / "evaluation_metrics.json", 'w') as f:
            import json
            json.dump(all_metrics, f, indent=2)
        
        logger.info("All models trained and evaluated successfully.")
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

if __name__ == "__main__":
    main() 