import pandas as pd
import numpy as np
import logging
import pickle
import joblib
from pathlib import Path
import heapq
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

class RecommendationEngine:
    """
    Class to handle loading of trained models and generating recommendations.
    """
    
    def __init__(self):
        """
        Initialize the recommendation engine by loading all required models.
        """
        logger.info("Initializing recommendation engine...")
        
        try:
            # Create directories if they don't exist
            MODELS_DIR.mkdir(exist_ok=True)
            PROCESSED_DIR.mkdir(exist_ok=True)
            
            # Force using PyTorch CF to avoid issues with Implicit ALS
            self.use_pytorch_cf = True
            
            # Load collaborative filtering model and mappings
            try:
                with open(MODELS_DIR / "cf_model.pkl", "rb") as f:
                    self.cf_model = pickle.load(f)
                
                # If it's an Implicit ALS model, convert to matrix factorization format
                if not isinstance(self.cf_model, dict):
                    logger.info("Converting ALS model to matrix format for inference")
                    try:
                        user_factors = self.cf_model.user_factors
                        item_factors = self.cf_model.item_factors
                        self.cf_model = {
                            'user_factors': user_factors,
                            'item_factors': item_factors
                        }
                    except:
                        logger.warning("Could not convert ALS model, creating dummy factors")
                        self.cf_model = {"user_factors": np.random.rand(10, 10), "item_factors": np.random.rand(10, 10)}
                else:
                    logger.info("Loaded matrix factorization model")
            except:
                logger.warning("No collaborative filtering model found, creating dummy")
                # Create a dummy model
                self.cf_model = {"user_factors": np.random.rand(10, 10), "item_factors": np.random.rand(10, 10)}
            
            # Load ID mappings
            try:
                with open(MODELS_DIR / "user_id_map.pkl", "rb") as f:
                    self.user_id_map = pickle.load(f)
                
                with open(MODELS_DIR / "item_id_map.pkl", "rb") as f:
                    self.item_id_map = pickle.load(f)
                
                with open(MODELS_DIR / "reverse_user_map.pkl", "rb") as f:
                    self.reverse_user_map = pickle.load(f)
                
                with open(MODELS_DIR / "reverse_item_map.pkl", "rb") as f:
                    self.reverse_item_map = pickle.load(f)
            except:
                logger.warning("ID mappings not found, creating dummy mappings")
                self.user_id_map = {"1000": 0, "1001": 1}
                self.item_id_map = {"10000": 0, "10001": 1}
                self.reverse_user_map = {0: "1000", 1: "1001"}
                self.reverse_item_map = {0: "10000", 1: "10001"}
            
            # Load business recommendation model and mappings
            try:
                self.business_similarity = np.load(MODELS_DIR / "business_similarity_matrix.npy")
                
                with open(MODELS_DIR / "business_id_map.pkl", "rb") as f:
                    self.business_id_map = pickle.load(f)
                
                with open(MODELS_DIR / "business_idx_map.pkl", "rb") as f:
                    self.business_idx_map = pickle.load(f)
            except:
                logger.warning("Business similarity matrix not found, creating dummy")
                self.business_similarity = np.random.rand(10, 10)
                self.business_id_map = {"Business 1": 1, "Business 2": 2}
                self.business_idx_map = {1: 0, 2: 1}
            
            # Load economic context
            try:
                with open(MODELS_DIR / "economic_context.pkl", "rb") as f:
                    self.economic_context = pickle.load(f)
            except:
                logger.warning("Economic context not found, creating dummy")
                self.economic_context = {
                    'gdp_growth': 4.35,
                    'inflation': 5.04,
                    'population_growth': 1.73
                }
            
            # Load product info
            try:
                # Try to load product information from different potential sources
                if (PROCESSED_DIR / "retail_cleaned.csv").exists():
                    self.products_df = pd.read_csv(PROCESSED_DIR / "retail_cleaned.csv")[['StockCode', 'Description']].drop_duplicates()
                    self.products_df.set_index('StockCode', inplace=True)
                elif (DATA_DIR / "data.csv").exists():
                    # Load from original source with limited rows
                    self.products_df = pd.read_csv(DATA_DIR / "data.csv", encoding='ISO-8859-1', nrows=10000)[['StockCode', 'Description']].drop_duplicates()
                    self.products_df.set_index('StockCode', inplace=True)
            except:
                logger.warning("Product info not found, creating dummy")
                self.products_df = pd.DataFrame({
                    'StockCode': ["10000", "10001", "10002"],
                    'Description': ["Product 1", "Product 2", "Product 3"]
                }).set_index('StockCode')
            
            # Load business-product affinity mapping
            try:
                with open(MODELS_DIR / "business_product_affinity.pkl", "rb") as f:
                    self.business_product_affinity = pickle.load(f)
            except:
                logger.warning("Business-product affinity not found, creating dummy")
                self.business_product_affinity = {
                    "Business 1": [
                        {"StockCode": "10000", "Description": "Product 1", "Score": 0.8}
                    ]
                }
            
            # Load business data for additional info
            try:
                self.business_df = pd.read_csv(PROCESSED_DIR / "business_features.csv")
            except:
                logger.warning("Business features not found, creating dummy")
                self.business_df = pd.DataFrame({
                    'BusinessID': [1, 2],
                    'Business Name': ["Business 1", "Business 2"],
                    'Category': ["Electronics", "Textiles"],
                    'Location': ["Cairo, Egypt", "Alexandria, Egypt"],
                    'Trade Type': ["Importer", "Exporter"]
                })
            
            logger.info("Recommendation engine initialized successfully.")
        
        except Exception as e:
            logger.error(f"Error initializing recommendation engine: {e}")
            raise
    
    def recommend_products_for_customer(self, customer_id, num_recommendations=10):
        """
        Generate product recommendations for a specific customer.
        
        Args:
            customer_id (str): The customer ID
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended products with scores
        """
        logger.info(f"Generating recommendations for customer {customer_id}")
        
        try:
            # Convert to appropriate format if needed
            # Our model might have customer IDs stored as floats
            try:
                float_id = float(customer_id)
                if float_id in self.user_id_map:
                    customer_id = float_id
            except (ValueError, TypeError):
                pass
                
            # Check if customer exists in our model
            if customer_id not in self.user_id_map:
                logger.warning(f"Customer {customer_id} not found in model data.")
                return []
            
            # Get user index
            user_idx = self.user_id_map[customer_id]
            
            # Generate recommendations
            try:
                # PyTorch matrix factorization
                user_vector = self.cf_model['user_factors'][user_idx]
                item_vectors = self.cf_model['item_factors']
                
                # Calculate scores for each item - make sure dimensions align
                if item_vectors.shape[0] == user_vector.shape[0]:  # user_vector aligns with item_vector rows
                    scores = np.dot(user_vector, item_vectors)
                else:  # item_vectors needs transposition
                    scores = np.dot(user_vector, item_vectors.T)
                
                # Get top N items
                top_indices = np.argsort(-scores)[:num_recommendations]
                recommendations = [(idx, scores[idx]) for idx in top_indices]
            except Exception as e:
                logger.error(f"Error calculating recommendations: {e}")
                return []
            
            # Format results
            result = []
            for item_idx, score in recommendations:
                try:
                    stock_code = self.reverse_item_map[item_idx]
                    try:
                        description = self.products_df.loc[stock_code, 'Description']
                    except (KeyError, TypeError):
                        description = f"Product {stock_code}"
                        
                    result.append({
                        "StockCode": stock_code,
                        "Description": description,
                        "Score": float(score)
                    })
                except (KeyError, IndexError):
                    continue
            
            logger.info(f"Generated {len(result)} recommendations for customer {customer_id}")
            return result
        
        except Exception as e:
            logger.error(f"Error generating customer recommendations: {e}")
            raise
    
    def recommend_business_partners(self, business_name, num_recommendations=10):
        """
        Generate business partnership recommendations for a specific business.
        
        Args:
            business_name (str): The name of the business
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended businesses with similarity scores
        """
        logger.info(f"Generating partnership recommendations for business {business_name}")
        
        try:
            # Check if business exists in our model
            if business_name not in self.business_id_map:
                logger.warning(f"Business {business_name} not found in model data.")
                return []
            
            # Get business index
            business_id = self.business_id_map[business_name]
            business_idx = self.business_idx_map[business_id]
            
            # Get similarity scores for this business with all others
            similarity_scores = self.business_similarity[business_idx]
            
            # Get top N similar businesses (exclude self)
            top_indices = heapq.nlargest(num_recommendations + 1, 
                                         range(len(similarity_scores)), 
                                         key=lambda i: similarity_scores[i])
            
            # Remove the business itself from recommendations
            top_indices = [idx for idx in top_indices if idx != business_idx][:num_recommendations]
            
            # Format results
            result = []
            for idx in top_indices:
                try:
                    # Find the business name from the index
                    for b_name, b_id in self.business_id_map.items():
                        if self.business_idx_map.get(b_id) == idx:
                            # Look up additional business info
                            business_rows = self.business_df[self.business_df['Business Name'] == b_name]
                            if len(business_rows) > 0:
                                business_info = business_rows.iloc[0]
                                
                                result.append({
                                    "BusinessName": b_name,
                                    "Category": business_info.get('Category', 'Unknown'),
                                    "Location": business_info.get('Location', 'Unknown'),
                                    "TradeType": business_info.get('Trade Type', 'Unknown'),
                                    "SimilarityScore": float(similarity_scores[idx])
                                })
                            break
                except (KeyError, IndexError, ValueError):
                    continue
            
            logger.info(f"Generated {len(result)} partnership recommendations for business {business_name}")
            return result
        
        except Exception as e:
            logger.error(f"Error generating business partnership recommendations: {e}")
            raise
    
    def recommend_products_for_business(self, business_name, num_recommendations=10):
        """
        Generate product recommendations for a specific business.
        
        Args:
            business_name (str): The name of the business
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended products with scores
        """
        logger.info(f"Generating product recommendations for business {business_name}")
        
        try:
            # First check if we have pre-calculated affinity for this business
            if business_name in self.business_product_affinity:
                recommendations = self.business_product_affinity[business_name][:num_recommendations]
                logger.info(f"Found {len(recommendations)} pre-calculated recommendations for {business_name}")
                return recommendations
            
            # If no pre-calculated affinity, return empty list
            # In a complete implementation, we could generate on-the-fly recommendations
            logger.warning(f"No pre-calculated product recommendations found for {business_name}")
            return []
            
        except Exception as e:
            logger.error(f"Error generating business product recommendations: {e}")
            raise
    
    def combine_with_economic_context(self, recommendations, weight=0.2):
        """
        Adjust recommendation scores based on economic indicators.
        
        This is a simplified demonstration of how economic context could be used.
        In a complete implementation, this would be more sophisticated.
        
        Args:
            recommendations (list): Original recommendations
            weight (float): Weight of economic adjustment (0-1)
            
        Returns:
            list: Adjusted recommendations
        """
        # Extract economic indicators
        gdp_growth = self.economic_context.get('gdp_growth', 0)
        
        # Simple adjustment based on GDP growth
        # Positive growth → boost scores slightly
        # Negative growth → reduce scores slightly
        adjustment_factor = 1 + (gdp_growth / 100 * weight)
        
        # Apply adjustment
        for rec in recommendations:
            rec['Score'] = rec['Score'] * adjustment_factor
            rec['Score'] = min(rec['Score'], 1.0)  # Cap at 1.0
        
        return recommendations

def load_recommendation_engine():
    """
    Factory function to create and return a recommendation engine instance.
    """
    return RecommendationEngine() 