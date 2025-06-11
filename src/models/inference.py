import pandas as pd
import logging
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import json

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

class RegularizedContentModel(nn.Module):
    """
    Content-based model with strong regularization to prevent overfitting
    """
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
    """
    Collaborative Filtering Model for customer-product interactions
    """
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

class RecommendationEngine:
    """
    Recommendation engine using trained PyTorch models
    """

    def __init__(self):
        """
        Initialize the recommendation engine by loading trained PyTorch models
        """
        logger.info("Initializing Recommendation Engine...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize model containers
        self.content_model = None
        self.cf_model = None

        # Initialize encoders and scalers
        self.business_encoder = None
        self.product_encoder = None
        self.customer_encoder = None
        self.feature_scaler = None

        try:
            # Create directories if they don't exist
            MODELS_DIR.mkdir(exist_ok=True)
            PROCESSED_DIR.mkdir(exist_ok=True)

            # Load trained models
            self._load_models()

            # Load additional data
            self._load_additional_data()

            logger.info("Models loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.info("Falling back to basic recommendations...")
            self._load_fallback_models()

    def _load_models(self):
        """Load the trained PyTorch models"""
        logger.info("Loading trained PyTorch models...")

        # Load encoders and scalers
        try:
            with open(MODELS_DIR / "regularized_business_encoder.pkl", "rb") as f:
                self.business_encoder = pickle.load(f)
            logger.info("Loaded business encoder")

            with open(MODELS_DIR / "regularized_product_encoder.pkl", "rb") as f:
                self.product_encoder = pickle.load(f)
            logger.info("Loaded product encoder")

            with open(MODELS_DIR / "regularized_feature_scaler.pkl", "rb") as f:
                self.feature_scaler = pickle.load(f)
            logger.info("Loaded feature scaler")

        except Exception as e:
            logger.error(f"Error loading encoders: {e}")
            raise

        # Load content-based model
        try:
            n_businesses = len(self.business_encoder.classes_)
            n_products = len(self.product_encoder.classes_)
            n_features = 6  # Number of features we use

            self.content_model = RegularizedContentModel(
                n_businesses, n_products, n_features, embedding_dim=64
            ).to(self.device)

            self.content_model.load_state_dict(
                torch.load(MODELS_DIR / "regularized_content_model.pth", map_location=self.device)
            )
            self.content_model.eval()
            logger.info("Loaded content-based model")

        except Exception as e:
            logger.error(f"Error loading content model: {e}")
            raise

        # Load collaborative filtering model
        try:
            with open(MODELS_DIR / "cf_customer_encoder.pkl", "rb") as f:
                self.customer_encoder = pickle.load(f)

            with open(MODELS_DIR / "cf_model_info.json", "r") as f:
                cf_info = json.load(f)

            self.cf_model = CollaborativeFilteringModel(
                cf_info['n_customers'], cf_info['n_products'], cf_info['embedding_dim']
            ).to(self.device)

            self.cf_model.load_state_dict(
                torch.load(MODELS_DIR / "cf_model.pth", map_location=self.device)
            )
            self.cf_model.eval()
            logger.info("Loaded collaborative filtering model")

        except Exception as e:
            logger.warning(f"Could not load CF model: {e}")
            self.cf_model = None

    def _load_fallback_models(self):
        """Load fallback models if trained models fail"""
        logger.info("Loading fallback models...")

        try:
            # Load business-product affinity as fallback
            with open(MODELS_DIR / "business_product_affinity.pkl", "rb") as f:
                self.business_product_affinity = pickle.load(f)
            logger.info("Loaded fallback business-product affinity")
        except:
            logger.warning("No fallback models available")
            self.business_product_affinity = {}
            
    def _load_additional_data(self):
        """Load additional data needed for recommendations"""
        logger.info("Loading additional data...")

        try:
            # Load product mapping for descriptions
            with open(MODELS_DIR / "product_name_mapping.pkl", "rb") as f:
                self.product_mapping = pickle.load(f)
            logger.info("Loaded product name mapping")

            # Load business data
            self.business_df = pd.read_csv(PROCESSED_DIR / "business_cleaned.csv")
            logger.info("Loaded business data")

            # Load economic context
            self.economic_context = {
                'gdp_growth': 4.35,
                'inflation': 5.04,
                'population_growth': 1.73
            }
            logger.info("Loaded economic context")

        except Exception as e:
            logger.warning(f"Error loading additional data: {e}")
            self.product_mapping = {}
            self.business_df = pd.DataFrame()
            self.economic_context = {
                'gdp_growth': 4.35,
                'inflation': 5.04,
                'population_growth': 1.73
            }
    
    def recommend_products_for_customer(self, customer_id, num_recommendations=10):
        """
        Generate product recommendations using trained collaborative filtering model

        Args:
            customer_id (str): The customer ID
            num_recommendations (int): Number of recommendations to return

        Returns:
            list: List of recommended products with scores from trained model
        """
        logger.info(f"Generating recommendations for customer {customer_id}")

        if self.cf_model is None:
            logger.warning("No collaborative filtering model available")
            return []

        try:
            # Convert customer ID to appropriate format
            try:
                customer_id = float(customer_id)
            except (ValueError, TypeError):
                pass

            # Check if customer exists in trained model
            if customer_id not in self.customer_encoder.classes_:
                logger.warning(f"Customer {customer_id} not found in trained model")
                return []

            # Encode customer
            customer_encoded = self.customer_encoder.transform([customer_id])[0]
            customer_tensor = torch.LongTensor([customer_encoded]).to(self.device)

            # Get all products from the trained model
            all_products = self.product_encoder.classes_
            product_ids = torch.LongTensor(range(len(all_products))).to(self.device)

            # Repeat customer ID for all products
            customer_ids = customer_tensor.repeat(len(all_products))

            # Generate predictions using trained model
            with torch.no_grad():
                scores = self.cf_model(customer_ids, product_ids)

            # Get top recommendations
            top_indices = torch.argsort(scores, descending=True)[:num_recommendations]

            # Format results with product names
            result = []
            for idx in top_indices:
                product_code = all_products[idx.item()]
                score = scores[idx.item()].cpu().numpy()

                # Get Egyptian product name
                description = self.product_mapping.get(product_code, f"Product {product_code}")

                result.append({
                    "StockCode": product_code,
                    "Description": description,
                    "Score": float(score)
                })

            logger.info(f"Generated {len(result)} recommendations for customer {customer_id}")
            return result

        except Exception as e:
            logger.error(f"Error in customer recommendations: {e}")
            return []
    
    def recommend_business_partners(self, business_name, num_recommendations=10):
        """
        Generate business partnership recommendations based on category similarity

        Args:
            business_name (str): The name of the business
            num_recommendations (int): Number of recommendations to return

        Returns:
            list: List of recommended businesses with similarity scores
        """
        logger.info(f"Generating partnership recommendations for business {business_name}")

        try:
            # Find the business in our data
            business_row = self.business_df[self.business_df['Business Name'] == business_name]
            if len(business_row) == 0:
                logger.warning(f"Business {business_name} not found in data")
                return []

            business_info = business_row.iloc[0]
            business_category = business_info.get('Category', '')
            business_location = business_info.get('Location', '')

            # Find similar businesses based on category and location
            similar_businesses = []

            for _, other_business in self.business_df.iterrows():
                other_name = other_business['Business Name']

                # Skip self
                if other_name == business_name:
                    continue

                other_category = other_business.get('Category', '')
                other_location = other_business.get('Location', '')

                # Calculate similarity score
                similarity_score = 0.0

                # Category similarity (70% weight)
                if business_category and other_category:
                    if business_category.lower() == other_category.lower():
                        similarity_score += 0.7
                    elif any(word in other_category.lower() for word in business_category.lower().split()):
                        similarity_score += 0.4

                # Location similarity (30% weight)
                if business_location and other_location:
                    if business_location.lower() == other_location.lower():
                        similarity_score += 0.3
                    elif any(word in other_location.lower() for word in business_location.lower().split()):
                        similarity_score += 0.15

                if similarity_score > 0:
                    similar_businesses.append({
                        "BusinessName": other_name,
                        "Category": other_category,
                        "Location": other_location,
                        "TradeType": other_business.get('Trade Type', 'Unknown'),
                        "SimilarityScore": similarity_score
                    })

            # Sort by similarity score and return top N
            similar_businesses.sort(key=lambda x: x['SimilarityScore'], reverse=True)
            result = similar_businesses[:num_recommendations]

            logger.info(f"Generated {len(result)} partnership recommendations for business {business_name}")
            return result

        except Exception as e:
            logger.error(f"Error generating business partnership recommendations: {e}")
            return []
    
    def recommend_products_for_business(self, business_name, num_recommendations=10):
        """
        Generate product recommendations using trained content-based model

        Args:
            business_name (str): The name of the business
            num_recommendations (int): Number of recommendations to return

        Returns:
            list: List of recommended products with scores from trained model
        """
        logger.info(f"Generating recommendations for business {business_name}")

        if self.content_model is None:
            logger.warning("No content model available, using fallback")
            return self._fallback_business_recommendations(business_name, num_recommendations)

        try:
            # Check if business exists in trained model
            if business_name not in self.business_encoder.classes_:
                logger.warning(f"Business {business_name} not found in trained model")
                return self._fallback_business_recommendations(business_name, num_recommendations)

            # Encode business
            business_id = self.business_encoder.transform([business_name])[0]
            business_tensor = torch.LongTensor([business_id]).to(self.device)

            # Get business features from data
            business_row = self.business_df[self.business_df['Business Name'] == business_name]
            if len(business_row) == 0:
                logger.warning(f"Business data not found for {business_name}")
                return []

            business_info = business_row.iloc[0]

            # Extract features (same as training)
            business_features = [
                min(business_info.get('Annual Trade Volume (M USD)', 0) / 1000, 1.0),
                business_info.get('Trade Success Rate (%)', 50) / 100,
                business_info.get('EgyptianAdvantageScore', 0.5),
                business_info.get('MarketAccessScore', 0.5),
                0.5,  # Average description length
                0.5   # Average word count
            ]

            # Normalize features using trained scaler
            business_features = self.feature_scaler.transform([business_features])[0]

            # Get all products from trained model
            all_products = self.product_encoder.classes_
            product_ids = torch.LongTensor(range(len(all_products))).to(self.device)

            # Repeat business and features for all products
            business_ids = business_tensor.repeat(len(all_products))
            features = torch.FloatTensor([business_features] * len(all_products)).to(self.device)

            # Generate predictions using trained model
            with torch.no_grad():
                scores = self.content_model(business_ids, product_ids, features)

            # Get top recommendations
            top_indices = torch.argsort(scores, descending=True)[:num_recommendations]

            # Format results with Egyptian product names
            result = []
            for idx in top_indices:
                product_code = all_products[idx.item()]
                score = scores[idx.item()].cpu().numpy()

                # Get Egyptian product name
                description = self.product_mapping.get(product_code, f"Product {product_code}")

                result.append({
                    "StockCode": product_code,
                    "Description": description,
                    "Score": float(score)
                })

            logger.info(f"Generated {len(result)} recommendations for business {business_name}")
            return result

        except Exception as e:
            logger.error(f"Error in business recommendations: {e}")
            return self._fallback_business_recommendations(business_name, num_recommendations)

    def _fallback_business_recommendations(self, business_name, num_recommendations):
        """Fallback to pre-calculated recommendations if real model fails"""
        try:
            if hasattr(self, 'business_product_affinity') and business_name in self.business_product_affinity:
                recommendations = self.business_product_affinity[business_name][:num_recommendations]
                logger.info(f"Using fallback recommendations for {business_name}")
                return recommendations
        except:
            pass

        logger.warning(f"No recommendations available for {business_name}")
        return []
    
    def combine_with_economic_context(self, recommendations, weight=0.2):
        """
        Adjust recommendation scores based on Egyptian economic indicators

        Args:
            recommendations (list): Original recommendations from trained models
            weight (float): Weight of economic adjustment (0-1)

        Returns:
            list: Economically adjusted recommendations
        """
        logger.info("Applying Egyptian economic context to recommendations")

        # Extract Egyptian economic indicators
        gdp_growth = self.economic_context.get('gdp_growth', 4.35)

        # Egyptian economic adjustment
        # Positive growth → boost scores for Egyptian products
        # This simulates economic favorability
        adjustment_factor = 1 + (gdp_growth / 100 * weight)

        # Apply adjustment to generated scores
        for rec in recommendations:
            rec['Score'] = rec['Score'] * adjustment_factor
            rec['Score'] = min(rec['Score'], 1.0)  # Cap at 1.0

            # Add Egyptian context flag
            rec['EconomicAdjustment'] = adjustment_factor

        logger.info(f"Applied economic adjustment factor: {adjustment_factor:.3f}")
        return recommendations

    def get_model_status(self):
        """Get status of loaded models"""
        return {
            "content_model_loaded": self.content_model is not None,
            "cf_model_loaded": self.cf_model is not None,
            "business_encoder_loaded": self.business_encoder is not None,
            "product_encoder_loaded": self.product_encoder is not None,
            "customer_encoder_loaded": self.customer_encoder is not None,
            "device": str(self.device),
            "model_type": "Trained PyTorch Models" if self.content_model else "Fallback Models"
        }

def load_recommendation_engine():
    """
    Factory function to create and return a recommendation engine instance
    """
    logger.info("Loading Recommendation Engine...")
    return RecommendationEngine()