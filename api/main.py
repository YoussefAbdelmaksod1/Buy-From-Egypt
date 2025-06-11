from fastapi import FastAPI, HTTPException, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import logging
import uvicorn
import json
from datetime import datetime

from src.models.inference import load_recommendation_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define response models
class ProductRecommendation(BaseModel):
    StockCode: str
    Description: str
    Score: float
    EgyptRelevance: Optional[float] = None

class EgyptianContext(BaseModel):
    gdp_growth: Optional[float] = None
    inflation: Optional[float] = None
    population_growth: Optional[float] = None
    tourism_sensitivity: Optional[float] = None
    economic_stability_index: Optional[float] = None
    trade_balance: Optional[float] = None
    is_winter_tourism_season: Optional[int] = None
    is_ramadan_season: Optional[int] = None
    current_date: str

class CustomerRecommendationResponse(BaseModel):
    user_id: str
    recommended_products: List[ProductRecommendation]
    egyptian_context: Optional[EgyptianContext] = None

class BusinessPartnerRecommendation(BaseModel):
    BusinessName: str
    Category: str
    Location: str
    TradeType: str
    SimilarityScore: float
    Region: Optional[str] = None
    LogisticsAccess: Optional[int] = None
    MarketAccessScore: Optional[float] = None

class BusinessRecommendationResponse(BaseModel):
    business_name: str
    recommended_products: List[ProductRecommendation]
    recommended_partners: List[BusinessPartnerRecommendation]
    egyptian_context: EgyptianContext
    industry_weights: Optional[Dict[str, float]] = None

# Define additional models for database sync
class UserSync(BaseModel):
    userId: str
    name: str
    email: str
    type: str  # EXPORTER or IMPORTER
    industrySector: Optional[str] = None
    country: str
    active: bool = True

class ProductSync(BaseModel):
    productId: str
    name: str
    description: Optional[str] = None
    price: float
    currencyCode: str
    categoryId: str
    ownerId: str
    rating: Optional[float] = 0.0
    reviewCount: Optional[int] = 0
    active: bool = True
    available: bool = True

class OrderSync(BaseModel):
    orderId: str
    importerId: str
    exporterId: str
    products: List[str]  # List of product IDs
    totalPrice: float
    currencyCode: str
    createdAt: str

class SyncResponse(BaseModel):
    success: bool
    message: str
    syncedItem: Optional[Dict[str, Any]] = None

# Initialize FastAPI app
app = FastAPI(
    title="Egyptian Business Recommendation API",
    description="""
    # Egyptian Business Recommendation API
    
    This API provides recommendation services for Egyptian businesses and customers, integrating with the Buy-From-Egypt platform.
    
    ## Features
    
    - **Customer Product Recommendations**: Get personalized product recommendations for customers
    - **Business Recommendations**: Get product and business partnership recommendations for businesses
    - **Egyptian Economic Context**: Enhance recommendations with Egyptian economic indicators
    
    ## Integration with Buy-From-Egypt
    
    This recommendation system integrates with the Buy-From-Egypt platform database, which uses the following models:
    
    - **User**: Represents both importers and exporters
    - **Product**: Items that can be recommended to users
    - **Category**: Product categories that help with recommendations
    - **Order**: Historical transactions used to improve recommendation accuracy
    
    ## Authentication
    
    The recommendation API uses the same authentication mechanism as the main platform.
    
    ## Data Sources
    
    The system is trained on three primary data sources:
    
    1. **Retail Transaction Data**: User-product interactions
    2. **Business Profile Data**: Company-level attributes 
    3. **Egyptian Economic Indicators**: Macroeconomic contextual data
    
    ## Model Performance
    
    The recommendation system is evaluated using the following metrics:
    
    - **RMSE (Root Mean Square Error)**: Measures prediction accuracy
    - **Precision@k**: % of recommended items that are relevant
    - **Recall@k**: % of relevant items that were recommended
    - **F1 Score**: Harmonic mean of precision and recall
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation engine
recommendation_engine = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize resources on startup
    """
    global recommendation_engine
    logger.info("Initializing recommendation engine...")
    recommendation_engine = load_recommendation_engine()
    logger.info("API startup complete.")

@app.get("/", tags=["Health"])
async def root():
    """
    Health check endpoint.
    """
    return {"status": "Egyptian Recommendation API is running"}

def get_egyptian_context():
    """
    Helper function to get the current Egyptian economic context.
    """
    global recommendation_engine
    
    context = recommendation_engine.economic_context
    
    # Add current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    return EgyptianContext(
        gdp_growth=context.get('gdp_growth'),
        inflation=context.get('inflation'),
        population_growth=context.get('population_growth'),
        tourism_sensitivity=context.get('tourism_sensitivity'),
        economic_stability_index=context.get('economic_stability_index'),
        trade_balance=context.get('trade_balance'),
        is_winter_tourism_season=context.get('is_winter_tourism_season'),
        is_ramadan_season=context.get('is_ramadan_season'),
        current_date=current_date
    )

@app.get("/recommend/customer/{customer_id}", 
         response_model=CustomerRecommendationResponse,
         tags=["Customer Recommendations"])
async def recommend_for_customer(
    customer_id: str = Path(..., description="The unique customer ID"),
    num_recommendations: int = Query(10, description="Number of recommendations to return", ge=1, le=100),
    apply_economic_context: bool = Query(True, description="Whether to apply Egyptian economic context to adjust recommendations"),
    include_egyptian_context: bool = Query(True, description="Whether to include Egyptian economic context in response")
):
    """
    Get product recommendations for a specific customer with Egyptian context.
    
    - **customer_id**: The unique customer ID
    - **num_recommendations**: Number of recommendations to return (default: 10)
    - **apply_economic_context**: Whether to adjust recommendations based on Egyptian economic indicators
    - **include_egyptian_context**: Whether to include Egyptian context data in response
    """
    global recommendation_engine
    
    try:
        # Generate recommendations
        recommendations = recommendation_engine.recommend_products_for_customer(
            customer_id, num_recommendations
        )
        
        # Apply economic context if requested
        if apply_economic_context and recommendations:
            recommendations = recommendation_engine.combine_with_economic_context(recommendations)
        
        # Format response
        response = {
            "user_id": customer_id,
            "recommended_products": recommendations
        }
        
        # Add Egyptian context if requested
        if include_egyptian_context:
            response["egyptian_context"] = get_egyptian_context()
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating recommendations for customer {customer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/business/{business_name}", 
         response_model=BusinessRecommendationResponse,
         tags=["Business Recommendations"])
async def recommend_for_business(
    business_name: str = Path(..., description="The business name"),
    num_product_recommendations: int = Query(10, description="Number of product recommendations", ge=1, le=100),
    num_partner_recommendations: int = Query(5, description="Number of business partner recommendations", ge=1, le=50),
    apply_economic_context: bool = Query(True, description="Whether to apply Egyptian economic context to adjust recommendations"),
    include_industry_weights: bool = Query(True, description="Whether to include Egyptian industry weights")
):
    """
    Get product and business partnership recommendations for a specific Egyptian business.
    
    - **business_name**: The business name
    - **num_product_recommendations**: Number of product recommendations to return (default: 10)
    - **num_partner_recommendations**: Number of business partner recommendations to return (default: 5)
    - **apply_economic_context**: Whether to adjust recommendations based on Egyptian economic indicators (default: True)
    - **include_industry_weights**: Whether to include Egyptian industry weights in response
    """
    global recommendation_engine
    
    try:
        # Generate product recommendations
        product_recommendations = recommendation_engine.recommend_products_for_business(
            business_name, num_product_recommendations
        )
        
        # Generate business partner recommendations
        try:
            partner_recommendations = recommendation_engine.recommend_business_partners(
                business_name, num_partner_recommendations
            )
        except Exception as e:
            logger.warning(f"Business partner recommendations failed for {business_name}: {e}")
            partner_recommendations = []
        
        # Apply economic context if requested
        if apply_economic_context and product_recommendations:
            product_recommendations = recommendation_engine.combine_with_economic_context(product_recommendations)
        
        # Get Egyptian context
        egyptian_context = get_egyptian_context()
        
        # Format response
        response = {
            "business_name": business_name,
            "recommended_products": product_recommendations,
            "recommended_partners": partner_recommendations,
            "egyptian_context": egyptian_context
        }
        
        # Add industry weights if requested
        if include_industry_weights and 'industry_weights' in recommendation_engine.economic_context:
            response["industry_weights"] = recommendation_engine.economic_context.get('industry_weights')
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating recommendations for business {business_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/egyptian-economic-context", 
         response_model=EgyptianContext,
         tags=["Egyptian Context"])
async def get_economic_context():
    """
    Get current Egyptian economic context data.
    
    This endpoint provides the latest economic indicators for Egypt that are used
    to enrich the recommendation system.
    """
    try:
        return get_egyptian_context()
    except Exception as e:
        logger.error(f"Error retrieving Egyptian economic context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export/recommendations/customer/{customer_id}", 
         tags=["Export"])
async def export_customer_recommendations(
    customer_id: str = Path(..., description="The unique customer ID"),
    num_recommendations: int = Query(20, description="Number of recommendations to export", ge=1, le=100),
    format: str = Query("json", description="Export format (json or csv)")
):
    """
    Export product recommendations for a specific customer in JSON or CSV format.
    
    - **customer_id**: The unique customer ID
    - **num_recommendations**: Number of recommendations to export (default: 20)
    - **format**: Export format, either 'json' or 'csv' (default: json)
    """
    global recommendation_engine
    
    try:
        # Generate recommendations
        recommendations = recommendation_engine.recommend_products_for_customer(
            customer_id, num_recommendations
        )
        
        # Apply Egyptian economic context
        recommendations = recommendation_engine.combine_with_economic_context(recommendations)
        
        if format.lower() == "json":
            # Return JSON response
            return {
                "user_id": customer_id,
                "recommended_products": recommendations,
                "egyptian_context": get_egyptian_context().dict(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        elif format.lower() == "csv":
            # Generate CSV content
            headers = "StockCode,Description,Score,EgyptRelevance\n"
            rows = []
            for rec in recommendations:
                egypt_relevance = rec.get('EgyptRelevance', '')
                rows.append(f"{rec['StockCode']},\"{rec['Description']}\",{rec['Score']},{egypt_relevance}")
            
            csv_content = headers + "\n".join(rows)
            
            # Return as downloadable CSV
            from fastapi.responses import Response
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=recommendations_egyptian_customer_{customer_id}.csv"
                }
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'csv'.")
    
    except Exception as e:
        logger.error(f"Error exporting recommendations for customer {customer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export/recommendations/business/{business_name}", 
         tags=["Export"])
async def export_business_recommendations(
    business_name: str = Path(..., description="The business name"),
    num_product_recommendations: int = Query(20, description="Number of product recommendations", ge=1, le=100),
    num_partner_recommendations: int = Query(10, description="Number of business partner recommendations", ge=1, le=50),
    format: str = Query("json", description="Export format (json or csv)")
):
    """
    Export product and business partnership recommendations for a specific Egyptian business in JSON or CSV format.
    
    - **business_name**: The business name
    - **num_product_recommendations**: Number of product recommendations to export (default: 20)
    - **num_partner_recommendations**: Number of business partner recommendations to export (default: 10)
    - **format**: Export format, either 'json' or 'csv' (default: json)
    """
    global recommendation_engine
    
    try:
        # Generate recommendations
        product_recommendations = recommendation_engine.recommend_products_for_business(
            business_name, num_product_recommendations
        )
        
        try:
            partner_recommendations = recommendation_engine.recommend_business_partners(
                business_name, num_partner_recommendations
            )
        except Exception as e:
            logger.warning(f"Business partner recommendations failed for {business_name}: {e}")
            partner_recommendations = []
        
        # Apply Egyptian economic context
        product_recommendations = recommendation_engine.combine_with_economic_context(product_recommendations)
        
        if format.lower() == "json":
            # Return JSON response
            return {
                "business_name": business_name,
                "recommended_products": product_recommendations,
                "recommended_partners": partner_recommendations,
                "egyptian_context": get_egyptian_context().dict(),
                "industry_weights": recommendation_engine.economic_context.get('industry_weights'),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        elif format.lower() == "csv":
            # Generate CSV content for products
            product_headers = "StockCode,Description,Score,EgyptRelevance\n"
            product_rows = []
            for rec in product_recommendations:
                egypt_relevance = rec.get('EgyptRelevance', '')
                product_rows.append(f"{rec['StockCode']},\"{rec['Description']}\",{rec['Score']},{egypt_relevance}")
            
            product_csv = product_headers + "\n".join(product_rows)
            
            # Generate CSV content for business partners
            partner_headers = "BusinessName,Category,Location,TradeType,Region,SimilarityScore\n"
            partner_rows = []
            for rec in partner_recommendations:
                region = rec.get('Region', '')
                partner_rows.append(
                    f"\"{rec['BusinessName']}\",\"{rec['Category']}\",\"{rec['Location']}\","
                    f"\"{rec['TradeType']}\",\"{region}\",{rec['SimilarityScore']}"
                )
            
            partner_csv = partner_headers + "\n".join(partner_rows)
            
            # Return as downloadable CSV
            from fastapi.responses import Response
            return Response(
                content=f"EGYPTIAN BUSINESS PRODUCT RECOMMENDATIONS\n{product_csv}\n\nEGYPTIAN BUSINESS PARTNER RECOMMENDATIONS\n{partner_csv}",
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=recommendations_egyptian_business_{business_name}.csv"
                }
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'csv'.")
    
    except Exception as e:
        logger.error(f"Error exporting recommendations for business {business_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync/user", 
         response_model=SyncResponse,
         tags=["Sync"])
async def sync_user(
    user: UserSync = Body(..., description="User data to sync with recommendation system")
):
    """
    Sync a user profile with the recommendation system.
    
    This endpoint allows synchronizing user data from the main application with 
    the recommendation system to ensure recommendations stay current.
    
    The user data should match the Prisma User model structure.
    """
    global recommendation_engine
    
    try:
        # In a full implementation, this would update the user data in the recommendation system
        # For now, we'll just log the data and return success
        logger.info(f"Syncing user: {user.userId} ({user.name})")
        
        # Here you would implement the actual sync logic:
        # 1. Check if user exists in recommendation system
        # 2. If yes, update user data
        # 3. If no, add new user
        
        return {
            "success": True,
            "message": f"User {user.name} ({user.userId}) synced successfully",
            "syncedItem": {
                "userId": user.userId,
                "name": user.name,
                "type": user.type
            }
        }
    
    except Exception as e:
        logger.error(f"Error syncing user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync/product", 
         response_model=SyncResponse,
         tags=["Sync"])
async def sync_product(
    product: ProductSync = Body(..., description="Product data to sync with recommendation system")
):
    """
    Sync a product with the recommendation system.
    
    This endpoint allows synchronizing product data from the main application with 
    the recommendation system to ensure recommendations stay current.
    
    The product data should match the Prisma Product model structure.
    """
    global recommendation_engine
    
    try:
        # In a full implementation, this would update the product data in the recommendation system
        # For now, we'll just log the data and return success
        logger.info(f"Syncing product: {product.productId} ({product.name})")
        
        # Here you would implement the actual sync logic:
        # 1. Check if product exists in recommendation system
        # 2. If yes, update product data
        # 3. If no, add new product
        
        return {
            "success": True,
            "message": f"Product {product.name} ({product.productId}) synced successfully",
            "syncedItem": {
                "productId": product.productId,
                "name": product.name,
                "ownerId": product.ownerId
            }
        }
    
    except Exception as e:
        logger.error(f"Error syncing product: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync/order", 
         response_model=SyncResponse,
         tags=["Sync"])
async def sync_order(
    order: OrderSync = Body(..., description="Order data to sync with recommendation system")
):
    """
    Sync an order with the recommendation system.
    
    This endpoint allows synchronizing order data from the main application with 
    the recommendation system to improve collaborative filtering recommendations.
    
    The order data should match the Prisma Order model structure.
    """
    global recommendation_engine
    
    try:
        # In a full implementation, this would update the recommendation system with new interactions
        # For now, we'll just log the data and return success
        logger.info(f"Syncing order: {order.orderId} (Importer: {order.importerId}, Exporter: {order.exporterId})")
        
        # Here you would implement the actual sync logic:
        # 1. Add new user-item interactions based on order
        # 2. Update product popularity metrics
        # 3. Update user purchase history
        
        return {
            "success": True,
            "message": f"Order {order.orderId} synced successfully",
            "syncedItem": {
                "orderId": order.orderId,
                "importerId": order.importerId,
                "productCount": len(order.products)
            }
        }
    
    except Exception as e:
        logger.error(f"Error syncing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/retrain", 
         response_model=SyncResponse,
         tags=["Admin"])
async def retrain_models():
    """
    Trigger a full retraining of all recommendation models.
    
    This admin endpoint allows initiating a complete retraining of all recommendation models
    using the latest data. This is a resource-intensive operation and should be used sparingly.
    """
    try:
        # In a full implementation, this would trigger a background process to retrain models
        # For now, we'll just return success
        logger.info("Triggering full model retraining")
        
        # Here you would implement the actual retraining logic:
        # 1. Initiate background task to retrain all models
        # 2. Monitor progress
        # 3. Update models when complete
        
        return {
            "success": True,
            "message": "Model retraining initiated. This may take some time to complete.",
            "syncedItem": None
        }
    
    except Exception as e:
        logger.error(f"Error initiating model retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)