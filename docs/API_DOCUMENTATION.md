# Buy-From-Egypt Recommendation API Documentation

## Overview

The Buy-From-Egypt Recommendation API provides personalized recommendations for Egyptian businesses and customers, leveraging machine learning models that incorporate collaborative filtering, content-based filtering, and Egyptian-specific economic context.

## Base URL

```
http://localhost:8000
```

## Authentication

The API is designed to integrate with the authentication mechanism of the main Buy-From-Egypt platform. Currently, no authentication is required for development purposes.

## API Endpoints

### Health Check

```
GET /
```

**Description:** Verify if the API is running.

**Response:**
```json
{
  "status": "Egyptian Recommendation API is running"
}
```

### Customer Recommendations

```
GET /recommend/customer/{customer_id}
```

**Description:** Get personalized product recommendations for a specific customer.

**Path Parameters:**
- `customer_id` - The unique customer identifier

**Query Parameters:**
- `num_recommendations` (optional, integer, default: 10) - Number of recommendations to return (min: 1, max: 100)
- `apply_economic_context` (optional, boolean, default: true) - Whether to adjust recommendations based on Egyptian economic indicators
- `include_egyptian_context` (optional, boolean, default: true) - Whether to include economic context data in the response

**Response:**
```json
{
  "user_id": "12395",
  "recommended_products": [
    {
      "StockCode": "21977",
      "Description": "PACK OF 60 PINK PAISLEY CAKE CASES",
      "Score": 0.95,
      "EgyptRelevance": 0.8
    },
    {
      "StockCode": "22632",
      "Description": "HAND WARMER RED POLKA DOT",
      "Score": 0.87,
      "EgyptRelevance": 0.5
    }
  ],
  "egyptian_context": {
    "gdp_growth": 4.35,
    "inflation": 5.04,
    "population_growth": 1.73,
    "tourism_sensitivity": 0.85,
    "economic_stability_index": 0.65,
    "trade_balance": -0.12,
    "is_winter_tourism_season": 1,
    "is_ramadan_season": 0,
    "current_date": "2025-05-03"
  }
}
```

### Business Recommendations

```
GET /recommend/business/{business_name}
```

**Description:** Get product and business partnership recommendations for a specific business.

**Path Parameters:**
- `business_name` - The name of the business

**Query Parameters:**
- `num_product_recommendations` (optional, integer, default: 10) - Number of product recommendations (min: 1, max: 100)
- `num_partner_recommendations` (optional, integer, default: 5) - Number of business partner recommendations (min: 1, max: 50)
- `apply_economic_context` (optional, boolean, default: true) - Whether to adjust recommendations based on Egyptian economic indicators
- `include_industry_weights` (optional, boolean, default: true) - Whether to include Egyptian industry weights in the response

**Response:**
```json
{
  "business_name": "Egyptian Imports",
  "recommended_products": [
    {
      "StockCode": "47567B",
      "Description": "TEA TIME KITCHEN APRON",
      "Score": 0.42,
      "EgyptRelevance": 0.04
    }
  ],
  "recommended_partners": [
    {
      "BusinessName": "Sahara Enterprises",
      "Category": "Agriculture",
      "Location": "Cairo, Egypt",
      "TradeType": "Importer",
      "SimilarityScore": 0.95,
      "Region": "Greater Cairo",
      "LogisticsAccess": 4,
      "MarketAccessScore": 3
    }
  ],
  "egyptian_context": {
    "gdp_growth": 4.35,
    "inflation": 5.04,
    "population_growth": 1.73,
    "tourism_sensitivity": 0.85,
    "economic_stability_index": 0.65,
    "trade_balance": -0.12,
    "is_winter_tourism_season": 1,
    "is_ramadan_season": 0,
    "current_date": "2025-05-03"
  },
  "industry_weights": {
    "Textiles": 0.15,
    "Agriculture": 0.18,
    "Spices": 0.12,
    "Fruits & Vegetables": 0.15,
    "Chemicals": 0.08,
    "Pharmaceuticals": 0.07,
    "Electronics": 0.06,
    "Machinery": 0.05,
    "Metals": 0.08,
    "Automobiles": 0.03,
    "Seafood": 0.06,
    "Manufacturing": 0.10
  }
}
```

### Egyptian Economic Context

```
GET /egyptian-economic-context
```

**Description:** Get current Egyptian economic context data used by the recommendation system.

**Response:**
```json
{
  "gdp_growth": 4.35,
  "inflation": 5.04,
  "population_growth": 1.73,
  "tourism_sensitivity": 0.85,
  "economic_stability_index": 0.65,
  "trade_balance": -0.12,
  "is_winter_tourism_season": 1,
  "is_ramadan_season": 0,
  "current_date": "2025-05-03"
}
```

### Export Customer Recommendations

```
GET /export/recommendations/customer/{customer_id}
```

**Description:** Export product recommendations for a customer in JSON or CSV format.

**Path Parameters:**
- `customer_id` - The unique customer identifier

**Query Parameters:**
- `num_recommendations` (optional, integer, default: 20) - Number of recommendations to export (min: 1, max: 100)
- `format` (optional, string, default: "json") - Export format, either 'json' or 'csv'

**Response:**
A file download containing the recommendations in the specified format.

### Export Business Recommendations

```
GET /export/recommendations/business/{business_name}
```

**Description:** Export product and business partnership recommendations in JSON or CSV format.

**Path Parameters:**
- `business_name` - The name of the business

**Query Parameters:**
- `num_product_recommendations` (optional, integer, default: 20) - Number of product recommendations (min: 1, max: 100)
- `num_partner_recommendations` (optional, integer, default: 10) - Number of business partner recommendations (min: 1, max: 50)
- `format` (optional, string, default: "json") - Export format, either 'json' or 'csv'

**Response:**
A file download containing the recommendations in the specified format.

## Data Synchronization Endpoints

These endpoints allow synchronizing data from the Buy-From-Egypt platform with the recommendation system.

### Sync User

```
POST /sync/user
```

**Description:** Synchronize user data with the recommendation system.

**Request Body:**
```json
{
  "userId": "user123",
  "name": "Cairo Textiles",
  "email": "info@cairotextiles.com",
  "type": "EXPORTER",
  "industrySector": "Textiles",
  "country": "Egypt",
  "active": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "User Cairo Textiles (user123) synced successfully",
  "syncedItem": {
    "userId": "user123",
    "name": "Cairo Textiles",
    "type": "EXPORTER"
  }
}
```

### Sync Product

```
POST /sync/product
```

**Description:** Synchronize product data with the recommendation system.

**Request Body:**
```json
{
  "productId": "prod123",
  "name": "Egyptian Cotton Shirt",
  "description": "High-quality cotton shirt made in Egypt",
  "price": 29.99,
  "currencyCode": "USD",
  "categoryId": "category123",
  "ownerId": "user456",
  "rating": 4.5,
  "reviewCount": 120,
  "active": true,
  "available": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Product Egyptian Cotton Shirt (prod123) synced successfully",
  "syncedItem": {
    "productId": "prod123",
    "name": "Egyptian Cotton Shirt",
    "ownerId": "user456"
  }
}
```

### Sync Order

```
POST /sync/order
```

**Description:** Synchronize order data with the recommendation system.

**Request Body:**
```json
{
  "orderId": "order123",
  "importerId": "importer123",
  "exporterId": "exporter456",
  "products": ["prod1", "prod2", "prod3"],
  "totalPrice": 150.75,
  "currencyCode": "USD",
  "createdAt": "2025-05-03T12:30:45Z"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Order order123 synced successfully",
  "syncedItem": {
    "orderId": "order123",
    "importerId": "importer123",
    "productCount": 3
  }
}
```

### Retrain Models

```
POST /admin/retrain
```

**Description:** Trigger a full retraining of all recommendation models.

**Response:**
```json
{
  "success": true,
  "message": "Model retraining initiated. This may take some time to complete.",
  "syncedItem": null
}
```

## Error Handling

The API returns appropriate HTTP status codes:

- 200: Successful operation
- 400: Bad request (missing or invalid parameters)
- 404: Resource not found (customer ID or business name not in database)
- 500: Server error

Error responses include a detailed message:

```json
{
  "detail": "Customer ID 99999 not found in model data."
}
```

## Integration with Buy-From-Egypt Platform

The API integrates with the Buy-From-Egypt platform's database schema:

- Customer recommendations use the `User.userId` as the customer ID
- Business recommendations use the `User.name` as the business name
- Product recommendations include `Product.productId` as the stock code and `Product.name` as the description
- Orders are used to build the interaction matrix for collaborative filtering

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `/api/docs`
- ReDoc: `/api/redoc` 