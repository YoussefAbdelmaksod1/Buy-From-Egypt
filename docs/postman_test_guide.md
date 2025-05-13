# Buy-From-Egypt API Testing Guide

This document provides sample requests for testing the Buy-From-Egypt recommendation and chatbot APIs using Postman or similar tools.

## Recommendation API

### Base URL: `http://localhost:8000` (development) or `https://api.buyfromegypt.com/recommendations` (production)

### 1. Health Check

**Request:**
```
GET /
```

**Postman Example:**
- Method: `GET`
- URL: `http://localhost:8000/`
- Headers: `Accept: application/json`

**Expected Response:**
```json
{
  "status": "Egyptian Recommendation API is running"
}
```

### 2. Customer Recommendations

**Request:**
```
GET /recommend/customer/{customer_id}
```

**Postman Example:**
- Method: `GET`
- URL: `http://localhost:8000/recommend/customer/10001`
- Headers: `Accept: application/json`

**Expected Response:**
```json
{
  "user_id": "10001",
  "recommended_products": [
    {
      "StockCode": "EGY00063",
      "Description": "Food Processing Equipment",
      "UnitPrice": 239.99,
      "confidence_score": 0.92
    },
    {
      "StockCode": "EGY00127",
      "Description": "Packaging Materials",
      "UnitPrice": 45.50,
      "confidence_score": 0.87
    }
  ],
  "recommended_categories": [
    {
      "category": "Food Processing",
      "confidence_score": 0.87
    },
    {
      "category": "Packaging",
      "confidence_score": 0.75
    }
  ]
}
```

### 3. Business Recommendations

**Request:**
```
GET /recommend/business/{business_name}
```

**Postman Example:**
- Method: `GET`
- URL: `http://localhost:8000/recommend/business/International%20Fruits%20Agriculture%20Egypt`
- Headers: `Accept: application/json`

**Expected Response:**
```json
{
  "business_name": "International Fruits Agriculture Egypt",
  "recommended_products": [],
  "recommended_businesses": [
    {
      "business_name": "Cairo Packaging Solutions",
      "industry": "Packaging",
      "match_score": 0.95,
      "reason": "Complementary business that handles packaging of agricultural exports"
    },
    {
      "business_name": "Alexandria Shipping Services",
      "industry": "Logistics",
      "match_score": 0.92,
      "reason": "Provides shipping services for agricultural products"
    }
  ]
}
```

### 4. Egyptian Economic Context

**Request:**
```
GET /egyptian-economic-context
```

**Postman Example:**
- Method: `GET`
- URL: `http://localhost:8000/egyptian-economic-context`
- Headers: `Accept: application/json`

**Expected Response:**
```json
{
  "gdp_growth": 4.35,
  "inflation": 5.04,
  "population_growth": 1.73,
  "tourism_sensitivity": 0.85,
  "economic_sentiment": "positive",
  "major_exports": [
    "textiles",
    "agricultural products",
    "petroleum products",
    "furniture"
  ],
  "seasonal_factors": {
    "current_season": "winter",
    "high_season_industries": [
      "textiles",
      "citrus exports"
    ]
  }
}
```

### 5. Export Recommendations

**Request:**
```
GET /export/recommendations/customer/{customer_id}?format=json
```

**Postman Example:**
- Method: `GET`
- URL: `http://localhost:8000/export/recommendations/customer/10001?format=json`
- Headers: `Accept: application/json`

**Expected Response:**
Same as customer recommendations but in a downloadable format.

## Data Synchronization Endpoints

### 1. Sync User

**Request:**
```
POST /sync/user
```

**Postman Example:**
- Method: `POST`
- URL: `http://localhost:8000/sync/user`
- Headers: 
  - `Content-Type: application/json`
  - `Accept: application/json`
- Body (raw JSON):
```json
{
  "userId": "user123",
  "name": "Test User",
  "email": "test@example.com",
  "type": "IMPORTER",
  "industrySector": "Textiles",
  "country": "Egypt",
  "active": true
}
```

**Expected Response:**
```json
{
  "success": true,
  "message": "User Test User (user123) synced successfully",
  "syncedItem": {
    "userId": "user123",
    "name": "Test User",
    "type": "IMPORTER"
  }
}
```

### 2. Sync Product

**Request:**
```
POST /sync/product
```

**Postman Example:**
- Method: `POST`
- URL: `http://localhost:8000/sync/product`
- Headers: 
  - `Content-Type: application/json`
  - `Accept: application/json`
- Body (raw JSON):
```json
{
  "productId": "prod123",
  "name": "Egyptian Cotton Fabric",
  "description": "High quality Egyptian cotton fabric for clothing",
  "price": 25.99,
  "currencyCode": "USD",
  "categoryId": "textiles",
  "ownerId": "business1",
  "rating": 4.8,
  "reviewCount": 24,
  "active": true,
  "available": true
}
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Product Egyptian Cotton Fabric (prod123) synced successfully",
  "syncedItem": {
    "productId": "prod123",
    "name": "Egyptian Cotton Fabric",
    "ownerId": "business1"
  }
}
```

### 3. Sync Order

**Request:**
```
POST /sync/order
```

**Postman Example:**
- Method: `POST`
- URL: `http://localhost:8000/sync/order`
- Headers: 
  - `Content-Type: application/json`
  - `Accept: application/json`
- Body (raw JSON):
```json
{
  "orderId": "order123",
  "importerId": "user123",
  "exporterId": "business1",
  "products": ["prod123", "prod456"],
  "totalPrice": 149.99,
  "currencyCode": "USD",
  "createdAt": "2023-01-15T14:30:00Z"
}
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Order order123 synced successfully",
  "syncedItem": {
    "orderId": "order123",
    "importerId": "user123",
    "productCount": 2
  }
}
```

## Chatbot API

### Base URL: `http://localhost:8080` (development) or `https://api.buyfromegypt.com/chatbot` (production)

### 1. Health Check

**Request:**
```
GET /
```

**Postman Example:**
- Method: `GET`
- URL: `http://localhost:8080/`
- Headers: `Accept: application/json`

**Expected Response:**
```json
{
  "status": "Egyptian Business Chatbot API is running"
}
```

### 2. Chat Conversation

**Request:**
```
POST /chat
```

**Postman Example:**
- Method: `POST`
- URL: `http://localhost:8080/chat`
- Headers: 
  - `Content-Type: application/json`
  - `Accept: application/json`
- Body (raw JSON):
```json
{
  "user_id": "user123",
  "message": "What are the best seasons for exporting Egyptian citrus?",
  "conversation_id": "conv789"
}
```

**Expected Response:**
```json
{
  "response": "The best seasons for exporting Egyptian citrus are from December to April. During this period, navel oranges, mandarins, and grapefruits are harvested and available for export. Egypt is one of the world's largest citrus exporters, with main markets in Russia, Saudi Arabia, Netherlands, and UAE. The Mediterranean climate in the Nile Delta and North Coast provides ideal growing conditions for high-quality citrus fruits.",
  "conversation_id": "conv789",
  "sources": [
    {
      "title": "Egyptian Agricultural Export Guide",
      "url": "https://example.com/egyptian-agriculture"
    }
  ]
}
```

### 3. Get Industry Information

**Request:**
```
GET /industry/{industry_name}
```

**Postman Example:**
- Method: `GET`
- URL: `http://localhost:8080/industry/Textiles`
- Headers: `Accept: application/json`

**Expected Response:**
```json
{
  "industry": "Textiles",
  "description": "Egypt's textile industry is one of its oldest and most established sectors...",
  "key_regions": ["Alexandria", "Greater Cairo", "Mahalla al-Kubra"],
  "annual_export_value": "3.8 billion USD",
  "growth_rate": 5.7,
  "key_products": ["Cotton fabrics", "Ready-made garments", "Home textiles"],
  "seasonal_factors": {
    "peak_seasons": ["Winter", "Spring"],
    "notes": "Higher demand for Egyptian cotton products typically occurs in Q4 and Q1..."
  }
}
```

## Error Handling Tests

### 1. Invalid Customer ID

**Request:**
```
GET /recommend/customer/invalid_id
```

**Postman Example:**
- Method: `GET`
- URL: `http://localhost:8000/recommend/customer/invalid_id`
- Headers: `Accept: application/json`

**Expected Response:**
```json
{
  "error": {
    "code": "INVALID_CUSTOMER_ID",
    "message": "Invalid customer ID format",
    "details": "Customer ID must be a number"
  }
}
```

### 2. Non-existent Business

**Request:**
```
GET /recommend/business/Nonexistent%20Business
```

**Postman Example:**
- Method: `GET`
- URL: `http://localhost:8000/recommend/business/Nonexistent%20Business`
- Headers: `Accept: application/json`

**Expected Response:**
```json
{
  "error": {
    "code": "BUSINESS_NOT_FOUND",
    "message": "The requested business was not found",
    "details": "Business 'Nonexistent Business' does not exist in the system"
  }
}
```

## Postman Collection

A complete Postman collection file is available for import at:
[Buy-From-Egypt_API_Collection.json](https://github.com/buy-from-egypt/api-collection/raw/main/Buy-From-Egypt_API_Collection.json)

To import:
1. Open Postman
2. Click "Import" in the upper left
3. Upload the collection file or paste the URL
4. All requests will be available in your Postman workspace

## Environment Setup

For easy switching between development and production environments, we recommend creating environments in Postman:

### Development Environment Variables:
- `base_url`: http://localhost:8000
- `chatbot_url`: http://localhost:8080
- `api_key`: your_dev_api_key

### Production Environment Variables:
- `base_url`: https://api.buyfromegypt.com/recommendations
- `chatbot_url`: https://api.buyfromegypt.com/chatbot
- `api_key`: your_prod_api_key

This allows you to easily switch between environments without modifying each request. 