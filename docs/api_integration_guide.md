# Buy-From-Egypt API Integration Guide for Backend Developers

This guide provides detailed information for backend developers on how to integrate with the Buy-From-Egypt recommendation system and chatbot APIs.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Recommendation API](#recommendation-api)
   - [API Base URL](#api-base-url)
   - [Endpoints](#endpoints)
   - [Data Models](#data-models)
   - [Example Requests](#example-requests)
4. [Chatbot API](#chatbot-api)
   - [API Base URL](#chatbot-api-base-url)
   - [Endpoints](#chatbot-endpoints)
   - [Data Models](#chatbot-data-models)
   - [Example Requests](#chatbot-example-requests)
5. [Data Synchronization](#data-synchronization)
6. [Error Handling](#error-handling)
7. [Deployment Notes](#deployment-notes)

## Overview

The Buy-From-Egypt platform consists of two main API components:

1. **Recommendation API**: Provides personalized product and business recommendations for Egyptian customers and businesses, leveraging machine learning algorithms and Egyptian economic context.

2. **Business Chatbot API**: Provides an AI assistant specializing in Egyptian business knowledge, answering questions about industries, regions, and economic factors.

These APIs can be used independently or together to enhance your Egyptian e-commerce or business platform.

## Authentication

Both APIs currently use simple API key authentication via header:

```
X-API-Key: your_api_key_here
```

Contact the Buy-From-Egypt team for API key provisioning. In production, the APIs will support OAuth2 authentication.

## Recommendation API

### API Base URL

Development: `http://localhost:8000`  
Production: `https://api.buyfromegypt.com/recommendations`

### Endpoints

#### Health Check

```
GET /
```

**Response:**
```json
{
  "status": "Egyptian Recommendation API is running"
}
```

#### Get Customer Recommendations

```
GET /recommend/customer/{customer_id}
```

**Parameters:**
- `customer_id` (path): The ID of the customer to get recommendations for

**Response:**
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
    ...
  ],
  "recommended_categories": [
    {
      "category": "Food Processing",
      "confidence_score": 0.87
    },
    ...
  ]
}
```

#### Get Business Recommendations

```
GET /recommend/business/{business_name}
```

**Parameters:**
- `business_name` (path): The name of the business to get recommendations for
- `limit` (query, optional): Maximum number of recommendations to return (default: 5)

**Response:**
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
    ...
  ]
}
```

#### Get Egyptian Economic Context

```
GET /egyptian-economic-context
```

**Response:**
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

#### Export Recommendations as CSV

```
GET /export/recommendations/customer/{customer_id}?format=csv
```

**Parameters:**
- `customer_id` (path): The ID of the customer to export recommendations for
- `format` (query): Export format, either 'json' or 'csv'

**Response:**  
CSV file download or JSON response

### Data Models

#### Customer

```json
{
  "userId": "string",
  "name": "string",
  "email": "string",
  "type": "string",
  "industrySector": "string",
  "country": "string",
  "active": true
}
```

#### Product

```json
{
  "productId": "string",
  "name": "string",
  "description": "string",
  "price": 0,
  "currencyCode": "string",
  "categoryId": "string",
  "ownerId": "string",
  "rating": 0,
  "reviewCount": 0,
  "active": true,
  "available": true
}
```

#### Order

```json
{
  "orderId": "string",
  "importerId": "string",
  "exporterId": "string",
  "products": [
    "string"
  ],
  "totalPrice": 0,
  "currencyCode": "string",
  "createdAt": "2023-01-01T00:00:00Z"
}
```

### Example Requests

#### cURL

```bash
# Get customer recommendations
curl -X GET "http://localhost:8000/recommend/customer/10001" -H "accept: application/json" -H "X-API-Key: your_api_key_here"

# Get business recommendations
curl -X GET "http://localhost:8000/recommend/business/International%20Fruits%20Agriculture%20Egypt" -H "accept: application/json" -H "X-API-Key: your_api_key_here"
```

#### Python

```python
import requests

# Base URL
base_url = "http://localhost:8000"
headers = {"X-API-Key": "your_api_key_here"}

# Get customer recommendations
customer_id = "10001"
response = requests.get(f"{base_url}/recommend/customer/{customer_id}", headers=headers)
print(response.json())

# Get business recommendations
business_name = "International Fruits Agriculture Egypt"
response = requests.get(f"{base_url}/recommend/business/{business_name}", headers=headers)
print(response.json())
```

## Data Synchronization

### Sync User Data

```
POST /sync/user
```

**Request Body:**
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

**Response:**
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

### Sync Product Data

```
POST /sync/product
```

**Request Body:**
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

**Response:**
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

### Sync Order Data

```
POST /sync/order
```

**Request Body:**
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

**Response:**
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

### Chatbot API Base URL

Development: `http://localhost:8080`  
Production: `https://api.buyfromegypt.com/chatbot`

### Chatbot Endpoints

#### Health Check

```
GET /
```

**Response:**
```json
{
  "status": "Egyptian Business Chatbot API is running"
}
```

#### Chat Conversation

```
POST /chat
```

**Request Body:**
```json
{
  "user_id": "user123",
  "message": "What are the best seasons for exporting Egyptian citrus?",
  "conversation_id": "conv789"
}
```

**Response:**
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

#### Get Industry Information

```
GET /industry/{industry_name}
```

**Parameters:**
- `industry_name` (path): The name of the industry to get information about

**Response:**
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

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication failure
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error

Error responses follow this format:

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested resource was not found",
    "details": "Customer ID 99999 does not exist in the system"
  }
}
```

## Deployment Notes

### Requirements

- Python 3.8 or higher
- FastAPI
- PostgreSQL database (for production)
- Redis (optional, for caching)

### Environment Variables

The following environment variables need to be set:

- `API_KEY`: API key for authentication
- `DATABASE_URL`: Connection string for PostgreSQL (production only)
- `REDIS_URL`: Connection string for Redis (optional)
- `ENVIRONMENT`: Either "development" or "production"
- `GOOGLE_API_KEY`: API key for the Gemini chatbot

### Deployment Options

1. **Docker**: Docker images are available for both APIs
2. **Kubernetes**: Helm charts are provided for Kubernetes deployment
3. **Serverless**: AWS Lambda configurations are available

Contact the Buy-From-Egypt team for detailed deployment instructions. 