# Backend Implementation Guide for Buy-From-Egypt Recommendation System

This document outlines how to integrate the Buy-From-Egypt recommendation system with your backend services.

## API Integration

### Recommendation API Endpoints

The following endpoints are available for integration:

1. `/api/recommendations/user/{user_id}` - Get personalized recommendations for a user
2. `/api/recommendations/business/{business_id}` - Get similar businesses
3. `/api/recommendations/trending` - Get trending businesses based on recent activity
4. `/api/recommendations/regional/{region_id}` - Get recommendations specific to a region

### Data Synchronization Endpoints

To keep the recommendation system up-to-date:

1. `/api/sync/users` - Sync user profile updates
2. `/api/sync/businesses` - Sync business profile updates
3. `/api/sync/transactions` - Sync new transaction data

## Authentication

All API requests require a valid API key passed in the header:

```
Authorization: Bearer YOUR_API_KEY
```

## Request/Response Examples

### Example: Get User Recommendations

**Request:**
```
GET /api/recommendations/user/123
Authorization: Bearer YOUR_API_KEY
```

**Response:**
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "business_id": 45,
      "name": "Cairo Crafts",
      "category": "Handicrafts",
      "score": 0.85
    },
    {
      "business_id": 67,
      "name": "Alexandria Apparel",
      "category": "Clothing",
      "score": 0.79
    }
  ]
}
```

## Error Handling

The API uses standard HTTP status codes:

- 200: Success
- 400: Bad request
- 401: Unauthorized 
- 404: Resource not found
- 500: Server error

Each error response includes a message and error code:

```json
{
  "error": {
    "code": "INVALID_USER_ID",
    "message": "The provided user ID does not exist"
  }
}
```

## Rate Limiting

API requests are rate-limited to 100 requests per minute per API key. The response headers include rate limiting information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1621872000
```

## Testing with Postman

A Postman collection is available in `/docs/postman/buyFromEgypt.json`. Import this collection to test all API endpoints.

## Performance Considerations

- The recommendation API typically responds within 200ms
- For high-traffic scenarios, implement caching of recommendation results (TTL: 1 hour recommended)
- Batch synchronization requests when possible to reduce API calls 