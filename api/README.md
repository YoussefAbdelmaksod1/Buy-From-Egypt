# Buy-From-Egypt Recommendation API

This directory contains the FastAPI implementation of the recommendation system API.

## Structure

- `main.py`: Main entry point for the FastAPI application
- Response models and endpoint implementations

## API Features

The API provides several key endpoints:

- **Customer Recommendations**: Personalized product suggestions for customers
- **Business Recommendations**: Product and business partnership recommendations
- **Egyptian Economic Context**: Economic indicators affecting recommendations
- **Export Functionality**: Export recommendations in JSON or CSV formats
- **Data Synchronization**: Sync user, product, and order data with the recommendation system

## Running the API

To run the API:

```bash
# From the project root directory
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## Implementation Details

### Request Flow

1. Request is received at an endpoint
2. The recommendation engine processes the request
3. Economic context is applied if requested
4. Response is formatted and returned

### Error Handling

The API uses FastAPI's exception handling to provide detailed error messages:

- 400: Bad request (missing or invalid parameters)
- 404: Resource not found (customer ID or business name not in database)
- 500: Server error

### Security

In production, you should add authentication and rate limiting to the API. Currently, the API is designed for development and doesn't include authentication. 