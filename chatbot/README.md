# Buy-From-Egypt Business Chatbot

A Gemini-powered chatbot API designed specifically for Egyptian businesses using the Buy-From-Egypt recommendation system. The chatbot provides intelligent responses about product recommendations, business partnerships, and Egyptian market insights.

## Features

- **Egypt-Specific Knowledge**: Incorporates detailed information about Egyptian industries, regions, and business seasonality
- **Recommendation System Expertise**: Provides guidance on using the recommendation API
- **Business Context-Aware**: Tailors responses based on business industry, region, and attributes
- **Knowledge-Enhanced Responses**: Leverages structured knowledge for more accurate answers
- **Conversation History**: Maintains context throughout conversations
- **REST API Integration**: Easy to integrate with web and mobile applications

## Setup

### Prerequisites

- Python 3.8+
- Google API key for Gemini (get one from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Buy-From-Egypt/chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the chatbot directory with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

### Running the Chatbot API

Start the chatbot API:

```bash
python chatbot_api.py
```

The API will be available at `http://localhost:8080`

## API Documentation

### Chat Endpoint

```
POST /chat
```

Get a response from the chatbot about the Buy-From-Egypt recommendation system.

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "How can the recommendation system help my textile business find new partners?"
    }
  ],
  "business_context": {
    "name": "Cairo Textile Exports",
    "industry": "Textiles",
    "region": "Greater Cairo",
    "business_type": "Exporter",
    "size": "Medium",
    "target_markets": ["Europe", "Gulf Countries"]
  }
}
```

**Response:**
```json
{
  "response": "The Buy-From-Egypt recommendation system can help your textile business find new partners in several ways...",
  "sources": [
    "Industry: Textiles",
    "Region: Greater Cairo",
    "Technical: api_integration"
  ],
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "processing_time": 2.45
}
```

### Health Check

```
GET /
```

Verify if the chatbot API is running.

**Response:**
```json
{
  "status": "Buy-From-Egypt Business Chatbot API is running"
}
```

### Knowledge Topics

```
GET /knowledge-topics
```

Get a list of knowledge topics that the chatbot is trained on.

**Response:**
```json
{
  "topics": [
    "Egyptian Industry Sectors",
    "Regional Business Characteristics",
    "Seasonality Factors",
    "Recommendation System Technical Details",
    "Common Business Questions"
  ],
  "categories": {
    "Egyptian Industry Sectors": ["Textiles", "Agriculture", "Spices", "Fruits & Vegetables"],
    "Regional Business Characteristics": ["Greater Cairo", "Mediterranean Coast", "Suez Canal"]
  }
}
```

### Industry Information

```
GET /industry/{industry_name}
```

Get detailed information about a specific Egyptian industry.

### Region Information

```
GET /region/{region_name}
```

Get detailed information about a specific Egyptian region.

## Integration Example

```python
import requests
import json

def get_chatbot_response(message, business_context=None):
    url = "http://localhost:8080/chat"
    
    payload = {
        "messages": [{"role": "user", "content": message}],
        "business_context": business_context
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
business_context = {
    "name": "Cairo Spice Trading",
    "industry": "Spices",
    "region": "Greater Cairo",
    "business_type": "Both"
}

result = get_chatbot_response(
    "How can I find the best spice industry partners?", 
    business_context
)

print(result["response"])
```

## Knowledge Base

The chatbot includes a comprehensive knowledge base on:

- Egyptian industry sectors (Textiles, Agriculture, Spices, etc.)
- Regional business characteristics (Greater Cairo, Mediterranean Coast, etc.)
- Seasonality factors (Tourism Seasons, Ramadan, Agricultural Cycles)
- Recommendation system technical details
- Common business questions and answers

## Contributing

Contributions to improve the chatbot's knowledge base or functionality are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

[MIT License](LICENSE) 