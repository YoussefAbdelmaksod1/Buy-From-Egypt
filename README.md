# Buy-From-Egypt Recommendation System

## Overview

This repository contains two independent systems:

1. A recommendation system that provides personalized product and business partnership recommendations for Egyptian businesses and customers, leveraging machine learning algorithms and Egyptian economic context.

2. A Gemini-powered chatbot specialized in Egyptian business knowledge, providing information about industries, regions, and seasonality factors.

## System Components

### 1. Recommendation Engine
- **Hybrid Recommendation Approach**: Combines collaborative filtering, content-based filtering, and economic context
- **Customer Recommendations**: Personalized product suggestions based on purchasing patterns
- **Business Recommendations**: Product and business partnership suggestions
- **Egyptian Context**: Incorporates economic indicators and seasonality factors

### 2. API Service
- **RESTful Endpoints**: API for retrieving and managing recommendations
- **Export Capabilities**: Tools for exporting recommendations in various formats
- **Economic Context**: Access to Egyptian economic indicators

### 3. Business Chatbot (Independent System)
- **Gemini-Powered Assistant**: AI chatbot with Egyptian business expertise
- **Egypt-Specific Knowledge**: Information about industries, regions, and business seasonality
- **Recommendation System Guidance**: Assistance with using the recommendation API

## Data Sources

The recommendation system uses three primary data sources:

1. **Retail Transaction Data**: User-product interactions
2. **Business Profile Data**: Business attributes and categories
3. **Egyptian Economic Indicators**: GDP growth, inflation, and seasonality factors

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YoussefAbdelmaksod1/Buy-From-Egypt.git
cd Buy-From-Egypt
```

2. Install dependencies:
```bash
# For recommendation system
pip install -r requirements.txt

# For chatbot system
cd chatbot
pip install -r requirements.txt
```

3. Set up environment variables:
   - For the chatbot functionality, create a `.env` file in the `chatbot` directory with your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## Usage

### Recommendation System

#### Data Processing

Process the raw data:

```bash
python main.py --process
```

#### Model Training

Train the hybrid recommendation model:

```bash
python main.py --train
```

#### Running the Recommendation API

Start the recommendation API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Business Chatbot (Independent)

Start the business chatbot API:

```bash
cd chatbot
python chatbot_api.py
```

The chatbot API will be available at `http://localhost:8080`

## API Documentation

### Recommendation API (Port 8000)

- `GET /recommend/customer/{customer_id}` - Get customer product recommendations
- `GET /recommend/business/{business_name}` - Get business product and partnership recommendations
- `GET /egyptian-economic-context` - Get current Egyptian economic context
- `GET /export/recommendations/customer/{customer_id}` - Export customer recommendations

### Business Chatbot API (Port 8080)

- `POST /chat` - Get a response from the Egyptian business chatbot
- `POST /chat/reset` - Reset a conversation session
- `GET /knowledge-topics` - Get a list of chatbot knowledge topics
- `GET /industry/{industry_name}` - Get information about a specific Egyptian industry
- `GET /region/{region_name}` - Get information about a specific Egyptian region 