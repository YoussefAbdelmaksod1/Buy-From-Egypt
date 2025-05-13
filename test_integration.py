#!/usr/bin/env python3
import requests
import json
import time
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "your_api_key_here"  # Replace with actual API key for testing

def test_api_health():
    """Test API health endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/", 
                               headers={"Authorization": f"Bearer {API_KEY}"})
        
        if response.status_code == 200:
            logger.info("✓ API health check: SUCCESS")
            return True
        else:
            logger.error(f"✗ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"✗ API health check error: {e}")
        return False

def test_recommendation_endpoints():
    """Test recommendation endpoints"""
    endpoints = {
        "user_recommendations": "/api/recommendations/user/12345",
        "business_recommendations": "/api/recommendations/business/123",
        "trending_recommendations": "/api/recommendations/trending",
        "regional_recommendations": "/api/recommendations/regional/cairo"
    }
    
    results = {}
    for name, endpoint in endpoints.items():
        try:
            response = requests.get(
                f"{API_BASE_URL}{endpoint}",
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                results[name] = {
                    "status": "SUCCESS",
                    "response_sample": str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
                }
                logger.info(f"✓ Endpoint {name}: SUCCESS")
            else:
                results[name] = {
                    "status": "FAILED",
                    "status_code": response.status_code,
                    "response": response.text
                }
                logger.error(f"✗ Endpoint {name} failed: {response.status_code}")
        except Exception as e:
            results[name] = {
                "status": "ERROR",
                "error": str(e)
            }
            logger.error(f"✗ Endpoint {name} error: {e}")
    
    return results

def test_data_sync_endpoints():
    """Test data synchronization endpoints"""
    # Sample data for syncing
    user_data = {
        "user_id": "test_user_123",
        "name": "Test User",
        "email": "test@example.com",
        "preferences": ["textiles", "handicrafts"]
    }
    
    business_data = {
        "business_id": "test_business_123",
        "name": "Test Egyptian Business",
        "category": "Handicrafts",
        "region": "Cairo",
        "description": "Authentic Egyptian handicrafts"
    }
    
    transaction_data = {
        "transaction_id": "test_transaction_123",
        "user_id": "test_user_123",
        "business_id": "test_business_123",
        "amount": 1500,
        "currency": "EGP",
        "timestamp": "2023-04-15T14:30:00Z"
    }
    
    sync_data = {
        "users": [user_data],
        "businesses": [business_data],
        "transactions": [transaction_data]
    }
    
    results = {}
    for data_type, data in sync_data.items():
        endpoint = f"/api/sync/{data_type}"
        try:
            response = requests.post(
                f"{API_BASE_URL}{endpoint}",
                json=data,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 200:
                results[f"sync_{data_type}"] = {
                    "status": "SUCCESS",
                    "response": response.json()
                }
                logger.info(f"✓ Sync {data_type}: SUCCESS")
            else:
                results[f"sync_{data_type}"] = {
                    "status": "FAILED",
                    "status_code": response.status_code,
                    "response": response.text
                }
                logger.error(f"✗ Sync {data_type} failed: {response.status_code}")
        except Exception as e:
            results[f"sync_{data_type}"] = {
                "status": "ERROR",
                "error": str(e)
            }
            logger.error(f"✗ Sync {data_type} error: {e}")
    
    return results

def verify_model_files():
    """Verify that model files exist"""
    model_files = [
        "models/collaborative_filtering_model.pkl",
        "models/business_recommendation_model.pkl",
        "models/model_info.json"
    ]
    
    results = {}
    for file_path in model_files:
        path = Path(file_path)
        if path.exists():
            results[path.name] = {
                "status": "SUCCESS",
                "size": path.stat().st_size
            }
            logger.info(f"✓ Model file {path.name} exists")
        else:
            results[path.name] = {
                "status": "FAILED",
                "error": "File not found"
            }
            logger.error(f"✗ Model file {path.name} not found")
    
    return results

def test_chatbot_endpoint():
    """Test chatbot API endpoint"""
    query = "Tell me about Egyptian textile businesses"
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chatbot",
            json={"query": query},
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            result = {
                "status": "SUCCESS",
                "response_sample": data.get("response", "")[:100] + "..." 
                    if len(data.get("response", "")) > 100 else data.get("response", "")
            }
            logger.info(f"✓ Chatbot endpoint: SUCCESS")
        else:
            result = {
                "status": "FAILED",
                "status_code": response.status_code,
                "response": response.text
            }
            logger.error(f"✗ Chatbot endpoint failed: {response.status_code}")
    except Exception as e:
        result = {
            "status": "ERROR",
            "error": str(e)
        }
        logger.error(f"✗ Chatbot endpoint error: {e}")
    
    return result

def run_integration_tests():
    """Run all integration tests"""
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Check API health
    api_health = test_api_health()
    if not api_health:
        logger.error("API is not running, cannot continue with tests")
        results["api_status"] = "FAILED"
        return results
    
    results["api_status"] = "SUCCESS"
    
    # Run tests
    results["recommendation_endpoints"] = test_recommendation_endpoints()
    results["data_sync_endpoints"] = test_data_sync_endpoints()
    results["model_files"] = verify_model_files()
    results["chatbot"] = test_chatbot_endpoint()
    
    # Calculate overall success rate
    successes = 0
    total = 0
    
    for category in ["recommendation_endpoints", "data_sync_endpoints", "model_files"]:
        if category in results:
            for key, result in results[category].items():
                if isinstance(result, dict) and "status" in result:
                    total += 1
                    if result["status"] == "SUCCESS":
                        successes += 1
    
    # Add chatbot result
    if "chatbot" in results and isinstance(results["chatbot"], dict) and "status" in results["chatbot"]:
        total += 1
        if results["chatbot"]["status"] == "SUCCESS":
            successes += 1
    
    results["success_rate"] = f"{successes}/{total} ({successes/total*100:.1f}%)" if total > 0 else "N/A"
    
    # Print summary
    logger.info(f"\n{'='*80}\nINTEGRATION TEST SUMMARY\n{'='*80}")
    logger.info(f"Success rate: {results['success_rate']}")
    logger.info(f"API status: {results['api_status']}")
    logger.info(f"{'='*80}\n")
    
    # Save results to file
    with open("integration_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Detailed results saved to integration_test_results.json")
    
    return results

if __name__ == "__main__":
    logger.info("Starting Buy-From-Egypt integration tests...")
    results = run_integration_tests()
    
    # Return non-zero exit code if tests failed
    if results.get("api_status") != "SUCCESS" or results.get("success_rate", "").startswith("0/"):
        sys.exit(1)
    
    logger.info("Integration tests completed successfully.") 