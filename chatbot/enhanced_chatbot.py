import os
import json
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from business_knowledge import EGYPTIAN_BUSINESS_KNOWLEDGE

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure the Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment variables. Please set it before using the chatbot.")

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    logger.error(f"Error configuring Gemini API: {e}")

class EnhancedBusinessChatbot:
    """
    Enhanced chatbot for Buy-From-Egypt recommendations that leverages structured
    Egyptian business knowledge for more accurate and context-aware responses.
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.history = []
        self.business_knowledge = EGYPTIAN_BUSINESS_KNOWLEDGE
        self.recommendation_api_docs = self._load_api_docs()
        logger.info("Enhanced Business Chatbot initialized")
        
    def _load_api_docs(self) -> Dict:
        """Load API documentation for reference in responses."""
        try:
            # In a real implementation, this would load from a file or API
            # Here we're creating a simplified version
            return {
                "endpoints": [
                    {
                        "path": "/recommend/customer/{customer_id}",
                        "method": "GET",
                        "description": "Get personalized product recommendations for a customer",
                        "parameters": [
                            {"name": "customer_id", "description": "Unique customer identifier"},
                            {"name": "num_recommendations", "description": "Number of recommendations to return (default: 10)"}
                        ]
                    },
                    {
                        "path": "/recommend/business/{business_name}",
                        "method": "GET",
                        "description": "Get product and business partnership recommendations",
                        "parameters": [
                            {"name": "business_name", "description": "Name of the business"},
                            {"name": "num_product_recommendations", "description": "Number of product recommendations (default: 10)"},
                            {"name": "num_partner_recommendations", "description": "Number of business partner recommendations (default: 5)"}
                        ]
                    },
                    {
                        "path": "/egyptian-economic-context",
                        "method": "GET",
                        "description": "Get current Egyptian economic context data"
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error loading API docs: {e}")
            return {}
    
    def _generate_system_prompt(self, business_context=None):
        """Generate a detailed system prompt with business knowledge."""
        
        system_prompt = """
        You are an AI assistant for Egyptian businesses using the Buy-From-Egypt recommendation system.
        Your purpose is to help businesses understand and effectively utilize the recommendation system
        to improve their operations, find partners, and select optimal products.
        
        IMPORTANT: Focus on being helpful, accurate, and business-oriented in your responses.
        When discussing recommendations, always emphasize their Egyptian context and relevance.
        """
        
        # Add knowledge about recommendation system
        system_prompt += "\n\n=== RECOMMENDATION SYSTEM KNOWLEDGE ===\n"
        system_prompt += self.business_knowledge["recommendation_technical"]["hybrid_approach_explanation"]
        
        # Add business context if provided
        if business_context:
            system_prompt += "\n\n=== SPECIFIC BUSINESS CONTEXT ===\n"
            
            # Add industry information if available
            if "industry" in business_context:
                industry = business_context["industry"]
                if industry in self.business_knowledge["industry_sectors"]:
                    industry_info = self.business_knowledge["industry_sectors"][industry]
                    system_prompt += f"\nIndustry: {industry}\n"
                    system_prompt += f"Description: {industry_info['description']}\n"
                    system_prompt += f"Recommendation focus: {industry_info['recommendation_focus']}\n"
                    system_prompt += f"Key regions: {', '.join(industry_info['key_regions'])}\n"
            
            # Add region information if available
            if "region" in business_context:
                region = business_context["region"]
                if region in self.business_knowledge["regional_characteristics"]:
                    region_info = self.business_knowledge["regional_characteristics"][region]
                    system_prompt += f"\nRegion: {region}\n"
                    system_prompt += f"Business density: {region_info['business_density']}\n"
                    system_prompt += f"Export access: {region_info['export_access']}\n"
                    system_prompt += f"Recommendation focus: {region_info['recommendation_focus']}\n"
            
            # Add other business details
            for key, value in business_context.items():
                if key not in ["industry", "region"]:
                    system_prompt += f"\n{key}: {value}"
        
        system_prompt += """
        
        RESPONSE GUIDELINES:
        1. Provide accurate information about the recommendation system
        2. When discussing API integration, reference specific endpoints and parameters
        3. Tailor responses to the Egyptian business context
        4. When discussing a specific industry, reference related business opportunities
        5. For seasonal questions, include relevant Egyptian seasonal factors
        6. Use clear business language and provide actionable insights
        """
        
        return system_prompt
    
    def _search_knowledge_base(self, query):
        """Search the knowledge base for relevant information about the query."""
        relevant_info = []
        
        # Check common business questions
        for question, answer in self.business_knowledge["common_business_questions"].items():
            if any(keyword in query.lower() for keyword in question.lower().split()):
                relevant_info.append(("Common Question", question, answer))
        
        # Check industry sectors
        for industry, details in self.business_knowledge["industry_sectors"].items():
            if industry.lower() in query.lower():
                relevant_info.append(("Industry", industry, json.dumps(details, indent=2)))
        
        # Check region information
        for region, details in self.business_knowledge["regional_characteristics"].items():
            if region.lower() in query.lower():
                relevant_info.append(("Region", region, json.dumps(details, indent=2)))
        
        # Check seasonality factors
        for season, details in self.business_knowledge["seasonality_factors"].items():
            if season.lower() in query.lower():
                relevant_info.append(("Seasonality", season, json.dumps(details, indent=2)))
        
        # Check technical recommendation information
        for topic, explanation in self.business_knowledge["recommendation_technical"].items():
            keywords = topic.replace("_", " ").split()
            if any(keyword in query.lower() for keyword in keywords):
                relevant_info.append(("Technical", topic, explanation))
        
        return relevant_info
    
    async def get_response(self, query, business_context=None):
        """
        Get a response from the chatbot for a user query.
        
        Args:
            query: The user's question or message
            business_context: Optional dict with business-specific information
            
        Returns:
            dict: Response containing the answer and relevant sources
        """
        try:
            # Search knowledge base for relevant information
            relevant_info = self._search_knowledge_base(query)
            
            # Generate system prompt
            system_prompt = self._generate_system_prompt(business_context)
            
            # Add relevant knowledge to the prompt
            if relevant_info:
                system_prompt += "\n\n=== RELEVANT KNOWLEDGE ===\n"
                for info_type, topic, details in relevant_info:
                    system_prompt += f"\n{info_type}: {topic}\n{details}\n"
            
            # Create messages for Gemini
            messages = [
                {"role": "system", "parts": [system_prompt]},
                {"role": "user", "parts": [query]}
            ]
            
            # Add conversation history
            for msg in self.history[-5:]:  # Include only last 5 messages to avoid exceeding context window
                messages.append(msg)
            
            # Generate response from Gemini
            response = self.model.generate_content([part for msg in messages for part in msg["parts"]])
            response_text = response.text
            
            # Store in history
            self.history.append({"role": "user", "parts": [query]})
            self.history.append({"role": "model", "parts": [response_text]})
            
            # Extract sources from relevant info
            sources = []
            for info_type, topic, _ in relevant_info:
                sources.append(f"{info_type}: {topic}")
            
            return {
                "response": response_text,
                "sources": sources if sources else None
            }
            
        except Exception as e:
            logger.error(f"Error generating chatbot response: {e}")
            return {
                "response": "I'm sorry, I encountered an error while processing your request. Please try again later.",
                "sources": None
            }
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.history = []
        return {"status": "Conversation history reset successfully"}


# Example usage
async def test_chatbot():
    chatbot = EnhancedBusinessChatbot()
    
    # Example business context
    business_context = {
        "name": "Cairo Textile Exports",
        "industry": "Textiles",
        "region": "Greater Cairo",
        "business_type": "Exporter",
        "size": "Medium",
        "target_markets": ["Europe", "Gulf Countries"]
    }
    
    # Example query
    query = "How can the recommendation system help my textile business find new partners?"
    
    response = await chatbot.get_response(query, business_context)
    print("Query:", query)
    print("\nResponse:", response["response"])
    print("\nSources:", response["sources"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_chatbot()) 