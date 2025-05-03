import os
import json
import time
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import uvicorn
from enhanced_chatbot import EnhancedBusinessChatbot

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize enhanced chatbot
chatbot = None

# Define request and response models
class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender (user or assistant)")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    business_context: Optional[Dict[str, Any]] = Field(None, description="Optional business-specific context")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation tracking")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The chatbot's response")
    sources: Optional[List[str]] = Field(None, description="Sources of information used in the response")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    processing_time: float = Field(..., description="Processing time in seconds")

class StatusResponse(BaseModel):
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Additional information")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")

class KnowledgeTopicsResponse(BaseModel):
    topics: List[str] = Field(..., description="List of knowledge topics")
    categories: Dict[str, List[str]] = Field(..., description="Categories of knowledge by topic")

# Create FastAPI application
app = FastAPI(
    title="Buy-From-Egypt Business Chatbot API",
    description="""
    # Egyptian Business Chatbot API
    
    A powerful Gemini-powered chatbot API designed specifically for Egyptian businesses 
    using the Buy-From-Egypt recommendation system. The chatbot provides intelligent 
    responses about product recommendations, business partnerships, and Egyptian market insights.
    
    ## Key Features
    
    * **Egypt-specific knowledge** about industries, regions, and business seasonality
    * **Business context-aware responses** tailored to specific companies
    * **Recommendation system expertise** to help businesses utilize the platform effectively
    * **Conversation history** to maintain context throughout interactions
    
    ## Integration
    
    Backend developers can integrate this API with the Buy-From-Egypt platform to provide 
    AI-powered assistance to business users. The API is designed to be easy to use and
    provides comprehensive documentation to help developers get started quickly.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session store (in-memory for simplicity; use Redis or another solution in production)
session_store = {}

# Initialize chatbot instance
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    global chatbot
    logger.info("Initializing enhanced business chatbot...")
    try:
        chatbot = EnhancedBusinessChatbot()
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        # Continue anyway to allow the API to start; will return errors on chatbot calls

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global session_store
    logger.info("Cleaning up resources...")
    # Clear session store
    session_store.clear()
    logger.info("Shutdown complete")

# Middleware for request timing and logging
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers and log requests"""
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"Request to {request.url.path} completed in {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error processing request to {request.url.path}: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error", "details": str(e)},
        )

# Helper function to generate a unique session ID
def generate_session_id():
    """Generate a unique session ID"""
    import uuid
    return str(uuid.uuid4())

# Helper to check if chatbot is initialized
def get_chatbot():
    """Get the chatbot instance or raise an exception if not initialized"""
    if chatbot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot service is not initialized yet. Please try again later."
        )
    return chatbot

# Main chat endpoint
@app.post(
    "/chat", 
    response_model=ChatResponse,
    responses={
        200: {"description": "Successful response from the chatbot"},
        400: {"description": "Bad request", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    },
    tags=["Chat"]
)
async def chat(request: ChatRequest, bot: EnhancedBusinessChatbot = Depends(get_chatbot)):
    """
    Get a response from the Buy-From-Egypt business chatbot.
    
    This endpoint processes a conversation with the chatbot and returns a response.
    You can include business context to get more tailored responses.
    
    - **messages**: List of conversation messages (required)
    - **business_context**: Optional context about the business
    - **session_id**: Optional session ID for continuing conversations
    
    If you provide a session_id that exists, the conversation history will be maintained.
    If you don't provide a session_id, a new one will be created for you.
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No messages provided in the request"
            )
        
        # Get or create session ID
        session_id = request.session_id
        if not session_id or session_id not in session_store:
            session_id = generate_session_id()
            session_store[session_id] = bot
            logger.info(f"Created new session: {session_id}")
        else:
            # Reuse existing chatbot instance from session
            bot = session_store[session_id]
            logger.info(f"Using existing session: {session_id}")
        
        # Process the last user message
        last_user_message = next((msg.content for msg in reversed(request.messages) 
                               if msg.role.lower() == "user"), None)
        
        if not last_user_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message found in the conversation"
            )
        
        # Get response from chatbot
        chat_response = await bot.get_response(last_user_message, request.business_context)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        return ChatResponse(
            response=chat_response["response"],
            sources=chat_response["sources"],
            session_id=session_id,
            processing_time=process_time
        )
    
    except HTTPException as e:
        # Re-raise HTTP exceptions directly
        raise
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )

@app.post(
    "/chat/reset",
    response_model=StatusResponse,
    responses={
        200: {"description": "Conversation history reset successfully"},
        404: {"description": "Session not found", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    },
    tags=["Chat"]
)
async def reset_conversation(
    session_id: str, 
    bot: EnhancedBusinessChatbot = Depends(get_chatbot)
):
    """
    Reset conversation history for a specific session.
    
    This endpoint clears the conversation history for the specified session,
    allowing you to start a new conversation while keeping the same session ID.
    
    - **session_id**: Session ID for the conversation to reset (required)
    """
    try:
        if session_id not in session_store:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session ID {session_id} not found"
            )
        
        # Reset conversation for this session
        session_store[session_id].reset_conversation()
        
        return StatusResponse(
            status="success",
            message=f"Conversation history for session {session_id} reset successfully"
        )
    
    except HTTPException as e:
        raise
    
    except Exception as e:
        logger.error(f"Error resetting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resetting conversation: {str(e)}"
        )

@app.get(
    "/", 
    response_model=StatusResponse,
    tags=["Health"]
)
async def health_check():
    """
    Health check endpoint to verify the chatbot API is running.
    
    This simple endpoint can be used to check if the API is available
    and respond to requests.
    """
    return StatusResponse(
        status="online",
        message="Buy-From-Egypt Business Chatbot API is running"
    )

@app.get(
    "/health", 
    response_model=Dict[str, Any],
    tags=["Health"]
)
async def detailed_health():
    """
    Detailed health check with API and dependency status.
    
    This endpoint provides more detailed information about the health
    of the API and its dependencies.
    """
    global chatbot
    
    # Check Gemini API
    gemini_status = "unavailable"
    try:
        if os.getenv("GOOGLE_API_KEY"):
            gemini_status = "available"
    except:
        pass
    
    return {
        "status": "online",
        "chatbot_initialized": chatbot is not None,
        "gemini_api": gemini_status,
        "active_sessions": len(session_store),
        "timestamp": time.time()
    }

@app.get(
    "/knowledge-topics", 
    response_model=KnowledgeTopicsResponse,
    tags=["Knowledge"]
)
async def get_knowledge_topics(bot: EnhancedBusinessChatbot = Depends(get_chatbot)):
    """
    Get a list of knowledge topics that the chatbot is trained on.
    
    This endpoint returns the categories of knowledge that the chatbot
    has been trained on, which can be useful for understanding the
    chatbot's capabilities.
    """
    # Get knowledge categories from the chatbot's business knowledge
    knowledge = bot.business_knowledge
    
    topics = [
        "Egyptian Industry Sectors",
        "Regional Business Characteristics",
        "Seasonality Factors",
        "Recommendation System Technical Details",
        "Common Business Questions"
    ]
    
    categories = {
        "Egyptian Industry Sectors": list(knowledge["industry_sectors"].keys()),
        "Regional Business Characteristics": list(knowledge["regional_characteristics"].keys()),
        "Seasonality Factors": list(knowledge["seasonality_factors"].keys()),
        "Recommendation System Technical Details": list(knowledge["recommendation_technical"].keys()),
        "Common Business Questions": list(knowledge["common_business_questions"].keys())
    }
    
    return KnowledgeTopicsResponse(
        topics=topics,
        categories=categories
    )

@app.get(
    "/industry/{industry_name}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Industry information retrieved successfully"},
        404: {"description": "Industry not found", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    },
    tags=["Knowledge"]
)
async def get_industry_info(
    industry_name: str,
    bot: EnhancedBusinessChatbot = Depends(get_chatbot)
):
    """
    Get information about a specific Egyptian industry.
    
    This endpoint returns detailed information about the specified industry,
    including recommendations, key regions, and economic factors.
    
    - **industry_name**: Name of the industry to get information about (required)
    """
    try:
        industry_sectors = bot.business_knowledge["industry_sectors"]
        if industry_name not in industry_sectors:
            # Try case-insensitive match
            for industry in industry_sectors:
                if industry.lower() == industry_name.lower():
                    return industry_sectors[industry]
            
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Industry '{industry_name}' not found"
            )
        
        return industry_sectors[industry_name]
    
    except HTTPException as e:
        raise
    
    except Exception as e:
        logger.error(f"Error getting industry info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting industry info: {str(e)}"
        )

@app.get(
    "/region/{region_name}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Region information retrieved successfully"},
        404: {"description": "Region not found", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    },
    tags=["Knowledge"]
)
async def get_region_info(
    region_name: str,
    bot: EnhancedBusinessChatbot = Depends(get_chatbot)
):
    """
    Get information about a specific Egyptian region.
    
    This endpoint returns detailed information about the specified region,
    including business density, infrastructure, and export access.
    
    - **region_name**: Name of the region to get information about (required)
    """
    try:
        regions = bot.business_knowledge["regional_characteristics"]
        if region_name not in regions:
            # Try case-insensitive match
            for region in regions:
                if region.lower() == region_name.lower():
                    return regions[region]
            
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Region '{region_name}' not found"
            )
        
        return regions[region_name]
    
    except HTTPException as e:
        raise
    
    except Exception as e:
        logger.error(f"Error getting region info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting region info: {str(e)}"
        )

@app.get(
    "/recommendation-integration",
    response_model=Dict[str, str],
    tags=["Integration"]
)
async def get_recommendation_integration(bot: EnhancedBusinessChatbot = Depends(get_chatbot)):
    """
    Get information on how to integrate with the recommendation API.
    
    This endpoint returns technical details about integrating with the
    Buy-From-Egypt recommendation API, including endpoints and usage.
    """
    return {
        "integration_guide": bot.business_knowledge["recommendation_technical"]["api_integration"]
    }

@app.post(
    "/feedback",
    response_model=StatusResponse,
    tags=["Feedback"]
)
async def submit_feedback(
    feedback: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """
    Submit feedback about a chatbot response.
    
    This endpoint allows users to provide feedback on chatbot responses,
    which can be used to improve the chatbot's performance.
    
    - **session_id**: Session ID for the conversation (required)
    - **rating**: User rating of the response (1-5) (required)
    - **comment**: Optional user comment on the response
    - **query**: The original user query
    - **response**: The chatbot's response
    """
    try:
        # Validate feedback
        if "session_id" not in feedback:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id is required"
            )
        
        if "rating" not in feedback:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="rating is required"
            )
        
        # In a real implementation, we would store this feedback
        # For now, we'll just log it
        def log_feedback(feedback):
            logger.info(f"Received feedback: {json.dumps(feedback)}")
            # In production, store in database or send to analytics service
        
        background_tasks.add_task(log_feedback, feedback)
        
        return StatusResponse(
            status="success",
            message="Feedback submitted successfully"
        )
    
    except HTTPException as e:
        raise
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting feedback: {str(e)}"
        )

# Custom exception handler for more friendly error messages
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Generic exception handler for all other exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An unexpected error occurred", "details": str(exc)}
    )

def main():
    """Run the API server"""
    uvicorn.run(
        "chatbot_api:app", 
        host="0.0.0.0", 
        port=8080, 
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 