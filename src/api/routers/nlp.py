"""
Natural Language Processing router for query processing.

This module provides endpoints for processing natural language queries,
query suggestions, and feedback collection.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import uuid
import time

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import json

from src.api.models import (
    NLPQueryRequest, NLPQueryResponse, QuerySuggestion, 
    QuerySuggestionsResponse, QueryFeedback, BaseResponse
)
from src.database.connection import db_session_scope
from src.database.operations import get_flights_dataframe
# NLP components will be imported dynamically
NLP_AVAILABLE = False
FlightQueryProcessor = None
ResponseGenerator = None
LangChainOrchestrator = None
from src.utils.logging import logger
from src.config import get_settings

router = APIRouter()
settings = get_settings()

# Global instances (in production, these would be dependency injected)
query_processor = None
response_generator = None
langchain_orchestrator = None

def get_db():
    """Dependency to get database session"""
    with db_session_scope() as session:
        yield session


def get_nlp_components():
    """Initialize NLP components if not already done"""
    global query_processor, response_generator, langchain_orchestrator, NLP_AVAILABLE
    global FlightQueryProcessor, ResponseGenerator, LangChainOrchestrator
    
    if query_processor is None:
        # Try to import NLP components
        if not NLP_AVAILABLE:
            try:
                from src.nlp.query_processor import FlightQueryProcessor
                from src.nlp.response_generator import ResponseGenerator
                from src.nlp.langchain_orchestrator import LangChainOrchestrator
                NLP_AVAILABLE = True
                logger.info("NLP components imported successfully")
            except ImportError as e:
                logger.warning(f"NLP components not available: {str(e)}. Using mock implementations.")
                NLP_AVAILABLE = False
        
        if NLP_AVAILABLE and settings.openai_api_key:
            try:
                query_processor = FlightQueryProcessor(
                    openai_api_key=settings.openai_api_key
                )
                response_generator = ResponseGenerator()
                langchain_orchestrator = LangChainOrchestrator(
                    openai_api_key=settings.openai_api_key
                )
                logger.info("NLP components initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize NLP components: {str(e)}")
                # Create mock components for testing
                query_processor = MockQueryProcessor()
                response_generator = MockResponseGenerator()
                langchain_orchestrator = MockLangChainOrchestrator()
        else:
            # Create mock components when NLP is not available
            logger.info("Using mock NLP components (OpenAI not configured or spaCy not available)")
            query_processor = MockQueryProcessor()
            response_generator = MockResponseGenerator()
            langchain_orchestrator = MockLangChainOrchestrator()
    
    return query_processor, response_generator, langchain_orchestrator


@router.get("/")
async def nlp_root():
    """NLP root endpoint with available operations."""
    return {
        "success": True,
        "message": "Natural Language Processing API",
        "data": {
            "available_endpoints": [
                "POST /query - Process natural language queries about flight data",
                "GET /suggestions - Get query suggestions and examples",
                "POST /feedback - Submit feedback on query responses",
                "GET /history - Get query history (if session provided)",
                "POST /stream - Stream query responses in real-time"
            ],
            "supported_queries": [
                "What's the best time to fly from Mumbai to Delhi?",
                "Which flights cause the most delays?",
                "Show me congestion patterns for BOM airport",
                "What's the average delay for flights?",
                "Which time slots should I avoid at Delhi airport?"
            ]
        }
    }


@router.post("/query", response_model=NLPQueryResponse)
async def process_nlp_query(
    request: NLPQueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Process natural language queries about flight data.
    
    This endpoint implements EXPECTATION 1: OpenAI-powered interface for querying
    processed flight information using NLP prompts.
    """
    try:
        # Initialize NLP components
        processor, generator, orchestrator = get_nlp_components()
        
        # Generate query ID for tracking
        query_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Processing NLP query: {request.query[:100]}...")
        
        # Process the query using LangChain orchestrator for complex queries
        if len(request.query.split()) > 10 or any(word in request.query.lower() 
                                                 for word in ['analyze', 'compare', 'predict', 'recommend']):
            # Use LangChain for complex queries
            response_data = await orchestrator.process_complex_query(
                query=request.query,
                context=request.context or {},
                session_id=request.session_id
            )
        else:
            # Use simple query processor for basic queries
            query_response = processor.process_query(
                query=request.query,
                context=request.context or {}
            )
            response_data = {
                "query_type": query_response.query_type if hasattr(query_response, 'query_type') else 'general_info',
                "data": query_response.data if hasattr(query_response, 'data') else {},
                "confidence": query_response.confidence if hasattr(query_response, 'confidence') else 0.5
            }
        
        # Generate response using response generator
        formatted_response = generator.generate_response(
            query=request.query,
            analysis_results=response_data,
            response_type=response_data.get('query_type', 'general_info')
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        nlp_response = NLPQueryResponse(
            query=request.query,
            response=formatted_response.get('answer', 'I apologize, but I could not process your query.'),
            confidence=formatted_response.get('confidence', 0.5),
            data=response_data.get('data'),
            visualizations=formatted_response.get('visualizations', []),
            follow_up_questions=formatted_response.get('follow_up_questions', []),
            session_id=request.session_id or query_id
        )
        
        # Store query and response in background
        background_tasks.add_task(
            store_query_history,
            query_id,
            request.query,
            nlp_response.model_dump(),
            processing_time
        )
        
        logger.info(f"NLP query processed successfully in {processing_time:.2f}s")
        return nlp_response
        
    except Exception as e:
        logger.error(f"Error processing NLP query: {str(e)}")
        
        # Return error response
        return NLPQueryResponse(
            query=request.query,
            response=f"I encountered an error while processing your query: {str(e)}. Please try rephrasing your question.",
            confidence=0.0,
            data={"error": str(e)},
            session_id=request.session_id or str(uuid.uuid4())
        )


@router.get("/suggestions", response_model=QuerySuggestionsResponse)
async def get_query_suggestions(
    category: Optional[str] = None,
    limit: int = 10
):
    """
    Get query suggestions and examples.
    
    Provides users with example queries they can ask about flight data.
    """
    try:
        # Predefined query suggestions by category
        all_suggestions = {
            "delay_analysis": [
                "What's the average delay for flights from Mumbai?",
                "Which airline has the most delays?",
                "What are the best times to avoid delays?",
                "Show me delay patterns for the last week",
                "Which flights are consistently delayed?"
            ],
            "congestion_analysis": [
                "What are the busiest hours at Delhi airport?",
                "When should I avoid flying to minimize congestion?",
                "Show me traffic patterns for BOM airport",
                "Which time slots have the least congestion?",
                "What's the peak hour for departures?"
            ],
            "schedule_optimization": [
                "What would happen if I reschedule flight AI101?",
                "How would a 2-hour delay affect other flights?",
                "What's the best time to schedule a new flight?",
                "Show me the impact of changing departure times",
                "Which flights should be prioritized for on-time performance?"
            ],
            "general": [
                "How many flights operate between Mumbai and Delhi?",
                "What's the most popular aircraft type?",
                "Show me flight statistics for last month",
                "Which routes have the highest passenger volume?",
                "What's the average flight duration?"
            ]
        }
        
        # Filter by category if specified
        if category and category in all_suggestions:
            suggestions_list = all_suggestions[category]
        else:
            # Combine all suggestions
            suggestions_list = []
            for cat_suggestions in all_suggestions.values():
                suggestions_list.extend(cat_suggestions)
        
        # Create suggestion objects
        suggestions = []
        for i, suggestion_text in enumerate(suggestions_list[:limit]):
            # Determine category
            suggestion_category = category or "general"
            for cat, cat_suggestions in all_suggestions.items():
                if suggestion_text in cat_suggestions:
                    suggestion_category = cat
                    break
            
            suggestions.append(QuerySuggestion(
                text=suggestion_text,
                category=suggestion_category,
                confidence=0.9 - (i * 0.05)  # Decreasing confidence
            ))
        
        return QuerySuggestionsResponse(
            suggestions=suggestions,
            message=f"Retrieved {len(suggestions)} query suggestions"
        )
        
    except Exception as e:
        logger.error(f"Error getting query suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.post("/feedback", response_model=BaseResponse)
async def submit_query_feedback(
    feedback: QueryFeedback,
    background_tasks: BackgroundTasks
):
    """
    Submit feedback on query responses.
    
    Helps improve the NLP system by collecting user feedback.
    """
    try:
        # Store feedback in background
        background_tasks.add_task(
            store_query_feedback,
            feedback.query_id,
            feedback.rating,
            feedback.feedback
        )
        
        return BaseResponse(
            message="Thank you for your feedback! It helps us improve our responses."
        )
        
    except Exception as e:
        logger.error(f"Error storing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {str(e)}")


@router.get("/history")
async def get_query_history(
    session_id: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Get query history for a session.
    
    Returns previous queries and responses for context.
    """
    try:
        if not session_id:
            return {
                "success": True,
                "message": "No session ID provided",
                "data": {
                    "queries": [],
                    "total": 0
                }
            }
        
        # In a real implementation, this would query a query_history table
        # For now, return empty history
        return {
            "success": True,
            "message": f"Query history for session {session_id}",
            "data": {
                "session_id": session_id,
                "queries": [],
                "total": 0,
                "note": "Query history storage not yet implemented"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting query history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.post("/stream")
async def stream_query_response(
    request: NLPQueryRequest
):
    """
    Stream query responses in real-time.
    
    Provides streaming responses for long-running queries.
    """
    async def generate_response():
        try:
            # Initialize NLP components
            processor, generator, orchestrator = get_nlp_components()
            
            # Send initial response
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Processing your query...'})}\n\n"
            
            # Simulate processing steps
            steps = [
                "Analyzing query intent...",
                "Extracting entities and parameters...",
                "Retrieving relevant flight data...",
                "Performing analysis...",
                "Generating response..."
            ]
            
            for i, step in enumerate(steps):
                await asyncio.sleep(0.5)  # Simulate processing time
                progress = (i + 1) / len(steps) * 100
                yield f"data: {json.dumps({'status': 'processing', 'message': step, 'progress': progress})}\n\n"
            
            # Process the actual query
            if len(request.query.split()) > 10:
                response_data = await orchestrator.process_complex_query(
                    query=request.query,
                    context=request.context or {},
                    session_id=request.session_id
                )
            else:
                query_response = processor.process_query(
                    query=request.query,
                    context=request.context or {}
                )
                response_data = {
                    "query_type": query_response.query_type if hasattr(query_response, 'query_type') else 'general_info',
                    "data": query_response.data if hasattr(query_response, 'data') else {},
                    "confidence": query_response.confidence if hasattr(query_response, 'confidence') else 0.5
                }
            
            # Generate final response
            formatted_response = generator.generate_response(
                query=request.query,
                analysis_results=response_data,
                response_type=response_data.get('query_type', 'general_info')
            )
            
            # Send final response
            final_response = {
                'status': 'completed',
                'query': request.query,
                'response': formatted_response.get('answer', 'Query processed successfully.'),
                'confidence': formatted_response.get('confidence', 0.5),
                'data': response_data.get('data'),
                'visualizations': formatted_response.get('visualizations', []),
                'follow_up_questions': formatted_response.get('follow_up_questions', [])
            }
            
            yield f"data: {json.dumps(final_response)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            error_response = {
                'status': 'error',
                'message': f'Error processing query: {str(e)}'
            }
            yield f"data: {json.dumps(error_response)}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


async def store_query_history(query_id: str, query: str, response: Dict[str, Any], processing_time: float):
    """Background task to store query history."""
    try:
        # In a real implementation, this would store in a database
        logger.info(f"Query {query_id} processed in {processing_time:.2f}s: {query[:50]}...")
    except Exception as e:
        logger.error(f"Failed to store query history: {str(e)}")


async def store_query_feedback(query_id: str, rating: int, feedback: Optional[str]):
    """Background task to store query feedback."""
    try:
        # In a real implementation, this would store in a database
        logger.info(f"Feedback received for query {query_id}: rating={rating}, feedback={feedback}")
    except Exception as e:
        logger.error(f"Failed to store query feedback: {str(e)}")


# Mock classes for testing when OpenAI is not available
class MockQueryProcessor:
    """Mock query processor for testing."""
    
    async def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "query_type": "general_info",
            "data": {"mock": True, "query": query},
            "confidence": 0.8
        }


class MockResponseGenerator:
    """Mock response generator for testing."""
    
    def generate_response(self, query: str, analysis_results: Dict[str, Any], response_type: str) -> Dict[str, Any]:
        return {
            "answer": f"This is a mock response to your query: '{query}'. The NLP system is not fully configured.",
            "confidence": 0.5,
            "visualizations": ["bar_chart"],
            "follow_up_questions": ["Can you provide more specific details?"]
        }


class MockLangChainOrchestrator:
    """Mock LangChain orchestrator for testing."""
    
    async def process_complex_query(self, query: str, context: Dict[str, Any], session_id: Optional[str]) -> Dict[str, Any]:
        return {
            "query_type": "complex_analysis",
            "data": {"mock": True, "query": query, "session_id": session_id},
            "confidence": 0.7
        }


@router.options("/", include_in_schema=False)
async def nlp_options():
    """Handle OPTIONS requests for CORS preflight."""
    return {"message": "OK"}