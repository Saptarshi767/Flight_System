"""
OpenAI integration for natural language query processing.

This module implements the core NLP query processing functionality using OpenAI GPT-4
and spaCy for intent recognition and entity extraction.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import re

import openai
import spacy
from spacy import displacy
from pydantic import BaseModel, Field
import pandas as pd

from ..utils.logger import get_logger
from ..data.models import FlightData
from ..database.operations import DatabaseOperations

logger = get_logger(__name__)


class QueryIntent(BaseModel):
    """Represents the intent and entities extracted from a user query."""
    
    intent: str = Field(..., description="The main intent of the query")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    confidence: float = Field(..., description="Confidence score for intent recognition")
    query_type: str = Field(..., description="Type of query (delay, congestion, schedule, etc.)")
    airports: List[str] = Field(default_factory=list, description="Airport codes mentioned")
    time_range: Optional[Dict[str, datetime]] = Field(None, description="Time range if specified")
    airlines: List[str] = Field(default_factory=list, description="Airlines mentioned")


class QueryResponse(BaseModel):
    """Represents the response to a user query."""
    
    answer: str = Field(..., description="The main answer to the query")
    data: Optional[Dict[str, Any]] = Field(None, description="Supporting data")
    visualizations: List[str] = Field(default_factory=list, description="Suggested visualizations")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    confidence: float = Field(..., description="Confidence in the response")
    sources: List[str] = Field(default_factory=list, description="Data sources used")


class FlightQueryProcessor:
    """
    Main class for processing natural language queries about flight data.
    
    Uses OpenAI GPT-4 for query understanding and response generation,
    and spaCy for entity extraction and intent recognition.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the query processor.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment.
        """
        # Initialize OpenAI client
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.openai_api_key
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy English model not found. Please install with: python -m spacy download en_core_web_sm")
            raise
        
        # Initialize database operations
        self.db_ops = DatabaseOperations()
        
        # Define intent patterns and keywords
        self.intent_patterns = {
            'best_time': [
                'best time', 'optimal time', 'when to fly', 'least delays', 'avoid delays',
                'good time', 'recommended time', 'schedule flight'
            ],
            'delay_analysis': [
                'delays', 'delayed flights', 'delay patterns', 'late flights', 'on time',
                'punctuality', 'delay causes', 'delay statistics'
            ],
            'congestion': [
                'busy', 'congestion', 'peak hours', 'crowded', 'traffic', 'busiest time',
                'avoid busy', 'peak time', 'rush hour'
            ],
            'schedule_impact': [
                'schedule change', 'impact', 'what if', 'reschedule', 'schedule optimization',
                'schedule tuning', 'modify schedule'
            ],
            'cascading_impact': [
                'cascading', 'domino effect', 'critical flights', 'network impact',
                'connected flights', 'downstream effects'
            ],
            'general_info': [
                'flight info', 'flight details', 'airline', 'aircraft', 'route',
                'flight number', 'status'
            ]
        }
        
        # Airport code mapping
        self.airport_codes = {
            'mumbai': 'BOM',
            'delhi': 'DEL',
            'bom': 'BOM',
            'del': 'DEL',
            'bombay': 'BOM',
            'new delhi': 'DEL'
        }
        
        logger.info("FlightQueryProcessor initialized successfully")
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryResponse:
        """
        Process a natural language query about flight data.
        
        Args:
            query: The user's natural language query
            context: Optional context from previous queries
            
        Returns:
            QueryResponse object with answer and supporting data
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Step 1: Extract intent and entities using spaCy
            query_intent = self._extract_intent_and_entities(query)
            logger.info(f"Extracted intent: {query_intent.intent}")
            
            # Step 2: Get relevant flight data based on intent
            flight_data = self._get_relevant_data(query_intent)
            
            # Step 3: Generate response using OpenAI
            response = self._generate_response(query, query_intent, flight_data, context)
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return QueryResponse(
                answer=f"I'm sorry, I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                sources=[]
            )
    
    def _extract_intent_and_entities(self, query: str) -> QueryIntent:
        """
        Extract intent and entities from the query using spaCy and pattern matching.
        
        Args:
            query: The user's query
            
        Returns:
            QueryIntent object with extracted information
        """
        # Process with spaCy
        doc = self.nlp(query.lower())
        
        # Extract entities
        entities = {}
        airports = []
        airlines = []
        time_entities = []
        
        for ent in doc.ents:
            if ent.label_ == "GPE":  # Geopolitical entity (cities, countries)
                if ent.text.lower() in self.airport_codes:
                    airports.append(self.airport_codes[ent.text.lower()])
            elif ent.label_ == "ORG":  # Organizations (airlines)
                airlines.append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                time_entities.append(ent.text)
            
            entities[ent.label_] = entities.get(ent.label_, []) + [ent.text]
        
        # Check for airport codes directly mentioned
        for token in doc:
            if token.text.upper() in ['BOM', 'DEL', 'MUMBAI', 'DELHI']:
                if token.text.upper() in ['BOM', 'DEL']:
                    airports.append(token.text.upper())
                elif token.text.lower() in self.airport_codes:
                    airports.append(self.airport_codes[token.text.lower()])
        
        # Determine intent based on keywords
        intent = self._classify_intent(query.lower())
        
        # Determine query type
        query_type = intent
        
        # Extract time range if mentioned
        time_range = self._extract_time_range(query, time_entities)
        
        return QueryIntent(
            intent=intent,
            entities=entities,
            confidence=0.8,  # Default confidence, could be improved with ML model
            query_type=query_type,
            airports=list(set(airports)),  # Remove duplicates
            time_range=time_range,
            airlines=list(set(airlines))
        )
    
    def _classify_intent(self, query: str) -> str:
        """
        Classify the intent of the query based on keyword patterns.
        
        Args:
            query: Lowercase query string
            
        Returns:
            Intent classification
        """
        intent_scores = {}
        
        for intent, keywords in self.intent_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    score += 1
            intent_scores[intent] = score
        
        # Return intent with highest score, default to general_info
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        else:
            return 'general_info'
    
    def _extract_time_range(self, query: str, time_entities: List[str]) -> Optional[Dict[str, datetime]]:
        """
        Extract time range from query if specified.
        
        Args:
            query: Original query
            time_entities: Time entities extracted by spaCy
            
        Returns:
            Dictionary with start and end datetime if found
        """
        # Simple time range extraction - could be enhanced
        now = datetime.now()
        
        if 'today' in query.lower():
            return {
                'start': now.replace(hour=0, minute=0, second=0, microsecond=0),
                'end': now.replace(hour=23, minute=59, second=59, microsecond=999999)
            }
        elif 'yesterday' in query.lower():
            yesterday = now - timedelta(days=1)
            return {
                'start': yesterday.replace(hour=0, minute=0, second=0, microsecond=0),
                'end': yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
            }
        elif 'last week' in query.lower():
            week_ago = now - timedelta(days=7)
            return {
                'start': week_ago.replace(hour=0, minute=0, second=0, microsecond=0),
                'end': now
            }
        elif 'last month' in query.lower():
            month_ago = now - timedelta(days=30)
            return {
                'start': month_ago.replace(hour=0, minute=0, second=0, microsecond=0),
                'end': now
            }
        
        return None
    
    def _get_relevant_data(self, query_intent: QueryIntent) -> Optional[pd.DataFrame]:
        """
        Retrieve relevant flight data based on the query intent.
        
        Args:
            query_intent: Extracted intent and entities
            
        Returns:
            Relevant flight data as DataFrame
        """
        try:
            # Build filters based on intent
            filters = {}
            
            if query_intent.airports:
                filters['airports'] = query_intent.airports
            
            if query_intent.airlines:
                filters['airlines'] = query_intent.airlines
            
            if query_intent.time_range:
                filters['time_range'] = query_intent.time_range
            
            # Get data from database based on query type
            if query_intent.query_type in ['delay_analysis', 'best_time']:
                return self.db_ops.get_delay_analysis_data(filters)
            elif query_intent.query_type == 'congestion':
                return self.db_ops.get_congestion_data(filters)
            elif query_intent.query_type == 'schedule_impact':
                return self.db_ops.get_schedule_impact_data(filters)
            elif query_intent.query_type == 'cascading_impact':
                return self.db_ops.get_cascading_impact_data(filters)
            else:
                return self.db_ops.get_general_flight_data(filters)
                
        except Exception as e:
            logger.error(f"Error retrieving data: {str(e)}")
            return None
    
    def _generate_response(
        self, 
        query: str, 
        query_intent: QueryIntent, 
        flight_data: Optional[pd.DataFrame],
        context: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """
        Generate response using OpenAI GPT-4.
        
        Args:
            query: Original user query
            query_intent: Extracted intent and entities
            flight_data: Relevant flight data
            context: Optional conversation context
            
        Returns:
            QueryResponse with generated answer
        """
        try:
            # Prepare data summary for GPT
            data_summary = self._prepare_data_summary(flight_data, query_intent)
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with context
            user_prompt = self._create_user_prompt(query, query_intent, data_summary, context)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse response
            answer = response.choices[0].message.content
            
            # Generate follow-up questions and visualizations
            follow_ups = self._generate_follow_up_questions(query_intent)
            visualizations = self._suggest_visualizations(query_intent)
            
            return QueryResponse(
                answer=answer,
                data=data_summary if data_summary else None,
                visualizations=visualizations,
                follow_up_questions=follow_ups,
                confidence=0.9,
                sources=["Flight database", "Historical analysis"]
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return QueryResponse(
                answer="I'm sorry, I couldn't generate a proper response to your query. Please try rephrasing your question.",
                confidence=0.0,
                sources=[]
            )
    
    def _prepare_data_summary(self, flight_data: Optional[pd.DataFrame], query_intent: QueryIntent) -> Optional[Dict[str, Any]]:
        """
        Prepare a summary of the flight data for GPT processing.
        
        Args:
            flight_data: Flight data DataFrame
            query_intent: Query intent and entities
            
        Returns:
            Data summary dictionary
        """
        if flight_data is None or flight_data.empty:
            return None
        
        summary = {
            'total_flights': len(flight_data),
            'date_range': {
                'start': flight_data['scheduled_departure'].min().isoformat() if 'scheduled_departure' in flight_data.columns else None,
                'end': flight_data['scheduled_departure'].max().isoformat() if 'scheduled_departure' in flight_data.columns else None
            }
        }
        
        # Add specific metrics based on query type
        if query_intent.query_type == 'delay_analysis':
            if 'delay_minutes' in flight_data.columns:
                summary.update({
                    'average_delay': flight_data['delay_minutes'].mean(),
                    'median_delay': flight_data['delay_minutes'].median(),
                    'max_delay': flight_data['delay_minutes'].max(),
                    'on_time_percentage': (flight_data['delay_minutes'] <= 15).mean() * 100
                })
        
        elif query_intent.query_type == 'congestion':
            if 'scheduled_departure' in flight_data.columns:
                flight_data['hour'] = pd.to_datetime(flight_data['scheduled_departure']).dt.hour
                hourly_counts = flight_data.groupby('hour').size()
                summary.update({
                    'peak_hour': hourly_counts.idxmax(),
                    'peak_hour_flights': hourly_counts.max(),
                    'quietest_hour': hourly_counts.idxmin(),
                    'quietest_hour_flights': hourly_counts.min()
                })
        
        # Add airport-specific data if requested
        if query_intent.airports:
            for airport in query_intent.airports:
                airport_data = flight_data[
                    (flight_data.get('origin_airport') == airport) | 
                    (flight_data.get('destination_airport') == airport)
                ] if 'origin_airport' in flight_data.columns else pd.DataFrame()
                
                if not airport_data.empty:
                    summary[f'{airport}_flights'] = len(airport_data)
        
        return summary
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for OpenAI."""
        return """You are an expert flight scheduling analyst assistant. You help users understand flight data, delays, congestion patterns, and scheduling optimization.

Your responses should be:
1. Clear and actionable
2. Based on the provided data
3. Include specific numbers and insights
4. Suggest practical recommendations
5. Be conversational but professional

When discussing times, use 24-hour format and mention time zones when relevant.
When discussing delays, categorize them as: minimal (0-15 min), moderate (15-60 min), significant (60+ min).
Always provide context for your recommendations."""
    
    def _create_user_prompt(
        self, 
        query: str, 
        query_intent: QueryIntent, 
        data_summary: Optional[Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Create user prompt with query and data context."""
        prompt = f"User Query: {query}\n\n"
        
        prompt += f"Query Intent: {query_intent.intent}\n"
        prompt += f"Query Type: {query_intent.query_type}\n"
        
        if query_intent.airports:
            prompt += f"Airports mentioned: {', '.join(query_intent.airports)}\n"
        
        if query_intent.airlines:
            prompt += f"Airlines mentioned: {', '.join(query_intent.airlines)}\n"
        
        if data_summary:
            prompt += f"\nFlight Data Summary:\n{json.dumps(data_summary, indent=2, default=str)}\n"
        else:
            prompt += "\nNo specific flight data available for this query.\n"
        
        if context:
            prompt += f"\nConversation Context: {json.dumps(context, indent=2, default=str)}\n"
        
        prompt += "\nPlease provide a helpful, data-driven response to the user's query."
        
        return prompt
    
    def _generate_follow_up_questions(self, query_intent: QueryIntent) -> List[str]:
        """Generate relevant follow-up questions based on query intent."""
        follow_ups = []
        
        if query_intent.query_type == 'best_time':
            follow_ups = [
                "Would you like to see delay patterns for specific days of the week?",
                "Are you interested in weather impact on these recommendations?",
                "Would you like to compare different airlines for this route?"
            ]
        elif query_intent.query_type == 'delay_analysis':
            follow_ups = [
                "Would you like to see the main causes of these delays?",
                "Are you interested in seasonal delay patterns?",
                "Would you like delay comparisons between different airports?"
            ]
        elif query_intent.query_type == 'congestion':
            follow_ups = [
                "Would you like to see congestion patterns for different days?",
                "Are you interested in runway-specific congestion data?",
                "Would you like recommendations for alternative time slots?"
            ]
        
        return follow_ups[:3]  # Limit to 3 follow-ups
    
    def _suggest_visualizations(self, query_intent: QueryIntent) -> List[str]:
        """Suggest relevant visualizations based on query intent."""
        visualizations = []
        
        if query_intent.query_type == 'delay_analysis':
            visualizations = [
                "delay_histogram",
                "delay_by_hour_heatmap",
                "delay_trends_timeline"
            ]
        elif query_intent.query_type == 'congestion':
            visualizations = [
                "hourly_flight_volume",
                "congestion_heatmap",
                "peak_hours_comparison"
            ]
        elif query_intent.query_type == 'best_time':
            visualizations = [
                "optimal_time_slots",
                "delay_probability_by_hour",
                "recommendation_chart"
            ]
        
        return visualizations
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a query can be processed.
        
        Args:
            query: User query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query) > 1000:
            return False, "Query is too long (max 1000 characters)"
        
        # Check for potentially harmful content
        harmful_patterns = ['delete', 'drop', 'truncate', 'update', 'insert']
        query_lower = query.lower()
        
        for pattern in harmful_patterns:
            if pattern in query_lower:
                return False, f"Query contains potentially harmful content: {pattern}"
        
        return True, None