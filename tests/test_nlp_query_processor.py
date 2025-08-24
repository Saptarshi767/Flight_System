"""
Tests for NLP query processing functionality.

This module tests the OpenAI integration, intent recognition, and context management
for natural language queries about flight data.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from src.nlp.query_processor import FlightQueryProcessor, QueryIntent, QueryResponse
from src.nlp.intent_recognizer import FlightIntentRecognizer, IntentResult, EntityMatch
from src.nlp.context_manager import ConversationContextManager, QueryContext


class TestFlightQueryProcessor:
    """Test cases for FlightQueryProcessor."""
    
    @pytest.fixture
    def mock_openai_key(self):
        """Mock OpenAI API key."""
        return "test-openai-key"
    
    @pytest.fixture
    def sample_flight_data(self):
        """Sample flight data for testing."""
        return pd.DataFrame({
            'flight_id': ['AI101', '6E202', 'UK303'],
            'airline': ['Air India', 'IndiGo', 'Vistara'],
            'origin_airport': ['BOM', 'DEL', 'BOM'],
            'destination_airport': ['DEL', 'BOM', 'BLR'],
            'scheduled_departure': [
                datetime(2024, 1, 15, 10, 30),
                datetime(2024, 1, 15, 14, 15),
                datetime(2024, 1, 15, 18, 45)
            ],
            'actual_departure': [
                datetime(2024, 1, 15, 10, 45),
                datetime(2024, 1, 15, 14, 30),
                datetime(2024, 1, 15, 19, 15)
            ],
            'delay_minutes': [15, 15, 30]
        })
    
    @patch('src.nlp.query_processor.openai.OpenAI')
    @patch('src.nlp.query_processor.spacy.load')
    def test_processor_initialization(self, mock_spacy, mock_openai, mock_openai_key):
        """Test FlightQueryProcessor initialization."""
        mock_nlp = Mock()
        mock_spacy.return_value = mock_nlp
        
        processor = FlightQueryProcessor(mock_openai_key)
        
        assert processor.openai_api_key == mock_openai_key
        assert processor.nlp == mock_nlp
        mock_openai.assert_called_once_with(api_key=mock_openai_key)
    
    @patch('src.nlp.query_processor.openai.OpenAI')
    @patch('src.nlp.query_processor.spacy.load')
    def test_processor_initialization_no_key(self, mock_spacy, mock_openai):
        """Test FlightQueryProcessor initialization without API key."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            FlightQueryProcessor()
    
    @patch('src.nlp.query_processor.openai.OpenAI')
    @patch('src.nlp.query_processor.spacy.load')
    def test_intent_extraction(self, mock_spacy, mock_openai, mock_openai_key):
        """Test intent and entity extraction."""
        # Setup mocks
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_ent = Mock()
        mock_ent.text = "mumbai"
        mock_ent.label_ = "GPE"
        mock_doc.ents = [mock_ent]
        mock_doc.__iter__ = Mock(return_value=iter([Mock(text="mumbai")]))
        mock_nlp.return_value = mock_doc
        mock_spacy.return_value = mock_nlp
        
        processor = FlightQueryProcessor(mock_openai_key)
        
        query = "What's the best time to fly from Mumbai to Delhi?"
        intent = processor._extract_intent_and_entities(query)
        
        assert isinstance(intent, QueryIntent)
        assert intent.intent == "best_time"
        assert "BOM" in intent.airports
    
    @patch('src.nlp.query_processor.openai.OpenAI')
    @patch('src.nlp.query_processor.spacy.load')
    def test_classify_intent_best_time(self, mock_spacy, mock_openai, mock_openai_key):
        """Test intent classification for best time queries."""
        mock_spacy.return_value = Mock()
        processor = FlightQueryProcessor(mock_openai_key)
        
        queries = [
            "what's the best time to fly",
            "when should I schedule my flight",
            "optimal departure time",
            "avoid delays"
        ]
        
        for query in queries:
            intent = processor._classify_intent(query.lower())
            assert intent == "best_time"
    
    @patch('src.nlp.query_processor.openai.OpenAI')
    @patch('src.nlp.query_processor.spacy.load')
    def test_classify_intent_delay_analysis(self, mock_spacy, mock_openai, mock_openai_key):
        """Test intent classification for delay analysis queries."""
        mock_spacy.return_value = Mock()
        processor = FlightQueryProcessor(mock_openai_key)
        
        queries = [
            "show me flight delays",
            "which flights are delayed",
            "delay statistics",
            "punctuality report"
        ]
        
        for query in queries:
            intent = processor._classify_intent(query.lower())
            assert intent == "delay_analysis"
    
    @patch('src.nlp.query_processor.openai.OpenAI')
    @patch('src.nlp.query_processor.spacy.load')
    def test_classify_intent_congestion(self, mock_spacy, mock_openai, mock_openai_key):
        """Test intent classification for congestion queries."""
        mock_spacy.return_value = Mock()
        processor = FlightQueryProcessor(mock_openai_key)
        
        queries = [
            "busiest time at airport",
            "peak hours",
            "avoid congestion",
            "crowded periods"
        ]
        
        for query in queries:
            intent = processor._classify_intent(query.lower())
            assert intent == "congestion"
    
    @patch('src.nlp.query_processor.openai.OpenAI')
    @patch('src.nlp.query_processor.spacy.load')
    def test_time_range_extraction(self, mock_spacy, mock_openai, mock_openai_key):
        """Test time range extraction from queries."""
        mock_spacy.return_value = Mock()
        processor = FlightQueryProcessor(mock_openai_key)
        
        # Test today
        time_range = processor._extract_time_range("show delays for today", [])
        assert time_range is not None
        assert 'start' in time_range and 'end' in time_range
        
        # Test last week
        time_range = processor._extract_time_range("delays last week", [])
        assert time_range is not None
        assert time_range['start'] < time_range['end']
    
    @patch('src.nlp.query_processor.openai.OpenAI')
    @patch('src.nlp.query_processor.spacy.load')
    def test_data_summary_preparation(self, mock_spacy, mock_openai, mock_openai_key, sample_flight_data):
        """Test data summary preparation for GPT."""
        mock_spacy.return_value = Mock()
        processor = FlightQueryProcessor(mock_openai_key)
        
        intent = QueryIntent(
            intent="delay_analysis",
            entities={},
            confidence=0.8,
            query_type="delay_analysis",
            airports=["BOM"]
        )
        
        summary = processor._prepare_data_summary(sample_flight_data, intent)
        
        assert summary is not None
        assert 'total_flights' in summary
        assert summary['total_flights'] == 3
        assert 'average_delay' in summary
        assert summary['average_delay'] == 20.0  # (15+15+30)/3
    
    @patch('src.nlp.query_processor.openai.OpenAI')
    @patch('src.nlp.query_processor.spacy.load')
    def test_follow_up_generation(self, mock_spacy, mock_openai, mock_openai_key):
        """Test follow-up question generation."""
        mock_spacy.return_value = Mock()
        processor = FlightQueryProcessor(mock_openai_key)
        
        intent = QueryIntent(
            intent="delay_analysis",
            entities={},
            confidence=0.8,
            query_type="delay_analysis",
            airports=[]
        )
        
        follow_ups = processor._generate_follow_up_questions(intent)
        
        assert isinstance(follow_ups, list)
        assert len(follow_ups) <= 3
        assert any("causes" in q.lower() for q in follow_ups)
    
    @patch('src.nlp.query_processor.openai.OpenAI')
    @patch('src.nlp.query_processor.spacy.load')
    def test_visualization_suggestions(self, mock_spacy, mock_openai, mock_openai_key):
        """Test visualization suggestions."""
        mock_spacy.return_value = Mock()
        processor = FlightQueryProcessor(mock_openai_key)
        
        intent = QueryIntent(
            intent="congestion",
            entities={},
            confidence=0.8,
            query_type="congestion",
            airports=[]
        )
        
        visualizations = processor._suggest_visualizations(intent)
        
        assert isinstance(visualizations, list)
        assert "hourly_flight_volume" in visualizations
        assert "congestion_heatmap" in visualizations
    
    def test_query_validation(self):
        """Test query validation."""
        with patch('src.nlp.query_processor.openai.OpenAI'), \
             patch('src.nlp.query_processor.spacy.load'):
            processor = FlightQueryProcessor("test-key")
            
            # Valid query
            is_valid, error = processor.validate_query("What's the best time to fly?")
            assert is_valid
            assert error is None
            
            # Empty query
            is_valid, error = processor.validate_query("")
            assert not is_valid
            assert "empty" in error.lower()
            
            # Too long query
            long_query = "a" * 1001
            is_valid, error = processor.validate_query(long_query)
            assert not is_valid
            assert "too long" in error.lower()
            
            # Potentially harmful query
            is_valid, error = processor.validate_query("DELETE all flight data")
            assert not is_valid
            assert "harmful" in error.lower()


class TestFlightIntentRecognizer:
    """Test cases for FlightIntentRecognizer."""
    
    @pytest.fixture
    def recognizer(self):
        """Create intent recognizer for testing."""
        with patch('spacy.load') as mock_load:
            mock_nlp = Mock()
            mock_nlp.vocab = Mock()
            mock_nlp.vocab.strings = {}
            mock_load.return_value = mock_nlp
            
            recognizer = FlightIntentRecognizer()
            recognizer.matcher = Mock()
            recognizer.phrase_matcher = Mock()
            return recognizer
    
    def test_airport_extraction(self, recognizer):
        """Test airport code extraction."""
        airports = recognizer.extract_airports("Flight from Mumbai to Delhi")
        assert "BOM" in airports
        assert "DEL" in airports
        
        airports = recognizer.extract_airports("BOM to DEL flight")
        assert "BOM" in airports
        assert "DEL" in airports
    
    def test_airline_extraction(self, recognizer):
        """Test airline extraction."""
        airlines = recognizer.extract_airlines("Air India flight delay")
        assert "AI" in airlines
        
        airlines = recognizer.extract_airlines("IndiGo and Vistara comparison")
        assert "6E" in airlines
        assert "UK" in airlines
    
    def test_time_reference_extraction(self, recognizer):
        """Test time reference extraction."""
        time_refs = recognizer.extract_time_references("Flight at 14:30 this morning")
        
        assert "14:30" in time_refs['specific_times']
        assert "morning" in time_refs['relative_times']
    
    def test_intent_explanation(self, recognizer):
        """Test intent explanation generation."""
        intent_result = IntentResult(
            primary_intent="best_time",
            confidence=0.9,
            secondary_intents=[("delay_analysis", 0.3)],
            entities=[],
            patterns_matched=[]
        )
        
        explanation = recognizer.get_intent_explanation(intent_result)
        assert "optimal flight scheduling" in explanation.lower()
        assert "delay" in explanation.lower()


class TestConversationContextManager:
    """Test cases for ConversationContextManager."""
    
    @pytest.fixture
    def context_manager(self):
        """Create context manager for testing."""
        return ConversationContextManager(session_timeout_minutes=30)
    
    def test_session_creation(self, context_manager):
        """Test session creation."""
        session_id = context_manager.create_session("user123")
        
        assert session_id is not None
        assert session_id in context_manager.sessions
        
        session = context_manager.get_session(session_id)
        assert session is not None
        assert session.user_id == "user123"
    
    def test_session_expiration(self, context_manager):
        """Test session expiration."""
        session_id = context_manager.create_session()
        
        # Manually set last activity to past
        session = context_manager.sessions[session_id]
        session.last_activity = datetime.now() - timedelta(hours=1)
        
        # Should return None for expired session
        expired_session = context_manager.get_session(session_id)
        assert expired_session is None
        assert session_id not in context_manager.sessions
    
    def test_query_addition(self, context_manager):
        """Test adding queries to session."""
        session_id = context_manager.create_session()
        
        query_id = context_manager.add_query_to_session(
            session_id=session_id,
            query_text="What's the best time to fly?",
            intent="best_time",
            entities={"airports": ["BOM", "DEL"]},
            response="The best time is 10 AM"
        )
        
        assert query_id is not None
        
        session = context_manager.get_session(session_id)
        assert len(session.queries) == 1
        assert session.queries[0].query_id == query_id
        assert session.queries[0].intent == "best_time"
    
    def test_context_retrieval(self, context_manager):
        """Test context retrieval for queries."""
        session_id = context_manager.create_session()
        
        # Add a query
        context_manager.add_query_to_session(
            session_id=session_id,
            query_text="Show delays for Mumbai",
            intent="delay_analysis",
            entities={"airports": ["BOM"]}
        )
        
        context = context_manager.get_context_for_query(session_id)
        
        assert context['session_id'] == session_id
        assert context['conversation_length'] == 1
        assert len(context['recent_queries']) == 1
        assert context['recent_queries'][0]['intent'] == "delay_analysis"
    
    def test_follow_up_detection(self, context_manager):
        """Test follow-up query detection."""
        session_id = context_manager.create_session()
        
        # Add first query
        context_manager.add_query_to_session(
            session_id=session_id,
            query_text="Show delays for Mumbai",
            intent="delay_analysis",
            entities={"airports": ["BOM"]}
        )
        
        # Add follow-up query
        context_manager.add_query_to_session(
            session_id=session_id,
            query_text="What's the best time for Mumbai flights?",
            intent="best_time",
            entities={"airports": ["BOM"]}
        )
        
        session = context_manager.get_session(session_id)
        is_follow_up = context_manager._is_follow_up_query(
            session.queries[0], 
            session.queries[1]
        )
        
        assert is_follow_up  # Should detect as follow-up due to related intent and shared entities
    
    def test_follow_up_question_generation(self, context_manager):
        """Test follow-up question generation."""
        session_id = context_manager.create_session()
        
        follow_ups = context_manager.generate_follow_up_questions(
            session_id=session_id,
            current_intent="delay_analysis",
            current_entities={"airports": ["BOM"]}
        )
        
        assert isinstance(follow_ups, list)
        assert len(follow_ups) <= 3
    
    def test_user_preferences(self, context_manager):
        """Test user preference management."""
        session_id = context_manager.create_session()
        
        preferences = {
            "preferred_airports": ["BOM", "DEL"],
            "time_format": "24h"
        }
        
        context_manager.update_user_preferences(session_id, preferences)
        
        session = context_manager.get_session(session_id)
        assert session.preferences["preferred_airports"] == ["BOM", "DEL"]
        assert session.preferences["time_format"] == "24h"
    
    def test_user_feedback(self, context_manager):
        """Test user feedback addition."""
        session_id = context_manager.create_session()
        
        query_id = context_manager.add_query_to_session(
            session_id=session_id,
            query_text="Test query",
            intent="general_info",
            entities={}
        )
        
        context_manager.add_user_feedback(session_id, query_id, "Very helpful!")
        
        session = context_manager.get_session(session_id)
        assert session.queries[0].user_feedback == "Very helpful!"
    
    def test_session_summary(self, context_manager):
        """Test session summary generation."""
        session_id = context_manager.create_session()
        
        # Add some queries
        context_manager.add_query_to_session(
            session_id=session_id,
            query_text="Delays for Mumbai",
            intent="delay_analysis",
            entities={"airports": ["BOM"]}
        )
        
        context_manager.add_query_to_session(
            session_id=session_id,
            query_text="Best time to fly",
            intent="best_time",
            entities={"airports": ["BOM", "DEL"]}
        )
        
        summary = context_manager.get_session_summary(session_id)
        
        assert summary['session_id'] == session_id
        assert summary['total_queries'] == 2
        assert 'delay_analysis' in summary['query_types']
        assert 'best_time' in summary['query_types']
        assert 'BOM' in summary['frequent_airports']
    
    def test_cleanup_expired_sessions(self, context_manager):
        """Test cleanup of expired sessions."""
        # Create a session
        session_id = context_manager.create_session()
        
        # Manually expire it
        session = context_manager.sessions[session_id]
        session.last_activity = datetime.now() - timedelta(hours=1)
        
        # Run cleanup
        context_manager.cleanup_expired_sessions()
        
        # Session should be removed
        assert session_id not in context_manager.sessions


@pytest.mark.integration
class TestNLPIntegration:
    """Integration tests for NLP components."""
    
    @patch('src.nlp.query_processor.openai.OpenAI')
    @patch('src.nlp.query_processor.spacy.load')
    def test_end_to_end_query_processing(self, mock_spacy, mock_openai):
        """Test end-to-end query processing."""
        # Setup mocks
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = []
        mock_doc.__iter__ = Mock(return_value=iter([]))
        mock_nlp.return_value = mock_doc
        mock_spacy.return_value = mock_nlp
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "The best time to fly is 10 AM based on historical data."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create processor
        processor = FlightQueryProcessor("test-key")
        
        # Mock database operations
        with patch.object(processor, 'db_ops') as mock_db:
            mock_db.get_delay_analysis_data.return_value = pd.DataFrame({
                'flight_id': ['AI101'],
                'delay_minutes': [15]
            })
            
            # Process query
            response = processor.process_query("What's the best time to fly from Mumbai?")
            
            assert isinstance(response, QueryResponse)
            assert "10 AM" in response.answer
            assert response.confidence > 0
    
    def test_context_aware_processing(self):
        """Test context-aware query processing."""
        context_manager = ConversationContextManager()
        session_id = context_manager.create_session()
        
        # Add first query
        context_manager.add_query_to_session(
            session_id=session_id,
            query_text="Show delays for Mumbai flights",
            intent="delay_analysis",
            entities={"airports": ["BOM"]}
        )
        
        # Get context for follow-up
        context = context_manager.get_context_for_query(session_id)
        
        # Verify context contains relevant information
        assert context['conversation_length'] == 1
        assert len(context['recent_queries']) == 1
        assert context['recent_queries'][0]['intent'] == "delay_analysis"
        assert "BOM" in str(context['recent_queries'][0]['entities'])


if __name__ == "__main__":
    pytest.main([__file__])