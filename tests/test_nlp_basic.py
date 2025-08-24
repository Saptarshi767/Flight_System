"""
Basic tests for NLP functionality without requiring spaCy model loading.

This module tests the core NLP functionality using mocks to avoid
spaCy compatibility issues during development.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta

# Test the basic structure without importing spaCy-dependent modules
def test_nlp_module_structure():
    """Test that NLP module structure is correct."""
    try:
        from src.nlp import __all__
        expected_exports = [
            'FlightQueryProcessor',
            'QueryIntent', 
            'QueryResponse',
            'FlightIntentRecognizer',
            'IntentResult',
            'EntityMatch',
            'ConversationContextManager',
            'QueryContext',
            'ConversationSession'
        ]
        
        for export in expected_exports:
            assert export in __all__, f"Missing export: {export}"
        
        print("✓ NLP module structure is correct")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_query_intent_model():
    """Test QueryIntent model structure."""
    try:
        from src.nlp.query_processor import QueryIntent
        
        # Test creating a QueryIntent instance
        intent = QueryIntent(
            intent="best_time",
            entities={"airports": ["BOM", "DEL"]},
            confidence=0.8,
            query_type="best_time",
            airports=["BOM", "DEL"]
        )
        
        assert intent.intent == "best_time"
        assert intent.confidence == 0.8
        assert "BOM" in intent.airports
        assert "DEL" in intent.airports
        
        print("✓ QueryIntent model works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_query_response_model():
    """Test QueryResponse model structure."""
    try:
        from src.nlp.query_processor import QueryResponse
        
        # Test creating a QueryResponse instance
        response = QueryResponse(
            answer="The best time to fly is 10 AM",
            confidence=0.9,
            sources=["Flight database"]
        )
        
        assert response.answer == "The best time to fly is 10 AM"
        assert response.confidence == 0.9
        assert "Flight database" in response.sources
        
        print("✓ QueryResponse model works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


@patch('src.nlp.query_processor.spacy.load')
@patch('src.nlp.query_processor.openai.OpenAI')
def test_query_processor_initialization_mocked(mock_openai, mock_spacy):
    """Test FlightQueryProcessor initialization with mocks."""
    try:
        from src.nlp.query_processor import FlightQueryProcessor
        
        # Setup mocks
        mock_nlp = Mock()
        mock_spacy.return_value = mock_nlp
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Test initialization
        processor = FlightQueryProcessor("test-api-key")
        
        assert processor.openai_api_key == "test-api-key"
        assert processor.nlp == mock_nlp
        
        # Verify spaCy was loaded
        mock_spacy.assert_called_once_with("en_core_web_sm")
        
        # Verify OpenAI client was created
        mock_openai.assert_called_once_with(api_key="test-api-key")
        
        print("✓ FlightQueryProcessor initialization works with mocks")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_context_manager_basic():
    """Test ConversationContextManager basic functionality."""
    try:
        from src.nlp.context_manager import ConversationContextManager
        
        # Test initialization
        context_manager = ConversationContextManager(session_timeout_minutes=30)
        assert context_manager.session_timeout.total_seconds() == 30 * 60
        
        # Test session creation
        session_id = context_manager.create_session("test_user")
        assert session_id is not None
        assert session_id in context_manager.sessions
        
        # Test session retrieval
        session = context_manager.get_session(session_id)
        assert session is not None
        assert session.user_id == "test_user"
        
        print("✓ ConversationContextManager basic functionality works")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


@patch('src.nlp.query_processor.spacy.load')
@patch('src.nlp.query_processor.openai.OpenAI')
def test_intent_classification_logic(mock_openai, mock_spacy):
    """Test intent classification logic with mocks."""
    try:
        from src.nlp.query_processor import FlightQueryProcessor
        
        # Setup mocks
        mock_nlp = Mock()
        mock_spacy.return_value = mock_nlp
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        processor = FlightQueryProcessor("test-api-key")
        
        # Test different query types
        test_cases = [
            ("what's the best time to fly", "best_time"),
            ("show me flight delays", "delay_analysis"),
            ("busiest hours at airport", "congestion"),
            ("schedule change impact", "schedule_impact"),
            ("cascading effects", "cascading_impact"),
            ("flight information", "general_info")
        ]
        
        for query, expected_intent in test_cases:
            intent = processor._classify_intent(query.lower())
            assert intent == expected_intent, f"Query '{query}' should classify as '{expected_intent}', got '{intent}'"
        
        print("✓ Intent classification logic works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


@patch('src.nlp.query_processor.spacy.load')
@patch('src.nlp.query_processor.openai.OpenAI')
def test_time_range_extraction_logic(mock_openai, mock_spacy):
    """Test time range extraction logic."""
    try:
        from src.nlp.query_processor import FlightQueryProcessor
        
        # Setup mocks
        mock_nlp = Mock()
        mock_spacy.return_value = mock_nlp
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        processor = FlightQueryProcessor("test-api-key")
        
        # Test time range extraction
        test_cases = [
            ("show delays for today", "today"),
            ("flights yesterday", "yesterday"),
            ("data from last week", "last week"),
            ("analysis for last month", "last month")
        ]
        
        for query, time_keyword in test_cases:
            time_range = processor._extract_time_range(query, [])
            assert time_range is not None, f"Should extract time range for '{query}'"
            assert 'start' in time_range and 'end' in time_range
            assert time_range['start'] < time_range['end']
        
        print("✓ Time range extraction logic works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_database_operations_methods():
    """Test that DatabaseOperations has required methods."""
    try:
        from src.database.operations import DatabaseOperations
        
        db_ops = DatabaseOperations()
        
        # Check that required methods exist
        required_methods = [
            'get_delay_analysis_data',
            'get_congestion_data',
            'get_schedule_impact_data',
            'get_cascading_impact_data',
            'get_general_flight_data'
        ]
        
        for method_name in required_methods:
            assert hasattr(db_ops, method_name), f"Missing method: {method_name}"
            method = getattr(db_ops, method_name)
            assert callable(method), f"Method {method_name} is not callable"
        
        print("✓ DatabaseOperations has all required methods")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


@patch('src.nlp.query_processor.spacy.load')
@patch('src.nlp.query_processor.openai.OpenAI')
def test_query_validation_logic(mock_openai, mock_spacy):
    """Test query validation logic."""
    try:
        from src.nlp.query_processor import FlightQueryProcessor
        
        # Setup mocks
        mock_nlp = Mock()
        mock_spacy.return_value = mock_nlp
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        processor = FlightQueryProcessor("test-api-key")
        
        # Test valid query
        is_valid, error = processor.validate_query("What's the best time to fly?")
        assert is_valid
        assert error is None
        
        # Test empty query
        is_valid, error = processor.validate_query("")
        assert not is_valid
        assert "empty" in error.lower()
        
        # Test too long query
        long_query = "a" * 1001
        is_valid, error = processor.validate_query(long_query)
        assert not is_valid
        assert "too long" in error.lower()
        
        # Test potentially harmful query
        is_valid, error = processor.validate_query("DELETE all flight data")
        assert not is_valid
        assert "harmful" in error.lower()
        
        print("✓ Query validation logic works correctly")
        
    except ImportError as e:
        pytest.skip(f"Skipping due to import error: {e}")


if __name__ == "__main__":
    # Run tests manually
    test_functions = [
        test_nlp_module_structure,
        test_query_intent_model,
        test_query_response_model,
        test_query_processor_initialization_mocked,
        test_context_manager_basic,
        test_intent_classification_logic,
        test_time_range_extraction_logic,
        test_database_operations_methods,
        test_query_validation_logic
    ]
    
    print("Running NLP basic tests...")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {failed} tests failed")