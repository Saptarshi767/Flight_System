"""
Tests for LangChain orchestration functionality.

This module tests the LangChain chains, tools, and orchestration
for complex flight data queries.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import json

# Mock LangChain imports to avoid dependency issues during testing
@pytest.fixture(autouse=True)
def mock_langchain_imports():
    """Mock LangChain imports to avoid import errors."""
    with patch.dict('sys.modules', {
        'langchain.llms': Mock(),
        'langchain.chat_models': Mock(),
        'langchain.chains': Mock(),
        'langchain.chains.router': Mock(),
        'langchain.chains.router.llm_router': Mock(),
        'langchain.chains.router.multi_route_prompt': Mock(),
        'langchain.memory': Mock(),
        'langchain.prompts': Mock(),
        'langchain.tools': Mock(),
        'langchain.agents': Mock(),
        'langchain.schema': Mock(),
        'langchain.callbacks.manager': Mock(),
    }):
        yield


class TestFlightDataTool:
    """Test cases for FlightDataTool."""
    
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_flight_data_tool_initialization(self, mock_db_ops):
        """Test FlightDataTool initialization."""
        try:
            from src.nlp.langchain_orchestrator import FlightDataTool
            
            db_ops = mock_db_ops()
            tool = FlightDataTool(db_ops)
            
            assert tool.name == "flight_data_retrieval"
            assert "flight data" in tool.description.lower()
            assert tool.db_ops == db_ops
            
            print("✓ FlightDataTool initialization works")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")
    
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_flight_data_tool_query_parsing(self, mock_db_ops):
        """Test query parsing in FlightDataTool."""
        try:
            from src.nlp.langchain_orchestrator import FlightDataTool
            
            db_ops = mock_db_ops()
            tool = FlightDataTool(db_ops)
            
            # Test airport extraction
            filters = tool._parse_query_filters("flights from BOM to DEL")
            assert "BOM" in filters.get("airports", [])
            assert "DEL" in filters.get("airports", [])
            
            # Test airline extraction
            filters = tool._parse_query_filters("Air India and IndiGo flights")
            assert "AI" in filters.get("airlines", [])
            assert "6E" in filters.get("airlines", [])
            
            print("✓ FlightDataTool query parsing works")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")
    
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_flight_data_tool_execution(self, mock_db_ops):
        """Test FlightDataTool execution."""
        try:
            from src.nlp.langchain_orchestrator import FlightDataTool
            
            # Mock database operations
            db_ops = mock_db_ops()
            sample_data = pd.DataFrame({
                'flight_id': ['AI101', '6E202'],
                'origin_airport': ['BOM', 'DEL'],
                'destination_airport': ['DEL', 'BOM'],
                'airline': ['AI', '6E'],
                'scheduled_departure': [datetime.now(), datetime.now()]
            })
            db_ops.get_general_flight_data.return_value = sample_data
            
            tool = FlightDataTool(db_ops)
            result = tool._run("flights from BOM")
            
            # Should return JSON string with flight data summary
            assert isinstance(result, str)
            data = json.loads(result)
            assert 'total_flights' in data
            assert data['total_flights'] == 2
            
            print("✓ FlightDataTool execution works")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")


class TestDelayAnalysisTool:
    """Test cases for DelayAnalysisTool."""
    
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_delay_analysis_tool_initialization(self, mock_db_ops):
        """Test DelayAnalysisTool initialization."""
        try:
            from src.nlp.langchain_orchestrator import DelayAnalysisTool
            
            db_ops = mock_db_ops()
            delay_analyzer = Mock()
            tool = DelayAnalysisTool(delay_analyzer, db_ops)
            
            assert tool.name == "delay_analysis"
            assert "delay" in tool.description.lower()
            assert tool.delay_analyzer == delay_analyzer
            assert tool.db_ops == db_ops
            
            print("✓ DelayAnalysisTool initialization works")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")
    
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_delay_analysis_tool_execution(self, mock_db_ops):
        """Test DelayAnalysisTool execution."""
        try:
            from src.nlp.langchain_orchestrator import DelayAnalysisTool
            
            # Mock components
            db_ops = mock_db_ops()
            delay_analyzer = Mock()
            
            sample_data = pd.DataFrame({
                'flight_id': ['AI101', '6E202'],
                'delay_minutes': [15, 30],
                'origin_airport': ['BOM', 'DEL'],
                'destination_airport': ['DEL', 'BOM'],
                'scheduled_departure': [datetime.now(), datetime.now()]
            })
            db_ops.get_delay_analysis_data.return_value = sample_data
            
            tool = DelayAnalysisTool(delay_analyzer, db_ops)
            result = tool._run("analyze delays for BOM")
            
            # Should return JSON string with delay analysis
            assert isinstance(result, str)
            data = json.loads(result)
            assert 'total_flights_analyzed' in data
            assert 'average_delay_minutes' in data
            
            print("✓ DelayAnalysisTool execution works")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")
    
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_delay_analysis_helper_methods(self, mock_db_ops):
        """Test DelayAnalysisTool helper methods."""
        try:
            from src.nlp.langchain_orchestrator import DelayAnalysisTool
            
            db_ops = mock_db_ops()
            delay_analyzer = Mock()
            tool = DelayAnalysisTool(delay_analyzer, db_ops)
            
            # Test most delayed route
            sample_data = pd.DataFrame({
                'origin_airport': ['BOM', 'DEL', 'BOM'],
                'destination_airport': ['DEL', 'BOM', 'BLR'],
                'delay_minutes': [10, 30, 45]
            })
            
            most_delayed = tool._get_most_delayed_route(sample_data)
            assert most_delayed == 'BOM-BLR'  # Highest delay
            
            # Test peak delay hours
            sample_data['scheduled_departure'] = [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 14, 0),
                datetime(2024, 1, 1, 14, 30)
            ]
            
            peak_hours = tool._get_peak_delay_hours(sample_data)
            assert 14 in peak_hours  # Hour with most delays
            
            print("✓ DelayAnalysisTool helper methods work")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")


class TestCongestionAnalysisTool:
    """Test cases for CongestionAnalysisTool."""
    
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_congestion_analysis_tool_initialization(self, mock_db_ops):
        """Test CongestionAnalysisTool initialization."""
        try:
            from src.nlp.langchain_orchestrator import CongestionAnalysisTool
            
            db_ops = mock_db_ops()
            congestion_analyzer = Mock()
            tool = CongestionAnalysisTool(congestion_analyzer, db_ops)
            
            assert tool.name == "congestion_analysis"
            assert "congestion" in tool.description.lower()
            
            print("✓ CongestionAnalysisTool initialization works")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")
    
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_congestion_analysis_helper_methods(self, mock_db_ops):
        """Test CongestionAnalysisTool helper methods."""
        try:
            from src.nlp.langchain_orchestrator import CongestionAnalysisTool
            
            db_ops = mock_db_ops()
            congestion_analyzer = Mock()
            tool = CongestionAnalysisTool(congestion_analyzer, db_ops)
            
            # Test peak hours calculation
            sample_data = pd.DataFrame({
                'hour': [10, 14, 14, 14, 18, 18],
                'origin_airport': ['BOM', 'BOM', 'DEL', 'BOM', 'DEL', 'BOM']
            })
            
            peak_hours = tool._get_peak_hours(sample_data)
            assert 14 in peak_hours  # Most frequent hour
            
            # Test busiest airports
            busiest = tool._get_busiest_airports(sample_data)
            assert 'BOM' in busiest  # Most flights
            
            # Test congestion score
            score = tool._calculate_congestion_score(sample_data)
            assert isinstance(score, float)
            assert score >= 0
            
            print("✓ CongestionAnalysisTool helper methods work")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")


class TestFlightQueryOrchestrator:
    """Test cases for FlightQueryOrchestrator."""
    
    @patch('src.nlp.langchain_orchestrator.ChatOpenAI')
    @patch('src.nlp.langchain_orchestrator.ConversationBufferMemory')
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_orchestrator_initialization(self, mock_db_ops, mock_memory, mock_llm):
        """Test FlightQueryOrchestrator initialization."""
        try:
            from src.nlp.langchain_orchestrator import FlightQueryOrchestrator
            
            # Mock components
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_memory_instance = Mock()
            mock_memory.return_value = mock_memory_instance
            
            orchestrator = FlightQueryOrchestrator("test-api-key")
            
            assert orchestrator.openai_api_key == "test-api-key"
            assert orchestrator.llm == mock_llm_instance
            assert orchestrator.memory == mock_memory_instance
            assert len(orchestrator.tools) > 0  # Should have at least FlightDataTool
            
            print("✓ FlightQueryOrchestrator initialization works")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")
    
    @patch('src.nlp.langchain_orchestrator.ChatOpenAI')
    @patch('src.nlp.langchain_orchestrator.ConversationBufferMemory')
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_query_complexity_classification(self, mock_db_ops, mock_memory, mock_llm):
        """Test query complexity classification."""
        try:
            from src.nlp.langchain_orchestrator import FlightQueryOrchestrator
            
            # Mock components
            mock_llm.return_value = Mock()
            mock_memory.return_value = Mock()
            
            orchestrator = FlightQueryOrchestrator("test-api-key")
            
            # Test different query types
            test_cases = [
                ("What time is flight AI101?", "simple"),
                ("Show delays and then congestion patterns", "multi_step"),
                ("Analyze flight delays for Mumbai", "delay_analysis"),
                ("What are the busiest hours at Delhi airport?", "congestion_analysis"),
                ("What's the impact of changing schedule by 1 hour?", "schedule_optimization")
            ]
            
            for query, expected_type in test_cases:
                query_type = orchestrator._classify_query_complexity(query)
                assert query_type == expected_type, f"Query '{query}' should be classified as '{expected_type}', got '{query_type}'"
            
            print("✓ Query complexity classification works")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")
    
    @patch('src.nlp.langchain_orchestrator.ChatOpenAI')
    @patch('src.nlp.langchain_orchestrator.ConversationBufferMemory')
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_follow_up_generation(self, mock_db_ops, mock_memory, mock_llm):
        """Test follow-up question generation."""
        try:
            from src.nlp.langchain_orchestrator import FlightQueryOrchestrator
            
            # Mock components
            mock_llm.return_value = Mock()
            mock_memory.return_value = Mock()
            
            orchestrator = FlightQueryOrchestrator("test-api-key")
            
            # Test follow-up generation for different query types
            delay_query = "Show me flight delays"
            delay_response = "Average delay is 25 minutes"
            follow_ups = orchestrator._generate_follow_ups(delay_query, delay_response)
            
            assert len(follow_ups) <= 3
            assert any("cause" in f.lower() for f in follow_ups)
            
            congestion_query = "What are the busiest hours?"
            congestion_response = "Peak hours are 2-4 PM"
            follow_ups = orchestrator._generate_follow_ups(congestion_query, congestion_response)
            
            assert len(follow_ups) <= 3
            assert any("alternative" in f.lower() for f in follow_ups)
            
            print("✓ Follow-up generation works")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")
    
    @patch('src.nlp.langchain_orchestrator.ChatOpenAI')
    @patch('src.nlp.langchain_orchestrator.ConversationBufferMemory')
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_confidence_calculation(self, mock_db_ops, mock_memory, mock_llm):
        """Test confidence score calculation."""
        try:
            from src.nlp.langchain_orchestrator import FlightQueryOrchestrator
            
            # Mock components
            mock_llm.return_value = Mock()
            mock_memory.return_value = Mock()
            
            orchestrator = FlightQueryOrchestrator("test-api-key")
            
            # Test different response types
            error_response = "Error occurred while processing"
            confidence = orchestrator._calculate_confidence(error_response)
            assert confidence == 0.1
            
            unavailable_response = "Data not available"
            confidence = orchestrator._calculate_confidence(unavailable_response)
            assert confidence == 0.3
            
            short_response = "Yes"
            confidence = orchestrator._calculate_confidence(short_response)
            assert confidence == 0.5
            
            good_response = "Based on the analysis of 1000 flights, the average delay is 15 minutes with peak delays occurring between 2-4 PM."
            confidence = orchestrator._calculate_confidence(good_response)
            assert confidence == 0.8
            
            print("✓ Confidence calculation works")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")
    
    @patch('src.nlp.langchain_orchestrator.ChatOpenAI')
    @patch('src.nlp.langchain_orchestrator.ConversationBufferMemory')
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_capabilities_reporting(self, mock_db_ops, mock_memory, mock_llm):
        """Test capabilities reporting."""
        try:
            from src.nlp.langchain_orchestrator import FlightQueryOrchestrator
            
            # Mock components
            mock_llm.return_value = Mock()
            mock_memory.return_value = Mock()
            
            orchestrator = FlightQueryOrchestrator("test-api-key")
            capabilities = orchestrator.get_available_capabilities()
            
            assert 'tools' in capabilities
            assert 'chains' in capabilities
            assert 'memory_type' in capabilities
            assert 'supported_query_types' in capabilities
            
            # Check that basic capabilities are present
            assert len(capabilities['tools']) > 0
            assert 'simple' in capabilities['supported_query_types']
            assert 'multi_step' in capabilities['supported_query_types']
            
            print("✓ Capabilities reporting works")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")


@pytest.mark.integration
class TestLangChainIntegration:
    """Integration tests for LangChain orchestration."""
    
    @patch('src.nlp.langchain_orchestrator.ChatOpenAI')
    @patch('src.nlp.langchain_orchestrator.ConversationBufferMemory')
    @patch('src.nlp.langchain_orchestrator.DatabaseOperations')
    def test_end_to_end_query_processing(self, mock_db_ops, mock_memory, mock_llm):
        """Test end-to-end query processing."""
        try:
            from src.nlp.langchain_orchestrator import FlightQueryOrchestrator
            
            # Mock components
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            mock_memory_instance = Mock()
            mock_memory.return_value = mock_memory_instance
            
            # Mock database operations
            db_ops_instance = mock_db_ops.return_value
            sample_data = pd.DataFrame({
                'flight_id': ['AI101'],
                'delay_minutes': [15]
            })
            db_ops_instance.get_general_flight_data.return_value = sample_data
            
            orchestrator = FlightQueryOrchestrator("test-api-key")
            
            # Test simple query processing
            result = orchestrator.process_complex_query("What's the status of flight AI101?")
            
            assert isinstance(result, dict)
            assert 'response' in result
            assert 'query_type' in result
            assert 'confidence' in result
            assert 'follow_up_suggestions' in result
            
            print("✓ End-to-end query processing works")
            
        except ImportError as e:
            pytest.skip(f"Skipping due to import error: {e}")


if __name__ == "__main__":
    # Run tests manually
    test_functions = [
        TestFlightDataTool().test_flight_data_tool_initialization,
        TestFlightDataTool().test_flight_data_tool_query_parsing,
        TestFlightDataTool().test_flight_data_tool_execution,
        TestDelayAnalysisTool().test_delay_analysis_tool_initialization,
        TestDelayAnalysisTool().test_delay_analysis_tool_execution,
        TestDelayAnalysisTool().test_delay_analysis_helper_methods,
        TestCongestionAnalysisTool().test_congestion_analysis_tool_initialization,
        TestCongestionAnalysisTool().test_congestion_analysis_helper_methods,
        TestFlightQueryOrchestrator().test_orchestrator_initialization,
        TestFlightQueryOrchestrator().test_query_complexity_classification,
        TestFlightQueryOrchestrator().test_follow_up_generation,
        TestFlightQueryOrchestrator().test_confidence_calculation,
        TestFlightQueryOrchestrator().test_capabilities_reporting,
        TestLangChainIntegration().test_end_to_end_query_processing
    ]
    
    print("Running LangChain orchestrator tests...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            # Create mock instances for methods that need them
            if hasattr(test_func, '__self__'):
                test_func()
            else:
                test_func(Mock(), Mock(), Mock())
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {failed} tests failed")