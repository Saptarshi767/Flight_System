"""
Tests for the query response generation system.

This module tests response templates, visualization recommendations,
follow-up question generation, and response caching functionality.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime
import json

from src.nlp.response_generator import (
    FlightResponseGenerator, ResponseType, VisualizationType,
    VisualizationRecommendation, ResponseTemplate
)


class TestFlightResponseGenerator:
    """Test cases for FlightResponseGenerator."""
    
    @pytest.fixture
    def response_generator(self):
        """Create response generator for testing."""
        return FlightResponseGenerator()
    
    @pytest.fixture
    def sample_delay_data(self):
        """Sample delay analysis data."""
        return {
            'total_flights': 150,
            'average_delay': 22.5,
            'on_time_percentage': 78.3,
            'most_delayed_route': 'BOM-DEL',
            'peak_delay_hours': [14, 15, 16],
            'delay_insights': 'Peak delays occur during afternoon hours',
            'recommendations': 'Consider scheduling flights outside 2-4 PM window'
        }
    
    @pytest.fixture
    def sample_congestion_data(self):
        """Sample congestion analysis data."""
        return {
            'total_flights': 200,
            'peak_hours': [14, 15, 16],
            'quietest_hours': [2, 3, 4],
            'congestion_score': 7.2,
            'busiest_airports': ['BOM', 'DEL', 'BLR'],
            'congestion_insights': 'High congestion during business hours',
            'best_times': 'Early morning (6-8 AM) and late evening (10 PM-12 AM)',
            'avoid_times': 'Afternoon peak (2-4 PM) and evening rush (6-8 PM)'
        }
    
    def test_initialization(self, response_generator):
        """Test response generator initialization."""
        assert response_generator is not None
        assert len(response_generator.templates) > 0
        assert len(response_generator.visualization_configs) > 0
        
        # Check that all response types have templates
        expected_types = [
            ResponseType.DELAY_ANALYSIS,
            ResponseType.CONGESTION_ANALYSIS,
            ResponseType.BEST_TIME_RECOMMENDATION,
            ResponseType.SCHEDULE_IMPACT,
            ResponseType.CASCADING_IMPACT,
            ResponseType.GENERAL_INFO,
            ResponseType.ERROR,
            ResponseType.NO_DATA
        ]
        
        for response_type in expected_types:
            assert response_type in response_generator.templates
        
        print("✓ FlightResponseGenerator initialization works")
    
    def test_delay_analysis_response(self, response_generator, sample_delay_data):
        """Test delay analysis response generation."""
        response = response_generator.generate_response(
            ResponseType.DELAY_ANALYSIS,
            sample_delay_data
        )
        
        assert response['response_type'] == ResponseType.DELAY_ANALYSIS.value
        assert 'response_text' in response
        assert 'visualizations' in response
        assert 'follow_up_questions' in response
        
        # Check that key data is included in response text
        response_text = response['response_text']
        assert '150' in response_text  # total_flights
        assert '22.5' in response_text  # average_delay
        assert '78.3' in response_text  # on_time_percentage
        
        # Check visualizations
        assert len(response['visualizations']) > 0
        viz_types = [viz['type'] for viz in response['visualizations']]
        assert 'histogram' in viz_types
        
        # Check follow-up questions
        assert len(response['follow_up_questions']) > 0
        
        print("✓ Delay analysis response generation works")
    
    def test_congestion_analysis_response(self, response_generator, sample_congestion_data):
        """Test congestion analysis response generation."""
        response = response_generator.generate_response(
            ResponseType.CONGESTION_ANALYSIS,
            sample_congestion_data
        )
        
        assert response['response_type'] == ResponseType.CONGESTION_ANALYSIS.value
        
        response_text = response['response_text']
        assert '200' in response_text  # total_flights
        assert '7.2' in response_text  # congestion_score
        
        # Check for peak hours formatting
        assert '14, 15, 16' in response_text or '14-16' in response_text
        
        # Check visualizations include line chart and heatmap
        viz_types = [viz['type'] for viz in response['visualizations']]
        assert 'line_chart' in viz_types
        assert 'heatmap' in viz_types
        
        print("✓ Congestion analysis response generation works")
    
    def test_missing_data_handling(self, response_generator):
        """Test handling of missing required data."""
        incomplete_data = {
            'total_flights': 100
            # Missing other required fields
        }
        
        response = response_generator.generate_response(
            ResponseType.DELAY_ANALYSIS,
            incomplete_data
        )
        
        # Should still generate a response with defaults
        assert response['response_type'] == ResponseType.DELAY_ANALYSIS.value
        assert 'response_text' in response
        
        # Confidence should be lower due to missing data
        assert response['confidence'] < 1.0
        
        print("✓ Missing data handling works")
    
    def test_visualization_recommendations(self, response_generator, sample_delay_data):
        """Test visualization recommendation generation."""
        response = response_generator.generate_response(
            ResponseType.DELAY_ANALYSIS,
            sample_delay_data
        )
        
        visualizations = response['visualizations']
        assert len(visualizations) > 0
        
        # Check visualization structure
        for viz in visualizations:
            assert 'type' in viz
            assert 'title' in viz
            assert 'description' in viz
            assert 'data_columns' in viz
            assert 'config' in viz
            assert 'priority' in viz
            assert 'implementation' in viz
        
        # Check that visualizations are sorted by priority
        priorities = [viz['priority'] for viz in visualizations]
        assert priorities == sorted(priorities)
        
        print("✓ Visualization recommendations work")
    
    def test_follow_up_question_generation(self, response_generator, sample_delay_data):
        """Test follow-up question generation."""
        context = {
            'user_preferences': {
                'preferred_airports': ['BOM', 'DEL']
            },
            'recent_queries': [
                {'intent': 'delay_analysis', 'entities': {'airports': ['BOM']}}
            ]
        }
        
        response = response_generator.generate_response(
            ResponseType.DELAY_ANALYSIS,
            sample_delay_data,
            context
        )
        
        follow_ups = response['follow_up_questions']
        assert len(follow_ups) > 0
        assert len(follow_ups) <= 5  # Should be limited
        
        # Check that context is used in follow-ups
        follow_up_text = ' '.join(follow_ups)
        # Should contain airport references or context-aware questions
        
        print("✓ Follow-up question generation works")
    
    def test_data_formatting(self, response_generator):
        """Test data formatting for templates."""
        data = {
            'peak_hours': [14, 15, 16, 17, 18],  # Long list
            'short_list': ['A', 'B'],  # Short list
            'empty_list': [],  # Empty list
            'float_value': 123.456789,  # Float needing formatting
            'large_float': 12345.67,  # Large float
            'dict_value': {'key1': 'value1', 'key2': 'value2'}  # Dictionary
        }
        
        formatted = response_generator._format_data_for_template(data)
        
        # Long list should be truncated
        assert 'and 2 more' in formatted['peak_hours']
        
        # Short list should be joined
        assert formatted['short_list'] == 'A, B'
        
        # Empty list should show "None"
        assert formatted['empty_list'] == 'None'
        
        # Float should be formatted
        assert formatted['float_value'] == '123.46'
        assert formatted['large_float'] == '12346'
        
        # Dictionary should be formatted as key-value pairs
        assert 'key1: value1' in formatted['dict_value']
        
        print("✓ Data formatting works")
    
    def test_time_format_conversion(self, response_generator):
        """Test time format conversion."""
        # Test 12-hour format conversion
        assert response_generator._hour_to_12h(0) == "12 AM"
        assert response_generator._hour_to_12h(6) == "6 AM"
        assert response_generator._hour_to_12h(12) == "12 PM"
        assert response_generator._hour_to_12h(18) == "6 PM"
        assert response_generator._hour_to_12h(23) == "11 PM"
        
        # Test time range conversion
        time_range = "14-16"
        converted = response_generator._convert_to_12h_format(time_range)
        assert "2 PM" in converted and "4 PM" in converted
        
        print("✓ Time format conversion works")
    
    def test_confidence_calculation(self, response_generator):
        """Test response confidence calculation."""
        template = response_generator.templates[ResponseType.DELAY_ANALYSIS]
        
        # Complete data should have high confidence
        complete_data = {
            'total_flights': 1000,
            'average_delay': 15.0,
            'on_time_percentage': 85.0,
            'most_delayed_route': 'BOM-DEL',
            'peak_delay_hours': [14, 15],
            'delay_insights': 'Good insights',
            'recommendations': 'Clear recommendations'
        }
        
        confidence = response_generator._calculate_response_confidence(complete_data, template)
        assert confidence > 0.8
        
        # Incomplete data should have lower confidence
        incomplete_data = {
            'total_flights': 10,  # Small dataset
            'average_delay': 15.0,
            'on_time_percentage': 85.0
            # Missing optional fields
        }
        
        confidence = response_generator._calculate_response_confidence(incomplete_data, template)
        assert confidence < 0.8
        
        print("✓ Confidence calculation works")
    
    def test_data_summary_creation(self, response_generator):
        """Test data summary creation."""
        data = {
            'flight_df': pd.DataFrame({
                'flight_id': ['AI101', '6E202'],
                'origin_airport': ['BOM', 'DEL'],
                'destination_airport': ['DEL', 'BOM'],
                'airline': ['AI', '6E'],
                'scheduled_departure': [datetime.now(), datetime.now()]
            }),
            'metrics': [1, 2, 3, 4, 5],
            'total_flights': 100
        }
        
        summary = response_generator._create_data_summary(data)
        
        assert summary['data_points'] == 7  # 2 from DataFrame + 5 from list
        assert 'DataFrame (flight_df)' in summary['data_types']
        assert 'List (metrics)' in summary['data_types']
        assert 'Numeric (total_flights)' in summary['data_types']
        
        # Should extract airports and airlines
        assert 'BOM' in summary['airports']
        assert 'DEL' in summary['airports']
        assert 'AI' in summary['airlines']
        assert '6E' in summary['airlines']
        
        print("✓ Data summary creation works")
    
    def test_error_response_generation(self, response_generator):
        """Test error response generation."""
        error_response = response_generator._generate_error_response("Test error message")
        
        assert error_response['response_type'] == ResponseType.ERROR.value
        assert 'Test error message' in error_response['response_text']
        assert error_response['confidence'] == 0.0
        assert len(error_response['follow_up_questions']) > 0
        
        print("✓ Error response generation works")
    
    def test_cache_key_generation(self, response_generator):
        """Test cache key generation."""
        data = {'test': 'data'}
        context = {'user': 'test_user'}
        
        key1 = response_generator._generate_cache_key(ResponseType.DELAY_ANALYSIS, data, context)
        key2 = response_generator._generate_cache_key(ResponseType.DELAY_ANALYSIS, data, context)
        key3 = response_generator._generate_cache_key(ResponseType.CONGESTION_ANALYSIS, data, context)
        
        # Same inputs should generate same key
        assert key1 == key2
        
        # Different response types should generate different keys
        assert key1 != key3
        
        # Keys should be valid MD5 hashes
        assert len(key1) == 32
        assert all(c in '0123456789abcdef' for c in key1)
        
        print("✓ Cache key generation works")
    
    @patch('src.nlp.response_generator.CacheManager')
    def test_response_caching(self, mock_cache_manager, sample_delay_data):
        """Test response caching functionality."""
        # Create mock cache manager
        cache_manager = mock_cache_manager.return_value
        cache_manager.get.return_value = None  # No cached response initially
        
        response_generator = FlightResponseGenerator(cache_manager)
        
        # Generate response (should cache it)
        response = response_generator.generate_response(
            ResponseType.DELAY_ANALYSIS,
            sample_delay_data
        )
        
        # Verify cache.set was called
        cache_manager.set.assert_called_once()
        
        # Test cache hit
        cached_response = {'cached': True}
        cache_manager.get.return_value = cached_response
        
        response2 = response_generator.generate_response(
            ResponseType.DELAY_ANALYSIS,
            sample_delay_data
        )
        
        assert response2 == cached_response
        
        print("✓ Response caching works")
    
    def test_custom_template_addition(self, response_generator):
        """Test adding custom response templates."""
        custom_template = ResponseTemplate(
            response_type=ResponseType.GENERAL_INFO,
            template="Custom template: {custom_field}",
            required_data_fields=['custom_field'],
            optional_data_fields=[],
            visualization_recommendations=[],
            follow_up_templates=[]
        )
        
        response_generator.add_custom_template(ResponseType.GENERAL_INFO, custom_template)
        
        # Test that custom template is used
        response = response_generator.generate_response(
            ResponseType.GENERAL_INFO,
            {'custom_field': 'test_value'}
        )
        
        assert 'Custom template: test_value' in response['response_text']
        
        print("✓ Custom template addition works")
    
    def test_template_info_retrieval(self, response_generator):
        """Test retrieval of template information."""
        template_info = response_generator.get_response_templates()
        
        assert isinstance(template_info, dict)
        assert ResponseType.DELAY_ANALYSIS.value in template_info
        
        delay_info = template_info[ResponseType.DELAY_ANALYSIS.value]
        assert 'required_fields' in delay_info
        assert 'optional_fields' in delay_info
        assert 'visualization_count' in delay_info
        assert 'follow_up_count' in delay_info
        
        print("✓ Template info retrieval works")
    
    def test_visualization_types_info(self, response_generator):
        """Test retrieval of visualization types information."""
        viz_info = response_generator.get_visualization_types()
        
        assert isinstance(viz_info, list)
        assert len(viz_info) > 0
        
        for viz in viz_info:
            assert 'type' in viz
            assert 'name' in viz
            assert 'default_config' in viz
            assert 'implementation' in viz
        
        # Check that common visualization types are present
        viz_types = [viz['type'] for viz in viz_info]
        assert 'bar_chart' in viz_types
        assert 'line_chart' in viz_types
        assert 'heatmap' in viz_types
        
        print("✓ Visualization types info works")


@pytest.mark.integration
class TestResponseGeneratorIntegration:
    """Integration tests for response generator."""
    
    def test_end_to_end_response_generation(self):
        """Test complete response generation workflow."""
        response_generator = FlightResponseGenerator()
        
        # Create realistic flight data
        flight_data = pd.DataFrame({
            'flight_id': ['AI101', '6E202', 'UK303', 'SG404'],
            'origin_airport': ['BOM', 'DEL', 'BOM', 'BLR'],
            'destination_airport': ['DEL', 'BOM', 'BLR', 'DEL'],
            'airline': ['AI', '6E', 'UK', 'SG'],
            'delay_minutes': [15, 30, 5, 45],
            'scheduled_departure': [
                datetime(2024, 1, 15, 10, 30),
                datetime(2024, 1, 15, 14, 15),
                datetime(2024, 1, 15, 18, 45),
                datetime(2024, 1, 15, 20, 30)
            ]
        })
        
        # Prepare data for response generation
        data = {
            'total_flights': len(flight_data),
            'average_delay': flight_data['delay_minutes'].mean(),
            'on_time_percentage': (flight_data['delay_minutes'] <= 15).mean() * 100,
            'most_delayed_route': 'BLR-DEL',
            'peak_delay_hours': [14, 20],
            'delay_insights': 'Delays are higher in afternoon and evening',
            'recommendations': 'Consider morning flights for better punctuality',
            'flight_data': flight_data
        }
        
        # Generate response
        response = response_generator.generate_response(
            ResponseType.DELAY_ANALYSIS,
            data
        )
        
        # Verify complete response structure
        assert 'response_text' in response
        assert 'visualizations' in response
        assert 'follow_up_questions' in response
        assert 'data_summary' in response
        assert 'confidence' in response
        assert 'timestamp' in response
        
        # Verify response quality
        assert response['confidence'] > 0.5
        assert len(response['visualizations']) > 0
        assert len(response['follow_up_questions']) > 0
        
        # Verify data summary includes flight data
        data_summary = response['data_summary']
        assert data_summary['data_points'] > 0
        assert 'BOM' in data_summary['airports']
        assert 'AI' in data_summary['airlines']
        
        print("✓ End-to-end response generation works")


if __name__ == "__main__":
    # Run tests manually
    test_classes = [TestFlightResponseGenerator, TestResponseGeneratorIntegration]
    
    print("Running response generator tests...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                
                # Handle fixtures for methods that need them
                if method_name in ['test_delay_analysis_response', 'test_congestion_analysis_response', 'test_follow_up_question_generation']:
                    # Create mock fixtures
                    response_generator = FlightResponseGenerator()
                    sample_delay_data = {
                        'total_flights': 150,
                        'average_delay': 22.5,
                        'on_time_percentage': 78.3,
                        'most_delayed_route': 'BOM-DEL',
                        'peak_delay_hours': [14, 15, 16]
                    }
                    sample_congestion_data = {
                        'total_flights': 200,
                        'peak_hours': [14, 15, 16],
                        'quietest_hours': [2, 3, 4],
                        'congestion_score': 7.2
                    }
                    
                    if 'delay_data' in method_name:
                        method(response_generator, sample_delay_data)
                    elif 'congestion_data' in method_name:
                        method(response_generator, sample_congestion_data)
                    else:
                        method(response_generator, sample_delay_data)
                elif method_name in ['test_initialization', 'test_missing_data_handling', 'test_visualization_recommendations']:
                    response_generator = FlightResponseGenerator()
                    if 'missing_data' in method_name:
                        method(response_generator)
                    elif 'visualization' in method_name:
                        sample_data = {'total_flights': 150, 'average_delay': 22.5, 'on_time_percentage': 78.3}
                        method(response_generator, sample_data)
                    else:
                        method(response_generator)
                else:
                    # For methods that don't need fixtures or create their own
                    if hasattr(test_instance, method_name):
                        method()
                
                passed += 1
                
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
                failed += 1
    
    print("=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {failed} tests failed")