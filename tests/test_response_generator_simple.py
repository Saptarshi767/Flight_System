"""
Simple tests for response generator functionality without spaCy dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Test the response generator directly
def test_response_generator_basic():
    """Test basic response generator functionality."""
    try:
        # Import only the response generator module
        from src.nlp.response_generator import (
            ResponseType, VisualizationType, FlightResponseGenerator
        )
        
        # Test enum values
        assert ResponseType.DELAY_ANALYSIS.value == "delay_analysis"
        assert ResponseType.CONGESTION_ANALYSIS.value == "congestion_analysis"
        assert VisualizationType.BAR_CHART.value == "bar_chart"
        assert VisualizationType.HEATMAP.value == "heatmap"
        
        print("✓ Response generator enums work correctly")
        
        # Test initialization
        generator = FlightResponseGenerator()
        assert generator is not None
        assert len(generator.templates) > 0
        assert len(generator.visualization_configs) > 0
        
        print("✓ Response generator initialization works")
        
        # Test template info retrieval
        template_info = generator.get_response_templates()
        assert isinstance(template_info, dict)
        assert ResponseType.DELAY_ANALYSIS.value in template_info
        
        print("✓ Template info retrieval works")
        
        # Test visualization info retrieval
        viz_info = generator.get_visualization_types()
        assert isinstance(viz_info, list)
        assert len(viz_info) > 0
        
        print("✓ Visualization info retrieval works")
        
        # Test data formatting
        test_data = {
            'peak_hours': [14, 15, 16, 17, 18],
            'short_list': ['A', 'B'],
            'empty_list': [],
            'float_value': 123.456789,
            'dict_value': {'key1': 'value1', 'key2': 'value2'}
        }
        
        formatted = generator._format_data_for_template(test_data)
        assert 'and 2 more' in formatted['peak_hours']
        assert formatted['short_list'] == 'A, B'
        assert formatted['empty_list'] == 'None'
        assert formatted['float_value'] == '123.46'
        
        print("✓ Data formatting works")
        
        # Test time conversion
        assert generator._hour_to_12h(0) == "12 AM"
        assert generator._hour_to_12h(12) == "12 PM"
        assert generator._hour_to_12h(18) == "6 PM"
        
        print("✓ Time conversion works")
        
        # Test cache key generation
        data = {'test': 'data'}
        context = {'user': 'test_user'}
        
        key1 = generator._generate_cache_key(ResponseType.DELAY_ANALYSIS, data, context)
        key2 = generator._generate_cache_key(ResponseType.DELAY_ANALYSIS, data, context)
        
        assert key1 == key2  # Same inputs should generate same key
        assert len(key1) == 32  # MD5 hash length
        
        print("✓ Cache key generation works")
        
        # Test response generation with sample data
        sample_data = {
            'total_flights': 150,
            'average_delay': 22.5,
            'on_time_percentage': 78.3,
            'most_delayed_route': 'BOM-DEL',
            'peak_delay_hours': [14, 15, 16],
            'delay_insights': 'Peak delays occur during afternoon hours',
            'recommendations': 'Consider scheduling flights outside 2-4 PM window'
        }
        
        response = generator.generate_response(
            ResponseType.DELAY_ANALYSIS,
            sample_data
        )
        
        assert response['response_type'] == ResponseType.DELAY_ANALYSIS.value
        assert 'response_text' in response
        assert 'visualizations' in response
        assert 'follow_up_questions' in response
        assert 'confidence' in response
        
        # Check that data is included in response
        response_text = response['response_text']
        assert '150' in response_text
        assert '22.5' in response_text
        assert '78.3' in response_text
        
        print("✓ Response generation works")
        
        # Test error response
        error_response = generator._generate_error_response("Test error")
        assert error_response['response_type'] == ResponseType.ERROR.value
        assert 'Test error' in error_response['response_text']
        assert error_response['confidence'] == 0.0
        
        print("✓ Error response generation works")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False


def test_response_templates():
    """Test response template functionality."""
    try:
        from src.nlp.response_generator import FlightResponseGenerator, ResponseType
        
        generator = FlightResponseGenerator()
        
        # Test all response types have templates
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
            assert response_type in generator.templates
            template = generator.templates[response_type]
            assert hasattr(template, 'template')
            assert hasattr(template, 'required_data_fields')
            assert hasattr(template, 'visualization_recommendations')
            assert hasattr(template, 'follow_up_templates')
        
        print("✓ All response templates are properly defined")
        
        # Test template structure for delay analysis
        delay_template = generator.templates[ResponseType.DELAY_ANALYSIS]
        assert 'total_flights' in delay_template.required_data_fields
        assert 'average_delay' in delay_template.required_data_fields
        assert 'on_time_percentage' in delay_template.required_data_fields
        assert len(delay_template.visualization_recommendations) > 0
        assert len(delay_template.follow_up_templates) > 0
        
        print("✓ Template structure is correct")
        
        return True
        
    except Exception as e:
        print(f"Template test failed: {e}")
        return False


def test_visualization_recommendations():
    """Test visualization recommendation functionality."""
    try:
        from src.nlp.response_generator import FlightResponseGenerator, ResponseType, VisualizationType
        
        generator = FlightResponseGenerator()
        
        # Test visualization config initialization
        assert VisualizationType.BAR_CHART in generator.visualization_configs
        assert VisualizationType.LINE_CHART in generator.visualization_configs
        assert VisualizationType.HEATMAP in generator.visualization_configs
        
        # Test visualization recommendation generation
        sample_data = {
            'total_flights': 100,
            'average_delay': 15.0,
            'on_time_percentage': 85.0
        }
        
        response = generator.generate_response(ResponseType.DELAY_ANALYSIS, sample_data)
        visualizations = response['visualizations']
        
        assert len(visualizations) > 0
        
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
        
        print("✓ Visualization recommendations work correctly")
        
        return True
        
    except Exception as e:
        print(f"Visualization test failed: {e}")
        return False


if __name__ == "__main__":
    print("Running simple response generator tests...")
    print("=" * 50)
    
    tests = [
        test_response_generator_basic,
        test_response_templates,
        test_visualization_recommendations
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {failed} tests failed")