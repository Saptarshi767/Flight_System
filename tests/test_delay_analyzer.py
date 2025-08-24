"""
Unit tests for the Delay Analysis Engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.analysis.delay_analyzer import DelayAnalyzer
from src.data.models import FlightData, DelayCategory, AnalysisResult


class TestDelayAnalyzer:
    """Test cases for DelayAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a DelayAnalyzer instance for testing."""
        return DelayAnalyzer()
    
    @pytest.fixture
    def sample_flight_data(self):
        """Create sample flight data for testing."""
        base_time = datetime(2024, 1, 15, 10, 0)
        flights = []
        
        # Create flights with various delay patterns
        for i in range(50):
            scheduled_dep = base_time + timedelta(hours=i % 24, minutes=i * 10)
            actual_dep = scheduled_dep + timedelta(minutes=np.random.randint(-5, 60))
            scheduled_arr = scheduled_dep + timedelta(hours=2)
            actual_arr = actual_dep + timedelta(hours=2, minutes=np.random.randint(-10, 30))
            
            flight = FlightData(
                flight_id=f"FL{i:03d}",
                airline=f"AI{i % 5}",
                flight_number=f"AI{100 + i}",
                aircraft_type="B737" if i % 2 == 0 else "A320",
                origin_airport="BOM" if i % 3 == 0 else "DEL",
                destination_airport="DEL" if i % 3 == 0 else "BOM",
                scheduled_departure=scheduled_dep,
                actual_departure=actual_dep,
                scheduled_arrival=scheduled_arr,
                actual_arrival=actual_arr,
                delay_category=list(DelayCategory)[i % len(DelayCategory)],
                passenger_count=150 + (i % 50)
            )
            flights.append(flight)
        
        return flights
    
    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert not analyzer.is_trained
        assert analyzer.delay_predictor is not None
        assert analyzer.anomaly_detector is not None
        assert analyzer.time_clusterer is not None
        assert analyzer.scaler is not None
    
    def test_flights_to_dataframe(self, analyzer, sample_flight_data):
        """Test conversion of flight data to DataFrame."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_flight_data)
        assert 'flight_id' in df.columns
        assert 'delay_minutes' in df.columns
        assert 'departure_delay_minutes' in df.columns
        assert 'arrival_delay_minutes' in df.columns
        
        # Check that delays are calculated correctly
        assert df['delay_minutes'].notna().all()
        assert df['departure_delay_minutes'].notna().all()
    
    def test_calculate_delay_metrics(self, analyzer, sample_flight_data):
        """Test delay metrics calculation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        metrics = analyzer._calculate_delay_metrics(df)
        
        assert 'average_delay_minutes' in metrics
        assert 'median_delay_minutes' in metrics
        assert 'std_delay_minutes' in metrics
        assert 'on_time_percentage' in metrics
        assert 'severe_delay_percentage' in metrics
        assert 'total_flights' in metrics
        
        assert metrics['total_flights'] == len(sample_flight_data)
        assert 0 <= metrics['on_time_percentage'] <= 100
        assert 0 <= metrics['severe_delay_percentage'] <= 100
    
    def test_identify_optimal_time_slots(self, analyzer, sample_flight_data):
        """Test optimal time slot identification."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        optimal_times = analyzer._identify_optimal_time_slots(df)
        
        assert 'optimal_hours' in optimal_times
        assert 'worst_hours' in optimal_times
        assert 'hourly_delay_stats' in optimal_times
        
        assert isinstance(optimal_times['optimal_hours'], list)
        assert isinstance(optimal_times['worst_hours'], list)
        assert isinstance(optimal_times['hourly_delay_stats'], list)
    
    def test_categorize_delays(self, analyzer, sample_flight_data):
        """Test delay categorization."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        categories = analyzer._categorize_delays(df)
        
        assert 'category_counts' in categories
        assert 'category_statistics' in categories
        
        # Check that all delay categories are represented
        for category in DelayCategory:
            assert category.value in categories['category_statistics']
    
    def test_analyze_time_of_day_patterns(self, analyzer, sample_flight_data):
        """Test time of day pattern analysis."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        time_analysis = analyzer._analyze_time_of_day_patterns(df)
        
        assert 'period_statistics' in time_analysis
        assert 'best_period' in time_analysis
        assert 'worst_period' in time_analysis
        
        # Check that all time periods are represented
        expected_periods = ['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
        for period in expected_periods:
            if period in time_analysis['period_statistics']:
                stats = time_analysis['period_statistics'][period]
                assert 'mean' in stats
                assert 'count' in stats
    
    def test_extract_features(self, analyzer, sample_flight_data):
        """Test feature extraction for ML models."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        features = analyzer._extract_features(df)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_flight_data)
        assert features.shape[1] == 6  # Expected number of features
        
        # Check feature ranges
        assert np.all(features[:, 0] >= 0) and np.all(features[:, 0] <= 23)  # Hour
        assert np.all(features[:, 1] >= 0) and np.all(features[:, 1] <= 6)   # Weekday
        assert np.all(features[:, 2] >= 1) and np.all(features[:, 2] <= 12)  # Month
    
    def test_train_models(self, analyzer, sample_flight_data):
        """Test model training."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        
        # Train models
        analyzer._train_models(df)
        
        assert analyzer.is_trained
        
        # Test that models can make predictions
        features = analyzer._extract_features(df)
        if len(features) > 0:
            scaled_features = analyzer.scaler.transform(features[:1])
            prediction = analyzer.delay_predictor.predict(scaled_features)
            assert len(prediction) == 1
            assert isinstance(prediction[0], (int, float))
    
    def test_predict_delay(self, analyzer, sample_flight_data):
        """Test delay prediction functionality."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._train_models(df)
        
        if analyzer.is_trained:
            flight_features = {
                'scheduled_departure': datetime(2024, 1, 15, 14, 30),
                'flight_number': 'AI101',
                'aircraft_type': 'B737',
                'passenger_count': 180
            }
            
            predicted_delay, confidence = analyzer.predict_delay(flight_features)
            
            assert isinstance(predicted_delay, (int, float))
            assert isinstance(confidence, (int, float))
            assert 0 <= confidence <= 1
    
    def test_find_best_departure_times(self, analyzer, sample_flight_data):
        """Test finding best departure times."""
        best_times = analyzer.find_best_departure_times("BOM", sample_flight_data)
        
        assert isinstance(best_times, list)
        assert len(best_times) <= 12  # Should return top 12 hours
        
        if best_times:
            # Check that results are sorted by delay score
            for i in range(1, len(best_times)):
                assert best_times[i-1]['delay_score'] <= best_times[i]['delay_score']
            
            # Check required fields
            for time_slot in best_times:
                assert 'departure_hour' in time_slot
                assert 'avg_delay' in time_slot
                assert 'flight_count' in time_slot
                assert 'recommendation' in time_slot
    
    def test_analyze_delays_full_workflow(self, analyzer, sample_flight_data):
        """Test the complete delay analysis workflow."""
        result = analyzer.analyze_delays(sample_flight_data, "BOM")
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "delay_analysis"
        assert result.airport_code == "BOM"
        assert result.metrics is not None
        assert result.recommendations is not None
        assert isinstance(result.confidence_score, (int, float))
        assert 0 <= result.confidence_score <= 1
        
        # Check metrics structure
        metrics = result.metrics
        assert 'delay_metrics' in metrics
        assert 'optimal_time_slots' in metrics
        assert 'delay_categories' in metrics
        assert 'time_of_day_analysis' in metrics
        assert 'total_flights_analyzed' in metrics
    
    def test_analyze_delays_empty_data(self, analyzer):
        """Test delay analysis with empty data."""
        result = analyzer.analyze_delays([], "BOM")
        
        assert isinstance(result, AnalysisResult)
        assert result.confidence_score == 0.0
        assert 'error' in result.metrics
    
    def test_analyze_delays_no_airport_data(self, analyzer, sample_flight_data):
        """Test delay analysis with no data for specific airport."""
        result = analyzer.analyze_delays(sample_flight_data, "XYZ")
        
        assert isinstance(result, AnalysisResult)
        assert result.airport_code == "XYZ"
        assert result.confidence_score == 0.0
    
    def test_generate_delay_recommendations(self, analyzer):
        """Test recommendation generation."""
        delay_metrics = {
            'average_delay_minutes': 45.0,
            'on_time_percentage': 60.0
        }
        optimal_times = {
            'optimal_hours': [6, 7, 8],
            'worst_hours': [18, 19, 20]
        }
        delay_categories = {
            'category_statistics': {
                'weather': {'percentage': 35.0},
                'operational': {'percentage': 30.0}
            }
        }
        time_analysis = {
            'best_period': 'Morning (6-12)'
        }
        
        recommendations = analyzer._generate_delay_recommendations(
            delay_metrics, optimal_times, delay_categories, time_analysis
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check that recommendations contain expected content
        rec_text = ' '.join(recommendations)
        assert 'delay' in rec_text.lower()
    
    def test_confidence_score_calculation(self, analyzer, sample_flight_data):
        """Test confidence score calculation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        confidence = analyzer._calculate_confidence_score(df)
        
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1
    
    def test_time_slot_recommendation(self, analyzer):
        """Test time slot recommendation generation."""
        # Test excellent time slot
        excellent_slot = {'avg_delay': 5, 'flight_count': 10}
        rec = analyzer._get_time_slot_recommendation(excellent_slot)
        assert 'Excellent' in rec
        
        # Test poor time slot
        poor_slot = {'avg_delay': 50, 'flight_count': 15}
        rec = analyzer._get_time_slot_recommendation(poor_slot)
        assert 'Poor' in rec
    
    def test_extract_features_for_prediction(self, analyzer):
        """Test feature extraction for single flight prediction."""
        flight_features = {
            'scheduled_departure': datetime(2024, 1, 15, 14, 30),
            'flight_number': 'AI101',
            'aircraft_type': 'B737',
            'passenger_count': 180
        }
        
        features = analyzer._extract_features_for_prediction(flight_features)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 6
        assert features[0] == 14  # Hour
        assert features[1] == 0   # Monday (weekday)
        assert features[2] == 1   # January
    
    @pytest.mark.parametrize("delay_minutes,expected_category", [
        (5, "Excellent"),
        (15, "Good"), 
        (30, "Fair"),
        (60, "Poor")
    ])
    def test_time_slot_recommendations_parametrized(self, analyzer, delay_minutes, expected_category):
        """Test time slot recommendations with different delay values."""
        time_slot = {'avg_delay': delay_minutes, 'flight_count': 10}
        recommendation = analyzer._get_time_slot_recommendation(time_slot)
        assert expected_category in recommendation


if __name__ == "__main__":
    pytest.main([__file__])