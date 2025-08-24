"""
Unit tests for the Congestion Analysis Engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.analysis.congestion_analyzer import CongestionAnalyzer
from src.data.models import FlightData, AirportData, AnalysisResult


class TestCongestionAnalyzer:
    """Test cases for CongestionAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a CongestionAnalyzer instance for testing."""
        return CongestionAnalyzer(runway_capacity=60)
    
    @pytest.fixture
    def airport_data(self):
        """Create sample airport data for testing."""
        return AirportData(
            airport_code="BOM",
            airport_name="Mumbai Airport",
            city="Mumbai",
            country="India",
            runway_capacity=80,
            active_runways=["09L", "09R", "27L", "27R"],
            peak_hours=[8, 9, 10, 18, 19, 20]
        )
    
    @pytest.fixture
    def sample_flight_data(self):
        """Create sample flight data with congestion patterns."""
        base_time = datetime(2024, 1, 15, 6, 0)
        flights = []
        
        # Create flights with congestion patterns (more flights during peak hours)
        for day in range(7):  # One week of data
            for hour in range(24):
                # Simulate peak hours (8-10, 18-20) with more flights
                if hour in [8, 9, 10, 18, 19, 20]:
                    num_flights = np.random.randint(8, 15)  # Peak hours
                elif hour in [0, 1, 2, 3, 4, 5]:
                    num_flights = np.random.randint(1, 4)   # Night hours
                else:
                    num_flights = np.random.randint(3, 8)   # Regular hours
                
                for flight_num in range(num_flights):
                    flight_time = base_time + timedelta(days=day, hours=hour, minutes=flight_num * 5)
                    
                    flight = FlightData(
                        flight_id=f"FL{day:02d}{hour:02d}{flight_num:02d}",
                        airline=f"AI{flight_num % 3}",
                        flight_number=f"AI{1000 + day * 100 + hour * 10 + flight_num}",
                        aircraft_type="B737" if flight_num % 2 == 0 else "A320",
                        origin_airport="BOM" if flight_num % 2 == 0 else "DEL",
                        destination_airport="DEL" if flight_num % 2 == 0 else "BOM",
                        scheduled_departure=flight_time,
                        scheduled_arrival=flight_time + timedelta(hours=2),
                        passenger_count=150 + (flight_num % 50)
                    )
                    flights.append(flight)
        
        return flights
    
    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.runway_capacity == 60
        assert not analyzer.is_trained
        assert analyzer.prophet_model is None
        assert analyzer.scaler is not None
        assert analyzer.clusterer is not None
    
    def test_analyzer_with_airport_data(self, airport_data):
        """Test analyzer initialization with airport data."""
        analyzer = CongestionAnalyzer()
        
        # Test that runway capacity is updated from airport data
        result = analyzer.analyze_congestion([], "BOM", airport_data)
        assert analyzer.runway_capacity == airport_data.runway_capacity
    
    def test_flights_to_dataframe(self, analyzer, sample_flight_data):
        """Test conversion of flight data to DataFrame."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_flight_data)
        assert 'flight_id' in df.columns
        assert 'scheduled_departure' in df.columns
        assert 'origin_airport' in df.columns
        assert 'destination_airport' in df.columns
    
    def test_calculate_flight_density(self, analyzer, sample_flight_data):
        """Test flight density calculation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        density_metrics = analyzer._calculate_flight_density(df)
        
        assert 'average_flights_per_hour' in density_metrics
        assert 'max_flights_per_hour' in density_metrics
        assert 'min_flights_per_hour' in density_metrics
        assert 'runway_capacity' in density_metrics
        assert 'capacity_utilization' in density_metrics
        assert 'peak_utilization' in density_metrics
        
        assert density_metrics['runway_capacity'] == analyzer.runway_capacity
        assert 0 <= density_metrics['capacity_utilization'] <= 200  # Allow over-capacity
        assert density_metrics['max_flights_per_hour'] >= density_metrics['min_flights_per_hour']
    
    def test_identify_peak_hours(self, analyzer, sample_flight_data):
        """Test peak hours identification."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        peak_hours = analyzer._identify_peak_hours(df)
        
        assert 'peak_hours' in peak_hours
        assert 'off_peak_hours' in peak_hours
        assert 'busiest_hour' in peak_hours
        assert 'quietest_hour' in peak_hours
        assert 'hourly_flight_counts' in peak_hours
        
        assert isinstance(peak_hours['peak_hours'], list)
        assert isinstance(peak_hours['off_peak_hours'], list)
        assert 0 <= peak_hours['busiest_hour'] <= 23
        assert 0 <= peak_hours['quietest_hour'] <= 23
    
    def test_model_runway_capacity(self, analyzer, sample_flight_data):
        """Test runway capacity modeling."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        capacity_analysis = analyzer._model_runway_capacity(df)
        
        assert 'runway_capacity' in capacity_analysis
        assert 'over_capacity_instances' in capacity_analysis
        assert 'over_capacity_percentage' in capacity_analysis
        assert 'average_utilization' in capacity_analysis
        assert 'bottleneck_hours' in capacity_analysis
        
        assert capacity_analysis['runway_capacity'] == analyzer.runway_capacity
        assert 0 <= capacity_analysis['over_capacity_percentage'] <= 100
        assert isinstance(capacity_analysis['bottleneck_hours'], list)
    
    def test_calculate_congestion_scores(self, analyzer, sample_flight_data):
        """Test congestion scoring calculation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        congestion_scores = analyzer._calculate_congestion_scores(df)
        
        assert 'hourly_scores' in congestion_scores
        assert 'most_congested_hour' in congestion_scores
        assert 'least_congested_hour' in congestion_scores
        assert 'average_congestion_score' in congestion_scores
        
        assert isinstance(congestion_scores['hourly_scores'], list)
        assert 0 <= congestion_scores['most_congested_hour'] <= 23
        assert 0 <= congestion_scores['least_congested_hour'] <= 23
        assert 0 <= congestion_scores['average_congestion_score'] <= 1
        
        # Check hourly scores structure
        for score_data in congestion_scores['hourly_scores']:
            assert 'hour' in score_data
            assert 'congestion_score' in score_data
            assert 'congestion_level' in score_data
            assert 0 <= score_data['congestion_score'] <= 1
    
    def test_train_prophet_model(self, analyzer, sample_flight_data):
        """Test Prophet model training."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        forecast_data = analyzer._train_prophet_model(df)
        
        assert 'model_trained' in forecast_data
        assert 'forecast_available' in forecast_data
        
        if forecast_data['model_trained']:
            assert analyzer.is_trained
            assert analyzer.prophet_model is not None
            assert 'training_data_points' in forecast_data
            assert 'model_performance' in forecast_data
    
    def test_recommend_alternative_slots(self, analyzer, sample_flight_data):
        """Test alternative slot recommendations."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        congestion_scores = analyzer._calculate_congestion_scores(df)
        alternatives = analyzer._recommend_alternative_slots(df, congestion_scores)
        
        assert isinstance(alternatives, list)
        
        for alt in alternatives:
            assert 'congested_hour' in alt
            assert 'alternative_hour' in alt
            assert 'congested_score' in alt
            assert 'alternative_score' in alt
            assert 'improvement' in alt
            assert alt['improvement'] > 0  # Should be an improvement
    
    def test_predict_congestion(self, analyzer, sample_flight_data):
        """Test congestion prediction functionality."""
        target_time = datetime(2024, 1, 20, 14, 30)
        score, level = analyzer.predict_congestion("BOM", target_time, sample_flight_data)
        
        assert isinstance(score, (int, float))
        assert isinstance(level, str)
        assert 0 <= score <= 1
        assert level in ["Low", "Moderate", "High", "Critical", "Unknown"]
    
    def test_find_least_congested_slots(self, analyzer, sample_flight_data):
        """Test finding least congested time slots."""
        least_congested = analyzer.find_least_congested_slots("BOM", sample_flight_data)
        
        assert isinstance(least_congested, list)
        assert len(least_congested) <= 12  # Should return top 12 hours
        
        if least_congested:
            # Check that results are sorted by congestion score (least congested first)
            for i in range(1, len(least_congested)):
                assert least_congested[i-1]['congestion_score'] <= least_congested[i]['congestion_score']
            
            # Check required fields
            for slot in least_congested:
                assert 'hour' in slot
                assert 'congestion_score' in slot
                assert 'congestion_level' in slot
                assert 'recommendation' in slot
    
    def test_analyze_congestion_full_workflow(self, analyzer, sample_flight_data, airport_data):
        """Test the complete congestion analysis workflow."""
        result = analyzer.analyze_congestion(sample_flight_data, "BOM", airport_data)
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "congestion_analysis"
        assert result.airport_code == "BOM"
        assert result.metrics is not None
        assert result.recommendations is not None
        assert isinstance(result.confidence_score, (int, float))
        assert 0 <= result.confidence_score <= 1
        
        # Check metrics structure
        metrics = result.metrics
        assert 'flight_density_metrics' in metrics
        assert 'peak_hours_analysis' in metrics
        assert 'runway_capacity_analysis' in metrics
        assert 'congestion_scores' in metrics
        assert 'forecast_data' in metrics
        assert 'alternative_slots' in metrics
        assert 'total_flights_analyzed' in metrics
    
    def test_analyze_congestion_empty_data(self, analyzer):
        """Test congestion analysis with empty data."""
        result = analyzer.analyze_congestion([], "BOM")
        
        assert isinstance(result, AnalysisResult)
        assert result.confidence_score == 0.0
        assert 'error' in result.metrics
    
    def test_analyze_congestion_no_airport_data(self, analyzer, sample_flight_data):
        """Test congestion analysis with no data for specific airport."""
        result = analyzer.analyze_congestion(sample_flight_data, "XYZ")
        
        assert isinstance(result, AnalysisResult)
        assert result.airport_code == "XYZ"
        assert result.confidence_score == 0.0
    
    def test_generate_congestion_recommendations(self, analyzer):
        """Test recommendation generation."""
        density_metrics = {
            'capacity_utilization': 85.0,
            'peak_utilization': 120.0
        }
        peak_hours = {
            'peak_hours': [8, 9, 10],
            'off_peak_hours': [2, 3, 4]
        }
        capacity_analysis = {
            'over_capacity_percentage': 15.0,
            'bottleneck_hours': [8, 9, 19]
        }
        congestion_scores = {
            'average_congestion_score': 0.7
        }
        alternatives = [
            {'improvement': 0.3},
            {'improvement': 0.2}
        ]
        
        recommendations = analyzer._generate_congestion_recommendations(
            density_metrics, peak_hours, capacity_analysis, congestion_scores, alternatives
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check that recommendations contain expected content
        rec_text = ' '.join(recommendations)
        assert 'congestion' in rec_text.lower() or 'capacity' in rec_text.lower()
    
    def test_get_congestion_level(self, analyzer):
        """Test congestion level classification."""
        assert analyzer._get_congestion_level(0.1) == "Low"
        assert analyzer._get_congestion_level(0.4) == "Moderate"
        assert analyzer._get_congestion_level(0.7) == "High"
        assert analyzer._get_congestion_level(0.9) == "Critical"
    
    def test_get_slot_recommendation(self, analyzer):
        """Test slot recommendation generation."""
        # Test low congestion slot
        low_slot = {'congestion_score': 0.2, 'congestion_level': 'Low'}
        rec = analyzer._get_slot_recommendation(low_slot)
        assert 'Excellent' in rec
        
        # Test critical congestion slot
        critical_slot = {'congestion_score': 0.9, 'congestion_level': 'Critical'}
        rec = analyzer._get_slot_recommendation(critical_slot)
        assert 'Avoid' in rec
    
    def test_confidence_score_calculation(self, analyzer, sample_flight_data):
        """Test confidence score calculation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        confidence = analyzer._calculate_confidence_score(df)
        
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1
    
    @pytest.mark.parametrize("score,expected_level", [
        (0.1, "Low"),
        (0.4, "Moderate"),
        (0.7, "High"),
        (0.9, "Critical")
    ])
    def test_congestion_levels_parametrized(self, analyzer, score, expected_level):
        """Test congestion level classification with different scores."""
        level = analyzer._get_congestion_level(score)
        assert level == expected_level
    
    def test_prophet_model_with_insufficient_data(self, analyzer):
        """Test Prophet model training with insufficient data."""
        # Create minimal flight data
        minimal_flights = [
            FlightData(
                flight_id="FL001",
                airline="AI",
                flight_number="AI101",
                origin_airport="BOM",
                destination_airport="DEL",
                scheduled_departure=datetime(2024, 1, 15, 10, 0),
                scheduled_arrival=datetime(2024, 1, 15, 12, 0)
            )
        ]
        
        df = analyzer._flights_to_dataframe(minimal_flights)
        forecast_data = analyzer._train_prophet_model(df)
        
        assert not forecast_data['model_trained']
        assert not forecast_data['forecast_available']
    
    def test_runway_capacity_update(self, analyzer, airport_data):
        """Test that runway capacity is updated from airport data."""
        initial_capacity = analyzer.runway_capacity
        
        # Analyze with airport data
        analyzer.analyze_congestion([], "BOM", airport_data)
        
        assert analyzer.runway_capacity == airport_data.runway_capacity
        assert analyzer.runway_capacity != initial_capacity


if __name__ == "__main__":
    pytest.main([__file__])