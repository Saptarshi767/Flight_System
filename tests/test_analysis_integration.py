"""
Integration tests for all analysis engines working together
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.analysis import (
    DelayAnalyzer, CongestionAnalyzer, ScheduleImpactAnalyzer, 
    CascadingImpactAnalyzer, ScheduleChange
)
from src.data.models import FlightData, AirportData


class TestAnalysisIntegration:
    """Integration tests for all analysis engines."""
    
    @pytest.fixture
    def comprehensive_flight_data(self):
        """Create comprehensive flight data for integration testing."""
        base_time = datetime(2024, 1, 15, 6, 0)
        flights = []
        
        # Create a realistic flight network with delays and connections
        for day in range(3):  # 3 days of data
            for hour in range(18):  # 18 hours of operations (6 AM to 12 AM)
                # Vary number of flights by hour (peak hours have more flights)
                if 8 <= hour <= 10 or 18 <= hour <= 20:  # Peak hours
                    num_flights = np.random.randint(4, 8)
                elif 0 <= hour <= 5 or 22 <= hour <= 23:  # Off-peak hours
                    num_flights = np.random.randint(1, 3)
                else:  # Regular hours
                    num_flights = np.random.randint(2, 5)
                
                for flight_num in range(num_flights):
                    flight_time = base_time + timedelta(days=day, hours=hour, minutes=flight_num * 10)
                    
                    # Create realistic delays (more delays during peak hours)
                    if 8 <= hour <= 10 or 18 <= hour <= 20:
                        delay_minutes = np.random.randint(0, 90)  # Peak hour delays
                    else:
                        delay_minutes = np.random.randint(0, 30)  # Regular delays
                    
                    actual_departure = flight_time + timedelta(minutes=delay_minutes)
                    actual_arrival = actual_departure + timedelta(hours=2, minutes=np.random.randint(0, 20))
                    
                    flight = FlightData(
                        flight_id=f"FL{day:02d}{hour:02d}{flight_num:02d}",
                        airline=f"AI{flight_num % 4}",  # 4 different airlines
                        flight_number=f"AI{1000 + day * 100 + hour * 10 + flight_num}",
                        aircraft_type=f"B73{flight_num % 3}",  # 3 aircraft types
                        origin_airport="BOM" if flight_num % 2 == 0 else "DEL",
                        destination_airport="DEL" if flight_num % 2 == 0 else "BOM",
                        scheduled_departure=flight_time,
                        actual_departure=actual_departure,
                        scheduled_arrival=flight_time + timedelta(hours=2),
                        actual_arrival=actual_arrival,
                        passenger_count=120 + np.random.randint(0, 180),
                        delay_category=np.random.choice(['weather', 'operational', 'traffic', 'other'])
                    )
                    flights.append(flight)
        
        return flights
    
    @pytest.fixture
    def airport_data(self):
        """Create airport data for testing."""
        return AirportData(
            airport_code="BOM",
            airport_name="Mumbai Airport",
            city="Mumbai",
            country="India",
            runway_capacity=80,
            active_runways=["09L", "09R", "27L", "27R"],
            peak_hours=[8, 9, 10, 18, 19, 20]
        )
    
    def test_all_analyzers_integration(self, comprehensive_flight_data, airport_data):
        """Test that all analyzers work together and produce consistent results."""
        # Initialize all analyzers
        delay_analyzer = DelayAnalyzer()
        congestion_analyzer = CongestionAnalyzer()
        schedule_analyzer = ScheduleImpactAnalyzer()
        cascading_analyzer = CascadingImpactAnalyzer()
        
        airport_code = "BOM"
        
        # Run delay analysis
        delay_result = delay_analyzer.analyze_delays(comprehensive_flight_data, airport_code)
        assert delay_result.analysis_type == "delay_analysis"
        assert delay_result.airport_code == airport_code
        
        # Run congestion analysis
        congestion_result = congestion_analyzer.analyze_congestion(
            comprehensive_flight_data, airport_code, airport_data
        )
        assert congestion_result.analysis_type == "congestion_analysis"
        assert congestion_result.airport_code == airport_code
        
        # Create some schedule changes based on delay analysis insights
        optimal_hours = delay_result.metrics.get('optimal_time_slots', {}).get('optimal_hours', [])
        worst_hours = delay_result.metrics.get('optimal_time_slots', {}).get('worst_hours', [])
        
        schedule_changes = []
        if optimal_hours and worst_hours:
            # Create a schedule change moving a flight from worst hour to optimal hour
            schedule_changes.append(ScheduleChange(
                flight_id="FL000000",  # First flight
                original_departure=datetime(2024, 1, 15, worst_hours[0] if worst_hours else 10, 0),
                new_departure=datetime(2024, 1, 15, optimal_hours[0] if optimal_hours else 6, 0),
                change_reason="delay_optimization",
                priority=2
            ))
        
        # Run schedule impact analysis
        if schedule_changes:
            schedule_result = schedule_analyzer.analyze_schedule_impact(
                comprehensive_flight_data, schedule_changes, airport_code
            )
            assert schedule_result.analysis_type == "schedule_impact_analysis"
            assert schedule_result.airport_code == airport_code
        
        # Run cascading impact analysis
        cascading_result = cascading_analyzer.analyze_cascading_impact(
            comprehensive_flight_data, airport_code
        )
        assert cascading_result.analysis_type == "cascading_impact_analysis"
        assert cascading_result.airport_code == airport_code
        
        # Verify that all analyses have reasonable confidence scores
        assert 0 <= delay_result.confidence_score <= 1
        assert 0 <= congestion_result.confidence_score <= 1
        assert 0 <= cascading_result.confidence_score <= 1
        
        # Verify that all analyses have recommendations
        assert len(delay_result.recommendations) > 0
        assert len(congestion_result.recommendations) > 0
        assert len(cascading_result.recommendations) > 0
    
    def test_cross_analyzer_insights_consistency(self, comprehensive_flight_data, airport_data):
        """Test that insights from different analyzers are consistent with each other."""
        delay_analyzer = DelayAnalyzer()
        congestion_analyzer = CongestionAnalyzer()
        
        airport_code = "BOM"
        
        # Get delay analysis results
        delay_result = delay_analyzer.analyze_delays(comprehensive_flight_data, airport_code)
        delay_optimal_hours = delay_result.metrics.get('optimal_time_slots', {}).get('optimal_hours', [])
        
        # Get congestion analysis results
        congestion_result = congestion_analyzer.analyze_congestion(
            comprehensive_flight_data, airport_code, airport_data
        )
        
        # Find least congested slots
        least_congested = congestion_analyzer.find_least_congested_slots(
            airport_code, comprehensive_flight_data
        )
        
        if delay_optimal_hours and least_congested:
            # There should be some overlap between optimal delay hours and least congested hours
            # This tests that the analyses are producing logically consistent results
            least_congested_hours = [slot['hour'] for slot in least_congested[:5]]
            
            # Check if there's any overlap (not strict requirement due to different algorithms)
            overlap = set(delay_optimal_hours[:3]) & set(least_congested_hours)
            
            # At minimum, the analyses should not be completely contradictory
            # (i.e., delay optimal hours shouldn't all be in the most congested hours)
            most_congested_hours = [slot['hour'] for slot in least_congested[-3:]]
            contradiction = set(delay_optimal_hours[:3]) & set(most_congested_hours)
            
            # This is a soft check - some contradiction is acceptable due to different methodologies
            assert len(contradiction) < len(delay_optimal_hours[:3])
    
    def test_critical_flights_impact_correlation(self, comprehensive_flight_data):
        """Test that critical flights identified by cascading analyzer correlate with schedule impact."""
        cascading_analyzer = CascadingImpactAnalyzer()
        schedule_analyzer = ScheduleImpactAnalyzer()
        
        airport_code = "BOM"
        
        # Get critical flights
        critical_flights = cascading_analyzer.identify_most_critical_flights(
            comprehensive_flight_data, airport_code, top_n=5
        )
        
        if critical_flights:
            # Create schedule changes for critical flights
            schedule_changes = []
            for i, critical_flight in enumerate(critical_flights[:3]):  # Top 3 critical flights
                schedule_changes.append(ScheduleChange(
                    flight_id=critical_flight.flight_id,
                    original_departure=critical_flight.scheduled_departure,
                    new_departure=critical_flight.scheduled_departure + timedelta(minutes=30),
                    change_reason="critical_flight_adjustment",
                    priority=3
                ))
            
            # Analyze schedule impact
            if schedule_changes:
                schedule_result = schedule_analyzer.analyze_schedule_impact(
                    comprehensive_flight_data, schedule_changes, airport_code
                )
                
                # Critical flights should generally have higher impact scores
                individual_impacts = schedule_result.metrics.get('individual_change_impacts', [])
                
                if individual_impacts:
                    # At least one critical flight should have significant impact
                    max_impact = max(impact['total_impact'] for impact in individual_impacts)
                    assert max_impact > 0  # Should have some measurable impact
    
    def test_analyzer_performance_with_large_dataset(self, comprehensive_flight_data):
        """Test that all analyzers can handle the dataset efficiently."""
        delay_analyzer = DelayAnalyzer()
        congestion_analyzer = CongestionAnalyzer()
        
        airport_code = "BOM"
        
        # This is a basic performance test - should complete without timeout
        import time
        
        start_time = time.time()
        delay_result = delay_analyzer.analyze_delays(comprehensive_flight_data, airport_code)
        delay_time = time.time() - start_time
        
        start_time = time.time()
        congestion_result = congestion_analyzer.analyze_congestion(comprehensive_flight_data, airport_code)
        congestion_time = time.time() - start_time
        
        # Basic performance check - should complete within reasonable time
        assert delay_time < 30  # 30 seconds max
        assert congestion_time < 30  # 30 seconds max
        
        # Results should be valid
        assert delay_result.confidence_score > 0
        assert congestion_result.confidence_score > 0
    
    def test_empty_data_handling_across_analyzers(self):
        """Test that all analyzers handle empty data gracefully."""
        delay_analyzer = DelayAnalyzer()
        congestion_analyzer = CongestionAnalyzer()
        schedule_analyzer = ScheduleImpactAnalyzer()
        cascading_analyzer = CascadingImpactAnalyzer()
        
        airport_code = "BOM"
        empty_data = []
        empty_changes = []
        
        # All analyzers should handle empty data without crashing
        delay_result = delay_analyzer.analyze_delays(empty_data, airport_code)
        congestion_result = congestion_analyzer.analyze_congestion(empty_data, airport_code)
        schedule_result = schedule_analyzer.analyze_schedule_impact(empty_data, empty_changes, airport_code)
        cascading_result = cascading_analyzer.analyze_cascading_impact(empty_data, airport_code)
        
        # All should return valid results with zero confidence
        assert delay_result.confidence_score == 0.0
        assert congestion_result.confidence_score == 0.0
        assert schedule_result.confidence_score == 0.0
        assert cascading_result.confidence_score == 0.0
        
        # All should have error indicators in metrics
        assert 'error' in delay_result.metrics
        assert 'error' in congestion_result.metrics
        assert 'error' in schedule_result.metrics
        assert 'error' in cascading_result.metrics


if __name__ == "__main__":
    pytest.main([__file__])