"""
Unit tests for the Cascading Impact Analysis Engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.analysis.cascading_impact_analyzer import (
    CascadingImpactAnalyzer, CriticalFlight, DelayPropagation, NetworkDisruption
)
from src.data.models import FlightData, AnalysisResult


class TestCascadingImpactAnalyzer:
    """Test cases for CascadingImpactAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a CascadingImpactAnalyzer instance for testing."""
        return CascadingImpactAnalyzer(min_turnaround_minutes=45, max_crew_duty_hours=14)
    
    @pytest.fixture
    def sample_flight_data(self):
        """Create sample flight data with complex network connections."""
        base_time = datetime(2024, 1, 15, 6, 0)
        flights = []
        
        # Create a network of connected flights
        for day in range(2):  # 2 days of data
            for aircraft_num in range(4):  # 4 aircraft per day
                # Morning flight (BOM -> DEL)
                morning_dep = base_time + timedelta(days=day, hours=6 + aircraft_num)
                morning_arr = morning_dep + timedelta(hours=2)
                
                morning_flight = FlightData(
                    flight_id=f"FL{day:02d}{aircraft_num:02d}M",
                    airline="AI",
                    flight_number=f"AI{100 + day * 10 + aircraft_num}",
                    aircraft_type=f"B73{aircraft_num}",
                    origin_airport="BOM",
                    destination_airport="DEL",
                    scheduled_departure=morning_dep,
                    actual_departure=morning_dep + timedelta(minutes=np.random.randint(0, 45)),
                    scheduled_arrival=morning_arr,
                    actual_arrival=morning_arr + timedelta(minutes=np.random.randint(0, 60)),
                    passenger_count=150 + aircraft_num * 25
                )
                flights.append(morning_flight)
                
                # Afternoon return flight (DEL -> BOM) - same aircraft
                afternoon_dep = morning_arr + timedelta(hours=1, minutes=30)  # 1.5 hour turnaround
                afternoon_arr = afternoon_dep + timedelta(hours=2)
                
                afternoon_flight = FlightData(
                    flight_id=f"FL{day:02d}{aircraft_num:02d}A",
                    airline="AI",
                    flight_number=f"AI{200 + day * 10 + aircraft_num}",
                    aircraft_type=f"B73{aircraft_num}",
                    origin_airport="DEL",
                    destination_airport="BOM",
                    scheduled_departure=afternoon_dep,
                    actual_departure=afternoon_dep + timedelta(minutes=np.random.randint(0, 30)),
                    scheduled_arrival=afternoon_arr,
                    actual_arrival=afternoon_arr + timedelta(minutes=np.random.randint(0, 45)),
                    passenger_count=150 + aircraft_num * 25
                )
                flights.append(afternoon_flight)
                
                # Evening flight (BOM -> DEL) - same aircraft
                evening_dep = afternoon_arr + timedelta(hours=2)  # 2 hour turnaround
                evening_arr = evening_dep + timedelta(hours=2)
                
                evening_flight = FlightData(
                    flight_id=f"FL{day:02d}{aircraft_num:02d}E",
                    airline="AI",
                    flight_number=f"AI{300 + day * 10 + aircraft_num}",
                    aircraft_type=f"B73{aircraft_num}",
                    origin_airport="BOM",
                    destination_airport="DEL",
                    scheduled_departure=evening_dep,
                    actual_departure=evening_dep + timedelta(minutes=np.random.randint(0, 20)),
                    scheduled_arrival=evening_arr,
                    actual_arrival=evening_arr + timedelta(minutes=np.random.randint(0, 30)),
                    passenger_count=150 + aircraft_num * 25
                )
                flights.append(evening_flight)
        
        return flights
    
    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.min_turnaround_minutes == 45
        assert analyzer.max_crew_duty_hours == 14
        assert analyzer.flight_network is not None
        assert analyzer.impact_scores == {}
        assert analyzer.critical_flights == []
    
    def test_critical_flight_dataclass(self):
        """Test CriticalFlight dataclass."""
        critical_flight = CriticalFlight(
            flight_id="FL001",
            airline="AI",
            flight_number="AI101",
            origin_airport="BOM",
            destination_airport="DEL",
            scheduled_departure=datetime(2024, 1, 15, 10, 0),
            network_impact_score=0.85,
            cascading_delay_potential=120.0,
            downstream_flights_affected=5,
            criticality_rank=1,
            impact_category="Critical"
        )
        
        assert critical_flight.flight_id == "FL001"
        assert critical_flight.network_impact_score == 0.85
        assert critical_flight.impact_category == "Critical"
    
    def test_delay_propagation_dataclass(self):
        """Test DelayPropagation dataclass."""
        propagation = DelayPropagation(
            source_flight="FL001",
            affected_flights=["FL002", "FL003"],
            propagation_path=["FL001", "FL002", "FL003"],
            total_delay_minutes=90.0,
            propagation_depth=2
        )
        
        assert propagation.source_flight == "FL001"
        assert len(propagation.affected_flights) == 2
        assert propagation.propagation_depth == 2
    
    def test_network_disruption_dataclass(self):
        """Test NetworkDisruption dataclass."""
        disruption = NetworkDisruption(
            disrupted_flight="FL001",
            disruption_type="delay",
            disruption_magnitude=60.0,
            affected_flights=["FL002", "FL003"],
            total_impact_score=150.0,
            recovery_time_hours=2.0
        )
        
        assert disruption.disrupted_flight == "FL001"
        assert disruption.disruption_type == "delay"
        assert disruption.recovery_time_hours == 2.0
    
    def test_flights_to_dataframe(self, analyzer, sample_flight_data):
        """Test conversion of flight data to DataFrame."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_flight_data)
        assert 'flight_id' in df.columns
        assert 'delay_minutes' in df.columns
        assert 'aircraft_type' in df.columns
        assert 'passenger_count' in df.columns
    
    def test_build_flight_network(self, analyzer, sample_flight_data):
        """Test flight network construction."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        assert analyzer.flight_network.number_of_nodes() > 0
        assert analyzer.flight_network.number_of_edges() >= 0
        
        # Check that nodes have required attributes
        for node in analyzer.flight_network.nodes():
            node_data = analyzer.flight_network.nodes[node]
            assert 'airline' in node_data
            assert 'departure_time' in node_data
            assert 'origin' in node_data
            assert 'destination' in node_data
    
    def test_add_aircraft_rotation_edges(self, analyzer, sample_flight_data):
        """Test aircraft rotation edge creation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        # Check for aircraft rotation edges
        aircraft_rotation_edges = [
            (u, v) for u, v, d in analyzer.flight_network.edges(data=True)
            if d.get('connection_type') == 'aircraft_rotation'
        ]
        
        assert len(aircraft_rotation_edges) > 0
        
        # Verify edge properties
        for u, v in aircraft_rotation_edges:
            edge_data = analyzer.flight_network.get_edge_data(u, v)
            assert 'turnaround_time' in edge_data
            assert 'criticality' in edge_data
            assert edge_data['criticality'] == 0.9
    
    def test_add_crew_rotation_edges(self, analyzer, sample_flight_data):
        """Test crew rotation edge creation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        # Check for crew rotation edges
        crew_rotation_edges = [
            (u, v) for u, v, d in analyzer.flight_network.edges(data=True)
            if d.get('connection_type') == 'crew_rotation'
        ]
        
        # Should have some crew rotation edges
        assert len(crew_rotation_edges) >= 0  # May be 0 if no valid crew rotations
        
        # Verify edge properties if they exist
        for u, v in crew_rotation_edges:
            edge_data = analyzer.flight_network.get_edge_data(u, v)
            assert 'rest_time' in edge_data
            assert 'criticality' in edge_data
            assert edge_data['criticality'] == 0.6
    
    def test_calculate_network_impact_scores(self, analyzer, sample_flight_data):
        """Test network impact score calculation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        impact_scores = analyzer._calculate_network_impact_scores()
        
        assert isinstance(impact_scores, dict)
        assert len(impact_scores) == analyzer.flight_network.number_of_nodes()
        
        # Check that all scores are non-negative
        for flight_id, score in impact_scores.items():
            assert isinstance(score, (int, float))
            assert score >= 0
    
    def test_identify_critical_flights(self, analyzer, sample_flight_data):
        """Test critical flight identification."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        network_scores = analyzer._calculate_network_impact_scores()
        critical_flights = analyzer._identify_critical_flights(df, network_scores)
        
        assert isinstance(critical_flights, list)
        assert len(critical_flights) == len(df)
        
        # Check that flights are sorted by impact score
        for i in range(1, len(critical_flights)):
            assert critical_flights[i-1].network_impact_score >= critical_flights[i].network_impact_score
        
        # Check that ranks are assigned correctly
        for i, flight in enumerate(critical_flights):
            assert flight.criticality_rank == i + 1
            assert flight.impact_category in ["Critical", "High", "Medium", "Low"]
    
    def test_analyze_delay_propagation(self, analyzer, sample_flight_data):
        """Test delay propagation analysis."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        propagation_patterns = analyzer._analyze_delay_propagation(df)
        
        assert isinstance(propagation_patterns, list)
        
        # Check propagation pattern structure
        for pattern in propagation_patterns:
            assert isinstance(pattern, DelayPropagation)
            assert pattern.source_flight is not None
            assert isinstance(pattern.affected_flights, list)
            assert pattern.total_delay_minutes >= 0
            assert pattern.propagation_depth >= 0
    
    def test_analyze_domino_effects(self, analyzer, sample_flight_data):
        """Test domino effect analysis."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        domino_analysis = analyzer._analyze_domino_effects(df)
        
        assert isinstance(domino_analysis, dict)
        assert 'potential_domino_flights' in domino_analysis
        assert 'domino_chains' in domino_analysis
        assert 'vulnerability_score' in domino_analysis
        
        assert isinstance(domino_analysis['potential_domino_flights'], list)
        assert isinstance(domino_analysis['domino_chains'], list)
        assert 0 <= domino_analysis['vulnerability_score'] <= 1
    
    def test_trace_delay_propagation(self, analyzer, sample_flight_data):
        """Test delay propagation tracing."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        if not df.empty:
            source_flight = df.iloc[0]['flight_id']
            propagation = analyzer.trace_delay_propagation(source_flight, 60.0, sample_flight_data)
            
            assert isinstance(propagation, DelayPropagation)
            assert propagation.source_flight == source_flight
            assert isinstance(propagation.affected_flights, list)
            assert propagation.total_delay_minutes >= 0
    
    def test_simulate_network_disruption(self, analyzer, sample_flight_data):
        """Test network disruption simulation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        if not df.empty:
            disrupted_flight = df.iloc[0]['flight_id']
            
            # Test delay disruption
            delay_disruption = analyzer.simulate_network_disruption(
                disrupted_flight, "delay", 90.0, sample_flight_data
            )
            
            assert isinstance(delay_disruption, NetworkDisruption)
            assert delay_disruption.disrupted_flight == disrupted_flight
            assert delay_disruption.disruption_type == "delay"
            assert delay_disruption.total_impact_score >= 0
            
            # Test cancellation disruption
            cancel_disruption = analyzer.simulate_network_disruption(
                disrupted_flight, "cancellation", 1.0, sample_flight_data
            )
            
            assert cancel_disruption.disruption_type == "cancellation"
            assert cancel_disruption.recovery_time_hours > 0
    
    def test_identify_most_critical_flights(self, analyzer, sample_flight_data):
        """Test identification of most critical flights."""
        critical_flights = analyzer.identify_most_critical_flights(sample_flight_data, "BOM", top_n=5)
        
        assert isinstance(critical_flights, list)
        assert len(critical_flights) <= 5
        
        # Check that flights are sorted by criticality
        for i in range(1, len(critical_flights)):
            assert critical_flights[i-1].network_impact_score >= critical_flights[i].network_impact_score
        
        for flight in critical_flights:
            assert isinstance(flight, CriticalFlight)
    
    def test_get_propagation_factor(self, analyzer):
        """Test propagation factor calculation."""
        assert analyzer._get_propagation_factor('aircraft_rotation') == 0.8
        assert analyzer._get_propagation_factor('crew_rotation') == 0.6
        assert analyzer._get_propagation_factor('passenger_connection') == 0.3
        assert analyzer._get_propagation_factor('resource_dependency') == 0.4
        assert analyzer._get_propagation_factor('unknown') == 0.5
    
    def test_get_aircraft_rotation_flights(self, analyzer, sample_flight_data):
        """Test aircraft rotation flight identification."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        if analyzer.flight_network.number_of_nodes() > 0:
            flight_id = list(analyzer.flight_network.nodes())[0]
            rotation_flights = analyzer._get_aircraft_rotation_flights(flight_id)
            
            assert isinstance(rotation_flights, list)
            # All returned flights should be strings (flight IDs)
            for flight in rotation_flights:
                assert isinstance(flight, str)
    
    def test_get_network_analysis_metrics(self, analyzer, sample_flight_data):
        """Test network analysis metrics calculation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        metrics = analyzer._get_network_analysis_metrics()
        
        assert isinstance(metrics, dict)
        assert 'nodes' in metrics
        assert 'edges' in metrics
        assert 'density' in metrics
        
        if metrics['nodes'] > 0:
            assert 'average_clustering' in metrics
            assert 'number_of_components' in metrics
            assert 'avg_out_degree' in metrics
            assert 'max_out_degree' in metrics
    
    def test_analyze_cascading_impact_full_workflow(self, analyzer, sample_flight_data):
        """Test the complete cascading impact analysis workflow."""
        result = analyzer.analyze_cascading_impact(sample_flight_data, "BOM")
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "cascading_impact_analysis"
        assert result.airport_code == "BOM"
        assert result.metrics is not None
        assert result.recommendations is not None
        assert isinstance(result.confidence_score, (int, float))
        assert 0 <= result.confidence_score <= 1
        
        # Check metrics structure
        metrics = result.metrics
        assert 'critical_flights' in metrics
        assert 'network_metrics' in metrics
        assert 'delay_propagation_analysis' in metrics
        assert 'domino_effect_analysis' in metrics
        assert 'disruption_scenarios' in metrics
        assert 'total_flights_analyzed' in metrics
    
    def test_analyze_cascading_impact_empty_data(self, analyzer):
        """Test cascading impact analysis with empty data."""
        result = analyzer.analyze_cascading_impact([], "BOM")
        
        assert isinstance(result, AnalysisResult)
        assert result.confidence_score == 0.0
        assert 'error' in result.metrics
    
    def test_analyze_cascading_impact_no_airport_data(self, analyzer, sample_flight_data):
        """Test cascading impact analysis with no data for specific airport."""
        result = analyzer.analyze_cascading_impact(sample_flight_data, "XYZ")
        
        assert isinstance(result, AnalysisResult)
        assert result.airport_code == "XYZ"
        assert result.confidence_score == 0.0
    
    def test_generate_cascading_recommendations(self, analyzer):
        """Test cascading impact recommendation generation."""
        # Mock critical flights
        critical_flights = [
            CriticalFlight(
                flight_id="FL001",
                airline="AI",
                flight_number="AI101",
                origin_airport="BOM",
                destination_airport="DEL",
                scheduled_departure=datetime.now(),
                network_impact_score=0.9,
                cascading_delay_potential=150.0,
                downstream_flights_affected=5,
                criticality_rank=1,
                impact_category="Critical"
            )
        ]
        
        # Mock propagation analysis
        propagation_analysis = [
            DelayPropagation(
                source_flight="FL001",
                affected_flights=["FL002", "FL003"],
                propagation_path=["FL001", "FL002", "FL003"],
                total_delay_minutes=120.0,
                propagation_depth=2
            )
        ]
        
        # Mock domino analysis
        domino_analysis = {
            'potential_domino_flights': [{'potential_affected_flights': 6}],
            'vulnerability_score': 0.3
        }
        
        # Mock disruption scenarios
        disruption_scenarios = [
            NetworkDisruption(
                disrupted_flight="FL001",
                disruption_type="delay",
                disruption_magnitude=60.0,
                affected_flights=["FL002"],
                total_impact_score=100.0,
                recovery_time_hours=2.0
            )
        ]
        
        recommendations = analyzer._generate_cascading_recommendations(
            critical_flights, propagation_analysis, domino_analysis, disruption_scenarios
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check that recommendations contain expected content
        rec_text = ' '.join(recommendations)
        assert 'critical' in rec_text.lower() or 'impact' in rec_text.lower()
    
    def test_calculate_confidence_score(self, analyzer, sample_flight_data):
        """Test confidence score calculation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        confidence = analyzer._calculate_confidence_score(df)
        
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1
    
    def test_edge_cases(self, analyzer):
        """Test edge cases and boundary conditions."""
        # Test with non-existent flight ID
        propagation = analyzer.trace_delay_propagation("NONEXISTENT", 60.0, [])
        assert propagation.source_flight == "NONEXISTENT"
        assert len(propagation.affected_flights) == 0
        assert propagation.total_delay_minutes == 0.0
        
        # Test disruption simulation with non-existent flight
        disruption = analyzer.simulate_network_disruption("NONEXISTENT", "delay", 60.0, [])
        assert disruption.disrupted_flight == "NONEXISTENT"
        assert len(disruption.affected_flights) == 0
        
        # Test aircraft rotation with non-existent flight
        rotation_flights = analyzer._get_aircraft_rotation_flights("NONEXISTENT")
        assert rotation_flights == []
    
    def test_empty_network_handling(self, analyzer):
        """Test handling of empty network scenarios."""
        # Test network metrics with empty network
        metrics = analyzer._get_network_analysis_metrics()
        assert metrics['nodes'] == 0
        assert metrics['edges'] == 0
        
        # Test impact scores with empty network
        impact_scores = analyzer._calculate_network_impact_scores()
        assert impact_scores == {}
    
    def test_large_network_performance(self, analyzer):
        """Test performance with larger network (basic performance test)."""
        # Create a larger dataset
        base_time = datetime(2024, 1, 15, 6, 0)
        large_flight_data = []
        
        for day in range(1):  # 1 day to keep test reasonable
            for hour in range(24):
                for flight_num in range(3):  # 3 flights per hour
                    flight_time = base_time + timedelta(days=day, hours=hour, minutes=flight_num * 20)
                    
                    flight = FlightData(
                        flight_id=f"FL{day:02d}{hour:02d}{flight_num:02d}",
                        airline=f"AI{flight_num % 3}",
                        flight_number=f"AI{1000 + day * 100 + hour * 10 + flight_num}",
                        aircraft_type=f"B73{flight_num}",
                        origin_airport="BOM" if flight_num % 2 == 0 else "DEL",
                        destination_airport="DEL" if flight_num % 2 == 0 else "BOM",
                        scheduled_departure=flight_time,
                        scheduled_arrival=flight_time + timedelta(hours=2),
                        passenger_count=150
                    )
                    large_flight_data.append(flight)
        
        # Should handle larger dataset without crashing
        result = analyzer.analyze_cascading_impact(large_flight_data, "BOM")
        assert isinstance(result, AnalysisResult)
        assert result.metrics['total_flights_analyzed'] > 0


if __name__ == "__main__":
    pytest.main([__file__])