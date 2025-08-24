"""
Unit tests for the Schedule Impact Analysis Engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.analysis.schedule_impact_analyzer import (
    ScheduleImpactAnalyzer, ScheduleChange, ImpactScore, Scenario
)
from src.data.models import FlightData, AnalysisResult


class TestScheduleImpactAnalyzer:
    """Test cases for ScheduleImpactAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a ScheduleImpactAnalyzer instance for testing."""
        return ScheduleImpactAnalyzer(turnaround_time_minutes=45)
    
    @pytest.fixture
    def sample_flight_data(self):
        """Create sample flight data with connected flights."""
        base_time = datetime(2024, 1, 15, 8, 0)
        flights = []
        
        # Create connected flights (aircraft rotations)
        for day in range(3):  # 3 days of data
            for aircraft_num in range(3):  # 3 aircraft
                # Morning flight
                morning_dep = base_time + timedelta(days=day, hours=aircraft_num * 2)
                morning_arr = morning_dep + timedelta(hours=2)
                
                morning_flight = FlightData(
                    flight_id=f"FL{day:02d}{aircraft_num:02d}M",
                    airline="AI",
                    flight_number=f"AI{100 + day * 10 + aircraft_num}",
                    aircraft_type=f"B73{aircraft_num}",
                    origin_airport="BOM",
                    destination_airport="DEL",
                    scheduled_departure=morning_dep,
                    actual_departure=morning_dep + timedelta(minutes=np.random.randint(0, 30)),
                    scheduled_arrival=morning_arr,
                    actual_arrival=morning_arr + timedelta(minutes=np.random.randint(0, 45)),
                    passenger_count=150 + aircraft_num * 20
                )
                flights.append(morning_flight)
                
                # Afternoon return flight (same aircraft)
                afternoon_dep = morning_arr + timedelta(hours=1)  # 1 hour turnaround
                afternoon_arr = afternoon_dep + timedelta(hours=2)
                
                afternoon_flight = FlightData(
                    flight_id=f"FL{day:02d}{aircraft_num:02d}A",
                    airline="AI",
                    flight_number=f"AI{200 + day * 10 + aircraft_num}",
                    aircraft_type=f"B73{aircraft_num}",
                    origin_airport="DEL",
                    destination_airport="BOM",
                    scheduled_departure=afternoon_dep,
                    actual_departure=afternoon_dep + timedelta(minutes=np.random.randint(0, 20)),
                    scheduled_arrival=afternoon_arr,
                    actual_arrival=afternoon_arr + timedelta(minutes=np.random.randint(0, 30)),
                    passenger_count=150 + aircraft_num * 20
                )
                flights.append(afternoon_flight)
        
        return flights
    
    @pytest.fixture
    def sample_schedule_changes(self):
        """Create sample schedule changes for testing."""
        base_time = datetime(2024, 1, 15, 8, 0)
        
        return [
            ScheduleChange(
                flight_id="FL000M",
                original_departure=base_time,
                new_departure=base_time + timedelta(minutes=30),
                change_reason="delay_optimization",
                priority=2
            ),
            ScheduleChange(
                flight_id="FL010M",
                original_departure=base_time + timedelta(hours=2),
                new_departure=base_time + timedelta(hours=2, minutes=-15),
                change_reason="congestion_avoidance",
                priority=3
            ),
            ScheduleChange(
                flight_id="FL020M",
                original_departure=base_time + timedelta(hours=4),
                new_departure=base_time + timedelta(hours=4, minutes=45),
                change_reason="crew_optimization",
                priority=1
            )
        ]
    
    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.turnaround_time_minutes == 45
        assert not analyzer.is_trained
        assert analyzer.impact_predictor is not None
        assert analyzer.scaler is not None
        assert analyzer.flight_network is not None
    
    def test_schedule_change_dataclass(self):
        """Test ScheduleChange dataclass."""
        change = ScheduleChange(
            flight_id="FL001",
            original_departure=datetime(2024, 1, 15, 10, 0),
            new_departure=datetime(2024, 1, 15, 10, 30),
            change_reason="test",
            priority=2
        )
        
        assert change.flight_id == "FL001"
        assert change.priority == 2
        assert change.change_reason == "test"
    
    def test_impact_score_dataclass(self):
        """Test ImpactScore dataclass."""
        impact = ImpactScore(
            delay_impact=10.0,
            cascading_impact=5.0,
            operational_impact=3.0,
            total_impact=18.0,
            confidence=0.8
        )
        
        assert impact.delay_impact == 10.0
        assert impact.total_impact == 18.0
        assert impact.confidence == 0.8
    
    def test_flights_to_dataframe(self, analyzer, sample_flight_data):
        """Test conversion of flight data to DataFrame."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_flight_data)
        assert 'flight_id' in df.columns
        assert 'delay_minutes' in df.columns
        assert 'scheduled_departure' in df.columns
        assert 'aircraft_type' in df.columns
    
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
    
    def test_train_impact_model(self, analyzer, sample_flight_data):
        """Test impact model training."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        analyzer._train_impact_model(df)
        
        if len(df) >= 20:  # Only if sufficient data
            assert analyzer.is_trained
        else:
            # Should handle insufficient data gracefully
            assert not analyzer.is_trained
    
    def test_analyze_single_change_impact(self, analyzer, sample_flight_data, sample_schedule_changes):
        """Test single change impact analysis."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        change = sample_schedule_changes[0]
        impact = analyzer._analyze_single_change_impact(change, df)
        
        assert isinstance(impact, ImpactScore)
        assert impact.delay_impact >= 0
        assert impact.cascading_impact >= 0
        assert impact.operational_impact >= 0
        assert impact.total_impact >= 0
        assert 0 <= impact.confidence <= 1
    
    def test_calculate_cascading_impact(self, analyzer, sample_flight_data):
        """Test cascading impact calculation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        # Test with a flight that should have connections
        if analyzer.flight_network.number_of_nodes() > 0:
            flight_id = list(analyzer.flight_network.nodes())[0]
            cascading_impact = analyzer._calculate_cascading_impact(flight_id, 30.0)
            
            assert isinstance(cascading_impact, (int, float))
            assert cascading_impact >= 0
    
    def test_calculate_operational_impact(self, analyzer, sample_flight_data, sample_schedule_changes):
        """Test operational impact calculation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        change = sample_schedule_changes[0]
        
        # Find corresponding flight
        flight_data = df[df['flight_id'] == change.flight_id]
        if not flight_data.empty:
            flight = flight_data.iloc[0]
            operational_impact = analyzer._calculate_operational_impact(change, flight)
            
            assert isinstance(operational_impact, (int, float))
            assert operational_impact >= 0
    
    def test_perform_whatif_analysis(self, analyzer, sample_flight_data, sample_schedule_changes):
        """Test what-if analysis."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        whatif_results = analyzer._perform_whatif_analysis(sample_schedule_changes, df)
        
        assert 'baseline' in whatif_results
        assert 'with_changes' in whatif_results
        assert 'total_delay' in whatif_results['baseline']
        assert 'avg_delay' in whatif_results['baseline']
        assert 'operational_cost' in whatif_results['with_changes']
    
    def test_generate_scenarios(self, analyzer, sample_flight_data, sample_schedule_changes):
        """Test scenario generation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        scenarios = analyzer._generate_scenarios(sample_schedule_changes, df)
        
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        
        for scenario in scenarios:
            assert isinstance(scenario, Scenario)
            assert scenario.scenario_id is not None
            assert isinstance(scenario.changes, list)
            assert isinstance(scenario.predicted_impact, ImpactScore)
            assert 0 <= scenario.feasibility_score <= 1
    
    def test_analyze_constraints(self, analyzer, sample_flight_data, sample_schedule_changes):
        """Test constraint analysis."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        constraints = analyzer._analyze_constraints(sample_schedule_changes, df)
        
        assert 'turnaround_violations' in constraints
        assert 'crew_duty_violations' in constraints
        assert 'gate_conflicts' in constraints
        assert 'runway_capacity_issues' in constraints
        
        assert isinstance(constraints['turnaround_violations'], list)
    
    def test_predict_delay_impact(self, analyzer, sample_flight_data, sample_schedule_changes):
        """Test delay impact prediction."""
        change = sample_schedule_changes[0]
        impact = analyzer.predict_delay_impact(change, sample_flight_data)
        
        assert isinstance(impact, ImpactScore)
        assert impact.total_impact >= 0
        assert 0 <= impact.confidence <= 1
    
    def test_optimize_schedule(self, analyzer, sample_flight_data):
        """Test schedule optimization."""
        optimization_goals = {'delay_reduction': 0.3}
        
        recommended_changes = analyzer.optimize_schedule(
            sample_flight_data, optimization_goals, "BOM"
        )
        
        assert isinstance(recommended_changes, list)
        assert len(recommended_changes) <= 10  # Should return top 10
        
        for change in recommended_changes:
            assert isinstance(change, ScheduleChange)
            assert change.flight_id is not None
            assert change.original_departure != change.new_departure
    
    def test_find_optimal_time_slot(self, analyzer, sample_flight_data):
        """Test optimal time slot finding."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        
        if not df.empty:
            flight = df.iloc[0]
            optimal_time = analyzer._find_optimal_time_slot(flight, df)
            
            if optimal_time:
                assert isinstance(optimal_time, datetime)
                assert optimal_time != flight['scheduled_departure']
    
    def test_rank_changes_by_impact(self, analyzer, sample_flight_data, sample_schedule_changes):
        """Test ranking changes by impact."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        ranked_changes = analyzer._rank_changes_by_impact(sample_schedule_changes, df)
        
        assert isinstance(ranked_changes, list)
        assert len(ranked_changes) == len(sample_schedule_changes)
        
        for change in ranked_changes:
            assert isinstance(change, ScheduleChange)
    
    def test_analyze_schedule_impact_full_workflow(self, analyzer, sample_flight_data, sample_schedule_changes):
        """Test the complete schedule impact analysis workflow."""
        result = analyzer.analyze_schedule_impact(sample_flight_data, sample_schedule_changes, "BOM")
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == "schedule_impact_analysis"
        assert result.airport_code == "BOM"
        assert result.metrics is not None
        assert result.recommendations is not None
        assert isinstance(result.confidence_score, (int, float))
        assert 0 <= result.confidence_score <= 1
        
        # Check metrics structure
        metrics = result.metrics
        assert 'individual_change_impacts' in metrics
        assert 'whatif_analysis' in metrics
        assert 'scenario_comparison' in metrics
        assert 'constraint_analysis' in metrics
        assert 'network_metrics' in metrics
        assert 'total_flights_analyzed' in metrics
    
    def test_analyze_schedule_impact_empty_data(self, analyzer, sample_schedule_changes):
        """Test schedule impact analysis with empty data."""
        result = analyzer.analyze_schedule_impact([], sample_schedule_changes, "BOM")
        
        assert isinstance(result, AnalysisResult)
        assert result.confidence_score == 0.0
        assert 'error' in result.metrics
    
    def test_analyze_schedule_impact_no_airport_data(self, analyzer, sample_flight_data, sample_schedule_changes):
        """Test schedule impact analysis with no data for specific airport."""
        result = analyzer.analyze_schedule_impact(sample_flight_data, sample_schedule_changes, "XYZ")
        
        assert isinstance(result, AnalysisResult)
        assert result.airport_code == "XYZ"
        assert result.confidence_score == 0.0
    
    def test_get_network_metrics(self, analyzer, sample_flight_data):
        """Test network metrics calculation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        metrics = analyzer._get_network_metrics()
        
        assert 'nodes' in metrics
        assert 'edges' in metrics
        assert isinstance(metrics['nodes'], int)
        assert isinstance(metrics['edges'], int)
        
        if metrics['nodes'] > 0:
            assert 'density' in metrics
            assert 'average_clustering' in metrics
    
    def test_calculate_confidence_score(self, analyzer, sample_flight_data):
        """Test confidence score calculation."""
        df = analyzer._flights_to_dataframe(sample_flight_data)
        analyzer._build_flight_network(df)
        
        # Mock change impacts
        change_impacts = [
            {'impact': ImpactScore(10, 5, 3, 18, 0.8)}
        ]
        
        confidence = analyzer._calculate_confidence_score(df, change_impacts)
        
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1
    
    def test_generate_impact_recommendations(self, analyzer):
        """Test impact recommendation generation."""
        change_impacts = [
            {
                'change': ScheduleChange("FL001", datetime.now(), datetime.now() + timedelta(minutes=30), "test"),
                'impact': ImpactScore(10, 5, 3, 18, 0.8)
            },
            {
                'change': ScheduleChange("FL002", datetime.now(), datetime.now() + timedelta(minutes=-15), "test"),
                'impact': ImpactScore(-5, 2, 1, -2, 0.7)
            }
        ]
        
        whatif_results = {
            'baseline': {'total_delay': 100, 'avg_delay': 10},
            'with_changes': {'total_delay': 90, 'operational_cost': 5}
        }
        
        scenarios = [
            Scenario("test", [], ImpactScore(0, 0, 0, 0, 0.8), 10, 0.9)
        ]
        
        constraints = {
            'turnaround_violations': [],
            'crew_duty_violations': [],
            'gate_conflicts': [],
            'runway_capacity_issues': []
        }
        
        recommendations = analyzer._generate_impact_recommendations(
            change_impacts, whatif_results, scenarios, constraints
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check that recommendations contain expected content
        rec_text = ' '.join(recommendations)
        assert 'impact' in rec_text.lower() or 'change' in rec_text.lower()
    
    def test_insufficient_data_handling(self, analyzer):
        """Test handling of insufficient data scenarios."""
        # Test with very limited flight data
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
        
        changes = [
            ScheduleChange(
                flight_id="FL001",
                original_departure=datetime(2024, 1, 15, 10, 0),
                new_departure=datetime(2024, 1, 15, 10, 30),
                change_reason="test"
            )
        ]
        
        result = analyzer.analyze_schedule_impact(minimal_flights, changes, "BOM")
        
        # Should handle gracefully without crashing
        assert isinstance(result, AnalysisResult)
        assert result.confidence_score >= 0
    
    def test_edge_cases(self, analyzer, sample_flight_data):
        """Test edge cases and boundary conditions."""
        # Test with empty changes list
        result = analyzer.analyze_schedule_impact(sample_flight_data, [], "BOM")
        assert isinstance(result, AnalysisResult)
        
        # Test with non-existent flight ID in changes
        invalid_change = ScheduleChange(
            flight_id="NONEXISTENT",
            original_departure=datetime(2024, 1, 15, 10, 0),
            new_departure=datetime(2024, 1, 15, 10, 30),
            change_reason="test"
        )
        
        result = analyzer.analyze_schedule_impact(sample_flight_data, [invalid_change], "BOM")
        assert isinstance(result, AnalysisResult)
        
        # Test cascading impact with non-existent flight
        cascading_impact = analyzer._calculate_cascading_impact("NONEXISTENT", 30.0)
        assert cascading_impact == 0.0


if __name__ == "__main__":
    pytest.main([__file__])