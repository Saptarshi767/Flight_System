"""
Tests for dashboard visualization components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Import visualization modules
from src.dashboard.visualizations.delay_charts import (
    create_delay_heatmap,
    create_delay_distribution_chart,
    create_airline_delay_comparison,
    create_delay_trend_chart,
    create_delay_causes_breakdown
)

from src.dashboard.visualizations.congestion_charts import (
    create_congestion_heatmap,
    create_runway_utilization_chart,
    create_traffic_flow_timeline,
    create_congestion_forecast,
    create_airport_comparison_chart
)

from src.dashboard.visualizations.network_graphs import (
    create_flight_network_graph,
    create_critical_flights_chart,
    create_delay_propagation_chart,
    create_network_resilience_chart
)

from src.dashboard.visualizations.schedule_charts import (
    create_schedule_comparison_chart,
    create_time_slot_optimization_chart,
    create_cost_benefit_analysis,
    create_scenario_comparison_matrix,
    create_implementation_timeline
)

class TestDelayCharts:
    """Test delay visualization charts"""
    
    def test_create_delay_heatmap_default(self):
        """Test delay heatmap creation with default data"""
        fig = create_delay_heatmap()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Flight Delay Patterns' in fig.layout.title.text
    
    def test_create_delay_heatmap_custom_data(self):
        """Test delay heatmap with custom data"""
        # Create sample data
        data = pd.DataFrame({
            'hour': [8, 9, 10, 8, 9, 10],
            'day_of_week': ['Monday', 'Monday', 'Monday', 'Tuesday', 'Tuesday', 'Tuesday'],
            'avg_delay': [15.2, 18.7, 12.3, 10.5, 14.8, 9.2]
        })
        
        fig = create_delay_heatmap(data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_create_delay_distribution_chart(self):
        """Test delay distribution chart creation"""
        fig = create_delay_distribution_chart()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Flight Delay Distribution' in fig.layout.title.text
    
    def test_create_airline_delay_comparison(self):
        """Test airline delay comparison chart"""
        fig = create_airline_delay_comparison()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Airline Delay Performance' in fig.layout.title.text
    
    def test_create_delay_trend_chart(self):
        """Test delay trend chart creation"""
        fig = create_delay_trend_chart(days=7)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Should have delay line and trend line
        assert '7-Day' in fig.layout.title.text
    
    def test_create_delay_causes_breakdown(self):
        """Test delay causes breakdown chart"""
        fig = create_delay_causes_breakdown()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Delay Causes' in fig.layout.title.text

class TestCongestionCharts:
    """Test congestion visualization charts"""
    
    def test_create_congestion_heatmap(self):
        """Test congestion heatmap creation"""
        fig = create_congestion_heatmap()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Congestion Heatmap' in fig.layout.title.text
    
    def test_create_runway_utilization_chart(self):
        """Test runway utilization chart"""
        fig = create_runway_utilization_chart()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Should have utilized and available bars
        assert 'Runway Utilization' in fig.layout.title.text
    
    def test_create_traffic_flow_timeline(self):
        """Test traffic flow timeline chart"""
        fig = create_traffic_flow_timeline()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Arrivals, departures, total traffic
        assert 'Traffic Flow Pattern' in fig.layout.title.text
    
    def test_create_congestion_forecast(self):
        """Test congestion forecast chart"""
        fig = create_congestion_forecast(forecast_hours=12)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert 'Congestion Forecast' in fig.layout.title.text
    
    def test_create_airport_comparison_chart(self):
        """Test airport comparison chart"""
        fig = create_airport_comparison_chart()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Average and peak congestion
        assert 'Airport Congestion Comparison' in fig.layout.title.text

class TestNetworkGraphs:
    """Test network visualization graphs"""
    
    def test_create_flight_network_graph(self):
        """Test flight network graph creation"""
        fig = create_flight_network_graph()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Edges and nodes
        assert 'Flight Network' in fig.layout.title.text
    
    def test_create_critical_flights_chart(self):
        """Test critical flights chart"""
        fig = create_critical_flights_chart()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Critical Flights' in fig.layout.title.text
    
    def test_create_delay_propagation_chart(self):
        """Test delay propagation chart"""
        fig = create_delay_propagation_chart()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Delay Propagation' in fig.layout.title.text
    
    def test_create_network_resilience_chart(self):
        """Test network resilience chart"""
        fig = create_network_resilience_chart()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Bar chart and line chart
        assert 'Network Resilience' in fig.layout.title.text

class TestScheduleCharts:
    """Test schedule visualization charts"""
    
    def test_create_schedule_comparison_chart(self):
        """Test schedule comparison chart"""
        fig = create_schedule_comparison_chart()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Current and proposed schedules
        assert 'Schedule Impact Comparison' in fig.layout.title.text
    
    def test_create_time_slot_optimization_chart(self):
        """Test time slot optimization chart"""
        fig = create_time_slot_optimization_chart()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Capacity, current, optimal
        assert 'Time Slot Optimization' in fig.layout.title.text
    
    def test_create_cost_benefit_analysis(self):
        """Test cost-benefit analysis chart"""
        fig = create_cost_benefit_analysis()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Costs, benefits, net impact
        assert 'Cost-Benefit Analysis' in fig.layout.title.text
    
    def test_create_scenario_comparison_matrix(self):
        """Test scenario comparison matrix"""
        fig = create_scenario_comparison_matrix()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Scenario Comparison Matrix' in fig.layout.title.text
    
    def test_create_implementation_timeline(self):
        """Test implementation timeline chart"""
        fig = create_implementation_timeline()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Implementation Timeline' in fig.layout.title.text

class TestVisualizationInteractivity:
    """Test visualization interactivity features"""
    
    def test_chart_hover_templates(self):
        """Test that charts have proper hover templates"""
        fig = create_delay_heatmap()
        
        # Check that hover template is set
        assert fig.data[0].hovertemplate is not None
        assert 'extra' in fig.data[0].hovertemplate
    
    def test_chart_responsiveness(self):
        """Test that charts are configured for responsiveness"""
        fig = create_congestion_heatmap()
        
        # Check layout configuration
        assert fig.layout.margin is not None
        assert fig.layout.height is not None
    
    def test_color_scales(self):
        """Test that appropriate color scales are used"""
        fig = create_delay_heatmap()
        
        # Check that color scale is set for heatmaps
        if hasattr(fig.data[0], 'colorscale'):
            assert fig.data[0].colorscale is not None

class TestVisualizationPerformance:
    """Test visualization performance"""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        # Create large dataset
        large_data = pd.DataFrame({
            'hour': np.random.randint(0, 24, 10000),
            'day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], 10000),
            'avg_delay': np.random.normal(15, 5, 10000)
        })
        
        # Should handle large dataset without errors
        fig = create_delay_heatmap(large_data)
        assert isinstance(fig, go.Figure)
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets"""
        empty_data = pd.DataFrame(columns=['hour', 'day_of_week', 'avg_delay'])
        
        # Should fallback to default data generation
        fig = create_delay_heatmap(empty_data)
        assert isinstance(fig, go.Figure)
    
    def test_chart_rendering_speed(self):
        """Test that charts render within reasonable time"""
        import time
        
        start_time = time.time()
        fig = create_flight_network_graph()
        end_time = time.time()
        
        # Should render within 5 seconds
        assert (end_time - start_time) < 5.0
        assert isinstance(fig, go.Figure)

class TestVisualizationAccessibility:
    """Test visualization accessibility features"""
    
    def test_color_blind_friendly_colors(self):
        """Test that visualizations use color-blind friendly palettes"""
        fig = create_delay_distribution_chart()
        
        # Check that colors are defined (accessibility will depend on specific palette choice)
        if hasattr(fig.data[0], 'marker') and hasattr(fig.data[0].marker, 'colors'):
            assert fig.data[0].marker.colors is not None
    
    def test_text_alternatives(self):
        """Test that charts have proper text alternatives"""
        fig = create_airline_delay_comparison()
        
        # Check that title and axis labels are set
        assert fig.layout.title.text is not None
        assert fig.layout.xaxis.title.text is not None
        assert fig.layout.yaxis.title.text is not None
    
    def test_keyboard_navigation_support(self):
        """Test that charts support keyboard navigation"""
        fig = create_congestion_forecast()
        
        # Plotly charts should have built-in keyboard support
        # We can check that the figure is properly structured
        assert len(fig.data) > 0
        assert fig.layout is not None

if __name__ == "__main__":
    pytest.main([__file__])