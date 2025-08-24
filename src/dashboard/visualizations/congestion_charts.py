"""
Congestion analysis visualization charts
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

def create_congestion_heatmap(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create an interactive heatmap showing congestion patterns by hour and day
    
    Args:
        data: DataFrame with columns ['hour', 'day_of_week', 'congestion_score']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample congestion data
        hours = list(range(24))
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create realistic congestion patterns
        congestion_matrix = np.random.rand(len(days), len(hours)) * 3 + 2  # Base congestion 2-5
        
        # Add peak patterns
        for i, day in enumerate(days):
            # Morning peak (7-10 AM)
            congestion_matrix[i, 7:11] += np.random.normal(3, 0.5, 4)
            # Evening peak (5-8 PM)
            congestion_matrix[i, 17:21] += np.random.normal(2.5, 0.5, 4)
            # Late night low congestion
            congestion_matrix[i, 23:6] *= 0.3
            # Weekend patterns
            if day in ['Saturday', 'Sunday']:
                congestion_matrix[i] *= 0.8
        
        # Cap at maximum congestion level
        congestion_matrix = np.minimum(congestion_matrix, 10)
    else:
        pivot_data = data.pivot(index='day_of_week', columns='hour', values='congestion_score')
        congestion_matrix = pivot_data.values
        days = pivot_data.index.tolist()
        hours = pivot_data.columns.tolist()
    
    fig = px.imshow(
        congestion_matrix,
        x=hours,
        y=days,
        color_continuous_scale='RdYlGn_r',
        title="Airport Congestion Heatmap",
        labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Congestion Level'},
        aspect='auto'
    )
    
    # Add congestion level annotations
    fig.update_layout(
        title={
            'text': "Airport Congestion Heatmap",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2,
            title_font={'size': 14}
        ),
        yaxis=dict(
            title_font={'size': 14}
        ),
        coloraxis_colorbar=dict(
            title="Congestion Level (1-10)",
            title_font={'size': 12},
            tickvals=[1, 3, 5, 7, 9],
            ticktext=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        ),
        height=400,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    # Custom hover template
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>" +
                     "Hour: %{x}:00<br>" +
                     "Congestion: %{z:.1f}/10<br>" +
                     "<extra></extra>"
    )
    
    return fig

def create_runway_utilization_chart(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a chart showing runway utilization across different runways
    
    Args:
        data: DataFrame with columns ['runway', 'utilization_percent', 'capacity']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample runway data
        runways = ['09L/27R', '09R/27L', '14/32', '05/23']
        utilization = [89, 76, 45, 32]
        capacity = [100, 100, 80, 60]  # Different runway capacities
        
        data = pd.DataFrame({
            'runway': runways,
            'utilization_percent': utilization,
            'capacity': capacity
        })
    
    # Calculate remaining capacity
    data['remaining_capacity'] = data['capacity'] - data['utilization_percent']
    
    fig = go.Figure()
    
    # Add utilized capacity bars
    fig.add_trace(go.Bar(
        name='Utilized',
        x=data['runway'],
        y=data['utilization_percent'],
        marker_color='#FF6B6B',
        hovertemplate="<b>%{x}</b><br>" +
                     "Utilized: %{y}%<br>" +
                     "<extra></extra>"
    ))
    
    # Add remaining capacity bars
    fig.add_trace(go.Bar(
        name='Available',
        x=data['runway'],
        y=data['remaining_capacity'],
        marker_color='#4ECDC4',
        hovertemplate="<b>%{x}</b><br>" +
                     "Available: %{y}%<br>" +
                     "<extra></extra>"
    ))
    
    # Add capacity threshold line
    fig.add_hline(
        y=80, 
        line_dash="dash", 
        line_color="orange",
        annotation_text="High Utilization Threshold (80%)",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title={
            'text': "Runway Utilization Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Runway",
        yaxis_title="Utilization (%)",
        barmode='stack',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_traffic_flow_timeline(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a timeline showing traffic flow throughout the day
    
    Args:
        data: DataFrame with columns ['hour', 'arrivals', 'departures', 'total_traffic']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample traffic data
        hours = list(range(24))
        
        # Create realistic traffic patterns
        arrivals = []
        departures = []
        
        for hour in hours:
            # Morning peak for departures
            if 6 <= hour <= 10:
                dep = np.random.poisson(80) + 40
                arr = np.random.poisson(60) + 20
            # Evening peak for arrivals
            elif 17 <= hour <= 21:
                dep = np.random.poisson(60) + 20
                arr = np.random.poisson(80) + 40
            # Night hours
            elif 22 <= hour or hour <= 5:
                dep = np.random.poisson(15) + 5
                arr = np.random.poisson(15) + 5
            # Regular hours
            else:
                dep = np.random.poisson(50) + 25
                arr = np.random.poisson(50) + 25
            
            arrivals.append(arr)
            departures.append(dep)
        
        data = pd.DataFrame({
            'hour': hours,
            'arrivals': arrivals,
            'departures': departures,
            'total_traffic': [a + d for a, d in zip(arrivals, departures)]
        })
    
    fig = go.Figure()
    
    # Add arrivals
    fig.add_trace(go.Scatter(
        x=data['hour'],
        y=data['arrivals'],
        mode='lines+markers',
        name='Arrivals',
        line=dict(color='#4ECDC4', width=3),
        marker=dict(size=6),
        fill='tonexty',
        hovertemplate="<b>Arrivals</b><br>" +
                     "Hour: %{x}:00<br>" +
                     "Flights: %{y}<br>" +
                     "<extra></extra>"
    ))
    
    # Add departures
    fig.add_trace(go.Scatter(
        x=data['hour'],
        y=data['departures'],
        mode='lines+markers',
        name='Departures',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=6),
        hovertemplate="<b>Departures</b><br>" +
                     "Hour: %{x}:00<br>" +
                     "Flights: %{y}<br>" +
                     "<extra></extra>"
    ))
    
    # Add total traffic line
    fig.add_trace(go.Scatter(
        x=data['hour'],
        y=data['total_traffic'],
        mode='lines',
        name='Total Traffic',
        line=dict(color='#9B59B6', width=2, dash='dash'),
        hovertemplate="<b>Total Traffic</b><br>" +
                     "Hour: %{x}:00<br>" +
                     "Flights: %{y}<br>" +
                     "<extra></extra>"
    ))
    
    # Add capacity threshold
    fig.add_hline(
        y=150,
        line_dash="dot",
        line_color="red",
        annotation_text="Capacity Limit",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title={
            'text': "24-Hour Traffic Flow Pattern",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Hour of Day",
        yaxis_title="Number of Flights",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Customize x-axis
    fig.update_xaxis(
        tickmode='linear',
        tick0=0,
        dtick=2,
        range=[-0.5, 23.5]
    )
    
    return fig

def create_congestion_forecast(data: Optional[pd.DataFrame] = None, 
                             forecast_hours: int = 24) -> go.Figure:
    """
    Create a forecast chart showing predicted congestion levels
    
    Args:
        data: DataFrame with columns ['datetime', 'actual_congestion', 'predicted_congestion']
        forecast_hours: Number of hours to forecast
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample forecast data
        now = datetime.now()
        times = [now + timedelta(hours=i) for i in range(-12, forecast_hours)]
        
        actual_congestion = []
        predicted_congestion = []
        
        for i, time in enumerate(times):
            hour = time.hour
            
            # Base congestion pattern
            if 7 <= hour <= 10 or 17 <= hour <= 20:
                base_congestion = 7 + np.random.normal(0, 1)
            elif 22 <= hour or hour <= 5:
                base_congestion = 2 + np.random.normal(0, 0.5)
            else:
                base_congestion = 4 + np.random.normal(0, 0.8)
            
            base_congestion = max(1, min(10, base_congestion))
            
            # Historical data (with some noise)
            if i < 12:  # Past 12 hours
                actual_congestion.append(base_congestion + np.random.normal(0, 0.3))
                predicted_congestion.append(None)
            else:  # Future hours
                actual_congestion.append(None)
                # Forecast with some uncertainty
                forecast = base_congestion + np.random.normal(0, 0.5)
                predicted_congestion.append(max(1, min(10, forecast)))
        
        data = pd.DataFrame({
            'datetime': times,
            'actual_congestion': actual_congestion,
            'predicted_congestion': predicted_congestion
        })
    
    fig = go.Figure()
    
    # Add historical data
    historical_data = data[data['actual_congestion'].notna()]
    if not historical_data.empty:
        fig.add_trace(go.Scatter(
            x=historical_data['datetime'],
            y=historical_data['actual_congestion'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#3498DB', width=3),
            marker=dict(size=6),
            hovertemplate="<b>Historical</b><br>" +
                         "Time: %{x}<br>" +
                         "Congestion: %{y:.1f}/10<br>" +
                         "<extra></extra>"
        ))
    
    # Add forecast data
    forecast_data = data[data['predicted_congestion'].notna()]
    if not forecast_data.empty:
        fig.add_trace(go.Scatter(
            x=forecast_data['datetime'],
            y=forecast_data['predicted_congestion'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#E74C3C', width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate="<b>Forecast</b><br>" +
                         "Time: %{x}<br>" +
                         "Predicted: %{y:.1f}/10<br>" +
                         "<extra></extra>"
        ))
    
    # Add congestion level zones
    fig.add_hrect(
        y0=0, y1=3,
        fillcolor="green", opacity=0.1,
        annotation_text="Low Congestion", annotation_position="top left"
    )
    fig.add_hrect(
        y0=3, y1=7,
        fillcolor="yellow", opacity=0.1,
        annotation_text="Moderate Congestion", annotation_position="top left"
    )
    fig.add_hrect(
        y0=7, y1=10,
        fillcolor="red", opacity=0.1,
        annotation_text="High Congestion", annotation_position="top left"
    )
    
    # Add current time line
    fig.add_vline(
        x=datetime.now(),
        line_dash="dot",
        line_color="black",
        annotation_text="Now",
        annotation_position="top"
    )
    
    fig.update_layout(
        title={
            'text': f"Congestion Forecast - Next {forecast_hours} Hours",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Time",
        yaxis_title="Congestion Level (1-10)",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='x unified',
        yaxis=dict(range=[0, 10])
    )
    
    return fig

def create_airport_comparison_chart(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a comparison chart of congestion across different airports
    
    Args:
        data: DataFrame with columns ['airport', 'avg_congestion', 'peak_congestion', 'flights_per_hour']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample airport data
        airports = ['Mumbai (BOM)', 'Delhi (DEL)', 'Bangalore (BLR)', 'Chennai (MAA)', 'Kolkata (CCU)']
        
        airport_data = []
        for airport in airports:
            avg_congestion = np.random.uniform(4, 8)
            peak_congestion = avg_congestion + np.random.uniform(1, 3)
            flights_per_hour = np.random.uniform(80, 150)
            
            airport_data.append({
                'airport': airport,
                'avg_congestion': avg_congestion,
                'peak_congestion': peak_congestion,
                'flights_per_hour': flights_per_hour
            })
        
        data = pd.DataFrame(airport_data)
    
    fig = go.Figure()
    
    # Add average congestion bars
    fig.add_trace(go.Bar(
        name='Average Congestion',
        x=data['airport'],
        y=data['avg_congestion'],
        marker_color='#3498DB',
        hovertemplate="<b>%{x}</b><br>" +
                     "Avg Congestion: %{y:.1f}/10<br>" +
                     "<extra></extra>"
    ))
    
    # Add peak congestion markers
    fig.add_trace(go.Scatter(
        name='Peak Congestion',
        x=data['airport'],
        y=data['peak_congestion'],
        mode='markers',
        marker=dict(
            color='#E74C3C',
            size=12,
            symbol='diamond',
            line=dict(width=2, color='white')
        ),
        hovertemplate="<b>%{x}</b><br>" +
                     "Peak Congestion: %{y:.1f}/10<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            'text': "Airport Congestion Comparison",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Airport",
        yaxis_title="Congestion Level (1-10)",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        yaxis=dict(range=[0, 10])
    )
    
    return fig