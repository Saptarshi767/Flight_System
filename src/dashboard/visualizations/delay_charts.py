"""
Delay pattern visualization charts
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

def create_delay_heatmap(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create an interactive heatmap showing delay patterns by hour and day of week
    
    Args:
        data: DataFrame with columns ['hour', 'day_of_week', 'avg_delay']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample data
        hours = list(range(24))
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create realistic delay patterns
        delay_matrix = np.random.normal(10, 5, (len(days), len(hours)))
        
        # Add peak patterns
        for i, day in enumerate(days):
            # Morning peak (7-9 AM)
            delay_matrix[i, 7:10] += np.random.normal(15, 3, 3)
            # Evening peak (6-8 PM)
            delay_matrix[i, 18:21] += np.random.normal(12, 3, 3)
            # Weekend patterns (less congestion)
            if day in ['Saturday', 'Sunday']:
                delay_matrix[i] *= 0.7
        
        # Ensure no negative delays
        delay_matrix = np.maximum(delay_matrix, 0)
    else:
        # Use provided data
        pivot_data = data.pivot(index='day_of_week', columns='hour', values='avg_delay')
        delay_matrix = pivot_data.values
        days = pivot_data.index.tolist()
        hours = pivot_data.columns.tolist()
    
    fig = px.imshow(
        delay_matrix,
        x=hours,
        y=days,
        color_continuous_scale='RdYlGn_r',
        title="Flight Delay Patterns - Heatmap View",
        labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Avg Delay (min)'},
        aspect='auto'
    )
    
    # Customize layout
    fig.update_layout(
        title={
            'text': "Flight Delay Patterns - Heatmap View",
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
            title="Average Delay (minutes)",
            title_font={'size': 12}
        ),
        height=400,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>" +
                     "Hour: %{x}:00<br>" +
                     "Avg Delay: %{z:.1f} min<br>" +
                     "<extra></extra>"
    )
    
    return fig

def create_delay_distribution_chart(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a delay distribution chart showing frequency of different delay ranges
    
    Args:
        data: DataFrame with 'delay_minutes' column
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample delay data
        np.random.seed(42)
        delays = np.concatenate([
            np.random.exponential(8, 800),  # Most flights have small delays
            np.random.normal(25, 10, 150),  # Some moderate delays
            np.random.normal(60, 20, 50)    # Few large delays
        ])
        delays = np.maximum(delays, 0)  # No negative delays
        data = pd.DataFrame({'delay_minutes': delays})
    
    # Create delay ranges
    delay_ranges = [
        (0, 5, "0-5 min", "#2E8B57"),      # Green
        (5, 15, "5-15 min", "#FFD700"),    # Yellow
        (15, 30, "15-30 min", "#FF8C00"),  # Orange
        (30, 60, "30-60 min", "#FF4500"),  # Red-Orange
        (60, float('inf'), "60+ min", "#DC143C")  # Red
    ]
    
    # Count flights in each range
    range_counts = []
    range_labels = []
    colors = []
    
    for min_delay, max_delay, label, color in delay_ranges:
        if max_delay == float('inf'):
            count = len(data[data['delay_minutes'] >= min_delay])
        else:
            count = len(data[(data['delay_minutes'] >= min_delay) & 
                           (data['delay_minutes'] < max_delay)])
        range_counts.append(count)
        range_labels.append(label)
        colors.append(color)
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=range_labels,
        values=range_counts,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+percent+value',
        textposition='outside',
        hovertemplate="<b>%{label}</b><br>" +
                     "Flights: %{value}<br>" +
                     "Percentage: %{percent}<br>" +
                     "<extra></extra>"
    )])
    
    fig.update_layout(
        title={
            'text': "Flight Delay Distribution",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        annotations=[dict(text='Delay<br>Distribution', x=0.5, y=0.5, font_size=14, showarrow=False)],
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_airline_delay_comparison(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a comparison chart of delay performance across airlines
    
    Args:
        data: DataFrame with columns ['airline', 'delay_minutes']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample data
        airlines = ['Air India', 'IndiGo', 'SpiceJet', 'Vistara', 'GoAir', 'AirAsia India']
        
        # Generate realistic delay patterns for each airline
        airline_data = []
        for airline in airlines:
            base_delay = np.random.uniform(8, 20)  # Different base performance
            delays = np.random.exponential(base_delay, 200)
            for delay in delays:
                airline_data.append({'airline': airline, 'delay_minutes': delay})
        
        data = pd.DataFrame(airline_data)
    
    # Calculate statistics for each airline
    airline_stats = data.groupby('airline')['delay_minutes'].agg([
        'mean', 'median', 'std', 'count'
    ]).round(2)
    
    # Create box plot
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, airline in enumerate(airline_stats.index):
        airline_delays = data[data['airline'] == airline]['delay_minutes']
        
        fig.add_trace(go.Box(
            y=airline_delays,
            name=airline,
            marker_color=colors[i % len(colors)],
            boxpoints='outliers',
            hovertemplate="<b>%{fullData.name}</b><br>" +
                         "Delay: %{y:.1f} min<br>" +
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        title={
            'text': "Airline Delay Performance Comparison",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        yaxis_title="Delay (minutes)",
        xaxis_title="Airline",
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False
    )
    
    return fig

def create_delay_trend_chart(data: Optional[pd.DataFrame] = None, 
                           days: int = 30) -> go.Figure:
    """
    Create a time series chart showing delay trends over time
    
    Args:
        data: DataFrame with columns ['date', 'avg_delay', 'flight_count']
        days: Number of days to show in the trend
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample trend data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create realistic trend with some seasonality
        base_delay = 12
        trend_data = []
        
        for i, date in enumerate(dates):
            # Add weekly pattern (higher delays on Fridays)
            weekly_factor = 1.3 if date.weekday() == 4 else 1.0
            
            # Add some random variation
            daily_delay = base_delay * weekly_factor + np.random.normal(0, 3)
            daily_delay = max(daily_delay, 0)
            
            # Flight count varies by day
            flight_count = np.random.poisson(1200) + (50 if date.weekday() < 5 else -100)
            
            trend_data.append({
                'date': date,
                'avg_delay': daily_delay,
                'flight_count': flight_count
            })
        
        data = pd.DataFrame(trend_data)
    
    # Create subplot with secondary y-axis
    fig = go.Figure()
    
    # Add delay trend line
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['avg_delay'],
        mode='lines+markers',
        name='Average Delay',
        line=dict(color='red', width=3),
        marker=dict(size=6),
        hovertemplate="<b>Average Delay</b><br>" +
                     "Date: %{x}<br>" +
                     "Delay: %{y:.1f} min<br>" +
                     "<extra></extra>"
    ))
    
    # Add trend line
    z = np.polyfit(range(len(data)), data['avg_delay'], 1)
    p = np.poly1d(z)
    trend_line = p(range(len(data)))
    
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=trend_line,
        mode='lines',
        name='Trend',
        line=dict(color='blue', width=2, dash='dash'),
        hovertemplate="<b>Trend Line</b><br>" +
                     "Date: %{x}<br>" +
                     "Trend: %{y:.1f} min<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            'text': f"Flight Delay Trends - Last {days} Days",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Date",
        yaxis_title="Average Delay (minutes)",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='x unified'
    )
    
    return fig

def create_delay_causes_breakdown(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a stacked bar chart showing delay causes by time period
    
    Args:
        data: DataFrame with columns ['time_period', 'cause', 'delay_minutes']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample data
        time_periods = ['6-9 AM', '9-12 PM', '12-3 PM', '3-6 PM', '6-9 PM', '9-12 AM']
        causes = ['Weather', 'Air Traffic', 'Technical', 'Operational', 'Security']
        
        data_list = []
        for period in time_periods:
            for cause in causes:
                # Different causes dominate at different times
                if cause == 'Weather' and period in ['12-3 PM', '3-6 PM']:
                    delay = np.random.normal(15, 5)
                elif cause == 'Air Traffic' and period in ['6-9 AM', '6-9 PM']:
                    delay = np.random.normal(20, 7)
                elif cause == 'Technical':
                    delay = np.random.normal(8, 3)
                elif cause == 'Operational':
                    delay = np.random.normal(6, 2)
                else:
                    delay = np.random.normal(4, 2)
                
                delay = max(delay, 0)
                data_list.append({
                    'time_period': period,
                    'cause': cause,
                    'delay_minutes': delay
                })
        
        data = pd.DataFrame(data_list)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    colors = {
        'Weather': '#FF6B6B',
        'Air Traffic': '#4ECDC4',
        'Technical': '#45B7D1',
        'Operational': '#96CEB4',
        'Security': '#FFEAA7'
    }
    
    for cause in data['cause'].unique():
        cause_data = data[data['cause'] == cause]
        
        fig.add_trace(go.Bar(
            name=cause,
            x=cause_data['time_period'],
            y=cause_data['delay_minutes'],
            marker_color=colors.get(cause, '#BDC3C7'),
            hovertemplate="<b>%{fullData.name}</b><br>" +
                         "Time: %{x}<br>" +
                         "Avg Delay: %{y:.1f} min<br>" +
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        title={
            'text': "Delay Causes by Time Period",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Time Period",
        yaxis_title="Average Delay (minutes)",
        barmode='stack',
        height=500,
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