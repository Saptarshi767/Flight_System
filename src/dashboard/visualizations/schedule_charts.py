"""
Schedule impact comparison charts
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

def create_schedule_comparison_chart(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a comparison chart showing before/after schedule impact
    
    Args:
        data: DataFrame with columns ['metric', 'current_value', 'proposed_value', 'improvement']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample schedule comparison data
        metrics = [
            'Average Delay (min)',
            'On-Time Performance (%)',
            'Affected Flights',
            'Passenger Satisfaction',
            'Fuel Efficiency (%)',
            'Crew Utilization (%)'
        ]
        
        comparison_data = []
        for metric in metrics:
            if 'Delay' in metric:
                current = np.random.uniform(15, 25)
                proposed = current * np.random.uniform(0.6, 0.8)
            elif 'Performance' in metric or 'Satisfaction' in metric or 'Efficiency' in metric:
                current = np.random.uniform(70, 85)
                proposed = current * np.random.uniform(1.1, 1.2)
            else:
                current = np.random.uniform(20, 40)
                proposed = current * np.random.uniform(0.7, 0.9)
            
            improvement = ((proposed - current) / current) * 100
            
            comparison_data.append({
                'metric': metric,
                'current_value': current,
                'proposed_value': proposed,
                'improvement': improvement
            })
        
        data = pd.DataFrame(comparison_data)
    
    fig = go.Figure()
    
    # Add current values
    fig.add_trace(go.Bar(
        name='Current Schedule',
        x=data['metric'],
        y=data['current_value'],
        marker_color='#E74C3C',
        hovertemplate="<b>%{x}</b><br>" +
                     "Current: %{y:.1f}<br>" +
                     "<extra></extra>"
    ))
    
    # Add proposed values
    fig.add_trace(go.Bar(
        name='Proposed Schedule',
        x=data['metric'],
        y=data['proposed_value'],
        marker_color='#2ECC71',
        hovertemplate="<b>%{x}</b><br>" +
                     "Proposed: %{y:.1f}<br>" +
                     "<extra></extra>"
    ))
    
    # Add improvement percentages as text annotations
    for i, row in data.iterrows():
        improvement_text = f"{row['improvement']:+.1f}%"
        color = 'green' if row['improvement'] > 0 else 'red'
        
        fig.add_annotation(
            x=i,
            y=max(row['current_value'], row['proposed_value']) + 2,
            text=improvement_text,
            showarrow=False,
            font=dict(color=color, size=12, family="Arial Black")
        )
    
    fig.update_layout(
        title={
            'text': "Schedule Impact Comparison",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Performance Metrics",
        yaxis_title="Value",
        barmode='group',
        height=500,
        margin=dict(l=50, r=50, t=100, b=120),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Rotate x-axis labels
    fig.update_xaxis(tickangle=45)
    
    return fig

def create_time_slot_optimization_chart(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a chart showing optimal time slot recommendations
    
    Args:
        data: DataFrame with columns ['hour', 'current_flights', 'optimal_flights', 'capacity']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample time slot data
        hours = list(range(6, 23))
        
        optimization_data = []
        for hour in hours:
            # Current distribution (with peaks)
            if 8 <= hour <= 10 or 18 <= hour <= 20:
                current = np.random.poisson(80) + 40
            else:
                current = np.random.poisson(50) + 20
            
            # Optimal distribution (more balanced)
            optimal = min(current * np.random.uniform(0.8, 1.2), 100)
            capacity = 120  # Fixed capacity
            
            optimization_data.append({
                'hour': hour,
                'current_flights': current,
                'optimal_flights': optimal,
                'capacity': capacity
            })
        
        data = pd.DataFrame(optimization_data)
    
    fig = go.Figure()
    
    # Add capacity line
    fig.add_trace(go.Scatter(
        x=data['hour'],
        y=data['capacity'],
        mode='lines',
        name='Runway Capacity',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate="<b>Capacity</b><br>" +
                     "Hour: %{x}:00<br>" +
                     "Capacity: %{y} flights<br>" +
                     "<extra></extra>"
    ))
    
    # Add current flights
    fig.add_trace(go.Bar(
        name='Current Schedule',
        x=data['hour'],
        y=data['current_flights'],
        marker_color='rgba(231, 76, 60, 0.7)',
        hovertemplate="<b>Current Schedule</b><br>" +
                     "Hour: %{x}:00<br>" +
                     "Flights: %{y}<br>" +
                     "<extra></extra>"
    ))
    
    # Add optimal flights
    fig.add_trace(go.Bar(
        name='Optimized Schedule',
        x=data['hour'],
        y=data['optimal_flights'],
        marker_color='rgba(46, 204, 113, 0.7)',
        hovertemplate="<b>Optimized Schedule</b><br>" +
                     "Hour: %{x}:00<br>" +
                     "Flights: %{y}<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            'text': "Time Slot Optimization Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Hour of Day",
        yaxis_title="Number of Flights",
        barmode='group',
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
    
    # Customize x-axis
    fig.update_xaxis(
        tickmode='linear',
        tick0=6,
        dtick=2
    )
    
    return fig

def create_cost_benefit_analysis(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a cost-benefit analysis chart for schedule changes
    
    Args:
        data: DataFrame with columns ['category', 'cost', 'benefit', 'net_impact']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample cost-benefit data
        categories = ['Fuel Costs', 'Crew Scheduling', 'Passenger Compensation', 
                     'Airport Fees', 'Maintenance', 'Revenue Impact']
        
        cost_benefit_data = []
        for category in categories:
            cost = np.random.uniform(1000, 5000)
            benefit = cost * np.random.uniform(0.8, 1.5)
            net_impact = benefit - cost
            
            cost_benefit_data.append({
                'category': category,
                'cost': cost,
                'benefit': benefit,
                'net_impact': net_impact
            })
        
        data = pd.DataFrame(cost_benefit_data)
    
    fig = go.Figure()
    
    # Add costs (negative values for visual effect)
    fig.add_trace(go.Bar(
        name='Costs',
        x=data['category'],
        y=-data['cost'],
        marker_color='#E74C3C',
        hovertemplate="<b>%{x}</b><br>" +
                     "Cost: $%{customdata:,.0f}<br>" +
                     "<extra></extra>",
        customdata=data['cost']
    ))
    
    # Add benefits (positive values)
    fig.add_trace(go.Bar(
        name='Benefits',
        x=data['category'],
        y=data['benefit'],
        marker_color='#2ECC71',
        hovertemplate="<b>%{x}</b><br>" +
                     "Benefit: $%{y:,.0f}<br>" +
                     "<extra></extra>"
    ))
    
    # Add net impact line
    fig.add_trace(go.Scatter(
        name='Net Impact',
        x=data['category'],
        y=data['net_impact'],
        mode='lines+markers',
        line=dict(color='#3498DB', width=3),
        marker=dict(size=10),
        hovertemplate="<b>%{x}</b><br>" +
                     "Net Impact: $%{y:,.0f}<br>" +
                     "<extra></extra>"
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title={
            'text': "Cost-Benefit Analysis of Schedule Changes",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Cost/Benefit Category",
        yaxis_title="Amount ($)",
        height=500,
        margin=dict(l=50, r=50, t=80, b=120),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Rotate x-axis labels
    fig.update_xaxis(tickangle=45)
    
    return fig

def create_scenario_comparison_matrix(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a matrix comparing different schedule scenarios
    
    Args:
        data: DataFrame with columns ['scenario', 'metric', 'score']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample scenario data
        scenarios = ['Current', 'Option A', 'Option B', 'Option C', 'Recommended']
        metrics = ['Delay Reduction', 'Cost Efficiency', 'Passenger Satisfaction', 
                  'Operational Complexity', 'Risk Level']
        
        scenario_data = []
        for scenario in scenarios:
            for metric in metrics:
                if scenario == 'Current':
                    score = 5  # Baseline
                elif scenario == 'Recommended':
                    score = np.random.uniform(7, 9)  # Best option
                else:
                    score = np.random.uniform(4, 8)
                
                scenario_data.append({
                    'scenario': scenario,
                    'metric': metric,
                    'score': score
                })
        
        data = pd.DataFrame(scenario_data)
    
    # Pivot data for heatmap
    pivot_data = data.pivot(index='metric', columns='scenario', values='score')
    
    fig = px.imshow(
        pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale='RdYlGn',
        title="Schedule Scenario Comparison Matrix",
        labels={'x': 'Scenario', 'y': 'Evaluation Metric', 'color': 'Score (1-10)'},
        aspect='auto'
    )
    
    # Add text annotations
    for i, metric in enumerate(pivot_data.index):
        for j, scenario in enumerate(pivot_data.columns):
            score = pivot_data.iloc[i, j]
            fig.add_annotation(
                x=j, y=i,
                text=f"{score:.1f}",
                showarrow=False,
                font=dict(color='white' if score < 5 else 'black', size=12)
            )
    
    fig.update_layout(
        title={
            'text': "Schedule Scenario Comparison Matrix",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=400,
        margin=dict(l=150, r=50, t=80, b=50),
        coloraxis_colorbar=dict(
            title="Performance Score",
            tickvals=[1, 3, 5, 7, 9],
            ticktext=['Poor', 'Below Avg', 'Average', 'Good', 'Excellent']
        )
    )
    
    # Custom hover template
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>" +
                     "Scenario: %{x}<br>" +
                     "Score: %{z:.1f}/10<br>" +
                     "<extra></extra>"
    )
    
    return fig

def create_implementation_timeline(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a Gantt chart showing implementation timeline for schedule changes
    
    Args:
        data: DataFrame with columns ['task', 'start_date', 'end_date', 'status', 'department']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample implementation timeline
        base_date = datetime.now()
        
        tasks = [
            ('Stakeholder Approval', 0, 5, 'Completed', 'Management'),
            ('System Configuration', 3, 10, 'In Progress', 'IT'),
            ('Crew Schedule Updates', 7, 14, 'Planned', 'Operations'),
            ('Passenger Notifications', 10, 12, 'Planned', 'Customer Service'),
            ('Ground Operations Prep', 12, 18, 'Planned', 'Ground Ops'),
            ('Implementation', 18, 19, 'Planned', 'Operations'),
            ('Monitoring & Adjustment', 19, 30, 'Planned', 'Analytics')
        ]
        
        timeline_data = []
        for task, start_offset, end_offset, status, dept in tasks:
            timeline_data.append({
                'task': task,
                'start_date': base_date + timedelta(days=start_offset),
                'end_date': base_date + timedelta(days=end_offset),
                'status': status,
                'department': dept
            })
        
        data = pd.DataFrame(timeline_data)
    
    # Color mapping for status
    status_colors = {
        'Completed': '#2ECC71',
        'In Progress': '#F39C12',
        'Planned': '#3498DB',
        'Delayed': '#E74C3C'
    }
    
    fig = go.Figure()
    
    for i, row in data.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['start_date'], row['end_date']],
            y=[i, i],
            mode='lines',
            line=dict(
                color=status_colors.get(row['status'], '#BDC3C7'),
                width=20
            ),
            name=row['status'],
            showlegend=i == 0 or row['status'] not in [r['status'] for r in data.iloc[:i].to_dict('records')],
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                         "Department: %{customdata[1]}<br>" +
                         "Status: %{customdata[2]}<br>" +
                         "Duration: %{customdata[3]} days<br>" +
                         "<extra></extra>",
            customdata=[[
                row['task'],
                row['department'],
                row['status'],
                (row['end_date'] - row['start_date']).days
            ]] * 2
        ))
    
    # Add task labels
    for i, row in data.iterrows():
        fig.add_annotation(
            x=row['start_date'] + (row['end_date'] - row['start_date']) / 2,
            y=i,
            text=row['task'],
            showarrow=False,
            font=dict(color='white', size=10),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        )
    
    fig.update_layout(
        title={
            'text': "Schedule Change Implementation Timeline",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Date",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(data))),
            ticktext=data['task'].tolist(),
            title="Implementation Tasks"
        ),
        height=500,
        margin=dict(l=200, r=50, t=80, b=50),
        hovermode='closest'
    )
    
    # Add today's date line
    fig.add_vline(
        x=datetime.now(),
        line_dash="dot",
        line_color="red",
        annotation_text="Today",
        annotation_position="top"
    )
    
    return fig