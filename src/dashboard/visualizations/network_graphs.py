"""
Interactive flight network graphs for cascading impact analysis
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

def create_flight_network_graph(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create an interactive network graph showing flight connections and their impact scores
    
    Args:
        data: DataFrame with columns ['flight_id', 'origin', 'destination', 'impact_score', 'delay_minutes']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample flight network data
        airports = ['BOM', 'DEL', 'BLR', 'MAA', 'CCU', 'HYD', 'AMD', 'COK']
        flights_data = []
        
        # Create realistic flight connections
        for i in range(50):
            origin = np.random.choice(airports)
            destination = np.random.choice([a for a in airports if a != origin])
            
            flights_data.append({
                'flight_id': f'AI{100 + i}',
                'origin': origin,
                'destination': destination,
                'impact_score': np.random.uniform(1, 10),
                'delay_minutes': np.random.exponential(8)
            })
        
        data = pd.DataFrame(flights_data)
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes (airports)
    airports = list(set(data['origin'].tolist() + data['destination'].tolist()))
    for airport in airports:
        # Calculate airport metrics
        airport_flights = data[(data['origin'] == airport) | (data['destination'] == airport)]
        avg_impact = airport_flights['impact_score'].mean()
        total_flights = len(airport_flights)
        
        G.add_node(airport, impact=avg_impact, flights=total_flights)
    
    # Add edges (flight routes)
    for _, flight in data.iterrows():
        if G.has_edge(flight['origin'], flight['destination']):
            # Update existing edge
            G[flight['origin']][flight['destination']]['flights'] += 1
            G[flight['origin']][flight['destination']]['total_impact'] += flight['impact_score']
        else:
            # Add new edge
            G.add_edge(
                flight['origin'], 
                flight['destination'],
                flights=1,
                total_impact=flight['impact_score'],
                flight_ids=[flight['flight_id']]
            )
    
    # Calculate average impact for edges
    for edge in G.edges():
        edge_data = G[edge[0]][edge[1]]
        edge_data['avg_impact'] = edge_data['total_impact'] / edge_data['flights']
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Extract node and edge information
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_info = G.nodes[node]
        node_text.append(f"{node}<br>Flights: {node_info['flights']}<br>Avg Impact: {node_info['impact']:.1f}")
        node_size.append(max(20, node_info['flights'] * 2))  # Size based on flight count
        node_color.append(node_info['impact'])
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        edge_data = G[edge[0]][edge[1]]
        edge_info.append(f"{edge[0]} ↔ {edge[1]}<br>Flights: {edge_data['flights']}<br>Avg Impact: {edge_data['avg_impact']:.1f}")
    
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        text=[node for node in G.nodes()],
        textposition="middle center",
        textfont=dict(size=12, color='white'),
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='RdYlGn_r',
            colorbar=dict(
                title="Impact Score",
                titleside="right",
                tickmode="linear",
                tick0=0,
                dtick=2
            ),
            line=dict(width=2, color='white'),
            showscale=True
        ),
        showlegend=False
    ))
    
    fig.update_layout(
        title={
            'text': "Flight Network - Cascading Impact Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=80),
        annotations=[
            dict(
                text="Node size = Number of flights<br>Node color = Impact score<br>Click and drag to explore",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

def create_critical_flights_chart(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a chart showing the most critical flights by cascading impact
    
    Args:
        data: DataFrame with columns ['flight_id', 'route', 'impact_score', 'affected_flights', 'delay_risk']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample critical flights data
        routes = ['BOM-DEL', 'DEL-BOM', 'BOM-BLR', 'DEL-BLR', 'BLR-MAA', 'DEL-MAA', 'BOM-MAA', 'CCU-DEL']
        
        critical_flights = []
        for i in range(20):
            route = np.random.choice(routes)
            impact_score = np.random.uniform(5, 10)
            affected_flights = int(np.random.poisson(impact_score))
            delay_risk = np.random.uniform(0.3, 0.9)
            
            critical_flights.append({
                'flight_id': f'AI{200 + i}',
                'route': route,
                'impact_score': impact_score,
                'affected_flights': affected_flights,
                'delay_risk': delay_risk
            })
        
        data = pd.DataFrame(critical_flights)
    
    # Sort by impact score
    data = data.sort_values('impact_score', ascending=True)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Color based on delay risk
    colors = ['#2ECC71' if risk < 0.5 else '#F39C12' if risk < 0.7 else '#E74C3C' 
              for risk in data['delay_risk']]
    
    fig.add_trace(go.Bar(
        y=data['flight_id'],
        x=data['impact_score'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=1)
        ),
        hovertemplate="<b>%{y}</b><br>" +
                     "Route: %{customdata[0]}<br>" +
                     "Impact Score: %{x:.1f}<br>" +
                     "Affected Flights: %{customdata[1]}<br>" +
                     "Delay Risk: %{customdata[2]:.1%}<br>" +
                     "<extra></extra>",
        customdata=list(zip(data['route'], data['affected_flights'], data['delay_risk']))
    ))
    
    fig.update_layout(
        title={
            'text': "Critical Flights - Cascading Impact Ranking",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Impact Score (1-10)",
        yaxis_title="Flight ID",
        height=600,
        margin=dict(l=100, r=50, t=80, b=50),
        showlegend=False
    )
    
    # Add risk level annotations
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text="🟢 Low Risk (< 50%)<br>🟡 Medium Risk (50-70%)<br>🔴 High Risk (> 70%)",
        showarrow=False,
        align='left',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='gray',
        borderwidth=1
    )
    
    return fig

def create_delay_propagation_chart(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a chart showing how delays propagate through the network over time
    
    Args:
        data: DataFrame with columns ['time_step', 'flight_id', 'delay_minutes', 'propagation_level']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample delay propagation data
        time_steps = list(range(0, 120, 10))  # 2 hours in 10-minute intervals
        flights = ['AI101', 'AI102', 'AI103', 'AI104', 'AI105', 'AI106']
        
        propagation_data = []
        
        # Initial delay in AI101
        initial_delay = 30
        
        for t in time_steps:
            for i, flight in enumerate(flights):
                if flight == 'AI101':
                    # Original delayed flight
                    delay = initial_delay if t == 0 else max(0, initial_delay - t/4)
                    level = 0
                elif i == 1 and t >= 20:
                    # First propagation
                    delay = max(0, 15 - (t-20)/6)
                    level = 1
                elif i == 2 and t >= 40:
                    # Second propagation
                    delay = max(0, 10 - (t-40)/8)
                    level = 2
                elif i >= 3 and t >= 60:
                    # Further propagations
                    delay = max(0, 5 - (t-60)/10)
                    level = 3
                else:
                    delay = 0
                    level = 0
                
                if delay > 0:
                    propagation_data.append({
                        'time_step': t,
                        'flight_id': flight,
                        'delay_minutes': delay,
                        'propagation_level': level
                    })
        
        data = pd.DataFrame(propagation_data)
    
    fig = go.Figure()
    
    # Color scheme for propagation levels
    colors = {0: '#E74C3C', 1: '#F39C12', 2: '#F1C40F', 3: '#3498DB'}
    level_names = {0: 'Original Delay', 1: '1st Level', 2: '2nd Level', 3: '3rd Level'}
    
    # Add traces for each flight
    for flight in data['flight_id'].unique():
        flight_data = data[data['flight_id'] == flight].sort_values('time_step')
        
        if not flight_data.empty:
            level = flight_data['propagation_level'].iloc[0]
            
            fig.add_trace(go.Scatter(
                x=flight_data['time_step'],
                y=flight_data['delay_minutes'],
                mode='lines+markers',
                name=f"{flight} ({level_names[level]})",
                line=dict(color=colors[level], width=3),
                marker=dict(size=8),
                hovertemplate="<b>%{fullData.name}</b><br>" +
                             "Time: %{x} min<br>" +
                             "Delay: %{y:.1f} min<br>" +
                             "<extra></extra>"
            ))
    
    fig.update_layout(
        title={
            'text': "Delay Propagation Through Flight Network",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Time (minutes from initial delay)",
        yaxis_title="Delay (minutes)",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Add annotations for key events
    fig.add_annotation(
        x=0, y=30,
        text="Initial Delay<br>AI101",
        showarrow=True,
        arrowhead=2,
        arrowcolor="red",
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.add_annotation(
        x=20, y=15,
        text="1st Propagation<br>AI102",
        showarrow=True,
        arrowhead=2,
        arrowcolor="orange",
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    return fig

def create_network_resilience_chart(data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a chart showing network resilience metrics
    
    Args:
        data: DataFrame with columns ['scenario', 'affected_flights', 'total_delay', 'recovery_time']
    
    Returns:
        Plotly figure object
    """
    
    if data is None:
        # Generate sample resilience data
        scenarios = [
            'Single Flight Delay (15 min)',
            'Hub Airport Closure (30 min)',
            'Weather Disruption (60 min)',
            'Technical Issue (45 min)',
            'Air Traffic Control (90 min)'
        ]
        
        resilience_data = []
        for scenario in scenarios:
            affected_flights = np.random.randint(5, 50)
            total_delay = affected_flights * np.random.uniform(10, 30)
            recovery_time = np.random.uniform(30, 180)
            
            resilience_data.append({
                'scenario': scenario,
                'affected_flights': affected_flights,
                'total_delay': total_delay,
                'recovery_time': recovery_time
            })
        
        data = pd.DataFrame(resilience_data)
    
    # Create subplot with secondary y-axis
    fig = go.Figure()
    
    # Add affected flights bars
    fig.add_trace(go.Bar(
        name='Affected Flights',
        x=data['scenario'],
        y=data['affected_flights'],
        yaxis='y',
        marker_color='#3498DB',
        hovertemplate="<b>%{x}</b><br>" +
                     "Affected Flights: %{y}<br>" +
                     "<extra></extra>"
    ))
    
    # Add recovery time line
    fig.add_trace(go.Scatter(
        name='Recovery Time',
        x=data['scenario'],
        y=data['recovery_time'],
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#E74C3C', width=3),
        marker=dict(size=10),
        hovertemplate="<b>%{x}</b><br>" +
                     "Recovery Time: %{y:.0f} min<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            'text': "Network Resilience Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Disruption Scenario",
        yaxis=dict(
            title="Number of Affected Flights",
            side="left"
        ),
        yaxis2=dict(
            title="Recovery Time (minutes)",
            side="right",
            overlaying="y"
        ),
        height=500,
        margin=dict(l=50, r=80, t=80, b=120),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxis(tickangle=45)
    
    return fig