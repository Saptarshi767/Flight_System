"""
Cascading Impact Analyzer for Critical Flight Identification

This module implements flight network graph modeling using NetworkX to isolate flights
with the biggest cascading impact. Provides critical flight identification and ranking.
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

from ..data.models import FlightData, AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class CriticalFlight:
    """Represents a critical flight with impact metrics."""
    flight_id: str
    airline: str
    flight_number: str
    origin_airport: str
    destination_airport: str
    scheduled_departure: datetime
    network_impact_score: float
    cascading_delay_potential: float
    downstream_flights_affected: int
    criticality_rank: int
    impact_category: str  # "Critical", "High", "Medium", "Low"


@dataclass
class DelayPropagation:
    """Represents delay propagation through the network."""
    source_flight: str
    affected_flights: List[str]
    propagation_path: List[str]
    total_delay_minutes: float
    propagation_depth: int


@dataclass
class NetworkDisruption:
    """Represents a network disruption scenario."""
    disrupted_flight: str
    disruption_type: str  # "delay", "cancellation", "diversion"
    disruption_magnitude: float  # minutes for delay, 1.0 for cancellation
    affected_flights: List[str]
    total_impact_score: float
    recovery_time_hours: float


class CascadingImpactAnalyzer:
    """
    Cascading impact analyzer for critical flight identification.
    
    This class implements EXPECTATION 5: flight network graph modeling using NetworkX
    to isolate flights with biggest cascading impact and identify critical flights.
    """
    
    def __init__(self, min_turnaround_minutes: int = 45, max_crew_duty_hours: int = 14):
        """
        Initialize the cascading impact analyzer.
        
        Args:
            min_turnaround_minutes: Minimum aircraft turnaround time
            max_crew_duty_hours: Maximum crew duty hours
        """
        self.min_turnaround_minutes = min_turnaround_minutes
        self.max_crew_duty_hours = max_crew_duty_hours
        self.flight_network = nx.DiGraph()
        self.impact_scores = {}
        self.critical_flights = []
    
    def analyze_network_impact(self, flight_data: List[Dict], airport_code: str,
                             network_depth: int = 3) -> Dict:
        """
        Simplified network impact analysis for API usage.
        
        Args:
            flight_data: List of flight data dictionaries
            airport_code: Airport code to analyze
            network_depth: Depth of network analysis
            
        Returns:
            Dictionary with cascading impact analysis results
        """
        try:
            if not flight_data:
                return {
                    "airport_code": airport_code,
                    "analysis_period": {"start": None, "end": None},
                    "critical_flights": [],
                    "network_impact_scores": {},
                    "delay_propagation_paths": [],
                    "priority_recommendations": []
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(flight_data)
            
            # Build flight network based on aircraft rotations and crew connections
            flight_network = nx.DiGraph()
            
            # Add flights as nodes
            for _, flight in df.iterrows():
                flight_id = flight.get('flight_id')
                if flight_id:
                    flight_network.add_node(flight_id, **flight.to_dict())
            
            # Create connections based on aircraft turnaround
            # Group by aircraft type and airline for simplified analysis
            aircraft_groups = df.groupby(['airline_code', 'aircraft_type'])
            
            connections_created = 0
            for (airline, aircraft), group in aircraft_groups:
                # Sort by departure time
                sorted_flights = group.sort_values('scheduled_departure')
                
                # Connect consecutive flights (simplified aircraft rotation)
                for i in range(len(sorted_flights) - 1):
                    current_flight = sorted_flights.iloc[i]
                    next_flight = sorted_flights.iloc[i + 1]
                    
                    # Check if turnaround time is reasonable (2-8 hours)
                    current_arrival = pd.to_datetime(current_flight.get('scheduled_arrival', 
                                                   current_flight['scheduled_departure']) + pd.Timedelta(hours=2))
                    next_departure = pd.to_datetime(next_flight['scheduled_departure'])
                    
                    turnaround_hours = (next_departure - current_arrival).total_seconds() / 3600
                    
                    if 1 <= turnaround_hours <= 8:  # Reasonable turnaround
                        flight_network.add_edge(
                            current_flight['flight_id'],
                            next_flight['flight_id'],
                            connection_type='aircraft_rotation',
                            turnaround_hours=turnaround_hours
                        )
                        connections_created += 1
            
            # Calculate network impact scores
            impact_scores = {}
            critical_flights = []
            
            for flight_id in flight_network.nodes():
                # Calculate impact based on out-degree (flights this affects)
                out_degree = flight_network.out_degree(flight_id)
                
                # Calculate downstream impact using BFS
                downstream_flights = set()
                if out_degree > 0:
                    for successor in nx.bfs_tree(flight_network, flight_id, depth_limit=network_depth):
                        if successor != flight_id:
                            downstream_flights.add(successor)
                
                # Impact score based on number of affected flights
                impact_score = len(downstream_flights) * 10 + out_degree * 5
                impact_scores[flight_id] = impact_score
                
                # Identify critical flights (high impact)
                if impact_score > 20:
                    flight_info = flight_network.nodes[flight_id]
                    critical_flights.append({
                        "flight_id": flight_id,
                        "flight_number": flight_info.get('flight_number'),
                        "airline": flight_info.get('airline_code'),
                        "origin_airport": flight_info.get('origin_airport'),
                        "destination_airport": flight_info.get('destination_airport'),
                        "scheduled_departure": flight_info.get('scheduled_departure'),
                        "network_impact_score": impact_score,
                        "downstream_flights_affected": len(downstream_flights),
                        "criticality_rank": 1,  # Will be ranked later
                        "impact_category": "Critical" if impact_score > 50 else "High"
                    })
            
            # Sort critical flights by impact score
            critical_flights.sort(key=lambda x: x['network_impact_score'], reverse=True)
            
            # Assign ranks
            for i, flight in enumerate(critical_flights):
                flight['criticality_rank'] = i + 1
            
            # Generate delay propagation paths for top critical flights
            delay_propagation_paths = []
            for flight in critical_flights[:5]:  # Top 5 critical flights
                flight_id = flight['flight_id']
                if flight_id in flight_network:
                    # Find paths from this flight
                    paths = []
                    for target in flight_network.nodes():
                        if target != flight_id:
                            try:
                                path = nx.shortest_path(flight_network, flight_id, target)
                                if len(path) > 1:  # Has connections
                                    paths.append(path)
                            except nx.NetworkXNoPath:
                                continue
                    
                    if paths:
                        # Take the longest path as example
                        longest_path = max(paths, key=len)
                        delay_propagation_paths.append({
                            "source_flight": flight_id,
                            "affected_flights": longest_path[1:],
                            "propagation_path": longest_path,
                            "total_delay_minutes": len(longest_path) * 15,  # Estimated
                            "propagation_depth": len(longest_path) - 1
                        })
            
            # Generate recommendations
            recommendations = []
            if len(critical_flights) > 0:
                recommendations.append(f"Monitor {len(critical_flights)} critical flights for on-time performance")
            if connections_created > 0:
                recommendations.append(f"Network analysis identified {connections_created} aircraft rotation connections")
            if len(delay_propagation_paths) > 0:
                recommendations.append("Focus on preventing delays in flights with high cascading impact")
            
            # Analysis period
            dates = pd.to_datetime(df['scheduled_departure'])
            analysis_period = {
                "start": dates.min().isoformat() if not dates.empty else None,
                "end": dates.max().isoformat() if not dates.empty else None
            }
            
            return {
                "airport_code": airport_code,
                "analysis_period": analysis_period,
                "critical_flights": critical_flights[:10],  # Top 10
                "network_impact_scores": {k: float(v) for k, v in impact_scores.items()},
                "delay_propagation_paths": delay_propagation_paths,
                "priority_recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in cascading impact analysis: {str(e)}")
            return {
                "airport_code": airport_code,
                "error": str(e),
                "priority_recommendations": ["Analysis failed due to data processing error"]
            }
        
    def analyze_cascading_impact(self, flight_data: List[FlightData], 
                                airport_code: str) -> AnalysisResult:
        """
        Analyze cascading impact and identify critical flights.
        
        Args:
            flight_data: List of flight data objects
            airport_code: Airport code to analyze
            
        Returns:
            AnalysisResult with cascading impact analysis and critical flights
        """
        logger.info(f"Analyzing cascading impact for airport {airport_code}")
        
        # Convert to DataFrame for analysis
        df = self._flights_to_dataframe(flight_data)
        
        if df.empty:
            logger.warning("No flight data provided for analysis")
            return self._empty_analysis_result(airport_code)
        
        # Filter for specific airport
        airport_flights = df[
            (df['origin_airport'] == airport_code) | 
            (df['destination_airport'] == airport_code)
        ].copy()
        
        if airport_flights.empty:
            logger.warning(f"No flight data found for airport {airport_code}")
            return self._empty_analysis_result(airport_code)
        
        # Build comprehensive flight network
        self._build_flight_network(airport_flights)
        
        # Calculate network impact scores
        network_scores = self._calculate_network_impact_scores()
        
        # Identify critical flights
        critical_flights = self._identify_critical_flights(airport_flights, network_scores)
        
        # Analyze delay propagation patterns
        propagation_analysis = self._analyze_delay_propagation(airport_flights)
        
        # Perform domino effect analysis
        domino_analysis = self._analyze_domino_effects(airport_flights)
        
        # Generate network disruption scenarios
        disruption_scenarios = self._generate_disruption_scenarios(critical_flights[:5])  # Top 5 critical
        
        # Generate recommendations
        recommendations = self._generate_cascading_recommendations(
            critical_flights, propagation_analysis, domino_analysis, disruption_scenarios
        )
        
        # Compile results
        metrics = {
            'critical_flights': [
                {
                    'flight_id': cf.flight_id,
                    'flight_number': cf.flight_number,
                    'airline': cf.airline,
                    'route': f"{cf.origin_airport}-{cf.destination_airport}",
                    'network_impact_score': cf.network_impact_score,
                    'cascading_delay_potential': cf.cascading_delay_potential,
                    'downstream_flights_affected': cf.downstream_flights_affected,
                    'criticality_rank': cf.criticality_rank,
                    'impact_category': cf.impact_category
                }
                for cf in critical_flights[:20]  # Top 20 critical flights
            ],
            'network_metrics': self._get_network_analysis_metrics(),
            'delay_propagation_analysis': {
                'total_propagation_paths': len(propagation_analysis),
                'average_propagation_depth': np.mean([p.propagation_depth for p in propagation_analysis]) if propagation_analysis else 0,
                'max_affected_flights': max([p.total_delay_minutes for p in propagation_analysis]) if propagation_analysis else 0,
                'propagation_patterns': [
                    {
                        'source_flight': p.source_flight,
                        'affected_count': len(p.affected_flights),
                        'total_delay': p.total_delay_minutes,
                        'depth': p.propagation_depth
                    }
                    for p in propagation_analysis[:10]  # Top 10 propagation patterns
                ]
            },
            'domino_effect_analysis': domino_analysis,
            'disruption_scenarios': [
                {
                    'disrupted_flight': ds.disrupted_flight,
                    'disruption_type': ds.disruption_type,
                    'affected_flights_count': len(ds.affected_flights),
                    'total_impact_score': ds.total_impact_score,
                    'recovery_time_hours': ds.recovery_time_hours
                }
                for ds in disruption_scenarios
            ],
            'total_flights_analyzed': len(airport_flights)
        }
        
        return AnalysisResult(
            analysis_type="cascading_impact_analysis",
            airport_code=airport_code,
            metrics=metrics,
            recommendations=recommendations,
            confidence_score=self._calculate_confidence_score(airport_flights),
            data_sources=["flight_data", "network_analysis"]
        )
    
    def identify_most_critical_flights(self, flight_data: List[FlightData], 
                                     airport_code: str, top_n: int = 10) -> List[CriticalFlight]:
        """
        Identify the most critical flights based on cascading impact.
        
        Args:
            flight_data: List of flight data objects
            airport_code: Airport code to analyze
            top_n: Number of top critical flights to return
            
        Returns:
            List of most critical flights
        """
        df = self._flights_to_dataframe(flight_data)
        airport_flights = df[
            (df['origin_airport'] == airport_code) | 
            (df['destination_airport'] == airport_code)
        ].copy()
        
        if airport_flights.empty:
            return []
        
        self._build_flight_network(airport_flights)
        network_scores = self._calculate_network_impact_scores()
        critical_flights = self._identify_critical_flights(airport_flights, network_scores)
        
        return critical_flights[:top_n]
    
    def trace_delay_propagation(self, source_flight_id: str, 
                               delay_minutes: float,
                               flight_data: List[FlightData]) -> DelayPropagation:
        """
        Trace how a delay propagates through the flight network.
        
        Args:
            source_flight_id: ID of the flight with initial delay
            delay_minutes: Initial delay in minutes
            flight_data: Flight data for network context
            
        Returns:
            DelayPropagation object with propagation details
        """
        df = self._flights_to_dataframe(flight_data)
        self._build_flight_network(df)
        
        if source_flight_id not in self.flight_network:
            return DelayPropagation(
                source_flight=source_flight_id,
                affected_flights=[],
                propagation_path=[],
                total_delay_minutes=0.0,
                propagation_depth=0
            )
        
        affected_flights = []
        propagation_path = [source_flight_id]
        total_delay = delay_minutes
        current_delay = delay_minutes
        
        # Trace propagation through network
        visited = set()
        queue = [(source_flight_id, current_delay, 0)]  # (flight_id, delay, depth)
        max_depth = 0
        
        while queue:
            current_flight, current_delay, depth = queue.pop(0)
            
            if current_flight in visited or current_delay < 5:  # Stop if delay < 5 minutes
                continue
            
            visited.add(current_flight)
            max_depth = max(max_depth, depth)
            
            # Get downstream flights
            successors = list(self.flight_network.successors(current_flight))
            
            for successor in successors:
                if successor not in visited:
                    edge_data = self.flight_network.get_edge_data(current_flight, successor)
                    
                    # Calculate propagated delay based on connection type
                    if edge_data:
                        connection_type = edge_data.get('connection_type', 'unknown')
                        propagation_factor = self._get_propagation_factor(connection_type)
                        
                        propagated_delay = current_delay * propagation_factor
                        
                        if propagated_delay >= 5:  # Only propagate significant delays
                            affected_flights.append(successor)
                            propagation_path.append(successor)
                            total_delay += propagated_delay
                            queue.append((successor, propagated_delay, depth + 1))
        
        return DelayPropagation(
            source_flight=source_flight_id,
            affected_flights=affected_flights,
            propagation_path=propagation_path,
            total_delay_minutes=total_delay,
            propagation_depth=max_depth
        )
    
    def simulate_network_disruption(self, disrupted_flight_id: str,
                                  disruption_type: str,
                                  disruption_magnitude: float,
                                  flight_data: List[FlightData]) -> NetworkDisruption:
        """
        Simulate the impact of a network disruption.
        
        Args:
            disrupted_flight_id: ID of the disrupted flight
            disruption_type: Type of disruption ("delay", "cancellation", "diversion")
            disruption_magnitude: Magnitude of disruption
            flight_data: Flight data for network context
            
        Returns:
            NetworkDisruption object with simulation results
        """
        df = self._flights_to_dataframe(flight_data)
        self._build_flight_network(df)
        
        affected_flights = []
        total_impact = 0.0
        recovery_time = 0.0
        
        if disrupted_flight_id in self.flight_network:
            if disruption_type == "cancellation":
                # Cancellation affects all downstream flights
                affected_flights = list(nx.descendants(self.flight_network, disrupted_flight_id))
                total_impact = len(affected_flights) * 60  # 60 minutes impact per affected flight
                recovery_time = 4.0  # 4 hours to recover from cancellation
                
            elif disruption_type == "delay":
                # Delay propagates through network
                propagation = self.trace_delay_propagation(
                    disrupted_flight_id, disruption_magnitude, flight_data
                )
                affected_flights = propagation.affected_flights
                total_impact = propagation.total_delay_minutes
                recovery_time = disruption_magnitude / 60  # Recovery time proportional to delay
                
            elif disruption_type == "diversion":
                # Diversion affects aircraft rotation
                aircraft_flights = self._get_aircraft_rotation_flights(disrupted_flight_id)
                affected_flights = aircraft_flights
                total_impact = len(affected_flights) * 45  # 45 minutes impact per flight
                recovery_time = 6.0  # 6 hours to recover from diversion
        
        return NetworkDisruption(
            disrupted_flight=disrupted_flight_id,
            disruption_type=disruption_type,
            disruption_magnitude=disruption_magnitude,
            affected_flights=affected_flights,
            total_impact_score=total_impact,
            recovery_time_hours=recovery_time
        )
    
    def _flights_to_dataframe(self, flight_data: List[FlightData]) -> pd.DataFrame:
        """Convert flight data objects to pandas DataFrame."""
        data = []
        for flight in flight_data:
            # Calculate delays if actual times are available
            departure_delay = 0
            arrival_delay = 0
            
            if flight.actual_departure and flight.scheduled_departure:
                departure_delay = (flight.actual_departure - flight.scheduled_departure).total_seconds() / 60
            
            if flight.actual_arrival and flight.scheduled_arrival:
                arrival_delay = (flight.actual_arrival - flight.scheduled_arrival).total_seconds() / 60
            
            data.append({
                'flight_id': flight.flight_id,
                'airline': flight.airline,
                'flight_number': flight.flight_number,
                'aircraft_type': flight.aircraft_type or 'Unknown',
                'origin_airport': flight.origin_airport,
                'destination_airport': flight.destination_airport,
                'scheduled_departure': flight.scheduled_departure,
                'actual_departure': flight.actual_departure,
                'scheduled_arrival': flight.scheduled_arrival,
                'actual_arrival': flight.actual_arrival,
                'departure_delay_minutes': departure_delay,
                'arrival_delay_minutes': arrival_delay,
                'delay_minutes': max(departure_delay, arrival_delay),
                'passenger_count': flight.passenger_count or 0
            })
        
        return pd.DataFrame(data)
    
    def _build_flight_network(self, df: pd.DataFrame):
        """Build comprehensive flight network graph using NetworkX."""
        self.flight_network.clear()
        
        # Add nodes for each flight
        for _, flight in df.iterrows():
            self.flight_network.add_node(
                flight['flight_id'],
                airline=flight['airline'],
                flight_number=flight['flight_number'],
                aircraft_type=flight['aircraft_type'],
                departure_time=flight['scheduled_departure'],
                arrival_time=flight['scheduled_arrival'],
                origin=flight['origin_airport'],
                destination=flight['destination_airport'],
                delay=flight['delay_minutes'],
                passenger_count=flight['passenger_count']
            )
        
        # Add aircraft rotation edges
        self._add_aircraft_rotation_edges(df)
        
        # Add crew rotation edges
        self._add_crew_rotation_edges(df)
        
        # Add passenger connection edges
        self._add_passenger_connection_edges(df)
        
        # Add gate/slot dependency edges
        self._add_gate_slot_dependency_edges(df)
    
    def _add_aircraft_rotation_edges(self, df: pd.DataFrame):
        """Add edges for aircraft rotations."""
        if df.empty or 'airline' not in df.columns or 'aircraft_type' not in df.columns:
            return
            
        # Group by airline and aircraft type to identify potential rotations
        aircraft_groups = df.groupby(['airline', 'aircraft_type'])
        
        for (airline, aircraft), group in aircraft_groups:
            if aircraft == 'Unknown':
                continue
                
            # Sort by departure time
            sorted_flights = group.sort_values('scheduled_departure')
            
            # Connect sequential flights that could use the same aircraft
            for i in range(len(sorted_flights) - 1):
                current_flight = sorted_flights.iloc[i]
                next_flight = sorted_flights.iloc[i + 1]
                
                # Check if aircraft rotation is possible
                if (current_flight['destination_airport'] == next_flight['origin_airport']):
                    turnaround_time = (next_flight['scheduled_departure'] - 
                                     current_flight['scheduled_arrival']).total_seconds() / 60
                    
                    if turnaround_time >= self.min_turnaround_minutes:
                        self.flight_network.add_edge(
                            current_flight['flight_id'],
                            next_flight['flight_id'],
                            connection_type='aircraft_rotation',
                            turnaround_time=turnaround_time,
                            weight=1.0,
                            criticality=0.9  # High criticality for aircraft rotations
                        )
    
    def _add_crew_rotation_edges(self, df: pd.DataFrame):
        """Add edges for crew rotations."""
        if df.empty or 'airline' not in df.columns:
            return
            
        # Group by airline for potential crew rotations
        airline_groups = df.groupby('airline')
        
        for airline, group in airline_groups:
            sorted_flights = group.sort_values('scheduled_departure')
            
            # Connect flights that could share crew
            for i in range(len(sorted_flights) - 1):
                current_flight = sorted_flights.iloc[i]
                next_flight = sorted_flights.iloc[i + 1]
                
                # Check crew duty time constraints
                duty_time = (next_flight['scheduled_departure'] - 
                           current_flight['scheduled_departure']).total_seconds() / 3600
                
                if (duty_time <= self.max_crew_duty_hours and 
                    current_flight['destination_airport'] == next_flight['origin_airport']):
                    
                    rest_time = (next_flight['scheduled_departure'] - 
                               current_flight['scheduled_arrival']).total_seconds() / 60
                    
                    if 30 <= rest_time <= 480:  # 30 minutes to 8 hours rest
                        if not self.flight_network.has_edge(current_flight['flight_id'], 
                                                           next_flight['flight_id']):
                            self.flight_network.add_edge(
                                current_flight['flight_id'],
                                next_flight['flight_id'],
                                connection_type='crew_rotation',
                                rest_time=rest_time,
                                weight=0.7,
                                criticality=0.6  # Medium criticality for crew rotations
                            )
    
    def _add_passenger_connection_edges(self, df: pd.DataFrame):
        """Add edges for passenger connections."""
        if df.empty or 'origin_airport' not in df.columns or 'destination_airport' not in df.columns:
            return
            
        # Group by airport to find potential connections
        airports = set(df['origin_airport'].unique()) | set(df['destination_airport'].unique())
        
        for airport in airports:
            # Find arriving and departing flights at this airport
            arriving_flights = df[df['destination_airport'] == airport].sort_values('scheduled_arrival')
            departing_flights = df[df['origin_airport'] == airport].sort_values('scheduled_departure')
            
            # Connect arriving flights to departing flights (passenger connections)
            for _, arriving in arriving_flights.iterrows():
                for _, departing in departing_flights.iterrows():
                    connection_time = (departing['scheduled_departure'] - 
                                     arriving['scheduled_arrival']).total_seconds() / 60
                    
                    # Reasonable connection time for passengers (45 minutes to 6 hours)
                    if 45 <= connection_time <= 360:
                        if not self.flight_network.has_edge(arriving['flight_id'], 
                                                           departing['flight_id']):
                            self.flight_network.add_edge(
                                arriving['flight_id'],
                                departing['flight_id'],
                                connection_type='passenger_connection',
                                connection_time=connection_time,
                                weight=0.4,
                                criticality=0.3  # Lower criticality for passenger connections
                            )
    
    def _add_gate_slot_dependency_edges(self, df: pd.DataFrame):
        """Add edges for gate and slot dependencies."""
        if df.empty or 'origin_airport' not in df.columns or 'destination_airport' not in df.columns:
            return
            
        # Group by airport and time to identify potential conflicts
        airports = set(df['origin_airport'].unique()) | set(df['destination_airport'].unique())
        
        for airport in airports:
            airport_flights = df[
                (df['origin_airport'] == airport) | 
                (df['destination_airport'] == airport)
            ].sort_values('scheduled_departure')
            
            # Connect flights that might compete for same resources
            for i in range(len(airport_flights) - 1):
                current_flight = airport_flights.iloc[i]
                next_flight = airport_flights.iloc[i + 1]
                
                time_diff = (next_flight['scheduled_departure'] - 
                           current_flight['scheduled_departure']).total_seconds() / 60
                
                # Flights within 30 minutes might compete for gates/slots
                if 0 < time_diff <= 30:
                    if not self.flight_network.has_edge(current_flight['flight_id'], 
                                                       next_flight['flight_id']):
                        self.flight_network.add_edge(
                            current_flight['flight_id'],
                            next_flight['flight_id'],
                            connection_type='resource_dependency',
                            time_separation=time_diff,
                            weight=0.5,
                            criticality=0.4  # Medium criticality for resource dependencies
                        )
    
    def _calculate_network_impact_scores(self) -> Dict[str, float]:
        """Calculate network impact scores for all flights using NetworkX metrics."""
        impact_scores = {}
        
        if self.flight_network.number_of_nodes() == 0:
            return impact_scores
        
        # Calculate various centrality measures
        try:
            betweenness_centrality = nx.betweenness_centrality(self.flight_network, weight='weight')
            closeness_centrality = nx.closeness_centrality(self.flight_network)
            eigenvector_centrality = nx.eigenvector_centrality(self.flight_network, weight='weight', max_iter=1000)
            pagerank = nx.pagerank(self.flight_network, weight='weight')
        except:
            # Fallback to simpler metrics if complex ones fail
            betweenness_centrality = {node: 0 for node in self.flight_network.nodes()}
            closeness_centrality = {node: 0 for node in self.flight_network.nodes()}
            eigenvector_centrality = {node: 0 for node in self.flight_network.nodes()}
            pagerank = {node: 1/self.flight_network.number_of_nodes() for node in self.flight_network.nodes()}
        
        # Calculate out-degree (number of downstream flights)
        out_degrees = dict(self.flight_network.out_degree())
        
        # Calculate weighted impact scores
        for flight_id in self.flight_network.nodes():
            # Combine different centrality measures
            impact_score = (
                betweenness_centrality.get(flight_id, 0) * 0.3 +
                closeness_centrality.get(flight_id, 0) * 0.2 +
                eigenvector_centrality.get(flight_id, 0) * 0.2 +
                pagerank.get(flight_id, 0) * 0.2 +
                (out_degrees.get(flight_id, 0) / max(out_degrees.values()) if out_degrees else 0) * 0.1
            )
            
            # Adjust for flight characteristics
            node_data = self.flight_network.nodes[flight_id]
            passenger_factor = min(2.0, (node_data.get('passenger_count', 150) / 150))
            delay_factor = min(2.0, (node_data.get('delay', 0) / 60 + 1))
            
            impact_scores[flight_id] = impact_score * passenger_factor * delay_factor
        
        return impact_scores
    
    def _identify_critical_flights(self, df: pd.DataFrame, 
                                 network_scores: Dict[str, float]) -> List[CriticalFlight]:
        """Identify critical flights based on network impact scores."""
        critical_flights = []
        
        for _, flight in df.iterrows():
            flight_id = flight['flight_id']
            network_score = network_scores.get(flight_id, 0.0)
            
            # Calculate cascading delay potential
            downstream_flights = list(self.flight_network.successors(flight_id)) if flight_id in self.flight_network else []
            cascading_potential = len(downstream_flights) * network_score * 10  # Scale factor
            
            # Determine impact category
            if network_score > 0.8:
                impact_category = "Critical"
            elif network_score > 0.6:
                impact_category = "High"
            elif network_score > 0.3:
                impact_category = "Medium"
            else:
                impact_category = "Low"
            
            critical_flight = CriticalFlight(
                flight_id=flight_id,
                airline=flight['airline'],
                flight_number=flight['flight_number'],
                origin_airport=flight['origin_airport'],
                destination_airport=flight['destination_airport'],
                scheduled_departure=flight['scheduled_departure'],
                network_impact_score=network_score,
                cascading_delay_potential=cascading_potential,
                downstream_flights_affected=len(downstream_flights),
                criticality_rank=0,  # Will be set after sorting
                impact_category=impact_category
            )
            
            critical_flights.append(critical_flight)
        
        # Sort by network impact score and assign ranks
        critical_flights.sort(key=lambda x: x.network_impact_score, reverse=True)
        for i, flight in enumerate(critical_flights):
            flight.criticality_rank = i + 1
        
        return critical_flights
    
    def _analyze_delay_propagation(self, df: pd.DataFrame) -> List[DelayPropagation]:
        """Analyze delay propagation patterns in the network."""
        propagation_patterns = []
        
        # Find flights with significant delays
        delayed_flights = df[df['delay_minutes'] > 15]  # More than 15 minutes delay
        
        for _, delayed_flight in delayed_flights.iterrows():
            propagation = self._trace_delay_propagation_from_df(
                delayed_flight['flight_id'],
                delayed_flight['delay_minutes'],
                df
            )
            
            if len(propagation.affected_flights) > 0:
                propagation_patterns.append(propagation)
        
        # Sort by total delay impact
        propagation_patterns.sort(key=lambda x: x.total_delay_minutes, reverse=True)
        
        return propagation_patterns
    
    def _trace_delay_propagation_from_df(self, source_flight_id: str, 
                                        delay_minutes: float,
                                        df: pd.DataFrame) -> DelayPropagation:
        """Trace delay propagation using DataFrame instead of FlightData objects."""
        if source_flight_id not in self.flight_network:
            return DelayPropagation(
                source_flight=source_flight_id,
                affected_flights=[],
                propagation_path=[],
                total_delay_minutes=0.0,
                propagation_depth=0
            )
        
        affected_flights = []
        propagation_path = [source_flight_id]
        total_delay = delay_minutes
        current_delay = delay_minutes
        
        # Trace propagation through network
        visited = set()
        queue = [(source_flight_id, current_delay, 0)]  # (flight_id, delay, depth)
        max_depth = 0
        
        while queue:
            current_flight, current_delay, depth = queue.pop(0)
            
            if current_flight in visited or current_delay < 5:  # Stop if delay < 5 minutes
                continue
            
            visited.add(current_flight)
            max_depth = max(max_depth, depth)
            
            # Get downstream flights
            successors = list(self.flight_network.successors(current_flight))
            
            for successor in successors:
                if successor not in visited:
                    edge_data = self.flight_network.get_edge_data(current_flight, successor)
                    
                    # Calculate propagated delay based on connection type
                    if edge_data:
                        connection_type = edge_data.get('connection_type', 'unknown')
                        propagation_factor = self._get_propagation_factor(connection_type)
                        
                        propagated_delay = current_delay * propagation_factor
                        
                        if propagated_delay >= 5:  # Only propagate significant delays
                            affected_flights.append(successor)
                            propagation_path.append(successor)
                            total_delay += propagated_delay
                            queue.append((successor, propagated_delay, depth + 1))
        
        return DelayPropagation(
            source_flight=source_flight_id,
            affected_flights=affected_flights,
            propagation_path=propagation_path,
            total_delay_minutes=total_delay,
            propagation_depth=max_depth
        )
    
    def _analyze_domino_effects(self, df: pd.DataFrame) -> Dict:
        """Analyze domino effects in the flight network."""
        domino_analysis = {
            'potential_domino_flights': [],
            'domino_chains': [],
            'vulnerability_score': 0.0
        }
        
        if self.flight_network.number_of_nodes() == 0:
            return domino_analysis
        
        # Identify flights that could trigger domino effects
        for flight_id in self.flight_network.nodes():
            descendants = list(nx.descendants(self.flight_network, flight_id))
            
            if len(descendants) >= 3:  # Could affect 3+ downstream flights
                domino_analysis['potential_domino_flights'].append({
                    'flight_id': flight_id,
                    'potential_affected_flights': len(descendants),
                    'domino_risk_score': len(descendants) * 0.1
                })
        
        # Find longest dependency chains
        try:
            # Find strongly connected components
            scc = list(nx.strongly_connected_components(self.flight_network))
            largest_scc = max(scc, key=len) if scc else set()
            
            if len(largest_scc) > 1:
                domino_analysis['domino_chains'].append({
                    'chain_flights': list(largest_scc),
                    'chain_length': len(largest_scc),
                    'vulnerability_level': 'High' if len(largest_scc) > 5 else 'Medium'
                })
        except:
            pass  # Handle cases where SCC calculation fails
        
        # Calculate overall vulnerability score
        total_flights = len(df)
        high_impact_flights = len([f for f in domino_analysis['potential_domino_flights'] 
                                 if f['potential_affected_flights'] >= 5])
        
        domino_analysis['vulnerability_score'] = (high_impact_flights / total_flights) if total_flights > 0 else 0.0
        
        return domino_analysis
    
    def _generate_disruption_scenarios(self, critical_flights: List[CriticalFlight]) -> List[NetworkDisruption]:
        """Generate network disruption scenarios for critical flights."""
        scenarios = []
        
        for critical_flight in critical_flights:
            # Delay scenario
            delay_scenario = NetworkDisruption(
                disrupted_flight=critical_flight.flight_id,
                disruption_type="delay",
                disruption_magnitude=60.0,  # 1 hour delay
                affected_flights=list(self.flight_network.successors(critical_flight.flight_id)) 
                    if critical_flight.flight_id in self.flight_network else [],
                total_impact_score=critical_flight.cascading_delay_potential,
                recovery_time_hours=1.5
            )
            scenarios.append(delay_scenario)
            
            # Cancellation scenario
            cancellation_scenario = NetworkDisruption(
                disrupted_flight=critical_flight.flight_id,
                disruption_type="cancellation",
                disruption_magnitude=1.0,  # Complete cancellation
                affected_flights=list(nx.descendants(self.flight_network, critical_flight.flight_id))
                    if critical_flight.flight_id in self.flight_network else [],
                total_impact_score=critical_flight.network_impact_score * 100,
                recovery_time_hours=8.0
            )
            scenarios.append(cancellation_scenario)
        
        return scenarios
    
    def _get_propagation_factor(self, connection_type: str) -> float:
        """Get delay propagation factor based on connection type."""
        factors = {
            'aircraft_rotation': 0.8,  # High propagation for aircraft rotations
            'crew_rotation': 0.6,      # Medium propagation for crew rotations
            'passenger_connection': 0.3, # Low propagation for passenger connections
            'resource_dependency': 0.4   # Medium propagation for resource dependencies
        }
        return factors.get(connection_type, 0.5)
    
    def _get_aircraft_rotation_flights(self, flight_id: str) -> List[str]:
        """Get all flights in the same aircraft rotation chain."""
        if flight_id not in self.flight_network:
            return []
        
        rotation_flights = []
        
        # Get all flights connected by aircraft rotation
        for successor in self.flight_network.successors(flight_id):
            edge_data = self.flight_network.get_edge_data(flight_id, successor)
            if edge_data and edge_data.get('connection_type') == 'aircraft_rotation':
                rotation_flights.append(successor)
                # Recursively get downstream aircraft rotations
                rotation_flights.extend(self._get_aircraft_rotation_flights(successor))
        
        return rotation_flights
    
    def _get_network_analysis_metrics(self) -> Dict:
        """Get comprehensive network analysis metrics."""
        if self.flight_network.number_of_nodes() == 0:
            return {'nodes': 0, 'edges': 0}
        
        metrics = {
            'nodes': self.flight_network.number_of_nodes(),
            'edges': self.flight_network.number_of_edges(),
            'density': nx.density(self.flight_network),
            'average_clustering': nx.average_clustering(self.flight_network.to_undirected()),
            'number_of_components': nx.number_weakly_connected_components(self.flight_network)
        }
        
        # Add centrality statistics
        try:
            betweenness = list(nx.betweenness_centrality(self.flight_network).values())
            metrics['avg_betweenness_centrality'] = np.mean(betweenness)
            metrics['max_betweenness_centrality'] = np.max(betweenness)
        except:
            metrics['avg_betweenness_centrality'] = 0.0
            metrics['max_betweenness_centrality'] = 0.0
        
        # Add degree statistics
        out_degrees = list(dict(self.flight_network.out_degree()).values())
        in_degrees = list(dict(self.flight_network.in_degree()).values())
        
        metrics['avg_out_degree'] = np.mean(out_degrees) if out_degrees else 0
        metrics['max_out_degree'] = np.max(out_degrees) if out_degrees else 0
        metrics['avg_in_degree'] = np.mean(in_degrees) if in_degrees else 0
        metrics['max_in_degree'] = np.max(in_degrees) if in_degrees else 0
        
        return metrics
    
    def _generate_cascading_recommendations(self, critical_flights: List[CriticalFlight],
                                          propagation_analysis: List[DelayPropagation],
                                          domino_analysis: Dict,
                                          disruption_scenarios: List[NetworkDisruption]) -> List[str]:
        """Generate actionable recommendations based on cascading impact analysis."""
        recommendations = []
        
        # Critical flights recommendations
        if critical_flights:
            top_critical = critical_flights[:5]
            critical_ids = [cf.flight_number for cf in top_critical]
            recommendations.append(
                f"Top 5 critical flights requiring priority attention: {', '.join(critical_ids)}. "
                "These flights have the highest cascading impact potential."
            )
            
            # Category-based recommendations
            critical_count = len([cf for cf in critical_flights if cf.impact_category == "Critical"])
            if critical_count > 0:
                recommendations.append(
                    f"{critical_count} flights identified as 'Critical' impact level. "
                    "Implement enhanced monitoring and contingency planning for these flights."
                )
        
        # Propagation analysis recommendations
        if propagation_analysis:
            max_propagation = max(propagation_analysis, key=lambda x: x.total_delay_minutes)
            recommendations.append(
                f"Highest delay propagation risk: Flight {max_propagation.source_flight} "
                f"could cause {max_propagation.total_delay_minutes:.0f} minutes of total network delays. "
                "Consider schedule buffer adjustments."
            )
            
            deep_propagations = [p for p in propagation_analysis if p.propagation_depth >= 3]
            if deep_propagations:
                recommendations.append(
                    f"{len(deep_propagations)} flights show deep propagation patterns (3+ levels). "
                    "Review network connectivity to reduce cascading risks."
                )
        
        # Domino effect recommendations
        vulnerability_score = domino_analysis.get('vulnerability_score', 0.0)
        if vulnerability_score > 0.2:
            recommendations.append(
                f"Network vulnerability score: {vulnerability_score:.2f}. "
                "High vulnerability detected - implement network resilience measures."
            )
        
        potential_domino_flights = domino_analysis.get('potential_domino_flights', [])
        if potential_domino_flights:
            high_risk_domino = [f for f in potential_domino_flights if f['potential_affected_flights'] >= 5]
            if high_risk_domino:
                recommendations.append(
                    f"{len(high_risk_domino)} flights identified as high domino risk. "
                    "These flights could trigger widespread network disruptions."
                )
        
        # Disruption scenario recommendations
        if disruption_scenarios:
            worst_scenario = max(disruption_scenarios, key=lambda x: x.total_impact_score)
            recommendations.append(
                f"Worst-case disruption scenario: {worst_scenario.disruption_type} of flight "
                f"{worst_scenario.disrupted_flight} could affect {len(worst_scenario.affected_flights)} flights "
                f"with {worst_scenario.recovery_time_hours:.1f} hours recovery time."
            )
        
        # Network structure recommendations
        network_metrics = self._get_network_analysis_metrics()
        if network_metrics.get('density', 0) > 0.3:
            recommendations.append(
                "High network density detected. Consider reducing flight interdependencies "
                "to improve resilience against cascading failures."
            )
        
        return recommendations
    
    def _calculate_confidence_score(self, df: pd.DataFrame) -> float:
        """Calculate confidence score based on data quality and network completeness."""
        base_confidence = 0.6
        
        # Increase confidence with more data
        data_factor = min(0.2, len(df) / 500)
        
        # Increase confidence with network connectivity
        network_factor = 0.0
        if self.flight_network.number_of_nodes() > 0:
            connectivity = self.flight_network.number_of_edges() / max(1, self.flight_network.number_of_nodes())
            network_factor = min(0.2, connectivity / 5)  # Normalize by expected connectivity
        
        return min(0.95, base_confidence + data_factor + network_factor)
    
    def _empty_analysis_result(self, airport_code: str) -> AnalysisResult:
        """Return empty analysis result when no data is available."""
        return AnalysisResult(
            analysis_type="cascading_impact_analysis",
            airport_code=airport_code,
            metrics={'error': 'No flight data available for analysis'},
            recommendations=["Insufficient data for cascading impact analysis"],
            confidence_score=0.0,
            data_sources=[]
        )