"""
Schedule Impact Modeling System for Flight Tuning

This module implements schedule change simulation algorithms to tune flight schedules
and analyze delay impact. Uses NetworkX for cascading effect prediction.
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import logging

from ..data.models import FlightData, AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ScheduleChange:
    """Represents a proposed schedule change."""
    flight_id: str
    original_departure: datetime
    new_departure: datetime
    change_reason: str
    priority: int = 1  # 1=low, 2=medium, 3=high


@dataclass
class ImpactScore:
    """Represents the impact score of a schedule change."""
    delay_impact: float
    cascading_impact: float
    operational_impact: float
    total_impact: float
    confidence: float


@dataclass
class Scenario:
    """Represents a schedule optimization scenario."""
    scenario_id: str
    changes: List[ScheduleChange]
    predicted_impact: ImpactScore
    improvement_score: float
    feasibility_score: float


class ScheduleImpactAnalyzer:
    """
    Schedule impact modeling system for flight tuning.
    
    This class implements EXPECTATION 4: schedule change simulation algorithms
    to tune flight schedules and see delay impact using NetworkX for cascading analysis.
    """
    
    def __init__(self, turnaround_time_minutes: int = 45):
        """
        Initialize the schedule impact analyzer.
        
        Args:
            turnaround_time_minutes: Minimum aircraft turnaround time (default: 45 minutes)
        """
        self.turnaround_time_minutes = turnaround_time_minutes
        self.impact_predictor = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=12
        )
        self.scaler = StandardScaler()
        self.flight_network = nx.DiGraph()
        self.is_trained = False
    
    def analyze_schedule_change(self, schedule_change: Dict, related_flights: List[Dict]) -> Dict:
        """
        Simplified schedule change analysis for API usage.
        
        Args:
            schedule_change: Dictionary with schedule change details
            related_flights: List of related flight data
            
        Returns:
            Dictionary with impact analysis results
        """
        try:
            flight_id = schedule_change['flight_id']
            original_departure = schedule_change['original_departure']
            new_departure = schedule_change['new_departure']
            
            # Calculate time difference
            time_diff = (new_departure - original_departure).total_seconds() / 60
            
            # Analyze impact on related flights
            affected_flights = []
            total_delay_impact = 0
            
            for flight in related_flights:
                flight_departure = pd.to_datetime(flight.get('scheduled_departure'))
                
                # Check if flights are close in time (within 2 hours)
                time_gap = abs((flight_departure - new_departure).total_seconds() / 60)
                
                if time_gap < 120:  # Within 2 hours
                    # Estimate delay impact based on proximity
                    impact_factor = max(0, (120 - time_gap) / 120)
                    estimated_delay = abs(time_diff) * impact_factor * 0.3  # 30% propagation
                    
                    if estimated_delay > 5:  # Only significant impacts
                        affected_flights.append({
                            "flight_id": flight.get('flight_id'),
                            "estimated_delay_minutes": round(estimated_delay, 1),
                            "impact_reason": "Schedule proximity"
                        })
                        total_delay_impact += estimated_delay
            
            # Calculate confidence based on data availability
            confidence_score = min(0.9, len(related_flights) / 20)  # More data = higher confidence
            
            # Generate recommendations
            recommendations = []
            if abs(time_diff) > 60:
                recommendations.append("Large schedule change may cause significant disruption")
            if len(affected_flights) > 5:
                recommendations.append("Multiple flights may be affected by this change")
            if total_delay_impact < 30:
                recommendations.append("Schedule change appears to have minimal network impact")
            
            return {
                "flight_id": flight_id,
                "original_schedule": {
                    "departure": original_departure.isoformat(),
                    "arrival": schedule_change.get('original_arrival', original_departure + timedelta(hours=2)).isoformat()
                },
                "proposed_schedule": {
                    "departure": new_departure.isoformat(),
                    "arrival": schedule_change.get('new_arrival', new_departure + timedelta(hours=2)).isoformat()
                },
                "delay_impact": round(total_delay_impact, 1),
                "affected_flights": affected_flights,
                "confidence_score": round(confidence_score, 2),
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in schedule impact analysis: {str(e)}")
            return {
                "flight_id": schedule_change.get('flight_id', 'unknown'),
                "error": str(e),
                "recommendations": ["Analysis failed due to data processing error"]
            }
        
    def analyze_schedule_impact(self, flight_data: List[FlightData], 
                               proposed_changes: List[ScheduleChange],
                               airport_code: str) -> AnalysisResult:
        """
        Analyze the impact of proposed schedule changes.
        
        Args:
            flight_data: Historical flight data
            proposed_changes: List of proposed schedule changes
            airport_code: Airport code to analyze
            
        Returns:
            AnalysisResult with impact analysis and recommendations
        """
        logger.info(f"Analyzing schedule impact for {len(proposed_changes)} changes at {airport_code}")
        
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
        
        # Build flight network for cascading analysis
        self._build_flight_network(airport_flights)
        
        # Train impact prediction model
        if not self.is_trained:
            self._train_impact_model(airport_flights)
        
        # Analyze each proposed change
        change_impacts = []
        for change in proposed_changes:
            impact = self._analyze_single_change_impact(change, airport_flights)
            change_impacts.append({
                'change': change,
                'impact': impact
            })
        
        # Perform what-if analysis
        whatif_results = self._perform_whatif_analysis(proposed_changes, airport_flights)
        
        # Generate scenario comparisons
        scenarios = self._generate_scenarios(proposed_changes, airport_flights)
        
        # Analyze constraints
        constraint_analysis = self._analyze_constraints(proposed_changes, airport_flights)
        
        # Generate recommendations
        recommendations = self._generate_impact_recommendations(
            change_impacts, whatif_results, scenarios, constraint_analysis
        )
        
        # Compile results
        metrics = {
            'individual_change_impacts': [
                {
                    'flight_id': ci['change'].flight_id,
                    'time_change_minutes': (ci['change'].new_departure - ci['change'].original_departure).total_seconds() / 60,
                    'delay_impact': ci['impact'].delay_impact,
                    'cascading_impact': ci['impact'].cascading_impact,
                    'total_impact': ci['impact'].total_impact,
                    'confidence': ci['impact'].confidence
                }
                for ci in change_impacts
            ],
            'whatif_analysis': whatif_results,
            'scenario_comparison': [
                {
                    'scenario_id': s.scenario_id,
                    'num_changes': len(s.changes),
                    'improvement_score': s.improvement_score,
                    'feasibility_score': s.feasibility_score,
                    'total_impact': s.predicted_impact.total_impact
                }
                for s in scenarios
            ],
            'constraint_analysis': constraint_analysis,
            'network_metrics': self._get_network_metrics(),
            'total_flights_analyzed': len(airport_flights)
        }
        
        return AnalysisResult(
            analysis_type="schedule_impact_analysis",
            airport_code=airport_code,
            metrics=metrics,
            recommendations=recommendations,
            confidence_score=self._calculate_confidence_score(airport_flights, change_impacts),
            data_sources=["flight_data", "schedule_changes"]
        )
    
    def predict_delay_impact(self, change: ScheduleChange, 
                           historical_data: List[FlightData]) -> ImpactScore:
        """
        Predict the delay impact of a single schedule change.
        
        Args:
            change: Proposed schedule change
            historical_data: Historical flight data for context
            
        Returns:
            ImpactScore with predicted impacts
        """
        if not self.is_trained:
            df = self._flights_to_dataframe(historical_data)
            self._train_impact_model(df)
        
        return self._analyze_single_change_impact(change, 
                                                self._flights_to_dataframe(historical_data))
    
    def optimize_schedule(self, flight_data: List[FlightData], 
                         optimization_goals: Dict[str, float],
                         airport_code: str) -> List[ScheduleChange]:
        """
        Generate optimized schedule changes based on goals.
        
        Args:
            flight_data: Historical flight data
            optimization_goals: Dictionary of optimization goals (e.g., {'delay_reduction': 0.3})
            airport_code: Airport code to optimize
            
        Returns:
            List of recommended schedule changes
        """
        df = self._flights_to_dataframe(flight_data)
        airport_flights = df[
            (df['origin_airport'] == airport_code) | 
            (df['destination_airport'] == airport_code)
        ].copy()
        
        if airport_flights.empty:
            return []
        
        # Identify problematic flights (high delays)
        problematic_flights = airport_flights[
            airport_flights['delay_minutes'] > airport_flights['delay_minutes'].quantile(0.8)
        ]
        
        recommended_changes = []
        
        for _, flight in problematic_flights.iterrows():
            # Find optimal time slot for this flight
            optimal_time = self._find_optimal_time_slot(flight, airport_flights)
            
            if optimal_time and optimal_time != flight['scheduled_departure']:
                change = ScheduleChange(
                    flight_id=flight['flight_id'],
                    original_departure=flight['scheduled_departure'],
                    new_departure=optimal_time,
                    change_reason="delay_optimization",
                    priority=2
                )
                recommended_changes.append(change)
        
        # Rank changes by potential impact
        ranked_changes = self._rank_changes_by_impact(recommended_changes, airport_flights)
        
        return ranked_changes[:10]  # Return top 10 recommendations
    
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
        """Build flight network graph for cascading analysis using NetworkX."""
        self.flight_network.clear()
        
        # Add nodes for each flight
        for _, flight in df.iterrows():
            self.flight_network.add_node(
                flight['flight_id'],
                airline=flight['airline'],
                aircraft_type=flight['aircraft_type'],
                departure_time=flight['scheduled_departure'],
                origin=flight['origin_airport'],
                destination=flight['destination_airport'],
                delay=flight['delay_minutes']
            )
        
        # Add edges for aircraft connections (same aircraft, sequential flights)
        aircraft_flights = df.groupby(['airline', 'aircraft_type'])
        
        for (airline, aircraft), group in aircraft_flights:
            # Sort by departure time
            sorted_flights = group.sort_values('scheduled_departure')
            
            # Connect sequential flights with same aircraft
            for i in range(len(sorted_flights) - 1):
                current_flight = sorted_flights.iloc[i]
                next_flight = sorted_flights.iloc[i + 1]
                
                # Check if turnaround time is feasible
                turnaround_time = (next_flight['scheduled_departure'] - 
                                 current_flight['scheduled_arrival']).total_seconds() / 60
                
                if turnaround_time >= self.turnaround_time_minutes:
                    self.flight_network.add_edge(
                        current_flight['flight_id'],
                        next_flight['flight_id'],
                        connection_type='aircraft_rotation',
                        turnaround_time=turnaround_time,
                        weight=1.0 / max(turnaround_time, 1)  # Shorter turnaround = higher impact
                    )
        
        # Add edges for crew connections (estimated based on airline and timing)
        airline_flights = df.groupby('airline')
        
        for airline, group in airline_flights:
            sorted_flights = group.sort_values('scheduled_departure')
            
            # Connect flights that could share crew (same airline, reasonable timing)
            for i in range(len(sorted_flights) - 1):
                current_flight = sorted_flights.iloc[i]
                next_flight = sorted_flights.iloc[i + 1]
                
                time_diff = (next_flight['scheduled_departure'] - 
                           current_flight['scheduled_arrival']).total_seconds() / 60
                
                # Crew connection possible if 30 minutes to 8 hours between flights
                if 30 <= time_diff <= 480:
                    if not self.flight_network.has_edge(current_flight['flight_id'], 
                                                       next_flight['flight_id']):
                        self.flight_network.add_edge(
                            current_flight['flight_id'],
                            next_flight['flight_id'],
                            connection_type='crew_rotation',
                            crew_rest_time=time_diff,
                            weight=0.5 / max(time_diff, 1)
                        )
    
    def _train_impact_model(self, df: pd.DataFrame):
        """Train machine learning model for impact prediction."""
        logger.info("Training schedule impact prediction model")
        
        if len(df) < 20:  # Need minimum data for training
            logger.warning("Insufficient data for impact model training")
            return
        
        # Prepare features for training
        features = []
        targets = []
        
        for _, flight in df.iterrows():
            # Features: hour, day of week, airline, aircraft type, etc.
            feature_vector = [
                flight['scheduled_departure'].hour,
                flight['scheduled_departure'].weekday(),
                flight['scheduled_departure'].month,
                len(flight['flight_number']),
                flight['passenger_count'],
                1 if flight['aircraft_type'] != 'Unknown' else 0,
                # Network features
                self.flight_network.in_degree(flight['flight_id']) if flight['flight_id'] in self.flight_network else 0,
                self.flight_network.out_degree(flight['flight_id']) if flight['flight_id'] in self.flight_network else 0
            ]
            
            features.append(feature_vector)
            targets.append(flight['delay_minutes'])
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Remove rows with missing target values
        valid_mask = ~np.isnan(targets)
        features = features[valid_mask]
        targets = targets[valid_mask]
        
        if len(features) < 10:
            logger.warning("Insufficient valid data for model training")
            return
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.impact_predictor.fit(features_scaled, targets)
        
        # Evaluate model
        cv_scores = cross_val_score(self.impact_predictor, features_scaled, targets, cv=3)
        logger.info(f"Impact model training completed. CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.is_trained = True
    
    def _analyze_single_change_impact(self, change: ScheduleChange, df: pd.DataFrame) -> ImpactScore:
        """Analyze the impact of a single schedule change."""
        # Calculate direct delay impact
        time_change_minutes = (change.new_departure - change.original_departure).total_seconds() / 60
        
        # Find the flight in the data
        flight_data = df[df['flight_id'] == change.flight_id]
        
        if flight_data.empty:
            # Return default impact if flight not found
            return ImpactScore(
                delay_impact=abs(time_change_minutes) * 0.1,
                cascading_impact=0.0,
                operational_impact=abs(time_change_minutes) * 0.05,
                total_impact=abs(time_change_minutes) * 0.15,
                confidence=0.3
            )
        
        flight = flight_data.iloc[0]
        
        # Predict delay impact using ML model
        if self.is_trained:
            # Create feature vector for new time
            new_time = change.new_departure
            feature_vector = np.array([[
                new_time.hour,
                new_time.weekday(),
                new_time.month,
                len(flight['flight_number']),
                flight['passenger_count'],
                1 if flight['aircraft_type'] != 'Unknown' else 0,
                self.flight_network.in_degree(change.flight_id) if change.flight_id in self.flight_network else 0,
                self.flight_network.out_degree(change.flight_id) if change.flight_id in self.flight_network else 0
            ]])
            
            feature_vector_scaled = self.scaler.transform(feature_vector)
            predicted_delay = self.impact_predictor.predict(feature_vector_scaled)[0]
            delay_impact = abs(predicted_delay - flight['delay_minutes'])
        else:
            # Fallback to simple heuristic
            delay_impact = abs(time_change_minutes) * 0.2
        
        # Calculate cascading impact using network analysis
        cascading_impact = self._calculate_cascading_impact(change.flight_id, time_change_minutes)
        
        # Calculate operational impact (crew, aircraft, gates)
        operational_impact = self._calculate_operational_impact(change, flight)
        
        # Total impact
        total_impact = delay_impact + cascading_impact + operational_impact
        
        # Confidence based on data availability and model performance
        confidence = 0.8 if self.is_trained else 0.5
        if change.flight_id in self.flight_network:
            confidence += 0.1
        
        return ImpactScore(
            delay_impact=delay_impact,
            cascading_impact=cascading_impact,
            operational_impact=operational_impact,
            total_impact=total_impact,
            confidence=min(confidence, 0.95)
        )
    
    def _calculate_cascading_impact(self, flight_id: str, time_change_minutes: float) -> float:
        """Calculate cascading impact using NetworkX graph analysis."""
        if flight_id not in self.flight_network:
            return 0.0
        
        cascading_impact = 0.0
        
        # Analyze downstream flights (flights that depend on this one)
        downstream_flights = list(self.flight_network.successors(flight_id))
        
        for downstream_flight in downstream_flights:
            edge_data = self.flight_network.get_edge_data(flight_id, downstream_flight)
            
            if edge_data:
                # Impact depends on connection type and timing
                connection_weight = edge_data.get('weight', 0.5)
                
                if edge_data.get('connection_type') == 'aircraft_rotation':
                    # Aircraft rotation is more critical
                    impact = abs(time_change_minutes) * connection_weight * 1.5
                else:
                    # Crew rotation has moderate impact
                    impact = abs(time_change_minutes) * connection_weight * 0.8
                
                cascading_impact += impact
                
                # Recursively calculate second-order effects (limited depth)
                second_order_impact = self._calculate_cascading_impact(downstream_flight, 
                                                                     time_change_minutes * 0.5)
                cascading_impact += second_order_impact * 0.3  # Diminished second-order effect
        
        return cascading_impact
    
    def _calculate_operational_impact(self, change: ScheduleChange, flight: pd.Series) -> float:
        """Calculate operational impact (crew, aircraft, gates)."""
        time_change_minutes = abs((change.new_departure - change.original_departure).total_seconds() / 60)
        
        # Base operational impact
        operational_impact = time_change_minutes * 0.1
        
        # Increase impact for peak hours
        if 7 <= change.new_departure.hour <= 10 or 17 <= change.new_departure.hour <= 20:
            operational_impact *= 1.5
        
        # Increase impact for large aircraft (more passengers affected)
        if flight['passenger_count'] > 200:
            operational_impact *= 1.3
        
        # Increase impact for significant time changes
        if time_change_minutes > 60:
            operational_impact *= 1.4
        
        return operational_impact
    
    def _perform_whatif_analysis(self, changes: List[ScheduleChange], df: pd.DataFrame) -> Dict:
        """Perform what-if analysis for schedule changes."""
        scenarios = {
            'baseline': {'total_delay': df['delay_minutes'].sum(), 'avg_delay': df['delay_minutes'].mean()},
            'with_changes': {}
        }
        
        # Calculate impact of all changes combined
        total_delay_reduction = 0
        total_operational_cost = 0
        
        for change in changes:
            impact = self._analyze_single_change_impact(change, df)
            
            # Assume positive impact reduces delays, negative increases them
            time_change = (change.new_departure - change.original_departure).total_seconds() / 60
            if time_change < 0:  # Earlier departure
                total_delay_reduction += impact.delay_impact * 0.7  # 70% effectiveness
            else:  # Later departure
                total_delay_reduction -= impact.delay_impact * 0.3  # May increase delays
            
            total_operational_cost += impact.operational_impact
        
        scenarios['with_changes'] = {
            'total_delay': max(0, scenarios['baseline']['total_delay'] - total_delay_reduction),
            'avg_delay': max(0, scenarios['baseline']['avg_delay'] - total_delay_reduction / len(df)),
            'operational_cost': total_operational_cost,
            'net_improvement': total_delay_reduction - total_operational_cost * 0.5
        }
        
        return scenarios
    
    def _generate_scenarios(self, changes: List[ScheduleChange], df: pd.DataFrame) -> List[Scenario]:
        """Generate and compare different scheduling scenarios."""
        scenarios = []
        
        # Scenario 1: All changes
        all_changes_impact = ImpactScore(0, 0, 0, 0, 0.8)
        for change in changes:
            impact = self._analyze_single_change_impact(change, df)
            all_changes_impact.delay_impact += impact.delay_impact
            all_changes_impact.cascading_impact += impact.cascading_impact
            all_changes_impact.operational_impact += impact.operational_impact
            all_changes_impact.total_impact += impact.total_impact
        
        scenarios.append(Scenario(
            scenario_id="all_changes",
            changes=changes,
            predicted_impact=all_changes_impact,
            improvement_score=max(0, -all_changes_impact.total_impact),
            feasibility_score=0.7
        ))
        
        # Scenario 2: High priority changes only
        high_priority_changes = [c for c in changes if c.priority >= 2]
        if high_priority_changes:
            hp_impact = ImpactScore(0, 0, 0, 0, 0.8)
            for change in high_priority_changes:
                impact = self._analyze_single_change_impact(change, df)
                hp_impact.delay_impact += impact.delay_impact
                hp_impact.cascading_impact += impact.cascading_impact
                hp_impact.operational_impact += impact.operational_impact
                hp_impact.total_impact += impact.total_impact
            
            scenarios.append(Scenario(
                scenario_id="high_priority_only",
                changes=high_priority_changes,
                predicted_impact=hp_impact,
                improvement_score=max(0, -hp_impact.total_impact),
                feasibility_score=0.9
            ))
        
        # Scenario 3: Minimal changes (top 3 by impact)
        if len(changes) > 3:
            # Sort changes by potential positive impact
            sorted_changes = sorted(changes, 
                                  key=lambda c: self._analyze_single_change_impact(c, df).total_impact)
            minimal_changes = sorted_changes[:3]
            
            min_impact = ImpactScore(0, 0, 0, 0, 0.9)
            for change in minimal_changes:
                impact = self._analyze_single_change_impact(change, df)
                min_impact.delay_impact += impact.delay_impact
                min_impact.cascading_impact += impact.cascading_impact
                min_impact.operational_impact += impact.operational_impact
                min_impact.total_impact += impact.total_impact
            
            scenarios.append(Scenario(
                scenario_id="minimal_changes",
                changes=minimal_changes,
                predicted_impact=min_impact,
                improvement_score=max(0, -min_impact.total_impact),
                feasibility_score=0.95
            ))
        
        return scenarios
    
    def _analyze_constraints(self, changes: List[ScheduleChange], df: pd.DataFrame) -> Dict:
        """Analyze constraints for proposed schedule changes."""
        constraints = {
            'turnaround_violations': [],
            'crew_duty_violations': [],
            'gate_conflicts': [],
            'runway_capacity_issues': []
        }
        
        for change in changes:
            # Check turnaround time constraints
            if change.flight_id in self.flight_network:
                predecessors = list(self.flight_network.predecessors(change.flight_id))
                successors = list(self.flight_network.successors(change.flight_id))
                
                for pred in predecessors:
                    edge_data = self.flight_network.get_edge_data(pred, change.flight_id)
                    if edge_data and edge_data.get('connection_type') == 'aircraft_rotation':
                        # Check if new timing violates turnaround
                        pred_flight = df[df['flight_id'] == pred]
                        if not pred_flight.empty:
                            turnaround = (change.new_departure - 
                                        pred_flight.iloc[0]['scheduled_arrival']).total_seconds() / 60
                            if turnaround < self.turnaround_time_minutes:
                                constraints['turnaround_violations'].append({
                                    'flight_id': change.flight_id,
                                    'predecessor': pred,
                                    'required_turnaround': self.turnaround_time_minutes,
                                    'actual_turnaround': turnaround
                                })
                
                # Similar check for successors
                for succ in successors:
                    edge_data = self.flight_network.get_edge_data(change.flight_id, succ)
                    if edge_data and edge_data.get('connection_type') == 'aircraft_rotation':
                        succ_flight = df[df['flight_id'] == succ]
                        if not succ_flight.empty:
                            # Estimate new arrival time (departure + flight duration)
                            flight_duration = (df[df['flight_id'] == change.flight_id].iloc[0]['scheduled_arrival'] - 
                                             df[df['flight_id'] == change.flight_id].iloc[0]['scheduled_departure'])
                            new_arrival = change.new_departure + flight_duration
                            
                            turnaround = (succ_flight.iloc[0]['scheduled_departure'] - 
                                        new_arrival).total_seconds() / 60
                            if turnaround < self.turnaround_time_minutes:
                                constraints['turnaround_violations'].append({
                                    'flight_id': change.flight_id,
                                    'successor': succ,
                                    'required_turnaround': self.turnaround_time_minutes,
                                    'actual_turnaround': turnaround
                                })
        
        return constraints
    
    def _find_optimal_time_slot(self, flight: pd.Series, df: pd.DataFrame) -> Optional[datetime]:
        """Find optimal time slot for a flight based on historical patterns."""
        # Analyze delay patterns by hour
        hourly_delays = df.groupby(df['scheduled_departure'].dt.hour)['delay_minutes'].mean()
        
        # Find the hour with minimum average delay
        optimal_hour = hourly_delays.idxmin()
        
        # Create new departure time with optimal hour
        original_time = flight['scheduled_departure']
        optimal_time = original_time.replace(hour=optimal_hour, minute=0, second=0, microsecond=0)
        
        # Ensure it's not the same as original
        if optimal_time == original_time.replace(minute=0, second=0, microsecond=0):
            return None
        
        return optimal_time
    
    def _rank_changes_by_impact(self, changes: List[ScheduleChange], df: pd.DataFrame) -> List[ScheduleChange]:
        """Rank schedule changes by their potential positive impact."""
        change_scores = []
        
        for change in changes:
            impact = self._analyze_single_change_impact(change, df)
            # Positive score for beneficial changes (negative total impact)
            score = -impact.total_impact * impact.confidence
            change_scores.append((change, score))
        
        # Sort by score (highest first)
        change_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [change for change, score in change_scores]
    
    def _generate_impact_recommendations(self, change_impacts: List[Dict], 
                                       whatif_results: Dict, scenarios: List[Scenario],
                                       constraints: Dict) -> List[str]:
        """Generate actionable recommendations based on impact analysis."""
        recommendations = []
        
        # Overall impact assessment
        total_positive_impact = sum(
            max(0, -ci['impact'].total_impact) for ci in change_impacts
        )
        total_negative_impact = sum(
            max(0, ci['impact'].total_impact) for ci in change_impacts
        )
        
        if total_positive_impact > total_negative_impact:
            recommendations.append(
                f"Proposed changes show net positive impact of {total_positive_impact - total_negative_impact:.1f} "
                "delay minutes. Implementation recommended."
            )
        else:
            recommendations.append(
                f"Proposed changes show net negative impact of {total_negative_impact - total_positive_impact:.1f} "
                "delay minutes. Reconsider implementation."
            )
        
        # Scenario recommendations
        if scenarios:
            best_scenario = max(scenarios, key=lambda s: s.improvement_score * s.feasibility_score)
            recommendations.append(
                f"Best scenario: '{best_scenario.scenario_id}' with {len(best_scenario.changes)} changes, "
                f"improvement score: {best_scenario.improvement_score:.1f}, "
                f"feasibility: {best_scenario.feasibility_score:.1f}"
            )
        
        # Constraint warnings
        if constraints['turnaround_violations']:
            recommendations.append(
                f"Warning: {len(constraints['turnaround_violations'])} turnaround time violations detected. "
                "Review aircraft rotation schedules."
            )
        
        # High-impact changes
        high_impact_changes = [
            ci for ci in change_impacts 
            if abs(ci['impact'].total_impact) > 30  # More than 30 minutes impact
        ]
        
        if high_impact_changes:
            recommendations.append(
                f"{len(high_impact_changes)} changes have high impact (>30 min). "
                "Prioritize careful review of these changes."
            )
        
        # Cascading effect warnings
        high_cascading_changes = [
            ci for ci in change_impacts 
            if ci['impact'].cascading_impact > 15  # More than 15 minutes cascading impact
        ]
        
        if high_cascading_changes:
            recommendations.append(
                f"{len(high_cascading_changes)} changes have significant cascading effects. "
                "Consider network-wide implications."
            )
        
        return recommendations
    
    def _get_network_metrics(self) -> Dict:
        """Get network analysis metrics."""
        if self.flight_network.number_of_nodes() == 0:
            return {'nodes': 0, 'edges': 0}
        
        return {
            'nodes': self.flight_network.number_of_nodes(),
            'edges': self.flight_network.number_of_edges(),
            'density': nx.density(self.flight_network),
            'average_clustering': nx.average_clustering(self.flight_network.to_undirected()),
            'strongly_connected_components': nx.number_strongly_connected_components(self.flight_network)
        }
    
    def _calculate_confidence_score(self, df: pd.DataFrame, change_impacts: List[Dict]) -> float:
        """Calculate confidence score based on data quality and model performance."""
        base_confidence = 0.5
        
        # Increase confidence with more data
        data_factor = min(0.2, len(df) / 500)
        
        # Increase confidence if model is trained
        model_factor = 0.2 if self.is_trained else 0.0
        
        # Increase confidence with network data
        network_factor = 0.1 if self.flight_network.number_of_nodes() > 0 else 0.0
        
        return min(0.95, base_confidence + data_factor + model_factor + network_factor)
    
    def _empty_analysis_result(self, airport_code: str) -> AnalysisResult:
        """Return empty analysis result when no data is available."""
        return AnalysisResult(
            analysis_type="schedule_impact_analysis",
            airport_code=airport_code,
            metrics={'error': 'No flight data available for analysis'},
            recommendations=["Insufficient data for schedule impact analysis"],
            confidence_score=0.0,
            data_sources=[]
        )