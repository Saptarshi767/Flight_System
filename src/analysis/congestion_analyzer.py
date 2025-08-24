"""
Congestion Analysis Engine for Flight Scheduling Optimization

This module implements flight density calculation algorithms to find busiest time slots to avoid.
Uses Prophet for time series analysis and provides congestion avoidance recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
import warnings

from ..data.models import FlightData, AirportData, AnalysisResult

# Suppress Prophet warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="prophet")
logging.getLogger('prophet').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class CongestionAnalyzer:
    """
    Congestion analysis engine for identifying busiest time slots to avoid.
    
    This class implements EXPECTATION 3: flight density calculation algorithms
    to find busiest time slots using open source AI tools (Prophet for time series analysis).
    """
    
    def __init__(self, runway_capacity: int = 60):
        """
        Initialize the congestion analyzer.
        
        Args:
            runway_capacity: Maximum flights per hour (default: 60)
        """
        self.runway_capacity = runway_capacity
        self.prophet_model = None
        self.scaler = StandardScaler()
        self.clusterer = KMeans(n_clusters=4, random_state=42)  # 4 congestion levels
        self.is_trained = False
        
    def analyze_congestion_from_data(self, flight_data: List[Dict], airport_code: str,
                                   runway_capacity: int = 60) -> Dict:
        """
        Simplified congestion analysis for API usage.
        
        Args:
            flight_data: List of flight data dictionaries
            airport_code: Airport code to analyze
            runway_capacity: Maximum flights per hour
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if not flight_data:
                return {
                    "airport_code": airport_code,
                    "analysis_period": {"start": None, "end": None},
                    "peak_hours": [],
                    "congestion_by_hour": {},
                    "capacity_utilization": {},
                    "busiest_time_slots": [],
                    "alternative_slots": []
                }
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(flight_data)
            
            # Add hour column
            df['hour'] = pd.to_datetime(df['scheduled_departure']).dt.hour
            
            # Count flights by hour
            flights_by_hour = df.groupby('hour').size()
            
            # Calculate capacity utilization
            capacity_utilization = (flights_by_hour / runway_capacity * 100).to_dict()
            
            # Find peak hours (>80% capacity)
            peak_hours = flights_by_hour[flights_by_hour > runway_capacity * 0.8].index.tolist()
            
            # Find busiest time slots
            busiest_slots = flights_by_hour.nlargest(5)
            busiest_time_slots = [
                {
                    "hour": int(hour),
                    "flight_count": int(count),
                    "capacity_utilization": float(count / runway_capacity * 100),
                    "congestion_level": "High" if count > runway_capacity * 0.8 else "Medium"
                }
                for hour, count in busiest_slots.items()
            ]
            
            # Find alternative slots (low congestion)
            low_congestion = flights_by_hour[flights_by_hour < runway_capacity * 0.5]
            alternative_slots = [
                {
                    "hour": int(hour),
                    "flight_count": int(count),
                    "available_capacity": int(runway_capacity - count),
                    "recommendation": f"Alternative time slot: {hour:02d}:00-{hour+1:02d}:00"
                }
                for hour, count in low_congestion.nsmallest(3).items()
            ]
            
            # Analysis period
            dates = pd.to_datetime(df['scheduled_departure'])
            analysis_period = {
                "start": dates.min().isoformat() if not dates.empty else None,
                "end": dates.max().isoformat() if not dates.empty else None
            }
            
            return {
                "airport_code": airport_code,
                "analysis_period": analysis_period,
                "peak_hours": peak_hours,
                "congestion_by_hour": {str(k): float(v) for k, v in flights_by_hour.items()},
                "capacity_utilization": {str(k): float(v) for k, v in capacity_utilization.items()},
                "busiest_time_slots": busiest_time_slots,
                "alternative_slots": alternative_slots
            }
            
        except Exception as e:
            logger.error(f"Error in congestion analysis: {str(e)}")
            return {
                "airport_code": airport_code,
                "error": str(e)
            }
    
    def analyze_congestion(self, flight_data: List[FlightData], airport_code: str,
                          airport_data: Optional[AirportData] = None) -> AnalysisResult:
        """
        Perform comprehensive congestion analysis for an airport.
        
        Args:
            flight_data: List of flight data objects
            airport_code: Airport code to analyze
            airport_data: Optional airport configuration data
            
        Returns:
            AnalysisResult with congestion patterns and recommendations
        """
        logger.info(f"Starting congestion analysis for airport {airport_code}")
        
        # Update runway capacity if airport data is provided
        if airport_data:
            self.runway_capacity = airport_data.runway_capacity
        
        # Convert to DataFrame for analysis
        df = self._flights_to_dataframe(flight_data)
        
        # Check if DataFrame is empty
        if df.empty:
            logger.warning(f"No flight data provided for analysis")
            return self._empty_analysis_result(airport_code)
        
        # Filter for specific airport
        airport_flights = df[
            (df['origin_airport'] == airport_code) | 
            (df['destination_airport'] == airport_code)
        ].copy()
        
        if airport_flights.empty:
            logger.warning(f"No flight data found for airport {airport_code}")
            return self._empty_analysis_result(airport_code)
        
        # Calculate flight density metrics
        density_metrics = self._calculate_flight_density(airport_flights)
        
        # Identify peak hours
        peak_hours = self._identify_peak_hours(airport_flights)
        
        # Create runway capacity model
        capacity_analysis = self._model_runway_capacity(airport_flights)
        
        # Generate congestion scoring
        congestion_scores = self._calculate_congestion_scores(airport_flights)
        
        # Train Prophet model for forecasting
        forecast_data = self._train_prophet_model(airport_flights)
        
        # Generate alternative time slot recommendations
        alternatives = self._recommend_alternative_slots(airport_flights, congestion_scores)
        
        # Generate recommendations
        recommendations = self._generate_congestion_recommendations(
            density_metrics, peak_hours, capacity_analysis, congestion_scores, alternatives
        )
        
        # Compile results
        metrics = {
            'flight_density_metrics': density_metrics,
            'peak_hours_analysis': peak_hours,
            'runway_capacity_analysis': capacity_analysis,
            'congestion_scores': congestion_scores,
            'forecast_data': forecast_data,
            'alternative_slots': alternatives,
            'total_flights_analyzed': len(airport_flights)
        }
        
        return AnalysisResult(
            analysis_type="congestion_analysis",
            airport_code=airport_code,
            metrics=metrics,
            recommendations=recommendations,
            confidence_score=self._calculate_confidence_score(airport_flights),
            data_sources=["flight_data"]
        )
    
    def predict_congestion(self, airport_code: str, target_datetime: datetime,
                          historical_data: List[FlightData]) -> Tuple[float, str]:
        """
        Predict congestion level for a specific time.
        
        Args:
            airport_code: Airport code
            target_datetime: Target datetime for prediction
            historical_data: Historical flight data for training
            
        Returns:
            Tuple of (congestion_score, congestion_level)
        """
        df = self._flights_to_dataframe(historical_data)
        airport_flights = df[
            (df['origin_airport'] == airport_code) | 
            (df['destination_airport'] == airport_code)
        ].copy()
        
        if airport_flights.empty:
            return 0.0, "Unknown"
        
        # Train model if not already trained
        if not self.is_trained:
            self._train_prophet_model(airport_flights)
        
        # Make prediction using Prophet
        if self.prophet_model:
            future_df = pd.DataFrame({
                'ds': [target_datetime],
                'hour': [target_datetime.hour],
                'day_of_week': [target_datetime.weekday()]
            })
            
            try:
                forecast = self.prophet_model.predict(future_df)
                predicted_flights = max(0, forecast['yhat'].iloc[0])
                
                # Convert to congestion score (0-1)
                congestion_score = min(1.0, predicted_flights / self.runway_capacity)
                congestion_level = self._get_congestion_level(congestion_score)
                
                return congestion_score, congestion_level
            except Exception as e:
                logger.warning(f"Prophet prediction failed: {e}")
        
        # Fallback to historical average
        hour = target_datetime.hour
        hourly_avg = airport_flights[
            airport_flights['scheduled_departure'].dt.hour == hour
        ].groupby(airport_flights['scheduled_departure'].dt.date).size().mean()
        
        congestion_score = min(1.0, hourly_avg / self.runway_capacity) if hourly_avg else 0.0
        congestion_level = self._get_congestion_level(congestion_score)
        
        return congestion_score, congestion_level
    
    def find_least_congested_slots(self, airport_code: str, flight_data: List[FlightData],
                                  duration_hours: int = 24) -> List[Dict]:
        """
        Find the least congested time slots for scheduling.
        
        Args:
            airport_code: Airport code to analyze
            flight_data: Historical flight data
            duration_hours: Time window to analyze (default 24 hours)
            
        Returns:
            List of least congested time slots with recommendations
        """
        df = self._flights_to_dataframe(flight_data)
        airport_flights = df[
            (df['origin_airport'] == airport_code) | 
            (df['destination_airport'] == airport_code)
        ].copy()
        
        if airport_flights.empty:
            return []
        
        # Group by hour of day
        airport_flights['hour'] = airport_flights['scheduled_departure'].dt.hour
        hourly_density = airport_flights.groupby('hour').size().reset_index(name='flight_count')
        
        # Calculate congestion scores
        hourly_density['congestion_score'] = hourly_density['flight_count'] / self.runway_capacity
        hourly_density['congestion_level'] = hourly_density['congestion_score'].apply(
            self._get_congestion_level
        )
        
        # Sort by congestion score (least congested first)
        least_congested = hourly_density.sort_values('congestion_score').to_dict('records')
        
        # Add recommendations
        for slot in least_congested:
            slot['recommendation'] = self._get_slot_recommendation(slot)
        
        return least_congested[:12]  # Return top 12 hours
    
    def _flights_to_dataframe(self, flight_data: List[FlightData]) -> pd.DataFrame:
        """Convert flight data objects to pandas DataFrame."""
        data = []
        for flight in flight_data:
            data.append({
                'flight_id': flight.flight_id,
                'airline': flight.airline,
                'flight_number': flight.flight_number,
                'origin_airport': flight.origin_airport,
                'destination_airport': flight.destination_airport,
                'scheduled_departure': flight.scheduled_departure,
                'scheduled_arrival': flight.scheduled_arrival,
                'runway_used': flight.runway_used,
                'passenger_count': flight.passenger_count or 0
            })
        
        return pd.DataFrame(data)
    
    def _calculate_flight_density(self, df: pd.DataFrame) -> Dict:
        """Calculate flight density metrics."""
        # Group by hour to calculate density
        df['hour'] = df['scheduled_departure'].dt.hour
        df['date'] = df['scheduled_departure'].dt.date
        
        # Calculate flights per hour statistics
        hourly_counts = df.groupby(['date', 'hour']).size()
        
        return {
            'average_flights_per_hour': float(hourly_counts.mean()),
            'max_flights_per_hour': int(hourly_counts.max()),
            'min_flights_per_hour': int(hourly_counts.min()),
            'std_flights_per_hour': float(hourly_counts.std()),
            'runway_capacity': self.runway_capacity,
            'capacity_utilization': float(hourly_counts.mean() / self.runway_capacity * 100),
            'peak_utilization': float(hourly_counts.max() / self.runway_capacity * 100)
        }
    
    def _identify_peak_hours(self, df: pd.DataFrame) -> Dict:
        """Identify peak hours using time series analysis."""
        # Group by hour of day
        df['hour'] = df['scheduled_departure'].dt.hour
        hourly_counts = df.groupby('hour').size().reset_index(name='flight_count')
        
        # Calculate statistics
        mean_flights = hourly_counts['flight_count'].mean()
        std_flights = hourly_counts['flight_count'].std()
        
        # Define peak hours as those above mean + 1 std
        peak_threshold = mean_flights + std_flights
        peak_hours = hourly_counts[hourly_counts['flight_count'] >= peak_threshold]
        
        # Define off-peak hours
        off_peak_threshold = mean_flights - 0.5 * std_flights
        off_peak_hours = hourly_counts[hourly_counts['flight_count'] <= off_peak_threshold]
        
        return {
            'peak_hours': peak_hours['hour'].tolist(),
            'off_peak_hours': off_peak_hours['hour'].tolist(),
            'peak_threshold': float(peak_threshold),
            'hourly_flight_counts': hourly_counts.to_dict('records'),
            'busiest_hour': int(hourly_counts.loc[hourly_counts['flight_count'].idxmax(), 'hour']),
            'quietest_hour': int(hourly_counts.loc[hourly_counts['flight_count'].idxmin(), 'hour'])
        }
    
    def _model_runway_capacity(self, df: pd.DataFrame) -> Dict:
        """Model runway capacity constraints."""
        # Group by hour and date to get actual utilization
        df['hour'] = df['scheduled_departure'].dt.hour
        df['date'] = df['scheduled_departure'].dt.date
        
        hourly_utilization = df.groupby(['date', 'hour']).size()
        
        # Calculate capacity metrics
        over_capacity_instances = (hourly_utilization > self.runway_capacity).sum()
        total_instances = len(hourly_utilization)
        
        # Identify capacity bottlenecks
        bottleneck_hours = df.groupby('hour').size()
        bottleneck_hours = bottleneck_hours[bottleneck_hours > self.runway_capacity * 0.8]
        
        return {
            'runway_capacity': self.runway_capacity,
            'over_capacity_instances': int(over_capacity_instances),
            'over_capacity_percentage': float(over_capacity_instances / total_instances * 100) if total_instances > 0 else 0.0,
            'average_utilization': float(hourly_utilization.mean()),
            'max_utilization': int(hourly_utilization.max()),
            'bottleneck_hours': bottleneck_hours.index.tolist(),
            'capacity_buffer_needed': max(0, int(hourly_utilization.max() - self.runway_capacity))
        }
    
    def _calculate_congestion_scores(self, df: pd.DataFrame) -> Dict:
        """Calculate hourly congestion scores and rankings."""
        # Group by hour
        df['hour'] = df['scheduled_departure'].dt.hour
        hourly_counts = df.groupby('hour').size().reset_index(name='flight_count')
        
        # Calculate congestion scores (0-1 scale)
        hourly_counts['congestion_score'] = hourly_counts['flight_count'] / self.runway_capacity
        hourly_counts['congestion_score'] = hourly_counts['congestion_score'].clip(0, 1)
        
        # Add congestion levels
        hourly_counts['congestion_level'] = hourly_counts['congestion_score'].apply(
            self._get_congestion_level
        )
        
        # Rank hours by congestion
        hourly_counts['congestion_rank'] = hourly_counts['congestion_score'].rank(
            method='dense', ascending=False
        ).astype(int)
        
        return {
            'hourly_scores': hourly_counts.to_dict('records'),
            'most_congested_hour': int(hourly_counts.loc[hourly_counts['congestion_score'].idxmax(), 'hour']),
            'least_congested_hour': int(hourly_counts.loc[hourly_counts['congestion_score'].idxmin(), 'hour']),
            'average_congestion_score': float(hourly_counts['congestion_score'].mean())
        }
    
    def _train_prophet_model(self, df: pd.DataFrame) -> Dict:
        """Train Prophet model for congestion forecasting."""
        try:
            # Prepare data for Prophet
            df['date'] = df['scheduled_departure'].dt.date
            df['hour'] = df['scheduled_departure'].dt.hour
            
            # Create hourly time series
            hourly_data = df.groupby([df['scheduled_departure'].dt.floor('h')]).size().reset_index()
            hourly_data.columns = ['ds', 'y']
            
            if len(hourly_data) < 10:  # Need minimum data for Prophet
                logger.warning("Insufficient data for Prophet model training")
                return {'model_trained': False, 'forecast_available': False}
            
            # Initialize and train Prophet model
            self.prophet_model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.8
            )
            
            # Add custom regressors
            self.prophet_model.add_regressor('hour')
            self.prophet_model.add_regressor('day_of_week')
            
            # Add hour and day_of_week features
            hourly_data['hour'] = hourly_data['ds'].dt.hour
            hourly_data['day_of_week'] = hourly_data['ds'].dt.dayofweek
            
            self.prophet_model.fit(hourly_data)
            
            # Generate forecast for next 24 hours
            future = self.prophet_model.make_future_dataframe(periods=24, freq='h')
            future['hour'] = future['ds'].dt.hour
            future['day_of_week'] = future['ds'].dt.dayofweek
            
            forecast = self.prophet_model.predict(future)
            
            self.is_trained = True
            
            return {
                'model_trained': True,
                'forecast_available': True,
                'training_data_points': len(hourly_data),
                'forecast_period_hours': 24,
                'model_performance': {
                    'mae': float(np.mean(np.abs(forecast['yhat'][-len(hourly_data):] - hourly_data['y']))),
                    'mape': float(np.mean(np.abs((forecast['yhat'][-len(hourly_data):] - hourly_data['y']) / hourly_data['y'])) * 100)
                }
            }
            
        except Exception as e:
            logger.error(f"Prophet model training failed: {e}")
            return {'model_trained': False, 'forecast_available': False, 'error': str(e)}
    
    def _recommend_alternative_slots(self, df: pd.DataFrame, congestion_scores: Dict) -> List[Dict]:
        """Recommend alternative time slots for congested periods."""
        hourly_scores = pd.DataFrame(congestion_scores['hourly_scores'])
        
        # Find highly congested hours (score > 0.7)
        congested_hours = hourly_scores[hourly_scores['congestion_score'] > 0.7]
        
        # Find low congestion alternatives (score < 0.4)
        alternatives = hourly_scores[hourly_scores['congestion_score'] < 0.4]
        
        recommendations = []
        
        for _, congested_hour in congested_hours.iterrows():
            # Find the best alternative within +/- 3 hours
            hour = congested_hour['hour']
            nearby_alternatives = alternatives[
                (alternatives['hour'] >= (hour - 3) % 24) |
                (alternatives['hour'] <= (hour + 3) % 24)
            ]
            
            if not nearby_alternatives.empty:
                best_alternative = nearby_alternatives.loc[
                    nearby_alternatives['congestion_score'].idxmin()
                ]
                
                recommendations.append({
                    'congested_hour': int(hour),
                    'congested_score': float(congested_hour['congestion_score']),
                    'alternative_hour': int(best_alternative['hour']),
                    'alternative_score': float(best_alternative['congestion_score']),
                    'improvement': float(congested_hour['congestion_score'] - best_alternative['congestion_score'])
                })
        
        return recommendations
    
    def _generate_congestion_recommendations(self, density_metrics: Dict, peak_hours: Dict,
                                           capacity_analysis: Dict, congestion_scores: Dict,
                                           alternatives: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on congestion analysis."""
        recommendations = []
        
        # Capacity utilization recommendations
        if density_metrics['capacity_utilization'] > 80:
            recommendations.append(
                f"High capacity utilization at {density_metrics['capacity_utilization']:.1f}%. "
                "Consider increasing runway capacity or redistributing flights."
            )
        
        # Peak hours recommendations
        if peak_hours['peak_hours']:
            peak_hours_str = ', '.join([f"{h:02d}:00" for h in peak_hours['peak_hours'][:3]])
            recommendations.append(
                f"Peak congestion hours identified: {peak_hours_str}. "
                "Avoid scheduling additional flights during these periods."
            )
        
        if peak_hours['off_peak_hours']:
            off_peak_str = ', '.join([f"{h:02d}:00" for h in peak_hours['off_peak_hours'][:3]])
            recommendations.append(
                f"Low congestion opportunities: {off_peak_str}. "
                "Consider moving flights to these time slots."
            )
        
        # Capacity constraint recommendations
        if capacity_analysis['over_capacity_percentage'] > 10:
            recommendations.append(
                f"Runway capacity exceeded in {capacity_analysis['over_capacity_percentage']:.1f}% of time periods. "
                "Urgent schedule optimization needed."
            )
        
        if capacity_analysis['bottleneck_hours']:
            bottleneck_str = ', '.join([f"{h:02d}:00" for h in capacity_analysis['bottleneck_hours'][:3]])
            recommendations.append(
                f"Capacity bottlenecks at: {bottleneck_str}. "
                "These hours require immediate attention."
            )
        
        # Alternative slot recommendations
        if alternatives:
            recommendations.append(
                f"Found {len(alternatives)} alternative time slots that could reduce congestion by "
                f"up to {max([alt['improvement'] for alt in alternatives]):.2f} points."
            )
        
        # Overall congestion level
        avg_congestion = congestion_scores['average_congestion_score']
        if avg_congestion > 0.6:
            recommendations.append(
                "Overall congestion level is high. Implement comprehensive schedule redistribution."
            )
        elif avg_congestion < 0.3:
            recommendations.append(
                "Overall congestion level is manageable. Focus on peak hour optimization."
            )
        
        return recommendations
    
    def _get_congestion_level(self, score: float) -> str:
        """Convert congestion score to descriptive level."""
        if score < 0.3:
            return "Low"
        elif score < 0.6:
            return "Moderate"
        elif score < 0.8:
            return "High"
        else:
            return "Critical"
    
    def _get_slot_recommendation(self, slot: Dict) -> str:
        """Generate recommendation for a specific time slot."""
        score = slot['congestion_score']
        level = slot['congestion_level']
        
        if score < 0.3:
            return f"{level} congestion - Excellent time for additional flights"
        elif score < 0.6:
            return f"{level} congestion - Good time for scheduling"
        elif score < 0.8:
            return f"{level} congestion - Consider alternative times"
        else:
            return f"{level} congestion - Avoid scheduling here"
    
    def _calculate_confidence_score(self, df: pd.DataFrame) -> float:
        """Calculate confidence score based on data quality and quantity."""
        base_confidence = 0.5
        
        # Increase confidence with more data
        data_factor = min(0.3, len(df) / 1000)
        
        # Increase confidence with data completeness
        completeness_factor = 0.2 * (
            df['scheduled_departure'].notna().mean()
        )
        
        return min(0.95, base_confidence + data_factor + completeness_factor)
    
    def _empty_analysis_result(self, airport_code: str) -> AnalysisResult:
        """Return empty analysis result when no data is available."""
        return AnalysisResult(
            analysis_type="congestion_analysis",
            airport_code=airport_code,
            metrics={'error': 'No flight data available for analysis'},
            recommendations=["Insufficient data for congestion analysis"],
            confidence_score=0.0,
            data_sources=[]
        )