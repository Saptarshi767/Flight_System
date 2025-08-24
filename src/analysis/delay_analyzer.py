"""
Delay Analysis Engine for Flight Scheduling Optimization

This module implements scheduled vs actual time comparison algorithms to find best takeoff/landing times.
Uses scikit-learn for delay pattern recognition and provides optimal time slot identification.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging

from ..data.models import FlightData, DelayCategory, AnalysisResult

logger = logging.getLogger(__name__)


class DelayAnalyzer:
    """
    Delay analysis engine for identifying optimal takeoff/landing times.
    
    This class implements EXPECTATION 2: scheduled vs actual time comparison algorithms
    to find best takeoff/landing times using open source AI tools (scikit-learn).
    """
    
    def __init__(self):
        """Initialize the delay analyzer with ML models."""
        self.delay_predictor = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.time_clusterer = KMeans(
            n_clusters=24,  # 24 hours
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def analyze_delays_from_data(self, flight_data: List[Dict], airport_code: str, 
                               include_weather: bool = True, granularity: str = "hourly") -> Dict:
        """
        Simplified delay analysis for API usage.
        
        Args:
            flight_data: List of flight data dictionaries
            airport_code: Airport code to analyze
            include_weather: Whether to include weather impact
            granularity: Analysis granularity (hourly, daily)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if not flight_data:
                return {
                    "airport_code": airport_code,
                    "analysis_period": {"start": None, "end": None},
                    "average_delay": 0,
                    "median_delay": 0,
                    "delay_by_hour": {},
                    "delay_by_category": {},
                    "optimal_time_slots": [],
                    "recommendations": ["No flight data available for analysis"]
                }
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(flight_data)
            
            # Calculate delays
            delays = []
            for _, row in df.iterrows():
                if row.get('actual_departure') and row.get('scheduled_departure'):
                    if isinstance(row['actual_departure'], str):
                        actual = pd.to_datetime(row['actual_departure'])
                        scheduled = pd.to_datetime(row['scheduled_departure'])
                    else:
                        actual = row['actual_departure']
                        scheduled = row['scheduled_departure']
                    
                    delay = (actual - scheduled).total_seconds() / 60
                    delays.append(max(0, delay))  # Only positive delays
                else:
                    delays.append(row.get('delay_minutes', 0))
            
            df['delay_minutes'] = delays
            
            # Add hour column for analysis
            df['hour'] = pd.to_datetime(df['scheduled_departure']).dt.hour
            
            # Calculate statistics
            avg_delay = df['delay_minutes'].mean()
            median_delay = df['delay_minutes'].median()
            
            # Delay by hour
            delay_by_hour = df.groupby('hour')['delay_minutes'].mean().to_dict()
            
            # Find optimal time slots (hours with lowest average delays)
            optimal_hours = df.groupby('hour')['delay_minutes'].mean().nsmallest(3)
            optimal_time_slots = [
                {
                    "hour": int(hour),
                    "average_delay_minutes": float(delay),
                    "recommendation": f"Optimal time slot: {hour:02d}:00-{hour+1:02d}:00"
                }
                for hour, delay in optimal_hours.items()
            ]
            
            # Generate recommendations
            recommendations = []
            if avg_delay > 15:
                recommendations.append("Consider rescheduling flights to off-peak hours")
            if len(optimal_time_slots) > 0:
                best_hour = optimal_time_slots[0]['hour']
                recommendations.append(f"Best time slot for minimal delays: {best_hour:02d}:00-{best_hour+1:02d}:00")
            
            # Analysis period
            dates = pd.to_datetime(df['scheduled_departure'])
            analysis_period = {
                "start": dates.min().isoformat() if not dates.empty else None,
                "end": dates.max().isoformat() if not dates.empty else None
            }
            
            return {
                "airport_code": airport_code,
                "analysis_period": analysis_period,
                "average_delay": float(avg_delay),
                "median_delay": float(median_delay),
                "delay_by_hour": {str(k): float(v) for k, v in delay_by_hour.items()},
                "delay_by_category": {"operational": float(avg_delay)},  # Simplified
                "optimal_time_slots": optimal_time_slots,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in delay analysis: {str(e)}")
            return {
                "airport_code": airport_code,
                "error": str(e),
                "recommendations": ["Analysis failed due to data processing error"]
            }
    
    def analyze_delays(self, flight_data: List[FlightData], airport_code: str) -> AnalysisResult:
        """
        Perform comprehensive delay analysis for an airport.
        
        Args:
            flight_data: List of flight data objects
            airport_code: Airport code to analyze
            
        Returns:
            AnalysisResult with delay patterns and recommendations
        """
        logger.info(f"Starting delay analysis for airport {airport_code}")
        
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
        
        # Calculate delay metrics
        delay_metrics = self._calculate_delay_metrics(airport_flights)
        
        # Identify optimal time slots
        optimal_times = self._identify_optimal_time_slots(airport_flights)
        
        # Categorize delays by cause
        delay_categories = self._categorize_delays(airport_flights)
        
        # Generate time-of-day analysis
        time_analysis = self._analyze_time_of_day_patterns(airport_flights)
        
        # Train models if not already trained
        if not self.is_trained:
            self._train_models(airport_flights)
        
        # Generate recommendations
        recommendations = self._generate_delay_recommendations(
            delay_metrics, optimal_times, delay_categories, time_analysis
        )
        
        # Compile results
        metrics = {
            'delay_metrics': delay_metrics,
            'optimal_time_slots': optimal_times,
            'delay_categories': delay_categories,
            'time_of_day_analysis': time_analysis,
            'total_flights_analyzed': len(airport_flights)
        }
        
        return AnalysisResult(
            analysis_type="delay_analysis",
            airport_code=airport_code,
            metrics=metrics,
            recommendations=recommendations,
            confidence_score=self._calculate_confidence_score(airport_flights),
            data_sources=["flight_data"]
        )
    
    def predict_delay(self, flight_features: Dict) -> Tuple[float, float]:
        """
        Predict delay for a flight based on features.
        
        Args:
            flight_features: Dictionary of flight features
            
        Returns:
            Tuple of (predicted_delay_minutes, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert features to array
        feature_array = self._extract_features_for_prediction(flight_features)
        
        # Make prediction
        predicted_delay = self.delay_predictor.predict([feature_array])[0]
        
        # Calculate confidence based on model performance
        confidence = min(0.95, max(0.1, 1.0 - abs(predicted_delay) / 120))  # Normalize by 2 hours
        
        return predicted_delay, confidence
    
    def find_best_departure_times(self, airport_code: str, flight_data: List[FlightData], 
                                 duration_hours: int = 24) -> List[Dict]:
        """
        Find the best departure times with minimal delays.
        
        Args:
            airport_code: Airport code to analyze
            flight_data: Historical flight data
            duration_hours: Time window to analyze (default 24 hours)
            
        Returns:
            List of optimal time slots with delay statistics
        """
        df = self._flights_to_dataframe(flight_data)
        airport_flights = df[df['origin_airport'] == airport_code].copy()
        
        if airport_flights.empty:
            return []
        
        # Group by hour of day
        airport_flights['departure_hour'] = airport_flights['scheduled_departure'].dt.hour
        hourly_stats = airport_flights.groupby('departure_hour').agg({
            'delay_minutes': ['mean', 'std', 'count'],
            'flight_id': 'count'
        }).round(2)
        
        # Flatten column names
        hourly_stats.columns = ['avg_delay', 'delay_std', 'delay_count', 'flight_count']
        hourly_stats = hourly_stats.reset_index()
        
        # Calculate delay score (lower is better)
        hourly_stats['delay_score'] = (
            hourly_stats['avg_delay'] + 
            hourly_stats['delay_std'].fillna(0) * 0.5
        )
        
        # Sort by delay score (best times first)
        best_times = hourly_stats.sort_values('delay_score').to_dict('records')
        
        # Add recommendations
        for time_slot in best_times:
            time_slot['recommendation'] = self._get_time_slot_recommendation(time_slot)
        
        return best_times[:12]  # Return top 12 hours
    
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
                'delay_minutes': max(departure_delay, arrival_delay),  # Use maximum delay
                'delay_category': flight.delay_category.value if flight.delay_category else 'other',
                'runway_used': flight.runway_used,
                'passenger_count': flight.passenger_count or 0
            })
        
        return pd.DataFrame(data)
    
    def _calculate_delay_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive delay metrics."""
        return {
            'average_delay_minutes': float(df['delay_minutes'].mean()),
            'median_delay_minutes': float(df['delay_minutes'].median()),
            'std_delay_minutes': float(df['delay_minutes'].std()),
            'max_delay_minutes': float(df['delay_minutes'].max()),
            'min_delay_minutes': float(df['delay_minutes'].min()),
            'on_time_percentage': float((df['delay_minutes'] <= 15).mean() * 100),
            'severe_delay_percentage': float((df['delay_minutes'] > 60).mean() * 100),
            'total_flights': len(df)
        }
    
    def _identify_optimal_time_slots(self, df: pd.DataFrame) -> Dict:
        """Identify optimal time slots based on historical delay patterns."""
        # Group by hour of day
        df['hour'] = df['scheduled_departure'].dt.hour
        hourly_delays = df.groupby('hour')['delay_minutes'].agg(['mean', 'count']).reset_index()
        
        # Find hours with lowest average delays and sufficient data
        min_flights_threshold = max(1, len(df) // 50)  # At least 2% of total flights
        valid_hours = hourly_delays[hourly_delays['count'] >= min_flights_threshold]
        
        if valid_hours.empty:
            return {'optimal_hours': [], 'worst_hours': []}
        
        # Sort by average delay
        sorted_hours = valid_hours.sort_values('mean')
        
        optimal_hours = sorted_hours.head(6)['hour'].tolist()  # Top 6 hours
        worst_hours = sorted_hours.tail(3)['hour'].tolist()    # Bottom 3 hours
        
        return {
            'optimal_hours': optimal_hours,
            'worst_hours': worst_hours,
            'hourly_delay_stats': hourly_delays.to_dict('records')
        }
    
    def _categorize_delays(self, df: pd.DataFrame) -> Dict:
        """Categorize delays by cause using pattern recognition."""
        delay_categories = df['delay_category'].value_counts().to_dict()
        
        # Calculate statistics for each category
        category_stats = {}
        for category in DelayCategory:
            category_flights = df[df['delay_category'] == category.value]
            if not category_flights.empty:
                category_stats[category.value] = {
                    'count': len(category_flights),
                    'avg_delay': float(category_flights['delay_minutes'].mean()),
                    'percentage': float(len(category_flights) / len(df) * 100)
                }
        
        return {
            'category_counts': delay_categories,
            'category_statistics': category_stats
        }
    
    def _analyze_time_of_day_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze delay patterns by time of day."""
        # Create time periods
        df['time_period'] = pd.cut(
            df['scheduled_departure'].dt.hour,
            bins=[0, 6, 12, 18, 24],
            labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'],
            include_lowest=True
        )
        
        period_stats = df.groupby('time_period', observed=True)['delay_minutes'].agg([
            'mean', 'median', 'std', 'count'
        ]).round(2)
        
        return {
            'period_statistics': period_stats.to_dict('index'),
            'best_period': period_stats['mean'].idxmin(),
            'worst_period': period_stats['mean'].idxmax()
        }
    
    def _train_models(self, df: pd.DataFrame):
        """Train ML models for delay prediction and pattern recognition."""
        logger.info("Training delay prediction models")
        
        # Prepare features for training
        features = self._extract_features(df)
        target = df['delay_minutes'].values
        
        # Remove rows with missing target values
        valid_mask = ~np.isnan(target)
        features = features[valid_mask]
        target = target[valid_mask]
        
        if len(features) < 10:  # Need minimum data for training
            logger.warning("Insufficient data for model training")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train delay predictor
        self.delay_predictor.fit(X_train_scaled, y_train)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_train_scaled)
        
        # Evaluate model
        y_pred = self.delay_predictor.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model training completed. MAE: {mae:.2f}, R2: {r2:.3f}")
        self.is_trained = True
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for ML model training."""
        features = []
        
        for _, row in df.iterrows():
            feature_vector = [
                row['scheduled_departure'].hour,  # Hour of day
                row['scheduled_departure'].weekday(),  # Day of week
                row['scheduled_departure'].month,  # Month
                len(row['flight_number']),  # Flight number length (proxy for airline)
                row['passenger_count'] if pd.notna(row['passenger_count']) else 150,  # Default passenger count
                1 if row['aircraft_type'] != 'Unknown' else 0,  # Has aircraft type info
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_features_for_prediction(self, flight_features: Dict) -> np.ndarray:
        """Extract features for a single flight prediction."""
        scheduled_time = flight_features.get('scheduled_departure', datetime.now())
        
        return np.array([
            scheduled_time.hour,
            scheduled_time.weekday(),
            scheduled_time.month,
            len(flight_features.get('flight_number', '')),
            flight_features.get('passenger_count', 150),
            1 if flight_features.get('aircraft_type') else 0
        ])
    
    def _generate_delay_recommendations(self, delay_metrics: Dict, optimal_times: Dict, 
                                      delay_categories: Dict, time_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on delay analysis."""
        recommendations = []
        
        # Overall delay recommendations
        if delay_metrics['average_delay_minutes'] > 30:
            recommendations.append(
                f"High average delay of {delay_metrics['average_delay_minutes']:.1f} minutes detected. "
                "Consider schedule optimization."
            )
        
        # Optimal time recommendations
        if optimal_times['optimal_hours']:
            optimal_hours_str = ', '.join([f"{h:02d}:00" for h in optimal_times['optimal_hours'][:3]])
            recommendations.append(
                f"Best departure times with minimal delays: {optimal_hours_str}"
            )
        
        if optimal_times['worst_hours']:
            worst_hours_str = ', '.join([f"{h:02d}:00" for h in optimal_times['worst_hours']])
            recommendations.append(
                f"Avoid scheduling during peak delay hours: {worst_hours_str}"
            )
        
        # Category-specific recommendations
        category_stats = delay_categories.get('category_statistics', {})
        if 'weather' in category_stats and category_stats['weather']['percentage'] > 30:
            recommendations.append(
                "Weather delays account for significant portion of delays. "
                "Consider weather-based scheduling adjustments."
            )
        
        if 'operational' in category_stats and category_stats['operational']['percentage'] > 25:
            recommendations.append(
                "High operational delays detected. Review ground operations and turnaround procedures."
            )
        
        # Time period recommendations
        best_period = time_analysis.get('best_period')
        if best_period:
            recommendations.append(f"Schedule more flights during {best_period} for optimal performance.")
        
        return recommendations
    
    def _get_time_slot_recommendation(self, time_slot: Dict) -> str:
        """Generate recommendation for a specific time slot."""
        avg_delay = time_slot['avg_delay']
        flight_count = time_slot['flight_count']
        
        if avg_delay < 10:
            return "Excellent - Very low delays"
        elif avg_delay < 20:
            return "Good - Below average delays"
        elif avg_delay < 40:
            return "Fair - Moderate delays expected"
        else:
            return "Poor - High delays, consider alternative times"
    
    def _calculate_confidence_score(self, df: pd.DataFrame) -> float:
        """Calculate confidence score based on data quality and quantity."""
        base_confidence = 0.5
        
        # Increase confidence with more data
        data_factor = min(0.3, len(df) / 1000)
        
        # Increase confidence with more complete data
        completeness_factor = 0.2 * (
            df['actual_departure'].notna().mean() + 
            df['actual_arrival'].notna().mean()
        ) / 2
        
        return min(0.95, base_confidence + data_factor + completeness_factor)
    
    def _empty_analysis_result(self, airport_code: str) -> AnalysisResult:
        """Return empty analysis result when no data is available."""
        return AnalysisResult(
            analysis_type="delay_analysis",
            airport_code=airport_code,
            metrics={'error': 'No flight data available for analysis'},
            recommendations=["Insufficient data for delay analysis"],
            confidence_score=0.0,
            data_sources=[]
        )