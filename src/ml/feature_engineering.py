"""
Feature engineering for flight delay prediction using scikit-learn
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering class for flight delay prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_selector = SelectKBest(score_func=f_regression, k='all')
        self.is_fitted = False
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime columns"""
        df = df.copy()
        
        # Extract features from scheduled_departure
        if 'scheduled_departure' in df.columns:
            df['departure_hour'] = pd.to_datetime(df['scheduled_departure']).dt.hour
            df['departure_day_of_week'] = pd.to_datetime(df['scheduled_departure']).dt.dayofweek
            df['departure_month'] = pd.to_datetime(df['scheduled_departure']).dt.month
            df['departure_quarter'] = pd.to_datetime(df['scheduled_departure']).dt.quarter
            df['is_weekend'] = (df['departure_day_of_week'] >= 5).astype(int)
            df['is_peak_hour'] = ((df['departure_hour'] >= 6) & (df['departure_hour'] <= 10) | 
                                 (df['departure_hour'] >= 17) & (df['departure_hour'] <= 21)).astype(int)
        
        # Extract features from scheduled_arrival
        if 'scheduled_arrival' in df.columns:
            df['arrival_hour'] = pd.to_datetime(df['scheduled_arrival']).dt.hour
            df['arrival_day_of_week'] = pd.to_datetime(df['scheduled_arrival']).dt.dayofweek
        
        # Calculate flight duration
        if 'scheduled_departure' in df.columns and 'scheduled_arrival' in df.columns:
            departure_dt = pd.to_datetime(df['scheduled_departure'])
            arrival_dt = pd.to_datetime(df['scheduled_arrival'])
            df['scheduled_duration_minutes'] = (arrival_dt - departure_dt).dt.total_seconds() / 60
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-based features"""
        df = df.copy()
        
        if 'weather_conditions' in df.columns:
            # Extract weather information if available
            weather_data = df['weather_conditions'].fillna({})
            
            # Create weather features
            df['temperature'] = weather_data.apply(lambda x: x.get('temperature', 20) if isinstance(x, dict) else 20)
            df['humidity'] = weather_data.apply(lambda x: x.get('humidity', 50) if isinstance(x, dict) else 50)
            df['wind_speed'] = weather_data.apply(lambda x: x.get('wind_speed', 10) if isinstance(x, dict) else 10)
            df['visibility'] = weather_data.apply(lambda x: x.get('visibility', 10) if isinstance(x, dict) else 10)
            df['precipitation'] = weather_data.apply(lambda x: x.get('precipitation', 0) if isinstance(x, dict) else 0)
            
            # Weather condition categories
            df['is_bad_weather'] = ((df['wind_speed'] > 25) | 
                                   (df['visibility'] < 5) | 
                                   (df['precipitation'] > 0)).astype(int)
        else:
            # Default weather features if no weather data available
            df['temperature'] = 20
            df['humidity'] = 50
            df['wind_speed'] = 10
            df['visibility'] = 10
            df['precipitation'] = 0
            df['is_bad_weather'] = 0
        
        return df
    
    def create_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create traffic and congestion features"""
        df = df.copy()
        
        # Calculate hourly flight counts for congestion
        if 'scheduled_departure' in df.columns and 'origin_airport' in df.columns:
            df['departure_datetime'] = pd.to_datetime(df['scheduled_departure'])
            df['departure_date'] = df['departure_datetime'].dt.date
            df['departure_hour'] = df['departure_datetime'].dt.hour
            
            # Count flights per hour per airport
            hourly_counts = df.groupby(['origin_airport', 'departure_date', 'departure_hour']).size().reset_index(name='hourly_departures')
            df = df.merge(hourly_counts, on=['origin_airport', 'departure_date', 'departure_hour'], how='left')
            df['hourly_departures'] = df['hourly_departures'].fillna(1)
            
            # Calculate congestion score (normalized by airport capacity)
            airport_capacity = {'BOM': 60, 'DEL': 70}  # flights per hour
            df['congestion_score'] = df.apply(
                lambda row: min(row['hourly_departures'] / airport_capacity.get(row['origin_airport'], 50), 1.0),
                axis=1
            )
        else:
            df['hourly_departures'] = 1
            df['congestion_score'] = 0.1
        
        return df
    
    def create_operational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create operational features"""
        df = df.copy()
        
        # Airline performance features
        if 'airline' in df.columns:
            # Calculate airline delay statistics
            airline_stats = df.groupby('airline')['delay_minutes'].agg(['mean', 'std']).reset_index()
            airline_stats.columns = ['airline', 'airline_avg_delay', 'airline_delay_std']
            df = df.merge(airline_stats, on='airline', how='left')
            df['airline_avg_delay'] = df['airline_avg_delay'].fillna(df['delay_minutes'].mean())
            df['airline_delay_std'] = df['airline_delay_std'].fillna(df['delay_minutes'].std())
        
        # Aircraft type features
        if 'aircraft_type' in df.columns:
            # Group similar aircraft types
            df['aircraft_category'] = df['aircraft_type'].apply(self._categorize_aircraft)
        else:
            df['aircraft_category'] = 'unknown'
        
        # Route features
        if 'origin_airport' in df.columns and 'destination_airport' in df.columns:
            df['route'] = df['origin_airport'] + '_' + df['destination_airport']
            
            # Calculate route statistics
            route_stats = df.groupby('route')['delay_minutes'].agg(['mean', 'count']).reset_index()
            route_stats.columns = ['route', 'route_avg_delay', 'route_frequency']
            df = df.merge(route_stats, on='route', how='left')
            df['route_avg_delay'] = df['route_avg_delay'].fillna(df['delay_minutes'].mean())
            df['route_frequency'] = df['route_frequency'].fillna(1)
        
        # Passenger load features
        if 'passenger_count' in df.columns:
            df['passenger_load'] = df['passenger_count'].fillna(df['passenger_count'].median())
            df['is_high_load'] = (df['passenger_load'] > df['passenger_load'].quantile(0.75)).astype(int)
        else:
            df['passenger_load'] = 150  # Default passenger count
            df['is_high_load'] = 0
        
        return df
    
    def _categorize_aircraft(self, aircraft_type: str) -> str:
        """Categorize aircraft types into groups"""
        if pd.isna(aircraft_type):
            return 'unknown'
        
        aircraft_type = str(aircraft_type).upper()
        
        if any(x in aircraft_type for x in ['A320', 'A321', 'A319', 'B737', 'B738']):
            return 'narrow_body'
        elif any(x in aircraft_type for x in ['A330', 'A340', 'A350', 'B777', 'B787', 'B747']):
            return 'wide_body'
        elif any(x in aircraft_type for x in ['ATR', 'DASH', 'CRJ', 'EMB']):
            return 'regional'
        else:
            return 'other'
    
    def create_lag_features(self, df: pd.DataFrame, lag_periods: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """Create lag features for time series analysis"""
        df = df.copy()
        df = df.sort_values(['origin_airport', 'scheduled_departure'])
        
        for lag in lag_periods:
            # Lag features for delay
            df[f'delay_lag_{lag}'] = df.groupby('origin_airport')['delay_minutes'].shift(lag)
            
            # Lag features for congestion
            if 'congestion_score' in df.columns:
                df[f'congestion_lag_{lag}'] = df.groupby('origin_airport')['congestion_score'].shift(lag)
        
        # Fill NaN values with 0 for lag features
        lag_columns = [col for col in df.columns if 'lag_' in col]
        df[lag_columns] = df[lag_columns].fillna(0)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """Create rolling window features"""
        df = df.copy()
        df = df.sort_values(['origin_airport', 'scheduled_departure'])
        
        for window in windows:
            # Rolling mean delay
            df[f'delay_rolling_mean_{window}'] = (
                df.groupby('origin_airport')['delay_minutes']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            # Rolling std delay
            df[f'delay_rolling_std_{window}'] = (
                df.groupby('origin_airport')['delay_minutes']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
        
        # Fill NaN values
        rolling_columns = [col for col in df.columns if 'rolling_' in col]
        df[rolling_columns] = df[rolling_columns].fillna(0)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'delay_minutes') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare all features for model training"""
        logger.info("Starting feature engineering process")
        
        # Create all feature types
        df_features = self.create_time_features(df)
        df_features = self.create_weather_features(df_features)
        df_features = self.create_traffic_features(df_features)
        df_features = self.create_operational_features(df_features)
        df_features = self.create_lag_features(df_features)
        df_features = self.create_rolling_features(df_features)
        
        # Select feature columns
        feature_columns = [
            # Time features
            'departure_hour', 'departure_day_of_week', 'departure_month', 'departure_quarter',
            'is_weekend', 'is_peak_hour', 'arrival_hour', 'scheduled_duration_minutes',
            
            # Weather features
            'temperature', 'humidity', 'wind_speed', 'visibility', 'precipitation', 'is_bad_weather',
            
            # Traffic features
            'hourly_departures', 'congestion_score',
            
            # Operational features
            'airline_avg_delay', 'airline_delay_std', 'route_avg_delay', 'route_frequency',
            'passenger_load', 'is_high_load',
            
            # Categorical features
            'aircraft_category', 'origin_airport', 'destination_airport', 'airline'
        ]
        
        # Add lag and rolling features
        lag_columns = [col for col in df_features.columns if 'lag_' in col]
        rolling_columns = [col for col in df_features.columns if 'rolling_' in col]
        feature_columns.extend(lag_columns)
        feature_columns.extend(rolling_columns)
        
        # Filter existing columns
        available_columns = [col for col in feature_columns if col in df_features.columns]
        
        X = df_features[available_columns].copy()
        y = df_features[target_column] if target_column in df_features.columns else None
        
        logger.info(f"Created {len(available_columns)} features")
        return X, y
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit transformers and transform features"""
        logger.info("Fitting feature transformers")
        
        # Separate categorical and numerical columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns)
            ]
        )
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(X)
        
        # Store the preprocessor
        self.preprocessor = preprocessor
        self.feature_names = (
            numerical_columns + 
            list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns))
        )
        
        self.is_fitted = True
        logger.info(f"Feature transformation complete. Shape: {X_transformed.shape}")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted transformers"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        return self.preprocessor.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before getting feature names")
        
        return self.feature_names
    
    def select_features(self, X: np.ndarray, y: np.ndarray, k: int = 50) -> Tuple[np.ndarray, List[str]]:
        """Select top k features using statistical tests"""
        if k > X.shape[1]:
            k = X.shape[1]
        
        self.feature_selector.k = k
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        logger.info(f"Selected {k} features out of {X.shape[1]}")
        return X_selected, selected_features