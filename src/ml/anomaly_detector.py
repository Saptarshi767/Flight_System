"""
Anomaly detection system for unusual flight patterns using open source AI tools
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports for anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix

# Statistical methods
from scipy import stats
from scipy.stats import zscore, iqr

# Local imports
from .feature_engineering import FeatureEngineer
from .model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Anomaly detection system using multiple algorithms"""
    
    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.1):
        """
        Initialize anomaly detector
        
        Args:
            method: Detection method ('isolation_forest', 'one_class_svm', 'elliptic_envelope', 
                   'local_outlier_factor', 'dbscan', 'statistical')
            contamination: Expected proportion of anomalies (0.0 to 0.5)
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.is_fitted = False
        self.anomaly_threshold = None
        self.feature_importance = None
        self.training_stats = {}
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the anomaly detection model"""
        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
        elif self.method == 'one_class_svm':
            self.model = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
        elif self.method == 'elliptic_envelope':
            self.model = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        elif self.method == 'local_outlier_factor':
            self.model = LocalOutlierFactor(
                contamination=self.contamination,
                n_jobs=-1
            )
        elif self.method == 'dbscan':
            self.model = DBSCAN(
                eps=0.5,
                min_samples=5,
                n_jobs=-1
            )
        elif self.method == 'statistical':
            # Statistical methods don't need a model object
            self.model = None
        else:
            raise ValueError(f"Unsupported anomaly detection method: {self.method}")
        
        logger.info(f"Initialized {self.method} anomaly detector")
    
    def prepare_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features specifically for anomaly detection"""
        logger.info("Preparing features for anomaly detection")
        
        # Use feature engineering to create comprehensive features
        X, _ = self.feature_engineer.prepare_features(df, target_column='delay_minutes')
        
        # Add anomaly-specific features
        anomaly_features = df.copy()
        
        # Time-based anomaly features
        if 'scheduled_departure' in df.columns:
            anomaly_features['departure_datetime'] = pd.to_datetime(df['scheduled_departure'])
            anomaly_features['hour'] = anomaly_features['departure_datetime'].dt.hour
            anomaly_features['day_of_week'] = anomaly_features['departure_datetime'].dt.dayofweek
            anomaly_features['month'] = anomaly_features['departure_datetime'].dt.month
            
            # Unusual time patterns
            anomaly_features['is_very_early'] = (anomaly_features['hour'] < 5).astype(int)
            anomaly_features['is_very_late'] = (anomaly_features['hour'] > 23).astype(int)
            anomaly_features['is_unusual_day'] = (anomaly_features['day_of_week'] == 6).astype(int)  # Sunday
        
        # Delay-based anomaly features
        if 'delay_minutes' in df.columns:
            delay_stats = df['delay_minutes'].describe()
            anomaly_features['delay_z_score'] = zscore(df['delay_minutes'])
            anomaly_features['is_extreme_delay'] = (df['delay_minutes'] > delay_stats['75%'] + 3 * iqr(df['delay_minutes'])).astype(int)
            anomaly_features['is_negative_delay'] = (df['delay_minutes'] < 0).astype(int)
        
        # Route-based anomaly features
        if 'origin_airport' in df.columns and 'destination_airport' in df.columns:
            # Unusual routes (less common combinations)
            route_counts = df.groupby(['origin_airport', 'destination_airport']).size()
            rare_routes = route_counts[route_counts <= route_counts.quantile(0.1)].index
            anomaly_features['is_rare_route'] = df.apply(
                lambda row: (row['origin_airport'], row['destination_airport']) in rare_routes, axis=1
            ).astype(int)
        
        # Aircraft-based anomaly features
        if 'aircraft_type' in df.columns:
            aircraft_counts = df['aircraft_type'].value_counts()
            rare_aircraft = aircraft_counts[aircraft_counts <= aircraft_counts.quantile(0.1)].index
            anomaly_features['is_rare_aircraft'] = df['aircraft_type'].isin(rare_aircraft).astype(int)
        
        # Passenger load anomaly features
        if 'passenger_count' in df.columns:
            passenger_stats = df['passenger_count'].describe()
            anomaly_features['passenger_z_score'] = zscore(df['passenger_count'].fillna(passenger_stats['mean']))
            anomaly_features['is_unusual_load'] = (
                np.abs(anomaly_features['passenger_z_score']) > 2
            ).astype(int)
        
        # Weather-based anomaly features
        if 'weather_conditions' in df.columns:
            weather_data = df['weather_conditions'].fillna({})
            
            # Extract weather anomalies
            temperatures = [w.get('temperature', 25) if isinstance(w, dict) else 25 for w in weather_data]
            wind_speeds = [w.get('wind_speed', 10) if isinstance(w, dict) else 10 for w in weather_data]
            
            temp_z = zscore(temperatures)
            wind_z = zscore(wind_speeds)
            
            anomaly_features['extreme_temperature'] = (np.abs(temp_z) > 2).astype(int)
            anomaly_features['extreme_wind'] = (np.abs(wind_z) > 2).astype(int)
        
        # Select only numeric features for anomaly detection
        numeric_columns = anomaly_features.select_dtypes(include=[np.number]).columns
        exclude_columns = ['flight_id', 'scheduled_departure', 'actual_departure', 'scheduled_arrival', 'actual_arrival']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        return anomaly_features[feature_columns]
    
    def fit(self, df: pd.DataFrame, labeled_anomalies: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Fit the anomaly detection model
        
        Args:
            df: Training data
            labeled_anomalies: Optional labels for known anomalies (for evaluation)
        """
        logger.info(f"Fitting {self.method} anomaly detection model")
        
        # Prepare features
        features = self.prepare_anomaly_features(df)
        
        # Handle missing values - separate numeric and categorical columns
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        categorical_columns = features.select_dtypes(include=['object', 'category']).columns
        
        # Fill numeric columns with median
        if len(numeric_columns) > 0:
            features[numeric_columns] = features[numeric_columns].fillna(features[numeric_columns].median())
        
        # Fill categorical columns with mode or default value
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                mode_value = features[col].mode()
                fill_value = mode_value.iloc[0] if len(mode_value) > 0 else 'unknown'
                features[col] = features[col].fillna(fill_value)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Store training statistics
        self.training_stats = {
            'n_samples': len(df),
            'n_features': X_scaled.shape[1],
            'feature_names': features.columns.tolist(),
            'contamination': self.contamination
        }
        
        # Fit model based on method
        if self.method == 'isolation_forest':
            self.model.fit(X_scaled)
            # Get anomaly scores
            anomaly_scores = self.model.decision_function(X_scaled)
            self.anomaly_threshold = np.percentile(anomaly_scores, self.contamination * 100)
            
        elif self.method == 'one_class_svm':
            self.model.fit(X_scaled)
            anomaly_scores = self.model.decision_function(X_scaled)
            self.anomaly_threshold = 0  # SVM threshold is 0
            
        elif self.method == 'elliptic_envelope':
            self.model.fit(X_scaled)
            anomaly_scores = self.model.decision_function(X_scaled)
            self.anomaly_threshold = 0  # Elliptic envelope threshold is 0
            
        elif self.method == 'local_outlier_factor':
            # LOF doesn't have a separate fit/predict, it computes during fit
            anomaly_scores = self.model.fit_predict(X_scaled)
            self.anomaly_threshold = -1  # LOF uses -1 for outliers
            
        elif self.method == 'dbscan':
            cluster_labels = self.model.fit_predict(X_scaled)
            # In DBSCAN, -1 indicates noise/anomalies
            anomaly_scores = (cluster_labels == -1).astype(int)
            self.anomaly_threshold = 0.5
            
        elif self.method == 'statistical':
            anomaly_scores = self._statistical_anomaly_detection(X_scaled)
            self.anomaly_threshold = np.percentile(anomaly_scores, (1 - self.contamination) * 100)
        
        # Calculate feature importance for tree-based methods
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        
        # Evaluate if labeled anomalies are provided
        evaluation_results = {}
        if labeled_anomalies is not None:
            predictions = self.predict(df)
            evaluation_results = self._evaluate_anomaly_detection(labeled_anomalies, predictions)
        
        fit_results = {
            'training_stats': self.training_stats,
            'anomaly_threshold': self.anomaly_threshold,
            'feature_importance': self.feature_importance,
            'evaluation': evaluation_results
        }
        
        logger.info(f"Anomaly detection model fitted successfully")
        return fit_results
    
    def _statistical_anomaly_detection(self, X: np.ndarray) -> np.ndarray:
        """Statistical anomaly detection using multiple methods"""
        n_samples, n_features = X.shape
        anomaly_scores = np.zeros(n_samples)
        
        # Z-score based detection
        z_scores = np.abs(zscore(X, axis=0))
        z_score_anomalies = np.mean(z_scores > 2, axis=1)  # Features with z-score > 2
        
        # IQR based detection
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_anomalies = np.mean((X < lower_bound) | (X > upper_bound), axis=1)
        
        # Mahalanobis distance
        try:
            cov_matrix = np.cov(X.T)
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
            mean_vector = np.mean(X, axis=0)
            
            mahalanobis_distances = []
            for i in range(n_samples):
                diff = X[i] - mean_vector
                mahalanobis_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                mahalanobis_distances.append(mahalanobis_dist)
            
            mahalanobis_distances = np.array(mahalanobis_distances)
            mahalanobis_anomalies = (mahalanobis_distances - np.mean(mahalanobis_distances)) / np.std(mahalanobis_distances)
        except:
            mahalanobis_anomalies = np.zeros(n_samples)
        
        # Combine scores
        anomaly_scores = (z_score_anomalies + iqr_anomalies + np.abs(mahalanobis_anomalies)) / 3
        
        return anomaly_scores
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomalies in new data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info("Predicting anomalies")
        
        # Prepare features
        features = self.prepare_anomaly_features(df)
        
        # Handle missing values - separate numeric and categorical columns
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        categorical_columns = features.select_dtypes(include=['object', 'category']).columns
        
        # Fill numeric columns with median
        if len(numeric_columns) > 0:
            features[numeric_columns] = features[numeric_columns].fillna(features[numeric_columns].median())
        
        # Fill categorical columns with mode or default value
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                mode_value = features[col].mode()
                fill_value = mode_value.iloc[0] if len(mode_value) > 0 else 'unknown'
                features[col] = features[col].fillna(fill_value)
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Make predictions based on method
        if self.method == 'isolation_forest':
            anomaly_scores = self.model.decision_function(X_scaled)
            predictions = (anomaly_scores < self.anomaly_threshold).astype(int)
            
        elif self.method in ['one_class_svm', 'elliptic_envelope']:
            predictions = (self.model.predict(X_scaled) == -1).astype(int)
            
        elif self.method == 'local_outlier_factor':
            # LOF requires recomputing on the combined dataset
            logger.warning("LOF requires refitting for new predictions")
            predictions = np.zeros(len(df))  # Return no anomalies
            
        elif self.method == 'dbscan':
            # DBSCAN requires refitting for new predictions
            logger.warning("DBSCAN requires refitting for new predictions")
            predictions = np.zeros(len(df))  # Return no anomalies
            
        elif self.method == 'statistical':
            anomaly_scores = self._statistical_anomaly_detection(X_scaled)
            predictions = (anomaly_scores > self.anomaly_threshold).astype(int)
        
        return predictions
    
    def predict_with_scores(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies with anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        features = self.prepare_anomaly_features(df)
        
        # Handle missing values - separate numeric and categorical columns
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        categorical_columns = features.select_dtypes(include=['object', 'category']).columns
        
        # Fill numeric columns with median
        if len(numeric_columns) > 0:
            features[numeric_columns] = features[numeric_columns].fillna(features[numeric_columns].median())
        
        # Fill categorical columns with mode or default value
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                mode_value = features[col].mode()
                fill_value = mode_value.iloc[0] if len(mode_value) > 0 else 'unknown'
                features[col] = features[col].fillna(fill_value)
        
        X_scaled = self.scaler.transform(features)
        
        # Get predictions and scores
        predictions = self.predict(df)
        
        if self.method == 'isolation_forest':
            scores = -self.model.decision_function(X_scaled)  # Negative for higher anomaly score
        elif self.method in ['one_class_svm', 'elliptic_envelope']:
            scores = -self.model.decision_function(X_scaled)
        elif self.method == 'statistical':
            scores = self._statistical_anomaly_detection(X_scaled)
        else:
            scores = predictions.astype(float)  # Use predictions as scores
        
        return predictions, scores
    
    def detect_flight_anomalies(self, df: pd.DataFrame, 
                               anomaly_types: List[str] = None) -> pd.DataFrame:
        """
        Detect specific types of flight anomalies
        
        Args:
            df: Flight data
            anomaly_types: Types of anomalies to detect 
                          ('delay', 'route', 'timing', 'weather', 'operational')
        """
        logger.info("Detecting flight-specific anomalies")
        
        if anomaly_types is None:
            anomaly_types = ['delay', 'route', 'timing', 'weather', 'operational']
        
        anomaly_results = df.copy()
        anomaly_results['anomaly_score'] = 0.0
        anomaly_results['anomaly_types'] = ''
        anomaly_results['is_anomaly'] = False
        
        # Delay anomalies
        if 'delay' in anomaly_types and 'delay_minutes' in df.columns:
            delay_z = np.abs(zscore(df['delay_minutes']))
            delay_anomalies = delay_z > 3  # 3 standard deviations
            
            anomaly_results.loc[delay_anomalies, 'anomaly_score'] += delay_z[delay_anomalies]
            anomaly_results.loc[delay_anomalies, 'anomaly_types'] += 'delay,'
            anomaly_results.loc[delay_anomalies, 'is_anomaly'] = True
        
        # Route anomalies
        if 'route' in anomaly_types and 'origin_airport' in df.columns and 'destination_airport' in df.columns:
            route_counts = df.groupby(['origin_airport', 'destination_airport']).size()
            rare_routes = route_counts[route_counts <= 2].index  # Routes with <= 2 flights
            
            route_anomalies = df.apply(
                lambda row: (row['origin_airport'], row['destination_airport']) in rare_routes, axis=1
            )
            
            anomaly_results.loc[route_anomalies, 'anomaly_score'] += 2.0
            anomaly_results.loc[route_anomalies, 'anomaly_types'] += 'route,'
            anomaly_results.loc[route_anomalies, 'is_anomaly'] = True
        
        # Timing anomalies
        if 'timing' in anomaly_types and 'scheduled_departure' in df.columns:
            departure_times = pd.to_datetime(df['scheduled_departure'])
            hours = departure_times.dt.hour
            
            # Very early (before 5 AM) or very late (after 11 PM) flights
            timing_anomalies = (hours < 5) | (hours > 23)
            
            anomaly_results.loc[timing_anomalies, 'anomaly_score'] += 1.5
            anomaly_results.loc[timing_anomalies, 'anomaly_types'] += 'timing,'
            anomaly_results.loc[timing_anomalies, 'is_anomaly'] = True
        
        # Weather anomalies
        if 'weather' in anomaly_types and 'weather_conditions' in df.columns:
            weather_data = df['weather_conditions'].fillna({})
            
            for idx, weather in enumerate(weather_data):
                if isinstance(weather, dict):
                    # Extreme weather conditions
                    temp = weather.get('temperature', 25)
                    wind = weather.get('wind_speed', 10)
                    precip = weather.get('precipitation', 0)
                    
                    if temp < 0 or temp > 45 or wind > 30 or precip > 10:
                        anomaly_results.loc[idx, 'anomaly_score'] += 2.0
                        anomaly_results.loc[idx, 'anomaly_types'] += 'weather,'
                        anomaly_results.loc[idx, 'is_anomaly'] = True
        
        # Operational anomalies
        if 'operational' in anomaly_types:
            # Unusual passenger loads
            if 'passenger_count' in df.columns:
                passenger_z = np.abs(zscore(df['passenger_count'].fillna(df['passenger_count'].median())))
                passenger_anomalies = passenger_z > 2.5
                
                anomaly_results.loc[passenger_anomalies, 'anomaly_score'] += passenger_z[passenger_anomalies]
                anomaly_results.loc[passenger_anomalies, 'anomaly_types'] += 'operational,'
                anomaly_results.loc[passenger_anomalies, 'is_anomaly'] = True
        
        # Clean up anomaly types string
        anomaly_results['anomaly_types'] = anomaly_results['anomaly_types'].str.rstrip(',')
        
        # Overall anomaly detection using the fitted model
        if self.is_fitted:
            ml_predictions, ml_scores = self.predict_with_scores(df)
            
            # Combine rule-based and ML-based detection
            combined_scores = anomaly_results['anomaly_score'] + ml_scores
            combined_anomalies = (anomaly_results['is_anomaly']) | (ml_predictions == 1)
            
            anomaly_results['ml_anomaly_score'] = ml_scores
            anomaly_results['combined_anomaly_score'] = combined_scores
            anomaly_results['ml_is_anomaly'] = ml_predictions
            anomaly_results['combined_is_anomaly'] = combined_anomalies
        
        logger.info(f"Detected {anomaly_results['is_anomaly'].sum()} rule-based anomalies")
        return anomaly_results
    
    def _evaluate_anomaly_detection(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Evaluate anomaly detection performance"""
        # Convert to binary if needed
        y_true_binary = (y_true == 1).astype(int)
        y_pred_binary = y_pred.astype(int)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        evaluation = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'true_positives': np.sum((y_true_binary == 1) & (y_pred_binary == 1)),
            'false_positives': np.sum((y_true_binary == 0) & (y_pred_binary == 1)),
            'true_negatives': np.sum((y_true_binary == 0) & (y_pred_binary == 0)),
            'false_negatives': np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        }
        
        return evaluation
    
    def create_alerting_system(self, alert_threshold: float = 0.8,
                              alert_types: List[str] = None) -> Dict[str, Any]:
        """Create alerting system configuration for detected anomalies"""
        if alert_types is None:
            alert_types = ['high_delay', 'unusual_route', 'extreme_weather', 'operational_issue']
        
        alerting_config = {
            'alert_threshold': alert_threshold,
            'alert_types': alert_types,
            'notification_channels': ['email', 'dashboard', 'api'],
            'alert_rules': {
                'high_delay': {
                    'condition': 'delay_minutes > 60',
                    'severity': 'high',
                    'message': 'Flight experiencing significant delay'
                },
                'unusual_route': {
                    'condition': 'is_rare_route == 1',
                    'severity': 'medium',
                    'message': 'Flight on unusual route detected'
                },
                'extreme_weather': {
                    'condition': 'extreme_temperature == 1 OR extreme_wind == 1',
                    'severity': 'high',
                    'message': 'Flight operating in extreme weather conditions'
                },
                'operational_issue': {
                    'condition': 'anomaly_score > alert_threshold',
                    'severity': 'medium',
                    'message': 'Operational anomaly detected'
                }
            },
            'escalation_rules': {
                'high_severity': {
                    'immediate_notification': True,
                    'escalate_after_minutes': 5
                },
                'medium_severity': {
                    'immediate_notification': False,
                    'escalate_after_minutes': 15
                }
            }
        }
        
        return alerting_config
    
    def update_model(self, new_data: pd.DataFrame, 
                    update_method: str = 'incremental') -> Dict[str, Any]:
        """Update anomaly detection model with new data"""
        logger.info(f"Updating model with {update_method} method")
        
        if update_method == 'incremental':
            # For methods that support incremental learning
            if self.method == 'isolation_forest':
                # Isolation Forest doesn't support incremental learning
                # Retrain with combined data
                logger.info("Retraining Isolation Forest with new data")
                return self.fit(new_data)
            else:
                logger.warning(f"{self.method} doesn't support incremental learning")
                return {}
        
        elif update_method == 'retrain':
            # Complete retraining
            return self.fit(new_data)
        
        else:
            raise ValueError(f"Unsupported update method: {update_method}")
    
    def get_anomaly_summary(self, anomaly_results: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of detected anomalies"""
        summary = {
            'total_flights': len(anomaly_results),
            'total_anomalies': anomaly_results['is_anomaly'].sum(),
            'anomaly_rate': anomaly_results['is_anomaly'].mean(),
            'anomaly_types_distribution': {},
            'top_anomalous_flights': [],
            'anomaly_score_stats': {
                'mean': anomaly_results['anomaly_score'].mean(),
                'std': anomaly_results['anomaly_score'].std(),
                'max': anomaly_results['anomaly_score'].max(),
                'min': anomaly_results['anomaly_score'].min()
            }
        }
        
        # Anomaly types distribution
        if 'anomaly_types' in anomaly_results.columns:
            all_types = []
            for types_str in anomaly_results['anomaly_types']:
                if types_str:
                    all_types.extend(types_str.split(','))
            
            from collections import Counter
            type_counts = Counter(all_types)
            summary['anomaly_types_distribution'] = dict(type_counts)
        
        # Top anomalous flights
        if 'flight_id' in anomaly_results.columns:
            top_anomalies = anomaly_results.nlargest(10, 'anomaly_score')
            summary['top_anomalous_flights'] = top_anomalies[['flight_id', 'anomaly_score', 'anomaly_types']].to_dict('records')
        
        return summary
    
    def save_model(self, filepath: str):
        """Save the fitted anomaly detection model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import joblib
        
        model_data = {
            'method': self.method,
            'contamination': self.contamination,
            'model': self.model,
            'scaler': self.scaler,
            'feature_engineer': self.feature_engineer,
            'is_fitted': self.is_fitted,
            'anomaly_threshold': self.anomaly_threshold,
            'feature_importance': self.feature_importance,
            'training_stats': self.training_stats
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Anomaly detection model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a fitted anomaly detection model"""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.method = model_data['method']
        self.contamination = model_data['contamination']
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_engineer = model_data['feature_engineer']
        self.is_fitted = model_data['is_fitted']
        self.anomaly_threshold = model_data['anomaly_threshold']
        self.feature_importance = model_data['feature_importance']
        self.training_stats = model_data['training_stats']
        
        logger.info(f"Anomaly detection model loaded from {filepath}")


class FlightAnomalyDetector(AnomalyDetector):
    """Specialized anomaly detector for flight operations"""
    
    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.05):
        super().__init__(method, contamination)
        self.flight_patterns = {}
        self.seasonal_patterns = {}
    
    def learn_flight_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Learn normal flight patterns from historical data"""
        logger.info("Learning normal flight patterns")
        
        patterns = {}
        
        # Learn airline patterns
        if 'airline' in df.columns and 'delay_minutes' in df.columns:
            airline_patterns = df.groupby('airline')['delay_minutes'].agg(['mean', 'std', 'count'])
            patterns['airlines'] = airline_patterns.to_dict('index')
        
        # Learn route patterns
        if 'origin_airport' in df.columns and 'destination_airport' in df.columns:
            route_patterns = df.groupby(['origin_airport', 'destination_airport']).agg({
                'delay_minutes': ['mean', 'std', 'count'],
                'scheduled_departure': lambda x: pd.to_datetime(x).dt.hour.mode().iloc[0] if len(x) > 0 else 12
            })
            patterns['routes'] = route_patterns.to_dict('index')
        
        # Learn temporal patterns
        if 'scheduled_departure' in df.columns:
            df['departure_hour'] = pd.to_datetime(df['scheduled_departure']).dt.hour
            df['departure_day'] = pd.to_datetime(df['scheduled_departure']).dt.dayofweek
            
            hourly_patterns = df.groupby('departure_hour')['delay_minutes'].agg(['mean', 'std', 'count'])
            daily_patterns = df.groupby('departure_day')['delay_minutes'].agg(['mean', 'std', 'count'])
            
            patterns['hourly'] = hourly_patterns.to_dict('index')
            patterns['daily'] = daily_patterns.to_dict('index')
        
        self.flight_patterns = patterns
        logger.info(f"Learned patterns for {len(patterns)} categories")
        
        return patterns
    
    def detect_pattern_deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect deviations from learned flight patterns"""
        if not self.flight_patterns:
            logger.warning("No flight patterns learned. Call learn_flight_patterns first.")
            return df
        
        logger.info("Detecting pattern deviations")
        
        results = df.copy()
        results['pattern_deviation_score'] = 0.0
        results['pattern_deviations'] = ''
        
        # Check airline deviations
        if 'airlines' in self.flight_patterns and 'airline' in df.columns:
            for idx, row in df.iterrows():
                airline = row['airline']
                if airline in self.flight_patterns['airlines']:
                    expected_delay = self.flight_patterns['airlines'][airline]['mean']
                    delay_std = self.flight_patterns['airlines'][airline]['std']
                    actual_delay = row.get('delay_minutes', 0)
                    
                    if delay_std > 0:
                        deviation = abs(actual_delay - expected_delay) / delay_std
                        if deviation > 2:  # More than 2 standard deviations
                            results.loc[idx, 'pattern_deviation_score'] += deviation
                            results.loc[idx, 'pattern_deviations'] += f'airline_delay({deviation:.1f}),'
        
        # Check route deviations
        if 'routes' in self.flight_patterns and 'origin_airport' in df.columns:
            for idx, row in df.iterrows():
                route = (row['origin_airport'], row['destination_airport'])
                if route in self.flight_patterns['routes']:
                    route_data = self.flight_patterns['routes'][route]
                    expected_delay = route_data[('delay_minutes', 'mean')]
                    delay_std = route_data[('delay_minutes', 'std')]
                    actual_delay = row.get('delay_minutes', 0)
                    
                    if delay_std > 0:
                        deviation = abs(actual_delay - expected_delay) / delay_std
                        if deviation > 2:
                            results.loc[idx, 'pattern_deviation_score'] += deviation
                            results.loc[idx, 'pattern_deviations'] += f'route_delay({deviation:.1f}),'
        
        # Clean up deviation strings
        results['pattern_deviations'] = results['pattern_deviations'].str.rstrip(',')
        
        return results