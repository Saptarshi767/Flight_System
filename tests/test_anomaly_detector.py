"""
Tests for anomaly detection system
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from src.ml.anomaly_detector import AnomalyDetector, FlightAnomalyDetector


@pytest.fixture
def sample_flight_data_with_anomalies():
    """Create sample flight data with known anomalies"""
    np.random.seed(42)
    n_samples = 500
    
    # Generate normal flight data
    data = {
        'flight_id': [f'FL{i:04d}' for i in range(n_samples)],
        'airline': np.random.choice(['AI', 'SG', '6E', 'UK'], n_samples),
        'aircraft_type': np.random.choice(['A320', 'B737', 'A321', 'B777'], n_samples),
        'origin_airport': np.random.choice(['BOM', 'DEL'], n_samples),
        'destination_airport': np.random.choice(['BOM', 'DEL', 'BLR', 'MAA'], n_samples),
        'passenger_count': np.random.randint(50, 300, n_samples),
    }
    
    # Generate datetime data
    base_date = datetime(2024, 1, 1)
    scheduled_departures = [
        base_date + timedelta(days=np.random.randint(0, 30), 
                             hours=np.random.randint(6, 22),
                             minutes=np.random.randint(0, 59))
        for _ in range(n_samples)
    ]
    
    data['scheduled_departure'] = scheduled_departures
    
    # Generate normal delays
    delays = np.random.exponential(10, n_samples)  # Exponential distribution for delays
    
    # Inject known anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    
    for idx in anomaly_indices:
        # Create different types of anomalies
        anomaly_type = np.random.choice(['extreme_delay', 'unusual_time', 'rare_route', 'extreme_passengers'])
        
        if anomaly_type == 'extreme_delay':
            delays[idx] = np.random.uniform(120, 300)  # Very high delay
        elif anomaly_type == 'unusual_time':
            # Very early or very late departure
            hour = np.random.choice([2, 3, 4, 24, 25])
            data['scheduled_departure'][idx] = data['scheduled_departure'][idx].replace(hour=min(hour, 23))
        elif anomaly_type == 'rare_route':
            data['origin_airport'][idx] = 'CCU'  # Rare airport
            data['destination_airport'][idx] = 'GOI'  # Rare airport
        elif anomaly_type == 'extreme_passengers':
            data['passenger_count'][idx] = np.random.choice([10, 500])  # Very low or very high
    
    data['delay_minutes'] = delays
    
    # Add weather conditions
    data['weather_conditions'] = [
        {
            'temperature': np.random.normal(25, 10),
            'humidity': np.random.normal(60, 20),
            'wind_speed': np.random.normal(15, 8),
            'visibility': np.random.normal(8, 3),
            'precipitation': np.random.exponential(2) if np.random.random() < 0.2 else 0
        }
        for _ in range(n_samples)
    ]
    
    # Inject extreme weather anomalies
    weather_anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    for idx in weather_anomaly_indices:
        data['weather_conditions'][idx]['temperature'] = np.random.choice([-5, 50])  # Extreme temperature
        data['weather_conditions'][idx]['wind_speed'] = np.random.uniform(40, 60)  # Extreme wind
    
    # Create labels for known anomalies (for evaluation)
    labels = np.zeros(n_samples)
    all_anomaly_indices = np.union1d(anomaly_indices, weather_anomaly_indices)
    labels[all_anomaly_indices] = 1
    
    df = pd.DataFrame(data)
    df['true_anomaly'] = labels
    
    return df


class TestAnomalyDetector:
    """Test anomaly detection functionality"""
    
    @pytest.mark.parametrize("method", ['isolation_forest', 'one_class_svm', 'elliptic_envelope', 'statistical'])
    def test_model_initialization(self, method):
        """Test model initialization for different methods"""
        detector = AnomalyDetector(method=method, contamination=0.1)
        assert detector.method == method
        assert detector.contamination == 0.1
        assert not detector.is_fitted
    
    def test_invalid_method(self):
        """Test invalid method raises error"""
        with pytest.raises(ValueError):
            AnomalyDetector(method='invalid_method')
    
    def test_prepare_anomaly_features(self, sample_flight_data_with_anomalies):
        """Test anomaly feature preparation"""
        detector = AnomalyDetector()
        features = detector.prepare_anomaly_features(sample_flight_data_with_anomalies)
        
        assert len(features) == len(sample_flight_data_with_anomalies)
        assert features.shape[1] > 0
        
        # Check for specific anomaly features
        expected_features = [
            'is_very_early', 'is_very_late', 'is_extreme_delay', 
            'is_rare_route', 'extreme_temperature', 'extreme_wind'
        ]
        
        for feature in expected_features:
            if feature in features.columns:
                assert features[feature].dtype in [int, bool, 'int64', 'bool']
    
    @pytest.mark.parametrize("method", ['isolation_forest', 'statistical'])
    def test_model_fitting(self, sample_flight_data_with_anomalies, method):
        """Test model fitting"""
        detector = AnomalyDetector(method=method, contamination=0.1)
        
        # Fit without labels
        fit_results = detector.fit(sample_flight_data_with_anomalies)
        
        assert detector.is_fitted
        assert 'training_stats' in fit_results
        assert 'anomaly_threshold' in fit_results
        assert fit_results['training_stats']['n_samples'] == len(sample_flight_data_with_anomalies)
    
    def test_model_fitting_with_labels(self, sample_flight_data_with_anomalies):
        """Test model fitting with labeled anomalies"""
        detector = AnomalyDetector(method='isolation_forest', contamination=0.1)
        
        # Fit with labels for evaluation
        fit_results = detector.fit(
            sample_flight_data_with_anomalies, 
            labeled_anomalies=sample_flight_data_with_anomalies['true_anomaly']
        )
        
        assert detector.is_fitted
        assert 'evaluation' in fit_results
        assert 'accuracy' in fit_results['evaluation']
        assert 'precision' in fit_results['evaluation']
        assert 'recall' in fit_results['evaluation']
    
    def test_prediction(self, sample_flight_data_with_anomalies):
        """Test anomaly prediction"""
        detector = AnomalyDetector(method='isolation_forest', contamination=0.1)
        detector.fit(sample_flight_data_with_anomalies)
        
        # Test prediction on same data
        predictions = detector.predict(sample_flight_data_with_anomalies.head(100))
        
        assert len(predictions) == 100
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should detect some anomalies
        assert np.sum(predictions) > 0
    
    def test_prediction_with_scores(self, sample_flight_data_with_anomalies):
        """Test prediction with anomaly scores"""
        detector = AnomalyDetector(method='isolation_forest', contamination=0.1)
        detector.fit(sample_flight_data_with_anomalies)
        
        predictions, scores = detector.predict_with_scores(sample_flight_data_with_anomalies.head(100))
        
        assert len(predictions) == 100
        assert len(scores) == 100
        assert all(isinstance(score, (int, float, np.number)) for score in scores)
    
    def test_prediction_without_fitting(self, sample_flight_data_with_anomalies):
        """Test that prediction fails without fitting"""
        detector = AnomalyDetector()
        
        with pytest.raises(ValueError):
            detector.predict(sample_flight_data_with_anomalies)
    
    def test_detect_flight_anomalies(self, sample_flight_data_with_anomalies):
        """Test flight-specific anomaly detection"""
        detector = AnomalyDetector(method='isolation_forest', contamination=0.1)
        detector.fit(sample_flight_data_with_anomalies)
        
        anomaly_results = detector.detect_flight_anomalies(
            sample_flight_data_with_anomalies,
            anomaly_types=['delay', 'route', 'timing', 'weather']
        )
        
        assert len(anomaly_results) == len(sample_flight_data_with_anomalies)
        assert 'anomaly_score' in anomaly_results.columns
        assert 'anomaly_types' in anomaly_results.columns
        assert 'is_anomaly' in anomaly_results.columns
        
        # Should detect some anomalies
        assert anomaly_results['is_anomaly'].sum() > 0
    
    def test_statistical_anomaly_detection(self, sample_flight_data_with_anomalies):
        """Test statistical anomaly detection methods"""
        detector = AnomalyDetector(method='statistical', contamination=0.1)
        detector.fit(sample_flight_data_with_anomalies)
        
        predictions = detector.predict(sample_flight_data_with_anomalies.head(100))
        
        assert len(predictions) == 100
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_create_alerting_system(self, sample_flight_data_with_anomalies):
        """Test alerting system creation"""
        detector = AnomalyDetector()
        
        alerting_config = detector.create_alerting_system(
            alert_threshold=0.8,
            alert_types=['high_delay', 'unusual_route']
        )
        
        assert 'alert_threshold' in alerting_config
        assert 'alert_types' in alerting_config
        assert 'alert_rules' in alerting_config
        assert 'escalation_rules' in alerting_config
        
        # Check specific alert rules
        assert 'high_delay' in alerting_config['alert_rules']
        assert 'unusual_route' in alerting_config['alert_rules']
    
    def test_get_anomaly_summary(self, sample_flight_data_with_anomalies):
        """Test anomaly summary generation"""
        detector = AnomalyDetector(method='isolation_forest', contamination=0.1)
        detector.fit(sample_flight_data_with_anomalies)
        
        anomaly_results = detector.detect_flight_anomalies(sample_flight_data_with_anomalies)
        summary = detector.get_anomaly_summary(anomaly_results)
        
        assert 'total_flights' in summary
        assert 'total_anomalies' in summary
        assert 'anomaly_rate' in summary
        assert 'anomaly_score_stats' in summary
        
        assert summary['total_flights'] == len(sample_flight_data_with_anomalies)
        assert 0 <= summary['anomaly_rate'] <= 1
    
    def test_save_load_model(self, sample_flight_data_with_anomalies):
        """Test model saving and loading"""
        detector = AnomalyDetector(method='isolation_forest', contamination=0.1)
        detector.fit(sample_flight_data_with_anomalies)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            detector.save_model(tmp.name)
            
            # Load model
            new_detector = AnomalyDetector()
            new_detector.load_model(tmp.name)
            
            # Check that loaded model works
            assert new_detector.is_fitted
            assert new_detector.method == 'isolation_forest'
            assert new_detector.contamination == 0.1
            
            predictions = new_detector.predict(sample_flight_data_with_anomalies.head(10))
            assert len(predictions) == 10
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_update_model(self, sample_flight_data_with_anomalies):
        """Test model updating"""
        detector = AnomalyDetector(method='isolation_forest', contamination=0.1)
        detector.fit(sample_flight_data_with_anomalies.head(300))
        
        # Update with new data
        new_data = sample_flight_data_with_anomalies.tail(200)
        update_results = detector.update_model(new_data, update_method='retrain')
        
        assert detector.is_fitted
        assert 'training_stats' in update_results


class TestFlightAnomalyDetector:
    """Test flight-specific anomaly detection"""
    
    def test_flight_anomaly_detector_initialization(self):
        """Test flight anomaly detector initialization"""
        detector = FlightAnomalyDetector()
        assert isinstance(detector, AnomalyDetector)
        assert detector.contamination == 0.05  # Default for flight detector
        assert hasattr(detector, 'flight_patterns')
        assert hasattr(detector, 'seasonal_patterns')
    
    def test_learn_flight_patterns(self, sample_flight_data_with_anomalies):
        """Test learning flight patterns"""
        detector = FlightAnomalyDetector()
        
        patterns = detector.learn_flight_patterns(sample_flight_data_with_anomalies)
        
        assert isinstance(patterns, dict)
        assert len(patterns) > 0
        
        # Check for expected pattern categories
        expected_categories = ['airlines', 'routes', 'hourly', 'daily']
        for category in expected_categories:
            if category in patterns:
                assert isinstance(patterns[category], dict)
                assert len(patterns[category]) > 0
    
    def test_detect_pattern_deviations(self, sample_flight_data_with_anomalies):
        """Test pattern deviation detection"""
        detector = FlightAnomalyDetector()
        
        # First learn patterns
        detector.learn_flight_patterns(sample_flight_data_with_anomalies.head(400))
        
        # Then detect deviations in new data
        test_data = sample_flight_data_with_anomalies.tail(100)
        deviation_results = detector.detect_pattern_deviations(test_data)
        
        assert len(deviation_results) == len(test_data)
        assert 'pattern_deviation_score' in deviation_results.columns
        assert 'pattern_deviations' in deviation_results.columns
        
        # Should detect some pattern deviations
        assert deviation_results['pattern_deviation_score'].sum() > 0
    
    def test_detect_pattern_deviations_without_learning(self, sample_flight_data_with_anomalies):
        """Test pattern deviation detection without learning patterns first"""
        detector = FlightAnomalyDetector()
        
        # Try to detect deviations without learning patterns
        deviation_results = detector.detect_pattern_deviations(sample_flight_data_with_anomalies)
        
        # Should return original data without modifications
        assert len(deviation_results) == len(sample_flight_data_with_anomalies)


class TestIntegration:
    """Integration tests for anomaly detection system"""
    
    def test_end_to_end_anomaly_detection_pipeline(self, sample_flight_data_with_anomalies):
        """Test complete anomaly detection pipeline"""
        # Split data
        train_data = sample_flight_data_with_anomalies.head(400)
        test_data = sample_flight_data_with_anomalies.tail(100)
        
        # Initialize detector
        detector = AnomalyDetector(method='isolation_forest', contamination=0.1)
        
        # Fit model
        fit_results = detector.fit(train_data, labeled_anomalies=train_data['true_anomaly'])
        assert detector.is_fitted
        assert 'evaluation' in fit_results
        
        # Make predictions
        predictions = detector.predict(test_data)
        assert len(predictions) == len(test_data)
        
        # Detect flight-specific anomalies
        anomaly_results = detector.detect_flight_anomalies(test_data)
        assert len(anomaly_results) == len(test_data)
        
        # Get summary
        summary = detector.get_anomaly_summary(anomaly_results)
        assert 'total_flights' in summary
        assert 'anomaly_rate' in summary
        
        # Create alerting system
        alerting_config = detector.create_alerting_system()
        assert 'alert_threshold' in alerting_config
    
    def test_multiple_method_comparison(self, sample_flight_data_with_anomalies):
        """Test comparing multiple anomaly detection methods"""
        # Use smaller dataset for faster testing
        data = sample_flight_data_with_anomalies.head(200)
        
        methods = ['isolation_forest', 'statistical']
        results = {}
        
        for method in methods:
            detector = AnomalyDetector(method=method, contamination=0.1)
            fit_results = detector.fit(data, labeled_anomalies=data['true_anomaly'])
            predictions = detector.predict(data)
            
            results[method] = {
                'fit_results': fit_results,
                'predictions': predictions,
                'detector': detector
            }
        
        # Check that all methods produced results
        assert len(results) == 2
        for method, result in results.items():
            assert result['detector'].is_fitted
            assert len(result['predictions']) == len(data)
            assert 'evaluation' in result['fit_results']
    
    def test_flight_pattern_learning_pipeline(self, sample_flight_data_with_anomalies):
        """Test complete flight pattern learning and deviation detection"""
        # Split data
        historical_data = sample_flight_data_with_anomalies.head(300)
        new_data = sample_flight_data_with_anomalies.tail(200)
        
        # Initialize flight-specific detector
        detector = FlightAnomalyDetector(method='isolation_forest', contamination=0.05)
        
        # Learn patterns from historical data
        patterns = detector.learn_flight_patterns(historical_data)
        assert len(patterns) > 0
        
        # Fit anomaly detection model
        fit_results = detector.fit(historical_data)
        assert detector.is_fitted
        
        # Detect anomalies in new data
        anomaly_results = detector.detect_flight_anomalies(new_data)
        assert len(anomaly_results) == len(new_data)
        
        # Detect pattern deviations
        deviation_results = detector.detect_pattern_deviations(new_data)
        assert len(deviation_results) == len(new_data)
        
        # Get comprehensive summary
        summary = detector.get_anomaly_summary(anomaly_results)
        assert summary['total_flights'] == len(new_data)
    
    def test_anomaly_detection_with_different_contamination_rates(self, sample_flight_data_with_anomalies):
        """Test anomaly detection with different contamination rates"""
        data = sample_flight_data_with_anomalies.head(200)
        contamination_rates = [0.05, 0.1, 0.15]
        
        results = {}
        
        for contamination in contamination_rates:
            detector = AnomalyDetector(method='isolation_forest', contamination=contamination)
            detector.fit(data)
            predictions = detector.predict(data)
            
            anomaly_rate = np.mean(predictions)
            results[contamination] = {
                'predicted_anomaly_rate': anomaly_rate,
                'total_anomalies': np.sum(predictions)
            }
        
        # Check that higher contamination rates detect more anomalies
        rates = [results[c]['predicted_anomaly_rate'] for c in contamination_rates]
        assert rates[0] <= rates[1] <= rates[2]  # Should be increasing
    
    def test_real_time_anomaly_detection_simulation(self, sample_flight_data_with_anomalies):
        """Test simulated real-time anomaly detection"""
        # Train on historical data
        historical_data = sample_flight_data_with_anomalies.head(300)
        detector = AnomalyDetector(method='isolation_forest', contamination=0.1)
        detector.fit(historical_data)
        
        # Simulate real-time detection on streaming data
        streaming_data = sample_flight_data_with_anomalies.tail(200)
        batch_size = 10
        
        all_predictions = []
        all_scores = []
        
        for i in range(0, len(streaming_data), batch_size):
            batch = streaming_data.iloc[i:i+batch_size]
            
            # Detect anomalies in batch
            predictions, scores = detector.predict_with_scores(batch)
            
            all_predictions.extend(predictions)
            all_scores.extend(scores)
            
            # Simulate alerting for high-score anomalies
            high_score_anomalies = batch[predictions == 1]
            if len(high_score_anomalies) > 0:
                # Would trigger alerts in real system
                pass
        
        assert len(all_predictions) == len(streaming_data)
        assert len(all_scores) == len(streaming_data)
        assert np.sum(all_predictions) > 0  # Should detect some anomalies