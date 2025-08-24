"""
Tests for delay prediction models
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from src.ml.delay_predictor import DelayPredictor, EnsembleDelayPredictor
from src.ml.feature_engineering import FeatureEngineer
from src.ml.model_evaluator import ModelEvaluator


@pytest.fixture
def sample_flight_data():
    """Create sample flight data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data
    data = {
        'flight_id': [f'FL{i:04d}' for i in range(n_samples)],
        'airline': np.random.choice(['AI', 'SG', '6E', 'UK'], n_samples),
        'flight_number': [f'{np.random.choice(["AI", "SG", "6E", "UK"])}{np.random.randint(100, 999)}' for _ in range(n_samples)],
        'aircraft_type': np.random.choice(['A320', 'B737', 'A321', 'B777'], n_samples),
        'origin_airport': np.random.choice(['BOM', 'DEL'], n_samples),
        'destination_airport': np.random.choice(['BOM', 'DEL', 'BLR', 'MAA'], n_samples),
        'passenger_count': np.random.randint(50, 300, n_samples),
    }
    
    # Generate datetime data
    base_date = datetime(2024, 1, 1)
    scheduled_departures = [
        base_date + timedelta(days=np.random.randint(0, 365), 
                             hours=np.random.randint(6, 23),
                             minutes=np.random.randint(0, 59))
        for _ in range(n_samples)
    ]
    
    data['scheduled_departure'] = scheduled_departures
    data['scheduled_arrival'] = [
        dep + timedelta(hours=np.random.randint(1, 8), minutes=np.random.randint(0, 59))
        for dep in scheduled_departures
    ]
    
    # Generate delays with some correlation to features
    delays = []
    for i in range(n_samples):
        base_delay = np.random.normal(10, 15)  # Base delay
        
        # Add hour-based delay (peak hours have more delays)
        hour = scheduled_departures[i].hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Peak hours
            base_delay += np.random.normal(10, 5)
        
        # Add airline-based delay
        if data['airline'][i] in ['AI', 'SG']:
            base_delay += np.random.normal(5, 3)
        
        # Add weather-based delay (random)
        if np.random.random() < 0.1:  # 10% chance of weather delay
            base_delay += np.random.normal(30, 10)
        
        delays.append(max(0, base_delay))  # No negative delays
    
    data['delay_minutes'] = delays
    
    # Generate actual times based on delays
    data['actual_departure'] = [
        sched + timedelta(minutes=delay) 
        for sched, delay in zip(scheduled_departures, delays)
    ]
    data['actual_arrival'] = [
        sched + timedelta(minutes=delay)
        for sched, delay in zip(data['scheduled_arrival'], delays)
    ]
    
    # Add delay categories
    data['delay_category'] = [
        'weather' if delay > 30 else 'operational' if delay > 15 else 'traffic' if delay > 5 else 'none'
        for delay in delays
    ]
    
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
    
    return pd.DataFrame(data)


class TestFeatureEngineer:
    """Test feature engineering functionality"""
    
    def test_create_time_features(self, sample_flight_data):
        """Test time feature creation"""
        fe = FeatureEngineer()
        df_with_features = fe.create_time_features(sample_flight_data)
        
        # Check that time features are created
        expected_features = [
            'departure_hour', 'departure_day_of_week', 'departure_month',
            'departure_quarter', 'is_weekend', 'is_peak_hour',
            'arrival_hour', 'arrival_day_of_week', 'scheduled_duration_minutes'
        ]
        
        for feature in expected_features:
            assert feature in df_with_features.columns
        
        # Check value ranges
        assert df_with_features['departure_hour'].min() >= 0
        assert df_with_features['departure_hour'].max() <= 23
        assert df_with_features['departure_day_of_week'].min() >= 0
        assert df_with_features['departure_day_of_week'].max() <= 6
        assert df_with_features['is_weekend'].isin([0, 1]).all()
        assert df_with_features['is_peak_hour'].isin([0, 1]).all()
    
    def test_create_weather_features(self, sample_flight_data):
        """Test weather feature creation"""
        fe = FeatureEngineer()
        df_with_features = fe.create_weather_features(sample_flight_data)
        
        # Check that weather features are created
        expected_features = [
            'temperature', 'humidity', 'wind_speed', 'visibility', 
            'precipitation', 'is_bad_weather'
        ]
        
        for feature in expected_features:
            assert feature in df_with_features.columns
        
        # Check that is_bad_weather is binary
        assert df_with_features['is_bad_weather'].isin([0, 1]).all()
    
    def test_create_traffic_features(self, sample_flight_data):
        """Test traffic feature creation"""
        fe = FeatureEngineer()
        df_with_features = fe.create_traffic_features(sample_flight_data)
        
        # Check that traffic features are created
        expected_features = ['hourly_departures', 'congestion_score']
        
        for feature in expected_features:
            assert feature in df_with_features.columns
        
        # Check value ranges
        assert df_with_features['hourly_departures'].min() >= 1
        assert df_with_features['congestion_score'].min() >= 0
        assert df_with_features['congestion_score'].max() <= 1
    
    def test_create_operational_features(self, sample_flight_data):
        """Test operational feature creation"""
        fe = FeatureEngineer()
        df_with_features = fe.create_operational_features(sample_flight_data)
        
        # Check that operational features are created
        expected_features = [
            'airline_avg_delay', 'airline_delay_std', 'aircraft_category',
            'route', 'route_avg_delay', 'route_frequency', 'passenger_load', 'is_high_load'
        ]
        
        for feature in expected_features:
            assert feature in df_with_features.columns
        
        # Check aircraft categorization
        valid_categories = ['narrow_body', 'wide_body', 'regional', 'other', 'unknown']
        assert df_with_features['aircraft_category'].isin(valid_categories).all()
        
        # Check binary features
        assert df_with_features['is_high_load'].isin([0, 1]).all()
    
    def test_prepare_features(self, sample_flight_data):
        """Test complete feature preparation"""
        fe = FeatureEngineer()
        X, y = fe.prepare_features(sample_flight_data)
        
        # Check that features and target are returned
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert len(X) == len(sample_flight_data)
        
        # Check that we have a reasonable number of features
        assert X.shape[1] > 10  # Should have many features
    
    def test_fit_transform(self, sample_flight_data):
        """Test feature transformation"""
        fe = FeatureEngineer()
        X, y = fe.prepare_features(sample_flight_data)
        X_transformed = fe.fit_transform(X, y)
        
        # Check transformation
        assert X_transformed is not None
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > 0
        assert fe.is_fitted
        
        # Check that we can transform new data
        X_new_transformed = fe.transform(X.head(10))
        assert X_new_transformed.shape[0] == 10
        assert X_new_transformed.shape[1] == X_transformed.shape[1]


class TestDelayPredictor:
    """Test delay prediction models"""
    
    @pytest.mark.parametrize("model_type", ['xgboost', 'random_forest', 'gradient_boosting', 'linear'])
    def test_model_initialization(self, model_type):
        """Test model initialization for different types"""
        predictor = DelayPredictor(model_type=model_type)
        assert predictor.model_type == model_type
        assert predictor.model is not None
        assert not predictor.is_trained
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error"""
        with pytest.raises(ValueError):
            DelayPredictor(model_type='invalid_model')
    
    def test_prepare_data(self, sample_flight_data):
        """Test data preparation"""
        predictor = DelayPredictor()
        X, y = predictor.prepare_data(sample_flight_data)
        
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert X.shape[1] > 0
    
    @pytest.mark.parametrize("model_type", ['xgboost', 'random_forest'])
    def test_model_training(self, sample_flight_data, model_type):
        """Test model training"""
        predictor = DelayPredictor(model_type=model_type)
        metrics = predictor.train(sample_flight_data)
        
        # Check that model is trained
        assert predictor.is_trained
        assert metrics is not None
        
        # Check metrics structure
        assert 'train_metrics' in metrics
        assert 'test_metrics' in metrics
        assert 'cv_scores' in metrics
        
        # Check that metrics are reasonable
        test_metrics = metrics['test_metrics']
        assert 'rmse' in test_metrics
        assert 'mae' in test_metrics
        assert 'r2' in test_metrics
        assert test_metrics['rmse'] > 0
        assert test_metrics['mae'] > 0
    
    def test_prediction(self, sample_flight_data):
        """Test making predictions"""
        predictor = DelayPredictor()
        predictor.train(sample_flight_data)
        
        # Test prediction on same data
        predictions = predictor.predict(sample_flight_data.head(100))
        assert len(predictions) == 100
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    
    def test_prediction_without_training(self, sample_flight_data):
        """Test that prediction fails without training"""
        predictor = DelayPredictor()
        
        with pytest.raises(ValueError):
            predictor.predict(sample_flight_data)
    
    def test_feature_importance(self, sample_flight_data):
        """Test feature importance calculation"""
        predictor = DelayPredictor(model_type='xgboost')
        predictor.train(sample_flight_data)
        
        importance = predictor.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
    
    def test_model_save_load(self, sample_flight_data):
        """Test model saving and loading"""
        predictor = DelayPredictor()
        predictor.train(sample_flight_data)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            predictor.save_model(tmp.name)
            
            # Load model
            new_predictor = DelayPredictor()
            new_predictor.load_model(tmp.name)
            
            # Check that loaded model works
            assert new_predictor.is_trained
            predictions = new_predictor.predict(sample_flight_data.head(10))
            assert len(predictions) == 10
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_hyperparameter_tuning(self, sample_flight_data):
        """Test hyperparameter tuning"""
        # Use smaller dataset for faster testing
        small_data = sample_flight_data.head(200)
        
        predictor = DelayPredictor(model_type='xgboost')
        tuning_results = predictor.hyperparameter_tuning(small_data)
        
        assert 'best_params' in tuning_results
        assert 'best_score' in tuning_results
        assert tuning_results['best_score'] > 0
    
    def test_prediction_with_confidence(self, sample_flight_data):
        """Test prediction with confidence intervals"""
        predictor = DelayPredictor(model_type='random_forest')
        predictor.train(sample_flight_data)
        
        predictions, confidence = predictor.predict_with_confidence(sample_flight_data.head(50))
        
        assert len(predictions) == 50
        assert len(confidence) == 50
        assert all(c >= 0 for c in confidence)  # Confidence should be non-negative
    
    def test_model_summary(self, sample_flight_data):
        """Test model summary generation"""
        predictor = DelayPredictor()
        
        # Test summary before training
        summary = predictor.get_model_summary()
        assert summary['status'] == 'Model not trained'
        
        # Test summary after training
        predictor.train(sample_flight_data)
        summary = predictor.get_model_summary()
        
        assert 'model_type' in summary
        assert 'training_status' in summary
        assert 'metrics' in summary
        assert summary['training_status'] == 'trained'


class TestEnsembleDelayPredictor:
    """Test ensemble delay prediction"""
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization"""
        ensemble = EnsembleDelayPredictor()
        assert len(ensemble.models) > 0
        assert not ensemble.is_trained
    
    def test_ensemble_training(self, sample_flight_data):
        """Test ensemble training"""
        # Use smaller dataset and fewer models for faster testing
        ensemble = EnsembleDelayPredictor(model_types=['xgboost', 'random_forest'])
        
        metrics = ensemble.train(sample_flight_data.head(300))
        
        assert ensemble.is_trained
        assert len(metrics) == 2  # Two models
        assert all(model_type in metrics for model_type in ['xgboost', 'random_forest'])
        
        # Check weights
        assert len(ensemble.weights) == 2
        assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-6  # Weights should sum to 1
    
    def test_ensemble_prediction(self, sample_flight_data):
        """Test ensemble prediction"""
        ensemble = EnsembleDelayPredictor(model_types=['xgboost', 'random_forest'])
        ensemble.train(sample_flight_data.head(300))
        
        predictions = ensemble.predict(sample_flight_data.head(50))
        assert len(predictions) == 50
        
        # Test individual predictions
        individual_preds = ensemble.get_individual_predictions(sample_flight_data.head(50))
        assert len(individual_preds) == 2
        assert 'xgboost' in individual_preds
        assert 'random_forest' in individual_preds


class TestModelEvaluator:
    """Test model evaluation functionality"""
    
    def test_calculate_regression_metrics(self):
        """Test regression metrics calculation"""
        evaluator = ModelEvaluator()
        
        # Create sample data
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])
        
        metrics = evaluator.calculate_regression_metrics(y_true, y_pred)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'rmse', 'mae', 'r2', 'mape', 'explained_variance',
            'mean_residual', 'std_residual', 'max_error',
            'delay_precision', 'delay_recall', 'delay_f1', 'delay_accuracy'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_evaluate_model_performance(self, sample_flight_data):
        """Test model performance evaluation"""
        evaluator = ModelEvaluator()
        predictor = DelayPredictor()
        predictor.train(sample_flight_data)
        
        # Prepare test data
        X, y = predictor.prepare_data(sample_flight_data.head(100))
        X_transformed = predictor.feature_engineer.transform(X)
        
        evaluation = evaluator.evaluate_model_performance(
            predictor.model, X_transformed, y, "Test Model"
        )
        
        assert 'model_name' in evaluation
        assert 'metrics' in evaluation
        assert 'sample_size' in evaluation
        assert evaluation['model_name'] == "Test Model"
        assert evaluation['sample_size'] == len(y)
    
    def test_compare_models(self):
        """Test model comparison"""
        evaluator = ModelEvaluator()
        
        # Create mock evaluations
        eval1 = {
            'model_name': 'Model1',
            'sample_size': 100,
            'metrics': {'rmse': 10.0, 'mae': 8.0, 'r2': 0.8}
        }
        eval2 = {
            'model_name': 'Model2', 
            'sample_size': 100,
            'metrics': {'rmse': 12.0, 'mae': 9.0, 'r2': 0.7}
        }
        
        comparison = evaluator.compare_models([eval1, eval2])
        
        assert len(comparison) == 2
        assert 'model_name' in comparison.columns
        assert 'rmse' in comparison.columns
        
        # Should be sorted by RMSE (lower is better)
        assert comparison.iloc[0]['model_name'] == 'Model1'
    
    def test_monitor_model_drift(self):
        """Test model drift monitoring"""
        evaluator = ModelEvaluator()
        
        baseline_metrics = {'rmse': 10.0, 'mae': 8.0, 'r2': 0.8}
        current_metrics = {'rmse': 12.0, 'mae': 9.0, 'r2': 0.7}  # Degraded performance
        
        drift_analysis = evaluator.monitor_model_drift(baseline_metrics, current_metrics, threshold=0.1)
        
        assert 'drift_detected' in drift_analysis
        assert 'degraded_metrics' in drift_analysis
        assert 'metric_changes' in drift_analysis
        
        # Should detect drift due to increased RMSE and decreased R2
        assert drift_analysis['drift_detected']
        assert 'rmse' in drift_analysis['degraded_metrics']
        assert 'r2' in drift_analysis['degraded_metrics']


class TestIntegration:
    """Integration tests for the complete delay prediction pipeline"""
    
    def test_end_to_end_prediction_pipeline(self, sample_flight_data):
        """Test complete end-to-end prediction pipeline"""
        # Split data
        train_data = sample_flight_data.head(800)
        test_data = sample_flight_data.tail(200)
        
        # Train model
        predictor = DelayPredictor(model_type='xgboost')
        train_metrics = predictor.train(train_data)
        
        # Make predictions
        predictions = predictor.predict(test_data)
        
        # Evaluate predictions
        evaluator = ModelEvaluator()
        X_test, y_test = predictor.prepare_data(test_data)
        X_test_transformed = predictor.feature_engineer.transform(X_test)
        
        evaluation = evaluator.evaluate_model_performance(
            predictor.model, X_test_transformed, y_test, "XGBoost"
        )
        
        # Check that everything works
        assert len(predictions) == len(test_data)
        assert evaluation['metrics']['rmse'] > 0
        assert evaluation['metrics']['r2'] <= 1.0
        
        # Check that model performs reasonably
        assert evaluation['metrics']['r2'] > 0.1  # At least some predictive power
        assert evaluation['metrics']['rmse'] < 100  # Reasonable error range
    
    def test_model_comparison_pipeline(self, sample_flight_data):
        """Test comparing multiple models"""
        # Use smaller dataset for faster testing
        data = sample_flight_data.head(400)
        
        models = ['xgboost', 'random_forest']
        evaluations = []
        
        for model_type in models:
            predictor = DelayPredictor(model_type=model_type)
            predictor.train(data)
            
            # Evaluate on same data (for testing purposes)
            X, y = predictor.prepare_data(data)
            X_transformed = predictor.feature_engineer.transform(X)
            
            evaluator = ModelEvaluator()
            evaluation = evaluator.evaluate_model_performance(
                predictor.model, X_transformed, y, model_type
            )
            evaluations.append(evaluation)
        
        # Compare models
        evaluator = ModelEvaluator()
        comparison = evaluator.compare_models(evaluations)
        
        assert len(comparison) == 2
        assert all(model in comparison['model_name'].values for model in models)
        
        # Check that comparison is sorted by performance
        assert comparison.iloc[0]['rmse'] <= comparison.iloc[1]['rmse']