"""
Tests for time series forecasting models
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from src.ml.time_series_forecaster import TimeSeriesForecaster, CongestionForecaster


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing"""
    np.random.seed(42)
    
    # Create 30 days of hourly data
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=30*24, freq='H')
    
    # Generate synthetic time series with trend and seasonality
    trend = np.linspace(10, 15, len(dates))  # Slight upward trend
    
    # Daily seasonality (higher delays during peak hours)
    daily_pattern = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)
    
    # Weekly seasonality (higher delays on weekdays)
    weekly_pattern = 2 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*7))
    
    # Random noise
    noise = np.random.normal(0, 2, len(dates))
    
    # Combine components
    values = trend + daily_pattern + weekly_pattern + noise
    values = np.maximum(values, 0)  # No negative delays
    
    # Create dataframe
    data = {
        'scheduled_departure': dates,
        'delay_minutes': values,
        'origin_airport': ['BOM'] * len(dates),
        'flight_count': np.random.poisson(5, len(dates)),  # Random flight counts
        'congestion_score': np.random.uniform(0, 1, len(dates))
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_flight_data_for_congestion():
    """Create sample flight data for congestion forecasting"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data
    base_date = datetime(2024, 1, 1)
    scheduled_departures = [
        base_date + timedelta(days=np.random.randint(0, 30), 
                             hours=np.random.randint(6, 23),
                             minutes=np.random.randint(0, 59))
        for _ in range(n_samples)
    ]
    
    data = {
        'flight_id': [f'FL{i:04d}' for i in range(n_samples)],
        'scheduled_departure': scheduled_departures,
        'origin_airport': np.random.choice(['BOM', 'DEL'], n_samples),
        'delay_minutes': np.random.exponential(10, n_samples),
        'congestion_score': np.random.uniform(0, 1, n_samples)
    }
    
    return pd.DataFrame(data)


class TestTimeSeriesForecaster:
    """Test time series forecasting functionality"""
    
    @pytest.mark.parametrize("model_type", ['seasonal_naive', 'arima'])
    def test_model_initialization(self, model_type):
        """Test model initialization for different types"""
        forecaster = TimeSeriesForecaster(model_type=model_type)
        assert forecaster.model_type == model_type
        assert not forecaster.is_fitted
    
    def test_prophet_initialization(self):
        """Test Prophet model initialization if available"""
        try:
            forecaster = TimeSeriesForecaster(model_type='prophet')
            assert forecaster.model_type == 'prophet'
            assert not forecaster.is_fitted
        except ImportError:
            pytest.skip("Prophet not available")
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error"""
        with pytest.raises(ValueError):
            TimeSeriesForecaster(model_type='invalid_model')
    
    def test_prepare_time_series_data(self, sample_time_series_data):
        """Test time series data preparation"""
        forecaster = TimeSeriesForecaster()
        
        # Test hourly aggregation
        ts_data = forecaster.prepare_time_series_data(
            sample_time_series_data, 
            aggregation='hourly'
        )
        
        assert 'ds' in ts_data.columns
        assert 'y' in ts_data.columns
        assert len(ts_data) > 0
        assert ts_data['ds'].dtype == 'datetime64[ns]'
        
        # Test daily aggregation
        ts_data_daily = forecaster.prepare_time_series_data(
            sample_time_series_data,
            aggregation='daily'
        )
        
        assert len(ts_data_daily) < len(ts_data)  # Should have fewer data points
    
    def test_seasonal_decomposition(self, sample_time_series_data):
        """Test seasonal decomposition"""
        forecaster = TimeSeriesForecaster()
        ts_data = forecaster.prepare_time_series_data(sample_time_series_data)
        
        components = forecaster.seasonal_decomposition(ts_data, period=24)
        
        if components:  # If decomposition was successful
            assert 'trend' in components
            assert 'seasonal' in components
            assert 'residual' in components
            assert 'observed' in components
    
    def test_check_stationarity(self, sample_time_series_data):
        """Test stationarity check"""
        forecaster = TimeSeriesForecaster()
        ts_data = forecaster.prepare_time_series_data(sample_time_series_data)
        
        stationarity = forecaster.check_stationarity(ts_data)
        
        assert 'is_stationary' in stationarity
        assert isinstance(stationarity['is_stationary'], bool)
        
        if 'p_value' in stationarity:
            assert isinstance(stationarity['p_value'], float)
    
    def test_fit_seasonal_naive(self, sample_time_series_data):
        """Test seasonal naive model fitting"""
        forecaster = TimeSeriesForecaster(model_type='seasonal_naive')
        
        fit_results = forecaster.fit(sample_time_series_data)
        
        assert forecaster.is_fitted
        assert 'seasonal_pattern' in fit_results
        assert 'train_size' in fit_results
        assert len(fit_results['seasonal_pattern']) > 0
    
    def test_fit_arima(self, sample_time_series_data):
        """Test ARIMA model fitting"""
        forecaster = TimeSeriesForecaster(model_type='arima')
        
        # Use smaller dataset for faster testing
        small_data = sample_time_series_data.head(100)
        fit_results = forecaster.fit(small_data)
        
        assert forecaster.is_fitted
        assert 'order' in fit_results
        assert 'aic' in fit_results
        assert 'train_size' in fit_results
    
    @pytest.mark.skipif(True, reason="Prophet may not be available in test environment")
    def test_fit_prophet(self, sample_time_series_data):
        """Test Prophet model fitting"""
        try:
            forecaster = TimeSeriesForecaster(model_type='prophet')
            fit_results = forecaster.fit(sample_time_series_data)
            
            assert forecaster.is_fitted
            assert 'forecast' in fit_results
            assert 'train_size' in fit_results
        except ImportError:
            pytest.skip("Prophet not available")
    
    def test_forecast_seasonal_naive(self, sample_time_series_data):
        """Test forecasting with seasonal naive model"""
        forecaster = TimeSeriesForecaster(model_type='seasonal_naive')
        forecaster.fit(sample_time_series_data)
        
        forecast = forecaster.forecast(periods=24)
        
        assert len(forecast) == 24
        assert 'datetime' in forecast.columns
        assert 'forecast' in forecast.columns
        assert 'forecast_lower' in forecast.columns
        assert 'forecast_upper' in forecast.columns
        
        # Check that forecasts are reasonable
        assert all(forecast['forecast'] >= 0)  # No negative forecasts
        assert all(forecast['forecast_lower'] <= forecast['forecast_upper'])
    
    def test_forecast_arima(self, sample_time_series_data):
        """Test forecasting with ARIMA model"""
        forecaster = TimeSeriesForecaster(model_type='arima')
        
        # Use smaller dataset for faster testing
        small_data = sample_time_series_data.head(100)
        forecaster.fit(small_data)
        
        forecast = forecaster.forecast(periods=12)
        
        assert len(forecast) == 12
        assert 'datetime' in forecast.columns
        assert 'forecast' in forecast.columns
        assert 'forecast_lower' in forecast.columns
        assert 'forecast_upper' in forecast.columns
    
    def test_forecast_without_fitting(self, sample_time_series_data):
        """Test that forecasting fails without fitting"""
        forecaster = TimeSeriesForecaster()
        
        with pytest.raises(ValueError):
            forecaster.forecast(periods=24)
    
    def test_evaluate_forecast(self, sample_time_series_data):
        """Test forecast evaluation"""
        forecaster = TimeSeriesForecaster(model_type='seasonal_naive')
        
        # Split data
        train_data = sample_time_series_data.head(600)  # 25 days
        test_data = sample_time_series_data.tail(120)   # 5 days
        
        # Fit and forecast
        forecaster.fit(train_data)
        forecast = forecaster.forecast(periods=len(test_data))
        
        # Prepare test data for evaluation
        test_ts = forecaster.prepare_time_series_data(test_data)
        
        # Evaluate
        metrics = forecaster.evaluate_forecast(test_ts, forecast)
        
        if metrics:  # If evaluation was successful
            assert 'mae' in metrics
            assert 'rmse' in metrics
            assert 'mape' in metrics
            assert all(isinstance(v, (int, float)) for v in metrics.values())
    
    def test_detect_anomalies(self, sample_time_series_data):
        """Test anomaly detection"""
        forecaster = TimeSeriesForecaster(model_type='seasonal_naive')
        forecaster.fit(sample_time_series_data)
        
        ts_data = forecaster.prepare_time_series_data(sample_time_series_data)
        anomalies = forecaster.detect_anomalies(ts_data)
        
        assert 'is_anomaly' in anomalies.columns
        assert 'anomaly_score' in anomalies.columns
        assert anomalies['is_anomaly'].dtype == bool
        assert len(anomalies) == len(ts_data)
    
    def test_get_trend_analysis(self, sample_time_series_data):
        """Test trend analysis"""
        forecaster = TimeSeriesForecaster(model_type='seasonal_naive')
        forecaster.fit(sample_time_series_data)
        
        trend_analysis = forecaster.get_trend_analysis()
        
        # May be empty if no trend components available
        if trend_analysis:
            assert isinstance(trend_analysis, dict)
    
    def test_create_retraining_pipeline(self, sample_time_series_data):
        """Test retraining pipeline creation"""
        forecaster = TimeSeriesForecaster()
        
        pipeline_config = forecaster.create_retraining_pipeline()
        
        assert 'retrain_frequency' in pipeline_config
        assert 'model_type' in pipeline_config
        assert 'performance_threshold' in pipeline_config
        assert 'data_requirements' in pipeline_config
        assert 'retraining_steps' in pipeline_config
    
    def test_save_load_model(self, sample_time_series_data):
        """Test model saving and loading"""
        forecaster = TimeSeriesForecaster(model_type='seasonal_naive')
        forecaster.fit(sample_time_series_data)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            forecaster.save_model(tmp.name)
            
            # Load model
            new_forecaster = TimeSeriesForecaster()
            new_forecaster.load_model(tmp.name)
            
            # Check that loaded model works
            assert new_forecaster.is_fitted
            assert new_forecaster.model_type == 'seasonal_naive'
            
            forecast = new_forecaster.forecast(periods=12)
            assert len(forecast) == 12
            
            # Clean up
            os.unlink(tmp.name)


class TestCongestionForecaster:
    """Test congestion-specific forecasting"""
    
    def test_congestion_forecaster_initialization(self):
        """Test congestion forecaster initialization"""
        forecaster = CongestionForecaster()
        assert isinstance(forecaster, TimeSeriesForecaster)
        assert hasattr(forecaster, 'airport_capacity')
        assert 'BOM' in forecaster.airport_capacity
        assert 'DEL' in forecaster.airport_capacity
    
    def test_prepare_congestion_data(self, sample_flight_data_for_congestion):
        """Test congestion data preparation"""
        forecaster = CongestionForecaster()
        
        congestion_data = forecaster.prepare_congestion_data(
            sample_flight_data_for_congestion, 
            airport_code='BOM'
        )
        
        assert 'ds' in congestion_data.columns
        assert 'y' in congestion_data.columns
        assert 'flight_count' in congestion_data.columns
        assert 'capacity_utilization' in congestion_data.columns
        
        # Check that congestion scores are between 0 and 1
        assert all(congestion_data['y'] >= 0)
        assert all(congestion_data['y'] <= 1)
    
    def test_forecast_congestion(self, sample_flight_data_for_congestion):
        """Test congestion forecasting"""
        forecaster = CongestionForecaster(model_type='seasonal_naive')
        
        forecast = forecaster.forecast_congestion(
            sample_flight_data_for_congestion,
            airport_code='BOM',
            periods=24
        )
        
        assert len(forecast) == 24
        assert 'datetime' in forecast.columns
        assert 'forecast' in forecast.columns
        assert 'congestion_level' in forecast.columns
        
        # Check congestion levels
        valid_levels = ['Low', 'Moderate', 'High', 'Critical']
        assert all(level in valid_levels for level in forecast['congestion_level'].dropna())


class TestIntegration:
    """Integration tests for time series forecasting"""
    
    def test_end_to_end_forecasting_pipeline(self, sample_time_series_data):
        """Test complete forecasting pipeline"""
        # Split data
        train_data = sample_time_series_data.head(600)
        test_data = sample_time_series_data.tail(120)
        
        # Initialize forecaster
        forecaster = TimeSeriesForecaster(model_type='seasonal_naive')
        
        # Fit model
        fit_results = forecaster.fit(train_data)
        assert forecaster.is_fitted
        
        # Generate forecast
        forecast = forecaster.forecast(periods=24)
        assert len(forecast) == 24
        
        # Detect anomalies
        ts_data = forecaster.prepare_time_series_data(train_data)
        anomalies = forecaster.detect_anomalies(ts_data)
        assert len(anomalies) == len(ts_data)
        
        # Get trend analysis
        trend_analysis = forecaster.get_trend_analysis()
        assert isinstance(trend_analysis, dict)
        
        # Evaluate if we have test data
        if len(test_data) > 0:
            test_ts = forecaster.prepare_time_series_data(test_data.head(24))
            if len(test_ts) > 0:
                metrics = forecaster.evaluate_forecast(test_ts, forecast)
                if metrics:
                    assert 'mae' in metrics
                    assert 'rmse' in metrics
    
    def test_multiple_model_comparison(self, sample_time_series_data):
        """Test comparing multiple forecasting models"""
        # Use smaller dataset for faster testing
        data = sample_time_series_data.head(200)
        
        models = ['seasonal_naive', 'arima']
        results = {}
        
        for model_type in models:
            forecaster = TimeSeriesForecaster(model_type=model_type)
            fit_results = forecaster.fit(data)
            forecast = forecaster.forecast(periods=12)
            
            results[model_type] = {
                'fit_results': fit_results,
                'forecast': forecast,
                'forecaster': forecaster
            }
        
        # Check that all models produced results
        assert len(results) == 2
        for model_type, result in results.items():
            assert result['forecaster'].is_fitted
            assert len(result['forecast']) == 12
    
    def test_congestion_forecasting_pipeline(self, sample_flight_data_for_congestion):
        """Test complete congestion forecasting pipeline"""
        forecaster = CongestionForecaster(model_type='seasonal_naive')
        
        # Test for different airports
        airports = ['BOM', 'DEL']
        
        for airport in airports:
            airport_data = sample_flight_data_for_congestion[
                sample_flight_data_for_congestion['origin_airport'] == airport
            ]
            
            if len(airport_data) > 50:  # Only test if we have enough data
                forecast = forecaster.forecast_congestion(
                    sample_flight_data_for_congestion,
                    airport_code=airport,
                    periods=12
                )
                
                assert len(forecast) == 12
                assert 'congestion_level' in forecast.columns
                
                # Check that congestion levels are assigned
                assert not forecast['congestion_level'].isna().all()