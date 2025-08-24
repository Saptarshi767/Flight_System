"""
Time series forecasting models using Prophet for congestion and delay forecasting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Prophet import
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    Prophet = None

# Statsmodels for seasonal decomposition (optional)
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    seasonal_decompose = None
    adfuller = None
    ARIMA = None

# Scikit-learn for additional time series utilities
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Local imports
from .model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """Time series forecasting using Prophet and statistical models"""
    
    def __init__(self, model_type: str = 'prophet'):
        """
        Initialize time series forecaster
        
        Args:
            model_type: Type of model ('prophet', 'arima', 'seasonal_naive')
        """
        if model_type == 'prophet' and not HAS_PROPHET:
            raise ImportError("Prophet is not installed. Please install it with: pip install prophet")
        
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.forecast_results = None
        self.seasonal_components = None
        self.trend_components = None
        self.evaluator = ModelEvaluator()
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the forecasting model"""
        if self.model_type == 'prophet':
            self.model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                interval_width=0.95
            )
        elif self.model_type == 'arima':
            # ARIMA model will be initialized during fitting
            self.model = None
        elif self.model_type == 'seasonal_naive':
            # Simple seasonal naive model
            self.model = None
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} forecasting model")
    
    def prepare_time_series_data(self, df: pd.DataFrame, 
                                date_column: str = 'scheduled_departure',
                                value_column: str = 'delay_minutes',
                                aggregation: str = 'hourly') -> pd.DataFrame:
        """
        Prepare time series data for forecasting
        
        Args:
            df: Input dataframe
            date_column: Column containing datetime values
            value_column: Column containing values to forecast
            aggregation: Time aggregation ('hourly', 'daily', 'weekly')
        """
        logger.info(f"Preparing time series data with {aggregation} aggregation")
        
        # Convert to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Set aggregation frequency
        freq_map = {
            'hourly': 'H',
            'daily': 'D',
            'weekly': 'W'
        }
        freq = freq_map.get(aggregation, 'H')
        
        # Aggregate data
        if aggregation == 'hourly':
            df['datetime_rounded'] = df[date_column].dt.floor('H')
        elif aggregation == 'daily':
            df['datetime_rounded'] = df[date_column].dt.floor('D')
        elif aggregation == 'weekly':
            df['datetime_rounded'] = df[date_column].dt.floor('W')
        
        # Group and aggregate
        agg_functions = {
            value_column: ['mean', 'count', 'std']
        }
        
        # Add congestion score if available
        if 'congestion_score' in df.columns:
            agg_functions['congestion_score'] = 'mean'
        
        ts_data = df.groupby('datetime_rounded').agg(agg_functions).reset_index()
        
        # Flatten column names
        if 'congestion_score' in df.columns:
            ts_data.columns = ['ds', 'y', 'flight_count', 'y_std', 'congestion']
        else:
            ts_data.columns = ['ds', 'y', 'flight_count', 'y_std']
            ts_data['congestion'] = 0
        
        # Fill missing values
        ts_data['y'] = ts_data['y'].fillna(ts_data['y'].mean())
        ts_data['y_std'] = ts_data['y_std'].fillna(ts_data['y_std'].mean())
        ts_data['congestion'] = ts_data['congestion'].fillna(0)
        
        # Create complete time range
        date_range = pd.date_range(
            start=ts_data['ds'].min(),
            end=ts_data['ds'].max(),
            freq=freq
        )
        
        complete_ts = pd.DataFrame({'ds': date_range})
        ts_data = complete_ts.merge(ts_data, on='ds', how='left')
        
        # Forward fill missing values
        ts_data = ts_data.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Prepared time series with {len(ts_data)} data points")
        return ts_data
    
    def seasonal_decomposition(self, ts_data: pd.DataFrame, 
                              period: int = 24) -> Dict[str, pd.Series]:
        """Perform seasonal decomposition of time series"""
        logger.info("Performing seasonal decomposition")
        
        if not HAS_STATSMODELS:
            logger.warning("Statsmodels not available, skipping seasonal decomposition")
            return {}
        
        # Ensure we have enough data points
        if len(ts_data) < 2 * period:
            logger.warning(f"Not enough data for seasonal decomposition (need at least {2*period} points)")
            return {}
        
        try:
            decomposition = seasonal_decompose(
                ts_data['y'], 
                model='additive', 
                period=period
            )
            
            self.seasonal_components = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }
            
            logger.info("Seasonal decomposition completed")
            return self.seasonal_components
        
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {e}")
            return {}
    
    def check_stationarity(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """Check stationarity of time series using Augmented Dickey-Fuller test"""
        logger.info("Checking time series stationarity")
        
        if not HAS_STATSMODELS:
            logger.warning("Statsmodels not available, skipping stationarity test")
            return {'is_stationary': False, 'error': 'statsmodels not available'}
        
        try:
            result = adfuller(ts_data['y'].dropna())
            
            stationarity_result = {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
            
            logger.info(f"Stationarity test - p-value: {result[1]:.4f}, stationary: {result[1] < 0.05}")
            return stationarity_result
        
        except Exception as e:
            logger.error(f"Error in stationarity test: {e}")
            return {'is_stationary': False, 'error': str(e)}
    
    def fit_prophet_model(self, ts_data: pd.DataFrame, 
                         add_regressors: List[str] = None) -> Dict[str, Any]:
        """Fit Prophet model"""
        logger.info("Fitting Prophet model")
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_data = ts_data[['ds', 'y']].copy()
        
        # Add additional regressors if specified
        if add_regressors:
            for regressor in add_regressors:
                if regressor in ts_data.columns:
                    prophet_data[regressor] = ts_data[regressor]
                    self.model.add_regressor(regressor)
        
        # Fit model
        self.model.fit(prophet_data)
        self.is_fitted = True
        
        # Get model components
        future = self.model.make_future_dataframe(periods=0)
        if add_regressors:
            for regressor in add_regressors:
                if regressor in ts_data.columns:
                    future[regressor] = ts_data[regressor]
        
        forecast = self.model.predict(future)
        
        # Store results
        fit_results = {
            'forecast': forecast,
            'model_params': self.model.params,
            'changepoints': self.model.changepoints,
            'train_size': len(prophet_data)
        }
        
        logger.info("Prophet model fitted successfully")
        return fit_results
    
    def fit_arima_model(self, ts_data: pd.DataFrame, 
                       order: Tuple[int, int, int] = None) -> Dict[str, Any]:
        """Fit ARIMA model"""
        logger.info("Fitting ARIMA model")
        
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels is required for ARIMA models. Please install it with: pip install statsmodels")
        
        # Auto-determine ARIMA order if not provided
        if order is None:
            order = self._auto_arima_order(ts_data['y'])
        
        try:
            self.model = ARIMA(ts_data['y'], order=order)
            fitted_model = self.model.fit()
            self.fitted_arima = fitted_model
            self.is_fitted = True
            
            fit_results = {
                'order': order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'params': fitted_model.params,
                'train_size': len(ts_data)
            }
            
            logger.info(f"ARIMA{order} model fitted successfully")
            return fit_results
        
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
    
    def _auto_arima_order(self, ts: pd.Series) -> Tuple[int, int, int]:
        """Automatically determine ARIMA order using AIC"""
        if not HAS_STATSMODELS:
            return (1, 1, 1)  # Default order
        
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        # Try different combinations
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        logger.info(f"Auto-selected ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def fit(self, df: pd.DataFrame, 
            date_column: str = 'scheduled_departure',
            value_column: str = 'delay_minutes',
            aggregation: str = 'hourly',
            add_regressors: List[str] = None) -> Dict[str, Any]:
        """
        Fit the time series model
        
        Args:
            df: Input dataframe
            date_column: Column containing datetime values
            value_column: Column containing values to forecast
            aggregation: Time aggregation level
            add_regressors: Additional regressors for Prophet model
        """
        logger.info(f"Fitting {self.model_type} time series model")
        
        # Prepare data
        ts_data = self.prepare_time_series_data(df, date_column, value_column, aggregation)
        
        # Perform seasonal decomposition
        self.seasonal_decomposition(ts_data)
        
        # Check stationarity
        stationarity = self.check_stationarity(ts_data)
        
        # Fit model based on type
        if self.model_type == 'prophet':
            fit_results = self.fit_prophet_model(ts_data, add_regressors)
        elif self.model_type == 'arima':
            fit_results = self.fit_arima_model(ts_data)
        elif self.model_type == 'seasonal_naive':
            fit_results = self._fit_seasonal_naive(ts_data)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Store training data
        self.training_data = ts_data
        
        # Add stationarity info to results
        fit_results['stationarity'] = stationarity
        fit_results['seasonal_components'] = self.seasonal_components
        
        return fit_results
    
    def _fit_seasonal_naive(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit seasonal naive model"""
        logger.info("Fitting seasonal naive model")
        
        # Store seasonal pattern (24-hour cycle for hourly data)
        season_length = 24
        if len(ts_data) >= season_length:
            self.seasonal_pattern = ts_data['y'].tail(season_length).values
        else:
            self.seasonal_pattern = ts_data['y'].values
        
        self.is_fitted = True
        
        return {
            'seasonal_pattern': self.seasonal_pattern,
            'season_length': season_length,
            'train_size': len(ts_data)
        }
    
    def forecast(self, periods: int = 24, 
                future_regressors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate forecasts
        
        Args:
            periods: Number of periods to forecast
            future_regressors: Future values of regressors (for Prophet)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        logger.info(f"Generating {periods} period forecast using {self.model_type}")
        
        if self.model_type == 'prophet':
            return self._forecast_prophet(periods, future_regressors)
        elif self.model_type == 'arima':
            return self._forecast_arima(periods)
        elif self.model_type == 'seasonal_naive':
            return self._forecast_seasonal_naive(periods)
        else:
            raise ValueError(f"Forecasting not implemented for {self.model_type}")
    
    def _forecast_prophet(self, periods: int, 
                         future_regressors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate Prophet forecasts"""
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Add future regressors if provided
        if future_regressors is not None:
            for col in future_regressors.columns:
                if col in future.columns:
                    future[col].iloc[-periods:] = future_regressors[col].values
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Extract forecast period
        forecast_period = forecast.tail(periods).copy()
        
        # Rename columns for consistency
        forecast_period = forecast_period.rename(columns={
            'ds': 'datetime',
            'yhat': 'forecast',
            'yhat_lower': 'forecast_lower',
            'yhat_upper': 'forecast_upper'
        })
        
        self.forecast_results = forecast_period
        return forecast_period
    
    def _forecast_arima(self, periods: int) -> pd.DataFrame:
        """Generate ARIMA forecasts"""
        # Generate forecast
        forecast_result = self.fitted_arima.forecast(steps=periods)
        conf_int = self.fitted_arima.get_forecast(steps=periods).conf_int()
        
        # Create datetime index
        last_date = self.training_data['ds'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=periods,
            freq='H'
        )
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'datetime': future_dates,
            'forecast': forecast_result,
            'forecast_lower': conf_int.iloc[:, 0],
            'forecast_upper': conf_int.iloc[:, 1]
        })
        
        self.forecast_results = forecast_df
        return forecast_df
    
    def _forecast_seasonal_naive(self, periods: int) -> pd.DataFrame:
        """Generate seasonal naive forecasts"""
        # Repeat seasonal pattern
        season_length = len(self.seasonal_pattern)
        forecasts = []
        
        for i in range(periods):
            forecast_value = self.seasonal_pattern[i % season_length]
            forecasts.append(forecast_value)
        
        # Create datetime index
        last_date = self.training_data['ds'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=periods,
            freq='H'
        )
        
        # Create forecast dataframe with simple confidence intervals
        forecast_std = np.std(self.seasonal_pattern)
        forecast_df = pd.DataFrame({
            'datetime': future_dates,
            'forecast': forecasts,
            'forecast_lower': np.array(forecasts) - 1.96 * forecast_std,
            'forecast_upper': np.array(forecasts) + 1.96 * forecast_std
        })
        
        self.forecast_results = forecast_df
        return forecast_df
    
    def evaluate_forecast(self, actual_data: pd.DataFrame, 
                         forecast_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate forecast accuracy"""
        logger.info("Evaluating forecast accuracy")
        
        # Align data by datetime
        merged = pd.merge(actual_data, forecast_data, 
                         left_on='ds', right_on='datetime', how='inner')
        
        if len(merged) == 0:
            logger.warning("No overlapping data for evaluation")
            return {}
        
        # Calculate metrics
        y_true = merged['y'].values
        y_pred = merged['forecast'].values
        
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'coverage': self._calculate_coverage(merged),
            'bias': np.mean(y_pred - y_true),
            'correlation': np.corrcoef(y_true, y_pred)[0, 1]
        }
        
        logger.info(f"Forecast evaluation - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}")
        return metrics
    
    def _calculate_coverage(self, merged_data: pd.DataFrame) -> float:
        """Calculate prediction interval coverage"""
        if 'forecast_lower' not in merged_data.columns or 'forecast_upper' not in merged_data.columns:
            return 0.0
        
        within_interval = (
            (merged_data['y'] >= merged_data['forecast_lower']) &
            (merged_data['y'] <= merged_data['forecast_upper'])
        )
        
        return within_interval.mean()
    
    def detect_anomalies(self, ts_data: pd.DataFrame, 
                        threshold: float = 2.0) -> pd.DataFrame:
        """Detect anomalies in time series data"""
        logger.info("Detecting time series anomalies")
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before anomaly detection")
        
        # Generate in-sample predictions
        if self.model_type == 'prophet':
            future = self.model.make_future_dataframe(periods=0)
            forecast = self.model.predict(future)
            predictions = forecast['yhat'].values
            prediction_std = (forecast['yhat_upper'] - forecast['yhat_lower']).values / 3.92  # ~2 std
        elif self.model_type == 'arima':
            predictions = self.fitted_arima.fittedvalues.values
            residuals = self.fitted_arima.resid
            prediction_std = np.std(residuals)
        else:
            # Simple anomaly detection for seasonal naive
            predictions = np.tile(self.seasonal_pattern, len(ts_data) // len(self.seasonal_pattern) + 1)[:len(ts_data)]
            prediction_std = np.std(ts_data['y'])
        
        # Calculate anomaly scores
        residuals = ts_data['y'].values - predictions[:len(ts_data)]
        anomaly_scores = np.abs(residuals) / prediction_std
        
        # Identify anomalies
        is_anomaly = anomaly_scores > threshold
        
        # Create anomaly dataframe
        anomalies = ts_data.copy()
        anomalies['predicted'] = predictions[:len(ts_data)]
        anomalies['residual'] = residuals
        anomalies['anomaly_score'] = anomaly_scores
        anomalies['is_anomaly'] = is_anomaly
        
        logger.info(f"Detected {np.sum(is_anomaly)} anomalies out of {len(ts_data)} data points")
        return anomalies
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Get trend analysis from fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before trend analysis")
        
        trend_analysis = {}
        
        if self.model_type == 'prophet' and self.seasonal_components:
            # Prophet trend analysis
            trend = self.seasonal_components.get('trend')
            if trend is not None:
                trend_clean = trend.dropna()
                if len(trend_clean) > 1:
                    trend_slope = (trend_clean.iloc[-1] - trend_clean.iloc[0]) / len(trend_clean)
                    trend_analysis = {
                        'overall_trend': 'increasing' if trend_slope > 0 else 'decreasing',
                        'trend_slope': trend_slope,
                        'trend_strength': abs(trend_slope),
                        'changepoints': len(self.model.changepoints) if hasattr(self.model, 'changepoints') else 0
                    }
        
        elif self.seasonal_components:
            # General trend analysis from seasonal decomposition
            trend = self.seasonal_components.get('trend')
            if trend is not None:
                trend_clean = trend.dropna()
                if len(trend_clean) > 1:
                    trend_slope = (trend_clean.iloc[-1] - trend_clean.iloc[0]) / len(trend_clean)
                    trend_analysis = {
                        'overall_trend': 'increasing' if trend_slope > 0 else 'decreasing',
                        'trend_slope': trend_slope,
                        'trend_strength': abs(trend_slope)
                    }
        
        return trend_analysis
    
    def create_retraining_pipeline(self, retrain_frequency: str = 'weekly') -> Dict[str, Any]:
        """Create automated model retraining pipeline configuration"""
        logger.info(f"Creating retraining pipeline with {retrain_frequency} frequency")
        
        pipeline_config = {
            'retrain_frequency': retrain_frequency,
            'model_type': self.model_type,
            'performance_threshold': {
                'mae_increase': 0.2,  # Retrain if MAE increases by 20%
                'rmse_increase': 0.2,  # Retrain if RMSE increases by 20%
                'coverage_decrease': 0.1  # Retrain if coverage decreases by 10%
            },
            'data_requirements': {
                'min_data_points': 168,  # At least 1 week of hourly data
                'max_missing_ratio': 0.1  # Max 10% missing data
            },
            'retraining_steps': [
                'validate_new_data',
                'check_performance_degradation',
                'retrain_model',
                'validate_new_model',
                'deploy_if_better'
            ]
        }
        
        return pipeline_config
    
    def save_model(self, filepath: str):
        """Save the fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import joblib
        
        model_data = {
            'model_type': self.model_type,
            'model': self.model,
            'is_fitted': self.is_fitted,
            'training_data': self.training_data,
            'seasonal_components': self.seasonal_components,
            'forecast_results': self.forecast_results
        }
        
        if self.model_type == 'arima':
            model_data['fitted_arima'] = self.fitted_arima
        elif self.model_type == 'seasonal_naive':
            model_data['seasonal_pattern'] = self.seasonal_pattern
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a fitted model"""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model_type = model_data['model_type']
        self.model = model_data['model']
        self.is_fitted = model_data['is_fitted']
        self.training_data = model_data['training_data']
        self.seasonal_components = model_data['seasonal_components']
        self.forecast_results = model_data['forecast_results']
        
        if self.model_type == 'arima':
            self.fitted_arima = model_data['fitted_arima']
        elif self.model_type == 'seasonal_naive':
            self.seasonal_pattern = model_data['seasonal_pattern']
        
        logger.info(f"Model loaded from {filepath}")


class CongestionForecaster(TimeSeriesForecaster):
    """Specialized forecaster for airport congestion patterns"""
    
    def __init__(self, model_type: str = 'prophet'):
        super().__init__(model_type)
        self.airport_capacity = {'BOM': 60, 'DEL': 70}  # flights per hour
    
    def prepare_congestion_data(self, df: pd.DataFrame, 
                               airport_code: str = 'BOM') -> pd.DataFrame:
        """Prepare congestion-specific time series data"""
        logger.info(f"Preparing congestion data for airport {airport_code}")
        
        # Filter by airport
        airport_data = df[df['origin_airport'] == airport_code].copy()
        
        # Calculate hourly flight counts
        airport_data['hour'] = pd.to_datetime(airport_data['scheduled_departure']).dt.floor('H')
        hourly_counts = airport_data.groupby('hour').size().reset_index(name='flight_count')
        
        # Calculate congestion score
        capacity = self.airport_capacity.get(airport_code, 50)
        hourly_counts['congestion_score'] = hourly_counts['flight_count'] / capacity
        hourly_counts['congestion_score'] = hourly_counts['congestion_score'].clip(0, 1)
        
        # Prepare for time series
        ts_data = hourly_counts.rename(columns={'hour': 'ds', 'congestion_score': 'y'})
        
        # Add additional features
        ts_data['flight_count'] = hourly_counts['flight_count']
        ts_data['capacity_utilization'] = ts_data['y']
        
        return ts_data
    
    def forecast_congestion(self, df: pd.DataFrame, 
                           airport_code: str = 'BOM',
                           periods: int = 24) -> pd.DataFrame:
        """Forecast congestion patterns for specific airport"""
        # Prepare data
        ts_data = self.prepare_congestion_data(df, airport_code)
        
        # Fit model
        self.fit(ts_data, date_column='ds', value_column='y')
        
        # Generate forecast
        forecast = self.forecast(periods)
        
        # Add congestion interpretation
        forecast['congestion_level'] = pd.cut(
            forecast['forecast'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low', 'Moderate', 'High', 'Critical']
        )
        
        return forecast