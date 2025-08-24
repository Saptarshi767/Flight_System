"""
Utility functions for time series queries and aggregations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import pandas as pd
from dataclasses import dataclass

from .influxdb_client import InfluxDBManager, get_influxdb_manager
from .time_series_models import TimeSeriesQueryBuilder, MeasurementType

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Container for query results with metadata."""
    data: List[Dict[str, Any]]
    query: str
    execution_time: float
    record_count: int
    timestamp: datetime


@dataclass
class AggregationConfig:
    """Configuration for time series aggregations."""
    window: str = "1h"  # Time window (e.g., '1h', '1d', '1w')
    function: str = "mean"  # Aggregation function
    group_by: Optional[List[str]] = None  # Tags to group by
    fill_missing: bool = True  # Fill missing values
    fill_value: Union[str, float] = "null"  # Value to use for missing data


class TimeSeriesQueryUtils:
    """
    Utility class for common time series queries and aggregations.
    
    Provides high-level methods for analyzing flight data stored in InfluxDB.
    """
    
    def __init__(self, influxdb_manager: Optional[InfluxDBManager] = None):
        """
        Initialize query utils.
        
        Args:
            influxdb_manager: InfluxDB manager instance. If None, uses global instance.
        """
        self.influxdb_manager = influxdb_manager or get_influxdb_manager()
        self.query_builder = TimeSeriesQueryBuilder()
    
    async def get_delay_statistics(
        self,
        airport_code: str,
        start_time: datetime,
        end_time: datetime,
        airline: Optional[str] = None,
        delay_category: Optional[str] = None
    ) -> QueryResult:
        """
        Get delay statistics for an airport.
        
        Args:
            airport_code: Airport code to analyze
            start_time: Start time for analysis
            end_time: End time for analysis
            airline: Optional airline filter
            delay_category: Optional delay category filter
            
        Returns:
            QueryResult containing delay statistics.
        """
        start_query_time = datetime.now()
        
        try:
            # Build base query
            filters = {}
            if airline:
                filters["airline"] = airline
            if delay_category:
                filters["delay_category"] = delay_category
            
            # Query delay metrics
            data = self.influxdb_manager.query_flight_metrics(
                measurement=MeasurementType.DELAY_METRICS.value,
                start_time=start_time,
                end_time=end_time,
                filters={
                    **filters,
                    "origin_airport": airport_code
                },
                fields=["delay_minutes"]
            )
            
            # Also query destination delays
            dest_data = self.influxdb_manager.query_flight_metrics(
                measurement=MeasurementType.DELAY_METRICS.value,
                start_time=start_time,
                end_time=end_time,
                filters={
                    **filters,
                    "destination_airport": airport_code
                },
                fields=["delay_minutes"]
            )
            
            # Combine data
            all_data = data + dest_data
            
            # Calculate statistics
            if all_data:
                df = pd.DataFrame(all_data)
                stats = self._calculate_delay_stats(df)
            else:
                stats = []
            
            execution_time = (datetime.now() - start_query_time).total_seconds()
            
            return QueryResult(
                data=stats,
                query=f"delay_statistics_{airport_code}",
                execution_time=execution_time,
                record_count=len(all_data),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting delay statistics: {e}")
            return QueryResult([], "", 0.0, 0, datetime.now())
    
    def _calculate_delay_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate delay statistics from DataFrame."""
        try:
            stats = []
            
            # Overall statistics
            overall_stats = {
                "category": "overall",
                "count": len(df),
                "mean_delay": df["value"].mean(),
                "median_delay": df["value"].median(),
                "std_delay": df["value"].std(),
                "min_delay": df["value"].min(),
                "max_delay": df["value"].max(),
                "on_time_percentage": (df["value"] <= 15).sum() / len(df) * 100,
                "delayed_percentage": (df["value"] > 15).sum() / len(df) * 100
            }
            stats.append(overall_stats)
            
            # Statistics by delay category if available
            if "delay_category" in df.columns:
                for category in df["delay_category"].unique():
                    if pd.isna(category):
                        continue
                    
                    cat_df = df[df["delay_category"] == category]
                    cat_stats = {
                        "category": category,
                        "count": len(cat_df),
                        "mean_delay": cat_df["value"].mean(),
                        "median_delay": cat_df["value"].median(),
                        "percentage_of_total": len(cat_df) / len(df) * 100
                    }
                    stats.append(cat_stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating delay stats: {e}")
            return []
    
    async def get_congestion_trends(
        self,
        airport_code: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: AggregationConfig = AggregationConfig()
    ) -> QueryResult:
        """
        Get congestion trends for an airport.
        
        Args:
            airport_code: Airport code
            start_time: Start time
            end_time: End time
            aggregation: Aggregation configuration
            
        Returns:
            QueryResult containing congestion trends.
        """
        start_query_time = datetime.now()
        
        try:
            data = self.influxdb_manager.query_flight_metrics(
                measurement=MeasurementType.CONGESTION_METRICS.value,
                start_time=start_time,
                end_time=end_time,
                filters={"airport_code": airport_code},
                fields=["congestion_score", "flight_count", "capacity_utilization"],
                aggregate_function=aggregation.function,
                window=aggregation.window,
                group_by=aggregation.group_by or ["runway"]
            )
            
            execution_time = (datetime.now() - start_query_time).total_seconds()
            
            return QueryResult(
                data=data,
                query=f"congestion_trends_{airport_code}",
                execution_time=execution_time,
                record_count=len(data),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting congestion trends: {e}")
            return QueryResult([], "", 0.0, 0, datetime.now())
    
    async def get_peak_hours_analysis(
        self,
        airport_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> QueryResult:
        """
        Analyze peak hours for an airport.
        
        Args:
            airport_code: Airport code
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            QueryResult containing peak hours analysis.
        """
        start_query_time = datetime.now()
        
        try:
            # Query hourly flight counts
            data = self.influxdb_manager.query_flight_metrics(
                measurement=MeasurementType.CONGESTION_METRICS.value,
                start_time=start_date,
                end_time=end_date,
                filters={"airport_code": airport_code},
                fields=["flight_count", "congestion_score"],
                aggregate_function="mean",
                window="1h"
            )
            
            # Analyze peak patterns
            peak_analysis = self._analyze_peak_patterns(data)
            
            execution_time = (datetime.now() - start_query_time).total_seconds()
            
            return QueryResult(
                data=peak_analysis,
                query=f"peak_hours_{airport_code}",
                execution_time=execution_time,
                record_count=len(data),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing peak hours: {e}")
            return QueryResult([], "", 0.0, 0, datetime.now())
    
    def _analyze_peak_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze peak hour patterns from congestion data."""
        try:
            if not data:
                return []
            
            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"])
            df["hour"] = df["time"].dt.hour
            
            # Group by hour and calculate statistics
            hourly_stats = df.groupby("hour").agg({
                "value": ["mean", "std", "count"]
            }).round(2)
            
            # Flatten column names
            hourly_stats.columns = ["_".join(col).strip() for col in hourly_stats.columns]
            hourly_stats = hourly_stats.reset_index()
            
            # Identify peak hours (top 25% by flight count)
            threshold = hourly_stats["value_mean"].quantile(0.75)
            hourly_stats["is_peak"] = hourly_stats["value_mean"] >= threshold
            
            # Convert to list of dictionaries
            peak_analysis = []
            for _, row in hourly_stats.iterrows():
                peak_analysis.append({
                    "hour": int(row["hour"]),
                    "average_flights": float(row["value_mean"]),
                    "std_deviation": float(row["value_std"]),
                    "data_points": int(row["value_count"]),
                    "is_peak_hour": bool(row["is_peak"]),
                    "peak_category": "peak" if row["is_peak"] else "normal"
                })
            
            return peak_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing peak patterns: {e}")
            return []
    
    async def get_runway_efficiency_metrics(
        self,
        airport_code: str,
        start_time: datetime,
        end_time: datetime
    ) -> QueryResult:
        """
        Get runway efficiency metrics.
        
        Args:
            airport_code: Airport code
            start_time: Start time
            end_time: End time
            
        Returns:
            QueryResult containing runway efficiency metrics.
        """
        start_query_time = datetime.now()
        
        try:
            data = self.influxdb_manager.query_flight_metrics(
                measurement=MeasurementType.RUNWAY_METRICS.value,
                start_time=start_time,
                end_time=end_time,
                filters={"airport_code": airport_code},
                fields=["efficiency_score", "operations_count", "average_turnaround_time"],
                group_by=["runway"]
            )
            
            execution_time = (datetime.now() - start_query_time).total_seconds()
            
            return QueryResult(
                data=data,
                query=f"runway_efficiency_{airport_code}",
                execution_time=execution_time,
                record_count=len(data),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting runway efficiency: {e}")
            return QueryResult([], "", 0.0, 0, datetime.now())
    
    async def get_weather_impact_analysis(
        self,
        airport_code: str,
        start_time: datetime,
        end_time: datetime
    ) -> QueryResult:
        """
        Analyze weather impact on flight operations.
        
        Args:
            airport_code: Airport code
            start_time: Start time
            end_time: End time
            
        Returns:
            QueryResult containing weather impact analysis.
        """
        start_query_time = datetime.now()
        
        try:
            # Get weather metrics
            weather_data = self.influxdb_manager.query_flight_metrics(
                measurement=MeasurementType.WEATHER_METRICS.value,
                start_time=start_time,
                end_time=end_time,
                filters={"airport_code": airport_code},
                fields=["impact_score", "visibility_km", "wind_speed_kmh", "precipitation_mm"],
                group_by=["weather_condition"]
            )
            
            # Get corresponding delay data
            delay_data = self.influxdb_manager.query_flight_metrics(
                measurement=MeasurementType.DELAY_METRICS.value,
                start_time=start_time,
                end_time=end_time,
                filters={
                    "origin_airport": airport_code,
                    "delay_category": "weather"
                },
                fields=["delay_minutes"]
            )
            
            # Combine and analyze
            impact_analysis = self._analyze_weather_impact(weather_data, delay_data)
            
            execution_time = (datetime.now() - start_query_time).total_seconds()
            
            return QueryResult(
                data=impact_analysis,
                query=f"weather_impact_{airport_code}",
                execution_time=execution_time,
                record_count=len(weather_data) + len(delay_data),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing weather impact: {e}")
            return QueryResult([], "", 0.0, 0, datetime.now())
    
    def _analyze_weather_impact(
        self,
        weather_data: List[Dict[str, Any]],
        delay_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze correlation between weather and delays."""
        try:
            analysis = []
            
            if weather_data:
                weather_df = pd.DataFrame(weather_data)
                weather_summary = {
                    "metric_type": "weather_conditions",
                    "average_impact_score": weather_df["value"].mean(),
                    "high_impact_periods": (weather_df["value"] > 50).sum(),
                    "total_weather_events": len(weather_df)
                }
                analysis.append(weather_summary)
            
            if delay_data:
                delay_df = pd.DataFrame(delay_data)
                delay_summary = {
                    "metric_type": "weather_delays",
                    "average_weather_delay": delay_df["value"].mean(),
                    "total_weather_delays": len(delay_df),
                    "severe_weather_delays": (delay_df["value"] > 60).sum()
                }
                analysis.append(delay_summary)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in weather impact analysis: {e}")
            return []
    
    async def get_time_series_comparison(
        self,
        airport_codes: List[str],
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: AggregationConfig = AggregationConfig()
    ) -> QueryResult:
        """
        Compare time series metrics across multiple airports.
        
        Args:
            airport_codes: List of airport codes to compare
            metric_name: Name of metric to compare
            start_time: Start time
            end_time: End time
            aggregation: Aggregation configuration
            
        Returns:
            QueryResult containing comparison data.
        """
        start_query_time = datetime.now()
        
        try:
            all_data = []
            
            for airport_code in airport_codes:
                data = self.influxdb_manager.query_flight_metrics(
                    measurement=MeasurementType.CONGESTION_METRICS.value,
                    start_time=start_time,
                    end_time=end_time,
                    filters={"airport_code": airport_code},
                    fields=[metric_name],
                    aggregate_function=aggregation.function,
                    window=aggregation.window
                )
                
                # Add airport identifier to each record
                for record in data:
                    record["airport_code"] = airport_code
                
                all_data.extend(data)
            
            execution_time = (datetime.now() - start_query_time).total_seconds()
            
            return QueryResult(
                data=all_data,
                query=f"comparison_{metric_name}",
                execution_time=execution_time,
                record_count=len(all_data),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in time series comparison: {e}")
            return QueryResult([], "", 0.0, 0, datetime.now())
    
    async def get_performance_metrics_summary(
        self,
        component: str,
        start_time: datetime,
        end_time: datetime
    ) -> QueryResult:
        """
        Get performance metrics summary for a system component.
        
        Args:
            component: Component name
            start_time: Start time
            end_time: End time
            
        Returns:
            QueryResult containing performance summary.
        """
        start_query_time = datetime.now()
        
        try:
            data = self.influxdb_manager.query_flight_metrics(
                measurement=MeasurementType.PERFORMANCE_METRICS.value,
                start_time=start_time,
                end_time=end_time,
                filters={"component": component},
                fields=["value"],
                group_by=["metric_name"]
            )
            
            execution_time = (datetime.now() - start_query_time).total_seconds()
            
            return QueryResult(
                data=data,
                query=f"performance_{component}",
                execution_time=execution_time,
                record_count=len(data),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return QueryResult([], "", 0.0, 0, datetime.now())
    
    def create_custom_query(
        self,
        measurement: str,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, str]] = None,
        fields: Optional[List[str]] = None,
        group_by: Optional[List[str]] = None,
        aggregate_function: Optional[str] = None,
        window: Optional[str] = None
    ) -> str:
        """
        Create custom Flux query for advanced use cases.
        
        Args:
            measurement: Measurement name
            start_time: Start time
            end_time: End time
            filters: Tag filters
            fields: Fields to select
            group_by: Tags to group by
            aggregate_function: Aggregation function
            window: Time window
            
        Returns:
            Flux query string.
        """
        return self.influxdb_manager._build_flux_query(
            measurement=measurement,
            start_time=start_time,
            end_time=end_time,
            filters=filters,
            fields=fields,
            group_by=group_by,
            aggregate_function=aggregate_function,
            window=window
        )


class TimeSeriesAggregator:
    """
    Utility class for advanced time series aggregations and calculations.
    """
    
    @staticmethod
    def calculate_moving_average(
        data: List[Dict[str, Any]],
        window_size: int = 5,
        value_field: str = "value"
    ) -> List[Dict[str, Any]]:
        """
        Calculate moving average for time series data.
        
        Args:
            data: Time series data
            window_size: Size of moving window
            value_field: Field name containing values
            
        Returns:
            Data with moving average added.
        """
        try:
            df = pd.DataFrame(data)
            if value_field in df.columns:
                df["moving_average"] = df[value_field].rolling(window=window_size).mean()
                return df.to_dict("records")
            return data
        except Exception as e:
            logger.error(f"Error calculating moving average: {e}")
            return data
    
    @staticmethod
    def detect_anomalies(
        data: List[Dict[str, Any]],
        value_field: str = "value",
        threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in time series data using standard deviation.
        
        Args:
            data: Time series data
            value_field: Field name containing values
            threshold_std: Standard deviation threshold for anomaly detection
            
        Returns:
            Data with anomaly flags added.
        """
        try:
            df = pd.DataFrame(data)
            if value_field in df.columns:
                mean_val = df[value_field].mean()
                std_val = df[value_field].std()
                
                df["is_anomaly"] = (
                    abs(df[value_field] - mean_val) > threshold_std * std_val
                )
                df["anomaly_score"] = abs(df[value_field] - mean_val) / std_val
                
                return df.to_dict("records")
            return data
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return data
    
    @staticmethod
    def calculate_trend(
        data: List[Dict[str, Any]],
        value_field: str = "value",
        time_field: str = "time"
    ) -> Dict[str, float]:
        """
        Calculate trend statistics for time series data.
        
        Args:
            data: Time series data
            value_field: Field name containing values
            time_field: Field name containing timestamps
            
        Returns:
            Dictionary containing trend statistics.
        """
        try:
            df = pd.DataFrame(data)
            if value_field in df.columns and time_field in df.columns:
                df[time_field] = pd.to_datetime(df[time_field])
                df["time_numeric"] = df[time_field].astype(int) / 10**9  # Convert to seconds
                
                # Calculate linear regression
                correlation = df["time_numeric"].corr(df[value_field])
                slope = df[value_field].diff().mean()
                
                return {
                    "correlation": correlation,
                    "slope": slope,
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                    "trend_strength": abs(correlation)
                }
            
            return {"error": "Required fields not found"}
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return {"error": str(e)}


# Global query utils instance
_query_utils: Optional[TimeSeriesQueryUtils] = None


def get_query_utils() -> TimeSeriesQueryUtils:
    """
    Get global query utils instance.
    
    Returns:
        TimeSeriesQueryUtils instance.
    """
    global _query_utils
    
    if _query_utils is None:
        _query_utils = TimeSeriesQueryUtils()
    
    return _query_utils