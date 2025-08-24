"""
Data ingestion pipeline for flight time series metrics.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

from .influxdb_client import InfluxDBManager, get_influxdb_manager
from .time_series_models import (
    FlightDelayMetric, CongestionMetric, RunwayMetric, WeatherMetric,
    PerformanceMetric, TimeSeriesDataBuilder, DelayCategory, MeasurementType
)

logger = logging.getLogger(__name__)


class TimeSeriesIngestionPipeline:
    """
    Pipeline for ingesting flight data into InfluxDB time series database.
    
    Handles batch processing, real-time ingestion, and data transformation
    from various sources into time series metrics.
    """
    
    def __init__(self, influxdb_manager: Optional[InfluxDBManager] = None):
        """
        Initialize ingestion pipeline.
        
        Args:
            influxdb_manager: InfluxDB manager instance. If None, uses global instance.
        """
        self.influxdb_manager = influxdb_manager or get_influxdb_manager()
        self.batch_size = 1000
        self.max_workers = 4
        self.retry_attempts = 3
        self.retry_delay = 1.0
        
        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "successful_writes": 0,
            "failed_writes": 0,
            "last_ingestion": None,
            "processing_time": 0.0
        }
    
    async def ingest_flight_data_batch(
        self,
        flight_data: List[Dict[str, Any]],
        calculate_delays: bool = True,
        include_congestion: bool = True
    ) -> bool:
        """
        Ingest batch of flight data into time series database.
        
        Args:
            flight_data: List of flight data dictionaries
            calculate_delays: Whether to calculate and store delay metrics
            include_congestion: Whether to calculate and store congestion metrics
            
        Returns:
            bool: True if ingestion successful, False otherwise.
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting batch ingestion of {len(flight_data)} flight records")
            
            # Process data in batches
            all_metrics = []
            
            for i in range(0, len(flight_data), self.batch_size):
                batch = flight_data[i:i + self.batch_size]
                
                # Process batch
                batch_metrics = await self._process_flight_batch(
                    batch, calculate_delays, include_congestion
                )
                all_metrics.extend(batch_metrics)
                
                logger.debug(f"Processed batch {i//self.batch_size + 1}, generated {len(batch_metrics)} metrics")
            
            # Write all metrics to InfluxDB
            success = await self._write_metrics_with_retry(all_metrics)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(len(flight_data), success, processing_time)
            
            if success:
                logger.info(f"Successfully ingested {len(all_metrics)} metrics in {processing_time:.2f}s")
            else:
                logger.error(f"Failed to ingest batch after {self.retry_attempts} attempts")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in batch ingestion: {e}")
            return False
    
    async def _process_flight_batch(
        self,
        flight_batch: List[Dict[str, Any]],
        calculate_delays: bool,
        include_congestion: bool
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of flight data into time series metrics.
        
        Args:
            flight_batch: Batch of flight data
            calculate_delays: Whether to calculate delay metrics
            include_congestion: Whether to calculate congestion metrics
            
        Returns:
            List of metric dictionaries ready for InfluxDB.
        """
        metrics = []
        
        # Use thread pool for CPU-intensive processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Process delay metrics
            if calculate_delays:
                delay_futures = [
                    executor.submit(self._create_delay_metric, flight)
                    for flight in flight_batch
                    if self._has_delay_data(flight)
                ]
                
                for future in delay_futures:
                    try:
                        delay_metric = future.result()
                        if delay_metric:
                            metrics.append(delay_metric.to_influx_dict())
                    except Exception as e:
                        logger.warning(f"Failed to create delay metric: {e}")
            
            # Process congestion metrics if requested
            if include_congestion:
                congestion_metrics = await self._calculate_congestion_metrics(flight_batch)
                metrics.extend([m.to_influx_dict() for m in congestion_metrics])
        
        return metrics
    
    def _create_delay_metric(self, flight_data: Dict[str, Any]) -> Optional[FlightDelayMetric]:
        """
        Create delay metric from flight data.
        
        Args:
            flight_data: Flight data dictionary
            
        Returns:
            FlightDelayMetric instance or None if data insufficient.
        """
        try:
            # Calculate delay
            delay_minutes = self._calculate_delay(flight_data)
            if delay_minutes is None:
                return None
            
            # Determine delay category
            delay_category = self._categorize_delay(flight_data, delay_minutes)
            
            # Create metric
            return TimeSeriesDataBuilder.create_flight_delay_metric(
                flight_data=flight_data,
                delay_minutes=delay_minutes,
                delay_category=delay_category.value
            )
            
        except Exception as e:
            logger.warning(f"Failed to create delay metric for flight {flight_data.get('flight_id', 'unknown')}: {e}")
            return None
    
    def _calculate_delay(self, flight_data: Dict[str, Any]) -> Optional[float]:
        """
        Calculate delay in minutes from flight data.
        
        Args:
            flight_data: Flight data dictionary
            
        Returns:
            Delay in minutes (negative for early) or None if cannot calculate.
        """
        try:
            # Try departure delay first
            if "actual_departure" in flight_data and "scheduled_departure" in flight_data:
                actual = pd.to_datetime(flight_data["actual_departure"])
                scheduled = pd.to_datetime(flight_data["scheduled_departure"])
                return (actual - scheduled).total_seconds() / 60.0
            
            # Try arrival delay
            if "actual_arrival" in flight_data and "scheduled_arrival" in flight_data:
                actual = pd.to_datetime(flight_data["actual_arrival"])
                scheduled = pd.to_datetime(flight_data["scheduled_arrival"])
                return (actual - scheduled).total_seconds() / 60.0
            
            # Check if delay is already calculated
            if "delay_minutes" in flight_data:
                return float(flight_data["delay_minutes"])
            
            return None
            
        except Exception as e:
            logger.warning(f"Error calculating delay: {e}")
            return None
    
    def _categorize_delay(self, flight_data: Dict[str, Any], delay_minutes: float) -> DelayCategory:
        """
        Categorize delay based on flight data and delay amount.
        
        Args:
            flight_data: Flight data dictionary
            delay_minutes: Delay in minutes
            
        Returns:
            DelayCategory enum value.
        """
        # Check if category is already specified
        if "delay_category" in flight_data:
            try:
                return DelayCategory(flight_data["delay_category"].lower())
            except ValueError:
                pass
        
        # Categorize based on available data and delay amount
        if delay_minutes <= 0:
            return DelayCategory.OTHER  # Early or on-time
        
        # Check weather conditions
        weather_condition = flight_data.get("weather_condition", "").lower()
        if any(condition in weather_condition for condition in ["rain", "storm", "fog", "wind"]):
            return DelayCategory.WEATHER
        
        # Check if it's a long delay (likely operational)
        if delay_minutes > 60:
            return DelayCategory.OPERATIONAL
        
        # Check if it's during peak hours (likely traffic)
        if "scheduled_departure" in flight_data:
            try:
                scheduled = pd.to_datetime(flight_data["scheduled_departure"])
                hour = scheduled.hour
                if 6 <= hour <= 9 or 17 <= hour <= 20:  # Peak hours
                    return DelayCategory.TRAFFIC
            except:
                pass
        
        # Default to operational for moderate delays
        return DelayCategory.OPERATIONAL
    
    def _has_delay_data(self, flight_data: Dict[str, Any]) -> bool:
        """
        Check if flight data has sufficient information for delay calculation.
        
        Args:
            flight_data: Flight data dictionary
            
        Returns:
            bool: True if delay can be calculated, False otherwise.
        """
        # Check for explicit delay field
        if "delay_minutes" in flight_data:
            return True
        
        # Check for departure times
        if "actual_departure" in flight_data and "scheduled_departure" in flight_data:
            return True
        
        # Check for arrival times
        if "actual_arrival" in flight_data and "scheduled_arrival" in flight_data:
            return True
        
        return False
    
    async def _calculate_congestion_metrics(
        self,
        flight_batch: List[Dict[str, Any]]
    ) -> List[CongestionMetric]:
        """
        Calculate congestion metrics from flight batch.
        
        Args:
            flight_batch: Batch of flight data
            
        Returns:
            List of CongestionMetric instances.
        """
        metrics = []
        
        try:
            # Group flights by airport and hour
            df = pd.DataFrame(flight_batch)
            
            if df.empty:
                return metrics
            
            # Process departure congestion
            if "scheduled_departure" in df.columns and "origin_airport" in df.columns:
                departure_metrics = self._calculate_airport_congestion(
                    df, "scheduled_departure", "origin_airport", "departure"
                )
                metrics.extend(departure_metrics)
            
            # Process arrival congestion
            if "scheduled_arrival" in df.columns and "destination_airport" in df.columns:
                arrival_metrics = self._calculate_airport_congestion(
                    df, "scheduled_arrival", "destination_airport", "arrival"
                )
                metrics.extend(arrival_metrics)
            
        except Exception as e:
            logger.warning(f"Error calculating congestion metrics: {e}")
        
        return metrics
    
    def _calculate_airport_congestion(
        self,
        df: pd.DataFrame,
        time_column: str,
        airport_column: str,
        operation_type: str
    ) -> List[CongestionMetric]:
        """
        Calculate congestion metrics for specific airport and operation type.
        
        Args:
            df: DataFrame with flight data
            time_column: Column name for time data
            airport_column: Column name for airport data
            operation_type: Type of operation (departure/arrival)
            
        Returns:
            List of CongestionMetric instances.
        """
        metrics = []
        
        try:
            # Convert time column to datetime
            df[time_column] = pd.to_datetime(df[time_column])
            
            # Group by airport and hour
            df["hour"] = df[time_column].dt.floor("H")
            grouped = df.groupby([airport_column, "hour"])
            
            for (airport, hour), group in grouped:
                flight_count = len(group)
                
                # Calculate average delay if available
                avg_delay = 0.0
                if "delay_minutes" in group.columns:
                    avg_delay = group["delay_minutes"].mean()
                
                # Simple congestion score based on flight count
                # This is a basic implementation - could be enhanced with runway capacity data
                congestion_score = min(flight_count * 10, 100)  # Scale to 0-100
                
                # Capacity utilization (assuming 10 flights per hour as 100% capacity)
                capacity_utilization = min(flight_count / 10 * 100, 100)
                
                # Determine if it's peak hour (6-9 AM or 5-8 PM)
                is_peak = hour.hour in range(6, 10) or hour.hour in range(17, 21)
                
                metric = CongestionMetric(
                    airport_code=airport,
                    runway=f"{operation_type}_all",
                    flight_count=flight_count,
                    congestion_score=congestion_score,
                    capacity_utilization=capacity_utilization,
                    average_delay=avg_delay,
                    timestamp=hour,
                    peak_hour=is_peak
                )
                
                metrics.append(metric)
        
        except Exception as e:
            logger.warning(f"Error in airport congestion calculation: {e}")
        
        return metrics
    
    async def _write_metrics_with_retry(self, metrics: List[Dict[str, Any]]) -> bool:
        """
        Write metrics to InfluxDB with retry logic.
        
        Args:
            metrics: List of metric dictionaries
            
        Returns:
            bool: True if write successful, False otherwise.
        """
        for attempt in range(self.retry_attempts):
            try:
                success = self.influxdb_manager.write_flight_metrics(metrics)
                if success:
                    return True
                
                logger.warning(f"Write attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Write attempt {attempt + 1} error: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        return False
    
    def _update_stats(self, processed_count: int, success: bool, processing_time: float) -> None:
        """
        Update ingestion statistics.
        
        Args:
            processed_count: Number of records processed
            success: Whether ingestion was successful
            processing_time: Time taken for processing
        """
        self.stats["total_processed"] += processed_count
        self.stats["last_ingestion"] = datetime.now()
        self.stats["processing_time"] = processing_time
        
        if success:
            self.stats["successful_writes"] += 1
        else:
            self.stats["failed_writes"] += 1
    
    async def ingest_real_time_flight(self, flight_data: Dict[str, Any]) -> bool:
        """
        Ingest single flight record in real-time.
        
        Args:
            flight_data: Flight data dictionary
            
        Returns:
            bool: True if ingestion successful, False otherwise.
        """
        try:
            metrics = await self._process_flight_batch([flight_data], True, False)
            
            if metrics:
                return await self._write_metrics_with_retry(metrics)
            
            return True  # No metrics to write, but not an error
            
        except Exception as e:
            logger.error(f"Error in real-time ingestion: {e}")
            return False
    
    async def ingest_weather_data(self, weather_data: List[Dict[str, Any]]) -> bool:
        """
        Ingest weather data into time series database.
        
        Args:
            weather_data: List of weather data dictionaries
            
        Returns:
            bool: True if ingestion successful, False otherwise.
        """
        try:
            metrics = []
            
            for weather in weather_data:
                metric = WeatherMetric(
                    airport_code=weather.get("airport_code", ""),
                    weather_condition=weather.get("condition", "unknown"),
                    visibility_km=weather.get("visibility_km", 0.0),
                    wind_speed_kmh=weather.get("wind_speed_kmh", 0.0),
                    precipitation_mm=weather.get("precipitation_mm", 0.0),
                    temperature_celsius=weather.get("temperature_celsius", 0.0),
                    impact_score=weather.get("impact_score", 0.0),
                    timestamp=pd.to_datetime(weather.get("timestamp", datetime.now())),
                    wind_direction=weather.get("wind_direction")
                )
                metrics.append(metric.to_influx_dict())
            
            return await self._write_metrics_with_retry(metrics)
            
        except Exception as e:
            logger.error(f"Error ingesting weather data: {e}")
            return False
    
    async def ingest_performance_metrics(
        self,
        component: str,
        metrics_data: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Ingest system performance metrics.
        
        Args:
            component: Component name
            metrics_data: Dictionary of metric names and values
            timestamp: Timestamp for metrics (defaults to now)
            
        Returns:
            bool: True if ingestion successful, False otherwise.
        """
        try:
            timestamp = timestamp or datetime.now()
            
            performance_metrics = TimeSeriesDataBuilder.create_performance_metrics_batch(
                component=component,
                metrics_data=metrics_data,
                timestamp=timestamp
            )
            
            influx_metrics = [m.to_influx_dict() for m in performance_metrics]
            return await self._write_metrics_with_retry(influx_metrics)
            
        except Exception as e:
            logger.error(f"Error ingesting performance metrics: {e}")
            return False
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion pipeline statistics.
        
        Returns:
            Dictionary containing ingestion statistics.
        """
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset ingestion statistics."""
        self.stats = {
            "total_processed": 0,
            "successful_writes": 0,
            "failed_writes": 0,
            "last_ingestion": None,
            "processing_time": 0.0
        }


# Global ingestion pipeline instance
_ingestion_pipeline: Optional[TimeSeriesIngestionPipeline] = None


def get_ingestion_pipeline() -> TimeSeriesIngestionPipeline:
    """
    Get global ingestion pipeline instance.
    
    Returns:
        TimeSeriesIngestionPipeline instance.
    """
    global _ingestion_pipeline
    
    if _ingestion_pipeline is None:
        _ingestion_pipeline = TimeSeriesIngestionPipeline()
    
    return _ingestion_pipeline


async def ingest_flight_data_async(
    flight_data: List[Dict[str, Any]],
    calculate_delays: bool = True,
    include_congestion: bool = True
) -> bool:
    """
    Convenience function for ingesting flight data asynchronously.
    
    Args:
        flight_data: List of flight data dictionaries
        calculate_delays: Whether to calculate delay metrics
        include_congestion: Whether to calculate congestion metrics
        
    Returns:
        bool: True if ingestion successful, False otherwise.
    """
    pipeline = get_ingestion_pipeline()
    return await pipeline.ingest_flight_data_batch(
        flight_data=flight_data,
        calculate_delays=calculate_delays,
        include_congestion=include_congestion
    )