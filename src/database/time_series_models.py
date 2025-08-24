"""
Time series data models for flight metrics in InfluxDB.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MeasurementType(Enum):
    """Enumeration of measurement types for flight metrics."""
    FLIGHT_METRICS = "flight_metrics"
    DELAY_METRICS = "delay_metrics"
    CONGESTION_METRICS = "congestion_metrics"
    RUNWAY_METRICS = "runway_metrics"
    WEATHER_METRICS = "weather_metrics"
    PERFORMANCE_METRICS = "performance_metrics"


class DelayCategory(Enum):
    """Enumeration of delay categories."""
    WEATHER = "weather"
    OPERATIONAL = "operational"
    TRAFFIC = "traffic"
    MECHANICAL = "mechanical"
    CREW = "crew"
    OTHER = "other"


@dataclass
class FlightMetric:
    """
    Base class for flight time series metrics.
    
    Represents a single data point in the time series with tags and fields
    that will be stored in InfluxDB.
    """
    measurement: str
    timestamp: datetime
    tags: Dict[str, str]
    fields: Dict[str, Union[int, float, str]]
    
    def to_influx_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format expected by InfluxDB client.
        
        Returns:
            Dictionary with measurement, tags, fields, and time.
        """
        return {
            "measurement": self.measurement,
            "tags": self.tags,
            "fields": self.fields,
            "time": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlightMetric':
        """
        Create FlightMetric from dictionary.
        
        Args:
            data: Dictionary containing metric data.
            
        Returns:
            FlightMetric instance.
        """
        return cls(
            measurement=data["measurement"],
            timestamp=data["timestamp"],
            tags=data.get("tags", {}),
            fields=data.get("fields", {})
        )


@dataclass
class FlightDelayMetric(FlightMetric):
    """Time series metric for flight delay data."""
    
    def __init__(
        self,
        flight_id: str,
        airline: str,
        aircraft_type: str,
        origin_airport: str,
        destination_airport: str,
        runway: str,
        delay_minutes: float,
        delay_category: DelayCategory,
        scheduled_time: datetime,
        actual_time: datetime,
        passenger_count: Optional[int] = None,
        weather_condition: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize flight delay metric.
        
        Args:
            flight_id: Unique flight identifier
            airline: Airline code
            aircraft_type: Aircraft type code
            origin_airport: Origin airport code
            destination_airport: Destination airport code
            runway: Runway identifier
            delay_minutes: Delay in minutes (negative for early)
            delay_category: Category of delay
            scheduled_time: Scheduled departure/arrival time
            actual_time: Actual departure/arrival time
            passenger_count: Number of passengers
            weather_condition: Weather condition description
            timestamp: Metric timestamp (defaults to actual_time)
        """
        tags = {
            "flight_id": flight_id,
            "airline": airline,
            "aircraft_type": aircraft_type,
            "origin_airport": origin_airport,
            "destination_airport": destination_airport,
            "runway": runway,
            "delay_category": delay_category.value
        }
        
        if weather_condition:
            tags["weather_condition"] = weather_condition
        
        fields = {
            "delay_minutes": float(delay_minutes),
            "scheduled_timestamp": int(scheduled_time.timestamp()),
            "actual_timestamp": int(actual_time.timestamp())
        }
        
        if passenger_count is not None:
            fields["passenger_count"] = int(passenger_count)
        
        super().__init__(
            measurement=MeasurementType.DELAY_METRICS.value,
            timestamp=timestamp or actual_time,
            tags=tags,
            fields=fields
        )


@dataclass
class CongestionMetric(FlightMetric):
    """Time series metric for airport congestion data."""
    
    def __init__(
        self,
        airport_code: str,
        runway: str,
        flight_count: int,
        congestion_score: float,
        capacity_utilization: float,
        average_delay: float,
        timestamp: datetime,
        weather_impact: Optional[float] = None,
        peak_hour: bool = False
    ):
        """
        Initialize congestion metric.
        
        Args:
            airport_code: Airport code
            runway: Runway identifier
            flight_count: Number of flights in time window
            congestion_score: Calculated congestion score (0-100)
            capacity_utilization: Runway capacity utilization percentage
            average_delay: Average delay in minutes for time window
            timestamp: Metric timestamp
            weather_impact: Weather impact factor (0-1)
            peak_hour: Whether this is a peak hour
        """
        tags = {
            "airport_code": airport_code,
            "runway": runway,
            "peak_hour": str(peak_hour).lower()
        }
        
        fields = {
            "flight_count": int(flight_count),
            "congestion_score": float(congestion_score),
            "capacity_utilization": float(capacity_utilization),
            "average_delay": float(average_delay)
        }
        
        if weather_impact is not None:
            fields["weather_impact"] = float(weather_impact)
        
        super().__init__(
            measurement=MeasurementType.CONGESTION_METRICS.value,
            timestamp=timestamp,
            tags=tags,
            fields=fields
        )


@dataclass
class RunwayMetric(FlightMetric):
    """Time series metric for runway performance data."""
    
    def __init__(
        self,
        airport_code: str,
        runway: str,
        operations_count: int,
        average_turnaround_time: float,
        efficiency_score: float,
        timestamp: datetime,
        runway_condition: Optional[str] = None,
        maintenance_status: Optional[str] = None
    ):
        """
        Initialize runway metric.
        
        Args:
            airport_code: Airport code
            runway: Runway identifier
            operations_count: Number of operations (takeoffs + landings)
            average_turnaround_time: Average time between operations
            efficiency_score: Runway efficiency score (0-100)
            timestamp: Metric timestamp
            runway_condition: Runway condition description
            maintenance_status: Maintenance status
        """
        tags = {
            "airport_code": airport_code,
            "runway": runway
        }
        
        if runway_condition:
            tags["runway_condition"] = runway_condition
        if maintenance_status:
            tags["maintenance_status"] = maintenance_status
        
        fields = {
            "operations_count": int(operations_count),
            "average_turnaround_time": float(average_turnaround_time),
            "efficiency_score": float(efficiency_score)
        }
        
        super().__init__(
            measurement=MeasurementType.RUNWAY_METRICS.value,
            timestamp=timestamp,
            tags=tags,
            fields=fields
        )


@dataclass
class WeatherMetric(FlightMetric):
    """Time series metric for weather impact data."""
    
    def __init__(
        self,
        airport_code: str,
        weather_condition: str,
        visibility_km: float,
        wind_speed_kmh: float,
        precipitation_mm: float,
        temperature_celsius: float,
        impact_score: float,
        timestamp: datetime,
        wind_direction: Optional[str] = None
    ):
        """
        Initialize weather metric.
        
        Args:
            airport_code: Airport code
            weather_condition: Weather condition description
            visibility_km: Visibility in kilometers
            wind_speed_kmh: Wind speed in km/h
            precipitation_mm: Precipitation in mm
            temperature_celsius: Temperature in Celsius
            impact_score: Weather impact score on operations (0-100)
            timestamp: Metric timestamp
            wind_direction: Wind direction
        """
        tags = {
            "airport_code": airport_code,
            "weather_condition": weather_condition
        }
        
        if wind_direction:
            tags["wind_direction"] = wind_direction
        
        fields = {
            "visibility_km": float(visibility_km),
            "wind_speed_kmh": float(wind_speed_kmh),
            "precipitation_mm": float(precipitation_mm),
            "temperature_celsius": float(temperature_celsius),
            "impact_score": float(impact_score)
        }
        
        super().__init__(
            measurement=MeasurementType.WEATHER_METRICS.value,
            timestamp=timestamp,
            tags=tags,
            fields=fields
        )


@dataclass
class PerformanceMetric(FlightMetric):
    """Time series metric for system performance data."""
    
    def __init__(
        self,
        component: str,
        metric_name: str,
        value: float,
        timestamp: datetime,
        unit: Optional[str] = None,
        threshold: Optional[float] = None,
        status: str = "normal"
    ):
        """
        Initialize performance metric.
        
        Args:
            component: System component name
            metric_name: Name of the performance metric
            value: Metric value
            timestamp: Metric timestamp
            unit: Unit of measurement
            threshold: Alert threshold value
            status: Status (normal, warning, critical)
        """
        tags = {
            "component": component,
            "metric_name": metric_name,
            "status": status
        }
        
        if unit:
            tags["unit"] = unit
        
        fields = {
            "value": float(value)
        }
        
        if threshold is not None:
            fields["threshold"] = float(threshold)
        
        super().__init__(
            measurement=MeasurementType.PERFORMANCE_METRICS.value,
            timestamp=timestamp,
            tags=tags,
            fields=fields
        )


class TimeSeriesDataBuilder:
    """Builder class for creating time series data points."""
    
    @staticmethod
    def create_flight_delay_metric(
        flight_data: Dict[str, Any],
        delay_minutes: float,
        delay_category: str,
        timestamp: Optional[datetime] = None
    ) -> FlightDelayMetric:
        """
        Create flight delay metric from flight data.
        
        Args:
            flight_data: Dictionary containing flight information
            delay_minutes: Calculated delay in minutes
            delay_category: Category of delay
            timestamp: Optional timestamp override
            
        Returns:
            FlightDelayMetric instance.
        """
        try:
            delay_cat = DelayCategory(delay_category.lower())
        except ValueError:
            delay_cat = DelayCategory.OTHER
        
        return FlightDelayMetric(
            flight_id=flight_data.get("flight_id", ""),
            airline=flight_data.get("airline", ""),
            aircraft_type=flight_data.get("aircraft_type", ""),
            origin_airport=flight_data.get("origin_airport", ""),
            destination_airport=flight_data.get("destination_airport", ""),
            runway=flight_data.get("runway", ""),
            delay_minutes=delay_minutes,
            delay_category=delay_cat,
            scheduled_time=flight_data.get("scheduled_departure") or flight_data.get("scheduled_arrival"),
            actual_time=flight_data.get("actual_departure") or flight_data.get("actual_arrival"),
            passenger_count=flight_data.get("passenger_count"),
            weather_condition=flight_data.get("weather_condition"),
            timestamp=timestamp
        )
    
    @staticmethod
    def create_congestion_metrics_batch(
        airport_code: str,
        hourly_data: List[Dict[str, Any]],
        base_timestamp: datetime
    ) -> List[CongestionMetric]:
        """
        Create batch of congestion metrics for hourly data.
        
        Args:
            airport_code: Airport code
            hourly_data: List of hourly congestion data
            base_timestamp: Base timestamp for metrics
            
        Returns:
            List of CongestionMetric instances.
        """
        metrics = []
        
        for i, hour_data in enumerate(hourly_data):
            timestamp = base_timestamp.replace(hour=i, minute=0, second=0, microsecond=0)
            
            metric = CongestionMetric(
                airport_code=airport_code,
                runway=hour_data.get("runway", "ALL"),
                flight_count=hour_data.get("flight_count", 0),
                congestion_score=hour_data.get("congestion_score", 0.0),
                capacity_utilization=hour_data.get("capacity_utilization", 0.0),
                average_delay=hour_data.get("average_delay", 0.0),
                timestamp=timestamp,
                weather_impact=hour_data.get("weather_impact"),
                peak_hour=hour_data.get("peak_hour", False)
            )
            
            metrics.append(metric)
        
        return metrics
    
    @staticmethod
    def create_performance_metrics_batch(
        component: str,
        metrics_data: Dict[str, float],
        timestamp: datetime
    ) -> List[PerformanceMetric]:
        """
        Create batch of performance metrics for a component.
        
        Args:
            component: Component name
            metrics_data: Dictionary of metric names and values
            timestamp: Timestamp for all metrics
            
        Returns:
            List of PerformanceMetric instances.
        """
        metrics = []
        
        for metric_name, value in metrics_data.items():
            metric = PerformanceMetric(
                component=component,
                metric_name=metric_name,
                value=value,
                timestamp=timestamp
            )
            metrics.append(metric)
        
        return metrics


class TimeSeriesQueryBuilder:
    """Builder class for creating InfluxDB queries."""
    
    @staticmethod
    def build_delay_analysis_query(
        airport_code: str,
        start_time: datetime,
        end_time: datetime,
        airline: Optional[str] = None
    ) -> str:
        """
        Build query for delay analysis.
        
        Args:
            airport_code: Airport code to analyze
            start_time: Start time for analysis
            end_time: End time for analysis
            airline: Optional airline filter
            
        Returns:
            Flux query string.
        """
        query = f'''
        from(bucket: "flight-metrics")
          |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
          |> filter(fn: (r) => r._measurement == "delay_metrics")
          |> filter(fn: (r) => r.origin_airport == "{airport_code}" or r.destination_airport == "{airport_code}")
        '''
        
        if airline:
            query += f'  |> filter(fn: (r) => r.airline == "{airline}")\n'
        
        query += '''
          |> filter(fn: (r) => r._field == "delay_minutes")
          |> group(columns: ["delay_category"])
          |> mean()
        '''
        
        return query.strip()
    
    @staticmethod
    def build_congestion_trend_query(
        airport_code: str,
        start_time: datetime,
        end_time: datetime,
        window: str = "1h"
    ) -> str:
        """
        Build query for congestion trend analysis.
        
        Args:
            airport_code: Airport code
            start_time: Start time
            end_time: End time
            window: Time window for aggregation
            
        Returns:
            Flux query string.
        """
        return f'''
        from(bucket: "flight-metrics")
          |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
          |> filter(fn: (r) => r._measurement == "congestion_metrics")
          |> filter(fn: (r) => r.airport_code == "{airport_code}")
          |> filter(fn: (r) => r._field == "congestion_score")
          |> aggregateWindow(every: {window}, fn: mean)
          |> yield(name: "congestion_trend")
        '''
    
    @staticmethod
    def build_runway_efficiency_query(
        airport_code: str,
        start_time: datetime,
        end_time: datetime
    ) -> str:
        """
        Build query for runway efficiency analysis.
        
        Args:
            airport_code: Airport code
            start_time: Start time
            end_time: End time
            
        Returns:
            Flux query string.
        """
        return f'''
        from(bucket: "flight-metrics")
          |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
          |> filter(fn: (r) => r._measurement == "runway_metrics")
          |> filter(fn: (r) => r.airport_code == "{airport_code}")
          |> filter(fn: (r) => r._field == "efficiency_score")
          |> group(columns: ["runway"])
          |> mean()
          |> yield(name: "runway_efficiency")
        '''