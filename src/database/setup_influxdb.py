"""
Setup script for InfluxDB initialization and retention policies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from .influxdb_client import InfluxDBManager, InfluxDBConfig
from .time_series_models import (
    FlightDelayMetric, CongestionMetric, RunwayMetric, WeatherMetric,
    PerformanceMetric, DelayCategory, TimeSeriesDataBuilder
)
from .time_series_ingestion import TimeSeriesIngestionPipeline

logger = logging.getLogger(__name__)


class InfluxDBSetup:
    """
    Setup and initialization class for InfluxDB flight analysis system.
    """
    
    def __init__(self, config: InfluxDBConfig = None):
        """
        Initialize InfluxDB setup.
        
        Args:
            config: InfluxDB configuration. If None, loads from environment.
        """
        self.config = config
        self.manager = InfluxDBManager(config)
        self.ingestion_pipeline = TimeSeriesIngestionPipeline(self.manager)
    
    async def initialize_database(self) -> bool:
        """
        Initialize InfluxDB database with proper configuration.
        
        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            logger.info("Starting InfluxDB initialization...")
            
            # Test connection
            if not self.manager.health_check():
                logger.error("InfluxDB health check failed")
                return False
            
            logger.info("InfluxDB connection verified")
            
            # Setup retention policies
            if not self.manager.setup_retention_policies():
                logger.warning("Failed to setup retention policies")
            else:
                logger.info("Retention policies configured")
            
            # Get bucket information
            bucket_info = self.manager.get_bucket_info()
            if bucket_info:
                logger.info(f"Bucket info: {bucket_info}")
            
            # Create sample data for testing
            await self._create_sample_data()
            
            logger.info("InfluxDB initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing InfluxDB: {e}")
            return False
    
    async def _create_sample_data(self) -> None:
        """Create sample time series data for testing."""
        try:
            logger.info("Creating sample time series data...")
            
            # Create sample flight delay metrics
            sample_delays = self._generate_sample_delay_metrics()
            delay_dicts = [metric.to_influx_dict() for metric in sample_delays]
            
            # Create sample congestion metrics
            sample_congestion = self._generate_sample_congestion_metrics()
            congestion_dicts = [metric.to_influx_dict() for metric in sample_congestion]
            
            # Create sample runway metrics
            sample_runway = self._generate_sample_runway_metrics()
            runway_dicts = [metric.to_influx_dict() for metric in sample_runway]
            
            # Create sample weather metrics
            sample_weather = self._generate_sample_weather_metrics()
            weather_dicts = [metric.to_influx_dict() for metric in sample_weather]
            
            # Create sample performance metrics
            sample_performance = self._generate_sample_performance_metrics()
            performance_dicts = [metric.to_influx_dict() for metric in sample_performance]
            
            # Combine all metrics
            all_metrics = (
                delay_dicts + congestion_dicts + runway_dicts + 
                weather_dicts + performance_dicts
            )
            
            # Write to InfluxDB
            success = self.manager.write_flight_metrics(all_metrics)
            
            if success:
                logger.info(f"Successfully created {len(all_metrics)} sample metrics")
            else:
                logger.warning("Failed to create sample metrics")
                
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
    
    def _generate_sample_delay_metrics(self) -> List[FlightDelayMetric]:
        """Generate sample flight delay metrics."""
        metrics = []
        base_time = datetime.now() - timedelta(days=7)
        
        airports = ["BOM", "DEL"]
        airlines = ["AI", "6E", "SG", "UK"]
        aircraft_types = ["A320", "B737", "A321", "B738"]
        
        for day in range(7):
            for hour in range(24):
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                # Generate 2-5 flights per hour
                for flight_num in range(2, 6):
                    flight_id = f"FL{day:02d}{hour:02d}{flight_num:02d}"
                    
                    # Simulate realistic delay patterns
                    if 6 <= hour <= 9 or 17 <= hour <= 20:  # Peak hours
                        delay_minutes = max(-5, min(120, 
                            15 + (hour - 12) * 2 + (flight_num - 3) * 5))
                    else:
                        delay_minutes = max(-10, min(60, 
                            5 + (flight_num - 3) * 3))
                    
                    # Determine delay category
                    if delay_minutes <= 0:
                        category = DelayCategory.OTHER
                    elif delay_minutes > 60:
                        category = DelayCategory.OPERATIONAL
                    elif 6 <= hour <= 9 or 17 <= hour <= 20:
                        category = DelayCategory.TRAFFIC
                    else:
                        category = DelayCategory.OPERATIONAL
                    
                    metric = FlightDelayMetric(
                        flight_id=flight_id,
                        airline=airlines[flight_num % len(airlines)],
                        aircraft_type=aircraft_types[flight_num % len(aircraft_types)],
                        origin_airport=airports[0],
                        destination_airport=airports[1],
                        runway=f"RW{(flight_num % 3) + 1:02d}",
                        delay_minutes=delay_minutes,
                        delay_category=category,
                        scheduled_time=timestamp,
                        actual_time=timestamp + timedelta(minutes=delay_minutes),
                        passenger_count=120 + flight_num * 20,
                        timestamp=timestamp + timedelta(minutes=delay_minutes)
                    )
                    
                    metrics.append(metric)
        
        return metrics
    
    def _generate_sample_congestion_metrics(self) -> List[CongestionMetric]:
        """Generate sample congestion metrics."""
        metrics = []
        base_time = datetime.now() - timedelta(days=7)
        
        airports = ["BOM", "DEL"]
        runways = ["RW01", "RW02", "RW03"]
        
        for day in range(7):
            for hour in range(24):
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                for airport in airports:
                    for runway in runways:
                        # Simulate realistic congestion patterns
                        if 6 <= hour <= 9 or 17 <= hour <= 20:  # Peak hours
                            flight_count = 8 + (hour % 4)
                            congestion_score = 70 + (hour % 3) * 10
                            capacity_utilization = 80 + (hour % 2) * 15
                            average_delay = 20 + (hour % 3) * 5
                            is_peak = True
                        else:
                            flight_count = 3 + (hour % 3)
                            congestion_score = 30 + (hour % 4) * 5
                            capacity_utilization = 40 + (hour % 3) * 10
                            average_delay = 5 + (hour % 2) * 3
                            is_peak = False
                        
                        metric = CongestionMetric(
                            airport_code=airport,
                            runway=runway,
                            flight_count=flight_count,
                            congestion_score=congestion_score,
                            capacity_utilization=capacity_utilization,
                            average_delay=average_delay,
                            timestamp=timestamp,
                            peak_hour=is_peak
                        )
                        
                        metrics.append(metric)
        
        return metrics
    
    def _generate_sample_runway_metrics(self) -> List[RunwayMetric]:
        """Generate sample runway metrics."""
        metrics = []
        base_time = datetime.now() - timedelta(days=7)
        
        airports = ["BOM", "DEL"]
        runways = ["RW01", "RW02", "RW03"]
        
        for day in range(7):
            for hour in range(0, 24, 4):  # Every 4 hours
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                for airport in airports:
                    for runway in runways:
                        operations_count = 15 + (hour // 4) * 3
                        turnaround_time = 8.5 + (hour % 3) * 1.2
                        efficiency_score = 85 - (hour % 4) * 5
                        
                        metric = RunwayMetric(
                            airport_code=airport,
                            runway=runway,
                            operations_count=operations_count,
                            average_turnaround_time=turnaround_time,
                            efficiency_score=efficiency_score,
                            timestamp=timestamp,
                            runway_condition="good",
                            maintenance_status="operational"
                        )
                        
                        metrics.append(metric)
        
        return metrics
    
    def _generate_sample_weather_metrics(self) -> List[WeatherMetric]:
        """Generate sample weather metrics."""
        metrics = []
        base_time = datetime.now() - timedelta(days=7)
        
        airports = ["BOM", "DEL"]
        weather_conditions = ["clear", "cloudy", "rain", "fog"]
        
        for day in range(7):
            for hour in range(0, 24, 2):  # Every 2 hours
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                for airport in airports:
                    condition = weather_conditions[hour % len(weather_conditions)]
                    
                    # Simulate weather impact
                    if condition == "clear":
                        visibility = 10.0
                        wind_speed = 15.0
                        precipitation = 0.0
                        impact_score = 10.0
                    elif condition == "cloudy":
                        visibility = 8.0
                        wind_speed = 20.0
                        precipitation = 0.0
                        impact_score = 25.0
                    elif condition == "rain":
                        visibility = 5.0
                        wind_speed = 25.0
                        precipitation = 5.0
                        impact_score = 60.0
                    else:  # fog
                        visibility = 2.0
                        wind_speed = 10.0
                        precipitation = 0.0
                        impact_score = 80.0
                    
                    metric = WeatherMetric(
                        airport_code=airport,
                        weather_condition=condition,
                        visibility_km=visibility,
                        wind_speed_kmh=wind_speed,
                        precipitation_mm=precipitation,
                        temperature_celsius=25.0 + (hour % 10),
                        impact_score=impact_score,
                        timestamp=timestamp,
                        wind_direction="NW"
                    )
                    
                    metrics.append(metric)
        
        return metrics
    
    def _generate_sample_performance_metrics(self) -> List[PerformanceMetric]:
        """Generate sample performance metrics."""
        metrics = []
        base_time = datetime.now() - timedelta(days=1)
        
        components = ["api", "database", "ingestion_pipeline", "ml_engine"]
        metric_names = {
            "api": ["response_time_ms", "requests_per_second", "error_rate"],
            "database": ["query_time_ms", "connection_count", "cpu_usage"],
            "ingestion_pipeline": ["records_per_second", "processing_time_ms", "queue_size"],
            "ml_engine": ["prediction_time_ms", "model_accuracy", "memory_usage_mb"]
        }
        
        for hour in range(24):
            timestamp = base_time + timedelta(hours=hour)
            
            for component in components:
                for metric_name in metric_names[component]:
                    # Simulate realistic performance values
                    if "time_ms" in metric_name:
                        value = 50 + (hour % 5) * 10
                    elif "per_second" in metric_name:
                        value = 100 + (hour % 8) * 20
                    elif "rate" in metric_name:
                        value = 0.5 + (hour % 3) * 0.2
                    elif "count" in metric_name:
                        value = 10 + (hour % 4) * 5
                    elif "usage" in metric_name:
                        value = 60 + (hour % 6) * 5
                    elif "accuracy" in metric_name:
                        value = 0.85 + (hour % 10) * 0.01
                    else:
                        value = 50 + (hour % 7) * 10
                    
                    metric = PerformanceMetric(
                        component=component,
                        metric_name=metric_name,
                        value=value,
                        timestamp=timestamp,
                        unit="ms" if "time_ms" in metric_name else None,
                        threshold=100.0 if "time_ms" in metric_name else None
                    )
                    
                    metrics.append(metric)
        
        return metrics
    
    async def verify_setup(self) -> Dict[str, Any]:
        """
        Verify InfluxDB setup and return status information.
        
        Returns:
            Dictionary containing setup verification results.
        """
        try:
            verification = {
                "connection_healthy": False,
                "bucket_exists": False,
                "retention_policies": False,
                "sample_data_count": 0,
                "measurements": [],
                "errors": []
            }
            
            # Check connection
            verification["connection_healthy"] = self.manager.health_check()
            
            if not verification["connection_healthy"]:
                verification["errors"].append("InfluxDB connection failed")
                return verification
            
            # Check bucket
            bucket_info = self.manager.get_bucket_info()
            verification["bucket_exists"] = bucket_info is not None
            
            if bucket_info:
                verification["retention_policies"] = len(bucket_info.get("retention_rules", [])) > 0
            
            # Query sample data
            try:
                from datetime import datetime, timedelta
                end_time = datetime.now()
                start_time = end_time - timedelta(days=1)
                
                # Check each measurement type
                measurements = ["delay_metrics", "congestion_metrics", "runway_metrics", 
                              "weather_metrics", "performance_metrics"]
                
                total_count = 0
                for measurement in measurements:
                    data = self.manager.query_flight_metrics(
                        measurement=measurement,
                        start_time=start_time,
                        end_time=end_time
                    )
                    count = len(data)
                    total_count += count
                    
                    if count > 0:
                        verification["measurements"].append({
                            "name": measurement,
                            "record_count": count
                        })
                
                verification["sample_data_count"] = total_count
                
            except Exception as e:
                verification["errors"].append(f"Error querying sample data: {e}")
            
            return verification
            
        except Exception as e:
            logger.error(f"Error in setup verification: {e}")
            return {
                "connection_healthy": False,
                "errors": [str(e)]
            }
    
    def close(self) -> None:
        """Close InfluxDB connections."""
        self.manager.close()


async def setup_influxdb() -> bool:
    """
    Main setup function for InfluxDB initialization.
    
    Returns:
        bool: True if setup successful, False otherwise.
    """
    try:
        logger.info("Starting InfluxDB setup process...")
        
        setup = InfluxDBSetup()
        
        # Initialize database
        success = await setup.initialize_database()
        
        if success:
            # Verify setup
            verification = await setup.verify_setup()
            
            logger.info("Setup verification results:")
            logger.info(f"  Connection healthy: {verification['connection_healthy']}")
            logger.info(f"  Bucket exists: {verification['bucket_exists']}")
            logger.info(f"  Retention policies: {verification['retention_policies']}")
            logger.info(f"  Sample data count: {verification['sample_data_count']}")
            logger.info(f"  Measurements: {len(verification['measurements'])}")
            
            if verification["errors"]:
                logger.warning(f"  Errors: {verification['errors']}")
        
        setup.close()
        return success
        
    except Exception as e:
        logger.error(f"Error in InfluxDB setup: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run setup
    success = asyncio.run(setup_influxdb())
    
    if success:
        print("InfluxDB setup completed successfully!")
    else:
        print("InfluxDB setup failed!")
        exit(1)