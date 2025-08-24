"""
InfluxDB client and connection management for flight time series data.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.query_api import QueryApi
from influxdb_client.client.delete_api import DeleteApi

logger = logging.getLogger(__name__)


@dataclass
class InfluxDBConfig:
    """InfluxDB configuration settings."""
    url: str
    token: str
    org: str
    bucket: str
    timeout: int = 10000


class InfluxDBManager:
    """
    InfluxDB client manager for flight time series data operations.
    
    Handles connection management, data ingestion, querying, and retention policies
    for flight metrics and time series data.
    """
    
    def __init__(self, config: Optional[InfluxDBConfig] = None):
        """
        Initialize InfluxDB manager with configuration.
        
        Args:
            config: InfluxDB configuration. If None, loads from environment variables.
        """
        if config is None:
            config = self._load_config_from_env()
        
        self.config = config
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.query_api: Optional[QueryApi] = None
        self.delete_api: Optional[DeleteApi] = None
        
        self._connect()
    
    def _load_config_from_env(self) -> InfluxDBConfig:
        """Load InfluxDB configuration from environment variables."""
        return InfluxDBConfig(
            url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
            token=os.getenv("INFLUXDB_TOKEN", "flight-analysis-token"),
            org=os.getenv("INFLUXDB_ORG", "flight-analysis"),
            bucket=os.getenv("INFLUXDB_BUCKET", "flight-metrics"),
            timeout=int(os.getenv("INFLUXDB_TIMEOUT", "10000"))
        )
    
    def _connect(self) -> None:
        """Establish connection to InfluxDB."""
        try:
            self.client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
                timeout=self.config.timeout
            )
            
            # Initialize APIs
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            self.delete_api = self.client.delete_api()
            
            # Test connection
            self.client.ping()
            logger.info(f"Successfully connected to InfluxDB at {self.config.url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check if InfluxDB connection is healthy.
        
        Returns:
            bool: True if connection is healthy, False otherwise.
        """
        try:
            if self.client:
                self.client.ping()
                return True
            return False
        except Exception as e:
            logger.error(f"InfluxDB health check failed: {e}")
            return False
    
    def write_flight_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """
        Write flight metrics to InfluxDB.
        
        Args:
            metrics: List of flight metric dictionaries containing:
                - measurement: Measurement name (e.g., 'flight_metrics')
                - tags: Dictionary of tag key-value pairs
                - fields: Dictionary of field key-value pairs
                - time: Timestamp (datetime object or string)
        
        Returns:
            bool: True if write successful, False otherwise.
        """
        try:
            points = []
            for metric in metrics:
                point = Point(metric["measurement"])
                
                # Add tags
                for tag_key, tag_value in metric.get("tags", {}).items():
                    point = point.tag(tag_key, str(tag_value))
                
                # Add fields
                for field_key, field_value in metric.get("fields", {}).items():
                    if isinstance(field_value, (int, float)):
                        point = point.field(field_key, field_value)
                    else:
                        point = point.field(field_key, str(field_value))
                
                # Set timestamp
                if "time" in metric:
                    if isinstance(metric["time"], datetime):
                        point = point.time(metric["time"], WritePrecision.S)
                    else:
                        point = point.time(metric["time"])
                
                points.append(point)
            
            # Write points to InfluxDB
            self.write_api.write(bucket=self.config.bucket, record=points)
            logger.info(f"Successfully wrote {len(points)} metrics to InfluxDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write metrics to InfluxDB: {e}")
            return False
    
    def query_flight_metrics(
        self,
        measurement: str,
        start_time: Union[str, datetime],
        end_time: Optional[Union[str, datetime]] = None,
        filters: Optional[Dict[str, str]] = None,
        fields: Optional[List[str]] = None,
        group_by: Optional[List[str]] = None,
        aggregate_function: Optional[str] = None,
        window: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query flight metrics from InfluxDB.
        
        Args:
            measurement: Measurement name to query
            start_time: Start time for query (datetime or RFC3339 string)
            end_time: End time for query (datetime or RFC3339 string)
            filters: Dictionary of tag filters
            fields: List of fields to select
            group_by: List of tags to group by
            aggregate_function: Aggregation function (mean, sum, count, etc.)
            window: Time window for aggregation (e.g., '1h', '1d')
        
        Returns:
            List of dictionaries containing query results.
        """
        try:
            # Build Flux query
            query = self._build_flux_query(
                measurement=measurement,
                start_time=start_time,
                end_time=end_time,
                filters=filters,
                fields=fields,
                group_by=group_by,
                aggregate_function=aggregate_function,
                window=window
            )
            
            logger.debug(f"Executing Flux query: {query}")
            
            # Execute query
            result = self.query_api.query(query)
            
            # Convert result to list of dictionaries
            data = []
            for table in result:
                for record in table.records:
                    row = {
                        "time": record.get_time(),
                        "measurement": record.get_measurement(),
                        "field": record.get_field(),
                        "value": record.get_value()
                    }
                    
                    # Add tags
                    for key, value in record.values.items():
                        if key not in ["_time", "_measurement", "_field", "_value", "_start", "_stop"]:
                            row[key] = value
                    
                    data.append(row)
            
            logger.info(f"Query returned {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Failed to query InfluxDB: {e}")
            return []
    
    def _build_flux_query(
        self,
        measurement: str,
        start_time: Union[str, datetime],
        end_time: Optional[Union[str, datetime]] = None,
        filters: Optional[Dict[str, str]] = None,
        fields: Optional[List[str]] = None,
        group_by: Optional[List[str]] = None,
        aggregate_function: Optional[str] = None,
        window: Optional[str] = None
    ) -> str:
        """Build Flux query string from parameters."""
        
        # Convert datetime to RFC3339 string if needed
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat() + "Z"
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat() + "Z"
        
        # Start building query
        query_parts = [
            f'from(bucket: "{self.config.bucket}")',
            f'|> range(start: {start_time}'
        ]
        
        if end_time:
            query_parts[-1] += f', stop: {end_time}'
        query_parts[-1] += ')'
        
        # Add measurement filter
        query_parts.append(f'|> filter(fn: (r) => r._measurement == "{measurement}")')
        
        # Add field filters
        if fields:
            field_filter = " or ".join([f'r._field == "{field}"' for field in fields])
            query_parts.append(f'|> filter(fn: (r) => {field_filter})')
        
        # Add tag filters
        if filters:
            for tag_key, tag_value in filters.items():
                query_parts.append(f'|> filter(fn: (r) => r.{tag_key} == "{tag_value}")')
        
        # Add aggregation
        if aggregate_function and window:
            if group_by:
                group_cols = '", "'.join(group_by)
                query_parts.append(f'|> group(columns: ["{group_cols}"])')
            
            query_parts.append(f'|> aggregateWindow(every: {window}, fn: {aggregate_function})')
        elif aggregate_function:
            if group_by:
                group_cols = '", "'.join(group_by)
                query_parts.append(f'|> group(columns: ["{group_cols}"])')
            query_parts.append(f'|> {aggregate_function}()')
        
        return '\n  '.join(query_parts)
    
    def delete_flight_metrics(
        self,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        predicate: Optional[str] = None
    ) -> bool:
        """
        Delete flight metrics from InfluxDB.
        
        Args:
            start_time: Start time for deletion
            end_time: End time for deletion
            predicate: Optional predicate for filtering data to delete
        
        Returns:
            bool: True if deletion successful, False otherwise.
        """
        try:
            # Convert datetime to RFC3339 string if needed
            if isinstance(start_time, datetime):
                start_time = start_time.isoformat() + "Z"
            if isinstance(end_time, datetime):
                end_time = end_time.isoformat() + "Z"
            
            self.delete_api.delete(
                start=start_time,
                stop=end_time,
                predicate=predicate,
                bucket=self.config.bucket,
                org=self.config.org
            )
            
            logger.info(f"Successfully deleted metrics from {start_time} to {end_time}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete metrics: {e}")
            return False
    
    def setup_retention_policies(self) -> bool:
        """
        Set up data retention policies for flight metrics.
        
        Returns:
            bool: True if setup successful, False otherwise.
        """
        try:
            buckets_api = self.client.buckets_api()
            
            # Get existing bucket
            bucket = buckets_api.find_bucket_by_name(self.config.bucket)
            
            if bucket:
                logger.info("Bucket found, retention policies managed by InfluxDB configuration")
                return True
            else:
                logger.error(f"Bucket {self.config.bucket} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup retention policies: {e}")
            return False
    
    def get_bucket_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current bucket.
        
        Returns:
            Dictionary containing bucket information or None if error.
        """
        try:
            buckets_api = self.client.buckets_api()
            bucket = buckets_api.find_bucket_by_name(self.config.bucket)
            
            if bucket:
                return {
                    "id": bucket.id,
                    "name": bucket.name,
                    "org_id": bucket.org_id,
                    "retention_rules": getattr(bucket, 'retention_rules', []),
                    "created_at": getattr(bucket, 'created_at', None),
                    "updated_at": getattr(bucket, 'updated_at', None)
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get bucket info: {e}")
            return None
    
    def close(self) -> None:
        """Close InfluxDB connection."""
        try:
            if self.client:
                self.client.close()
                logger.info("InfluxDB connection closed")
        except Exception as e:
            logger.error(f"Error closing InfluxDB connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global InfluxDB manager instance
_influxdb_manager: Optional[InfluxDBManager] = None


def get_influxdb_manager() -> InfluxDBManager:
    """
    Get global InfluxDB manager instance.
    
    Returns:
        InfluxDBManager instance.
    """
    global _influxdb_manager
    
    if _influxdb_manager is None:
        _influxdb_manager = InfluxDBManager()
    
    return _influxdb_manager


def close_influxdb_connection() -> None:
    """Close global InfluxDB connection."""
    global _influxdb_manager
    
    if _influxdb_manager:
        _influxdb_manager.close()
        _influxdb_manager = None