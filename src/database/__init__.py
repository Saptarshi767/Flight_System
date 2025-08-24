"""
Database module for flight scheduling analysis

This module provides database models, connection management, and CRUD operations
for the flight scheduling analysis system.
"""

from .models import (
    Base,
    Flight,
    Airport,
    Airline,
    Aircraft,
    AnalysisResult,
    DataIngestionLog
)

from .connection import (
    DatabaseManager,
    db_manager,
    get_db_session,
    db_session_scope,
    init_database,
    test_database_connection
)

from .operations import (
    flight_repo,
    airport_repo,
    airline_repo,
    aircraft_repo,
    analysis_result_repo,
    ingestion_log_repo,
    get_flights_dataframe,
    insert_sample_data
)

from .utils import (
    db_utils,
    init_database as init_db_with_utils,
    reset_database,
    check_database_health,
    create_migration,
    apply_migrations
)

# InfluxDB Time Series Components
from .influxdb_client import (
    InfluxDBManager,
    InfluxDBConfig,
    get_influxdb_manager,
    close_influxdb_connection
)

from .time_series_models import (
    FlightMetric,
    FlightDelayMetric,
    CongestionMetric,
    RunwayMetric,
    WeatherMetric,
    PerformanceMetric,
    TimeSeriesDataBuilder,
    TimeSeriesQueryBuilder,
    MeasurementType,
    DelayCategory
)

from .time_series_ingestion import (
    TimeSeriesIngestionPipeline,
    get_ingestion_pipeline,
    ingest_flight_data_async
)

from .time_series_utils import (
    TimeSeriesQueryUtils,
    QueryResult,
    AggregationConfig,
    TimeSeriesAggregator,
    get_query_utils
)

from .setup_influxdb import (
    InfluxDBSetup,
    setup_influxdb
)

__all__ = [
    # Models
    'Base',
    'Flight',
    'Airport',
    'Airline',
    'Aircraft',
    'AnalysisResult',
    'DataIngestionLog',
    
    # Connection
    'DatabaseManager',
    'db_manager',
    'get_db_session',
    'db_session_scope',
    'init_database',
    'test_database_connection',
    
    # Operations
    'flight_repo',
    'airport_repo',
    'airline_repo',
    'aircraft_repo',
    'analysis_result_repo',
    'ingestion_log_repo',
    'get_flights_dataframe',
    'insert_sample_data',
    
    # Utils
    'db_utils',
    'init_db_with_utils',
    'reset_database',
    'check_database_health',
    'create_migration',
    'apply_migrations',
    
    # InfluxDB Components
    'InfluxDBManager',
    'InfluxDBConfig',
    'get_influxdb_manager',
    'close_influxdb_connection',
    'FlightMetric',
    'FlightDelayMetric',
    'CongestionMetric',
    'RunwayMetric',
    'WeatherMetric',
    'PerformanceMetric',
    'TimeSeriesDataBuilder',
    'TimeSeriesQueryBuilder',
    'MeasurementType',
    'DelayCategory',
    'TimeSeriesIngestionPipeline',
    'get_ingestion_pipeline',
    'ingest_flight_data_async',
    'TimeSeriesQueryUtils',
    'QueryResult',
    'AggregationConfig',
    'TimeSeriesAggregator',
    'get_query_utils',
    'InfluxDBSetup',
    'setup_influxdb'
]