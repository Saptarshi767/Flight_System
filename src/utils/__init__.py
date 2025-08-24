"""Utility functions and helpers."""

from .cache import (
    CacheManager, cache_manager, cached, MLModelCache, ml_cache,
    CacheInvalidator, cache_invalidator, cache_key_generator
)

from .cache_strategies import (
    FlightDataCacheStrategy, AnalysisCacheStrategy, MLModelCacheStrategy,
    QueryCacheStrategy, ReportCacheStrategy,
    flight_data_cache, analysis_cache, ml_model_cache, query_cache, report_cache,
    cache_flight_data, cache_analysis_result, cache_ml_prediction,
    cache_query_result, cache_report_data
)

from .cache_monitor import (
    CacheMonitor, cache_monitor, MonitoredCacheManager, monitored_cache,
    get_cache_health, get_cache_metrics, reset_cache_monitoring,
    CacheAlerts, cache_alerts, setup_cache_monitoring
)

from .cache_config import (
    CacheConfig, CacheKeyConfig, CacheConfigManager, cache_config_manager,
    CacheStrategy, EvictionPolicy,
    get_cache_config, get_cache_ttl, format_cache_key, update_cache_config,
    load_environment_config
)

__all__ = [
    # Core cache functionality
    'CacheManager', 'cache_manager', 'cached', 'MLModelCache', 'ml_cache',
    'CacheInvalidator', 'cache_invalidator', 'cache_key_generator',
    
    # Cache strategies
    'FlightDataCacheStrategy', 'AnalysisCacheStrategy', 'MLModelCacheStrategy',
    'QueryCacheStrategy', 'ReportCacheStrategy',
    'flight_data_cache', 'analysis_cache', 'ml_model_cache', 'query_cache', 'report_cache',
    'cache_flight_data', 'cache_analysis_result', 'cache_ml_prediction',
    'cache_query_result', 'cache_report_data',
    
    # Monitoring
    'CacheMonitor', 'cache_monitor', 'MonitoredCacheManager', 'monitored_cache',
    'get_cache_health', 'get_cache_metrics', 'reset_cache_monitoring',
    'CacheAlerts', 'cache_alerts', 'setup_cache_monitoring',
    
    # Configuration
    'CacheConfig', 'CacheKeyConfig', 'CacheConfigManager', 'cache_config_manager',
    'CacheStrategy', 'EvictionPolicy',
    'get_cache_config', 'get_cache_ttl', 'format_cache_key', 'update_cache_config',
    'load_environment_config'
]