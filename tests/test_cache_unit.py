"""
Unit tests for cache functionality without requiring Redis
"""
import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd

from src.utils.cache import CacheManager, cache_key_generator
from src.utils.cache_strategies import FlightDataCacheStrategy, AnalysisCacheStrategy
from src.utils.cache_config import CacheConfig, CacheKeyConfig, load_environment_config
from src.utils.cache_monitor import CacheMonitor, CacheMetrics


class TestCacheManagerUnit:
    """Unit tests for CacheManager without Redis dependency"""
    
    def test_serialize_key(self):
        """Test key serialization"""
        cache = CacheManager()
        key = cache._serialize_key("test_key")
        assert key == "flight_analysis:test_key"
    
    def test_serialize_deserialize_simple_data(self):
        """Test serialization/deserialization of simple data"""
        cache = CacheManager()
        
        # Test dictionary
        test_data = {"key": "value", "number": 42}
        serialized = cache._serialize_value(test_data)
        deserialized = cache._deserialize_value(serialized)
        assert deserialized == test_data
        
        # Test list
        test_list = [1, 2, 3, "test"]
        serialized = cache._serialize_value(test_list)
        deserialized = cache._deserialize_value(serialized)
        assert deserialized == test_list
    
    def test_serialize_deserialize_complex_data(self):
        """Test serialization/deserialization of complex data"""
        cache = CacheManager()
        
        # Test DataFrame
        test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        serialized = cache._serialize_value(test_df)
        deserialized = cache._deserialize_value(serialized)
        pd.testing.assert_frame_equal(deserialized, test_df)


class TestCacheKeyGenerator:
    """Test cache key generation"""
    
    def test_consistent_key_generation(self):
        """Test that same arguments produce same keys"""
        key1 = cache_key_generator(1, 2, param="value")
        key2 = cache_key_generator(1, 2, param="value")
        assert key1 == key2
    
    def test_different_args_different_keys(self):
        """Test that different arguments produce different keys"""
        key1 = cache_key_generator(1, 2, param="value")
        key2 = cache_key_generator(1, 3, param="value")
        assert key1 != key2
    
    def test_kwargs_order_independence(self):
        """Test that kwargs order doesn't affect key"""
        key1 = cache_key_generator(a=1, b=2, c=3)
        key2 = cache_key_generator(c=3, a=1, b=2)
        assert key1 == key2


class TestCacheConfig:
    """Test cache configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CacheConfig()
        assert config.default_ttl == 3600
        assert config.namespace == "flight_analysis"
        assert config.max_connections == 20
        assert config.ttl_settings is not None
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = CacheConfig()
        assert config.validate() is True
        
        # Invalid config
        config.default_ttl = -1
        assert config.validate() is False
    
    def test_get_ttl(self):
        """Test TTL retrieval for different cache types"""
        config = CacheConfig()
        
        # Known cache type
        ttl = config.get_ttl("flight_data")
        assert ttl == 1800
        
        # Unknown cache type (should return default)
        ttl = config.get_ttl("unknown_type")
        assert ttl == config.default_ttl
    
    def test_config_to_from_dict(self):
        """Test configuration serialization"""
        config = CacheConfig(default_ttl=7200, namespace="test")
        config_dict = config.to_dict()
        
        assert config_dict["default_ttl"] == 7200
        assert config_dict["namespace"] == "test"
        
        # Test deserialization
        new_config = CacheConfig.from_dict(config_dict)
        assert new_config.default_ttl == 7200
        assert new_config.namespace == "test"


class TestCacheKeyConfig:
    """Test cache key configuration"""
    
    def test_default_patterns(self):
        """Test default key patterns"""
        key_config = CacheKeyConfig()
        assert "flight_data" in key_config.patterns
        assert "ml_model" in key_config.patterns
    
    def test_get_pattern(self):
        """Test pattern retrieval"""
        key_config = CacheKeyConfig()
        
        # Known pattern
        pattern = key_config.get_pattern("flight_data")
        assert pattern == "flight_data:{airport}:{date}"
        
        # Unknown pattern
        pattern = key_config.get_pattern("unknown")
        assert pattern == "{key_type}:{id}"
    
    def test_format_key(self):
        """Test key formatting"""
        key_config = CacheKeyConfig()
        
        # Valid formatting
        key = key_config.format_key("flight_data", airport="BOM", date="2024-01-01")
        assert key == "flight_data:BOM:2024-01-01"
        
        # Missing parameters (should handle gracefully)
        key = key_config.format_key("flight_data", airport="BOM")
        assert "flight_data" in key
        assert "BOM" in key


class TestEnvironmentConfigs:
    """Test environment-specific configurations"""
    
    def test_load_development_config(self):
        """Test loading development configuration"""
        config = load_environment_config("development")
        assert config.default_ttl == 300  # 5 minutes
        assert config.enable_monitoring is True
        assert config.max_connections == 5
    
    def test_load_production_config(self):
        """Test loading production configuration"""
        config = load_environment_config("production")
        assert config.default_ttl == 3600  # 1 hour
        assert config.enable_compression is True
        assert config.monitoring_sample_rate == 0.1
    
    def test_load_testing_config(self):
        """Test loading testing configuration"""
        config = load_environment_config("testing")
        assert config.default_ttl == 60  # 1 minute
        assert config.enable_monitoring is False
        assert config.namespace == "test_flight_analysis"


class TestCacheMonitor:
    """Test cache monitoring functionality"""
    
    def test_cache_metrics_initialization(self):
        """Test CacheMetrics initialization"""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.hit_rate == 0.0
        assert metrics.error_rate == 0.0
    
    def test_cache_metrics_calculation(self):
        """Test metrics calculation"""
        metrics = CacheMetrics()
        metrics.hits = 8
        metrics.misses = 2
        metrics.errors = 1
        metrics.total_operations = 11
        
        metrics.calculate_rates()
        
        assert metrics.hit_rate == 0.8  # 8/(8+2)
        assert metrics.error_rate == 1/11  # 1/11
    
    def test_cache_monitor_initialization(self):
        """Test CacheMonitor initialization"""
        monitor = CacheMonitor(max_history=100)
        assert monitor.max_history == 100
        assert len(monitor.operations_history) == 0
        assert monitor.metrics.total_operations == 0
    
    def test_record_operation(self):
        """Test operation recording"""
        monitor = CacheMonitor()
        
        # Record a successful get operation
        monitor.record_operation("get", "test_key", 0.05, True)
        
        assert monitor.metrics.total_operations == 1
        assert monitor.metrics.hits == 1
        assert monitor.metrics.misses == 0
        assert len(monitor.operations_history) == 1
        
        # Record a cache miss
        monitor.record_operation("get", "missing_key", 0.02, False)
        
        assert monitor.metrics.total_operations == 2
        assert monitor.metrics.hits == 1
        assert monitor.metrics.misses == 1
        assert monitor.metrics.hit_rate == 0.5
    
    def test_get_operation_stats(self):
        """Test operation statistics"""
        monitor = CacheMonitor()
        
        # Record some operations
        for i in range(5):
            monitor.record_operation("get", f"key_{i}", 0.01 * (i + 1), True)
        
        stats = monitor.get_operation_stats("get")
        assert stats["count"] == 5
        assert stats["avg_time"] == 0.03  # (0.01+0.02+0.03+0.04+0.05)/5
        assert stats["min_time"] == 0.01
        assert stats["max_time"] == 0.05
    
    def test_get_slow_operations(self):
        """Test slow operation detection"""
        monitor = CacheMonitor()
        monitor.slow_operation_threshold = 0.05
        
        # Record fast and slow operations
        monitor.record_operation("get", "fast_key", 0.01, True)
        monitor.record_operation("get", "slow_key", 0.1, True)
        
        slow_ops = monitor.get_slow_operations()
        assert len(slow_ops) == 1
        assert slow_ops[0].key == "slow_key"
    
    def test_get_error_operations(self):
        """Test error operation tracking"""
        monitor = CacheMonitor()
        
        # Record successful and failed operations
        monitor.record_operation("get", "success_key", 0.01, True)
        monitor.record_operation("get", "error_key", 0.02, False, "Connection failed")
        
        error_ops = monitor.get_error_operations()
        assert len(error_ops) == 1
        assert error_ops[0].key == "error_key"
        assert error_ops[0].error_message == "Connection failed"


class TestCacheStrategiesUnit:
    """Unit tests for cache strategies without Redis"""
    
    def test_flight_data_strategy_hash_params(self):
        """Test parameter hashing in analysis strategy"""
        strategy = AnalysisCacheStrategy()
        
        params1 = {"time_window": "1h", "threshold": 15}
        params2 = {"threshold": 15, "time_window": "1h"}  # Same params, different order
        params3 = {"time_window": "2h", "threshold": 15}  # Different params
        
        hash1 = strategy._hash_params(params1)
        hash2 = strategy._hash_params(params2)
        hash3 = strategy._hash_params(params3)
        
        assert hash1 == hash2  # Same params should produce same hash
        assert hash1 != hash3  # Different params should produce different hash
    
    def test_query_strategy_hash_query(self):
        """Test query hashing in query strategy"""
        from src.utils.cache_strategies import QueryCacheStrategy
        
        strategy = QueryCacheStrategy()
        
        query1 = "What is the average delay?"
        query2 = "what is the average delay?"  # Different case
        query3 = "What is the maximum delay?"  # Different query
        
        hash1 = strategy._hash_query(query1)
        hash2 = strategy._hash_query(query2)
        hash3 = strategy._hash_query(query3)
        
        assert hash1 == hash2  # Case insensitive
        assert hash1 != hash3  # Different queries


def test_import_structure():
    """Test that all imports work correctly"""
    from src.utils import (
        CacheManager, cache_manager, cached, MLModelCache, ml_cache,
        CacheInvalidator, cache_invalidator, cache_key_generator
    )
    
    from src.utils import (
        FlightDataCacheStrategy, AnalysisCacheStrategy, MLModelCacheStrategy,
        QueryCacheStrategy, ReportCacheStrategy
    )
    
    from src.utils import (
        CacheMonitor, cache_monitor, get_cache_health, get_cache_metrics
    )
    
    from src.utils import (
        CacheConfig, CacheKeyConfig, get_cache_config, get_cache_ttl
    )
    
    # Verify instances exist
    assert cache_manager is not None
    assert ml_cache is not None
    assert cache_invalidator is not None
    assert cache_monitor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])