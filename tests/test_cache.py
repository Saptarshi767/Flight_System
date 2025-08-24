"""
Tests for Redis caching functionality
"""
import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd

from src.utils.cache import (
    CacheManager, cache_manager, cached, MLModelCache, ml_cache,
    CacheInvalidator, cache_invalidator, cache_key_generator
)
from src.utils.cache_strategies import (
    FlightDataCacheStrategy, AnalysisCacheStrategy, MLModelCacheStrategy,
    QueryCacheStrategy, ReportCacheStrategy,
    flight_data_cache, analysis_cache, ml_model_cache, query_cache, report_cache
)
from src.database.redis_client import redis_client, get_redis_client


class TestCacheManager:
    """Test cases for CacheManager class"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing"""
        with patch('redis.Redis') as mock_redis_class:
            mock_client = Mock()
            mock_redis_class.return_value = mock_client
            mock_client.ping.return_value = True
            mock_client.setex.return_value = True
            mock_client.get.return_value = None
            mock_client.delete.return_value = 1
            mock_client.exists.return_value = False
            mock_client.expire.return_value = True
            mock_client.ttl.return_value = 3600
            mock_client.flushdb.return_value = True
            mock_client.keys.return_value = []
            mock_client.info.return_value = {
                'connected_clients': 1,
                'used_memory': 1024,
                'used_memory_human': '1K',
                'keyspace_hits': 10,
                'keyspace_misses': 5,
                'total_commands_processed': 100,
                'uptime_in_seconds': 3600
            }
            yield mock_client
    
    def test_cache_manager_initialization(self, mock_redis):
        """Test CacheManager initialization"""
        cache = CacheManager("redis://localhost:6379/0")
        assert cache.redis_url == "redis://localhost:6379/0"
        assert cache.default_ttl == 3600
    
    def test_redis_client_property(self, mock_redis):
        """Test Redis client property with connection pooling"""
        cache = CacheManager()
        client = cache.redis_client
        assert client is not None
        mock_redis.ping.assert_called_once()
    
    def test_is_connected(self, mock_redis):
        """Test connection status check"""
        cache = CacheManager()
        assert cache.is_connected() is True
        
        mock_redis.ping.side_effect = Exception("Connection failed")
        assert cache.is_connected() is False
    
    def test_serialize_key(self, mock_redis):
        """Test key serialization with namespace"""
        cache = CacheManager()
        key = cache._serialize_key("test_key")
        assert key == "flight_analysis:test_key"
    
    def test_serialize_deserialize_value(self, mock_redis):
        """Test value serialization and deserialization"""
        cache = CacheManager()
        
        # Test simple types (JSON)
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        serialized = cache._serialize_value(test_data)
        deserialized = cache._deserialize_value(serialized)
        assert deserialized == test_data
        
        # Test complex objects (pickle)
        test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        serialized = cache._serialize_value(test_df)
        deserialized = cache._deserialize_value(serialized)
        pd.testing.assert_frame_equal(deserialized, test_df)
    
    def test_set_get_operations(self, mock_redis):
        """Test basic set and get operations"""
        cache = CacheManager()
        
        # Test successful set
        result = cache.set("test_key", {"data": "value"}, 1800)
        assert result is True
        mock_redis.setex.assert_called()
        
        # Test get with cache hit
        mock_redis.get.return_value = json.dumps({"data": "value"}).encode('utf-8')
        result = cache.get("test_key")
        assert result == {"data": "value"}
        
        # Test get with cache miss
        mock_redis.get.return_value = None
        result = cache.get("test_key")
        assert result is None
    
    def test_delete_operations(self, mock_redis):
        """Test delete operations"""
        cache = CacheManager()
        
        # Test single key delete
        result = cache.delete("test_key")
        assert result is True
        mock_redis.delete.assert_called()
        
        # Test pattern delete
        mock_redis.keys.return_value = [b"flight_analysis:pattern:key1", b"flight_analysis:pattern:key2"]
        mock_redis.delete.return_value = 2
        result = cache.delete_pattern("pattern:*")
        assert result == 2
    
    def test_cache_utilities(self, mock_redis):
        """Test utility methods"""
        cache = CacheManager()
        
        # Test exists
        mock_redis.exists.return_value = True
        assert cache.exists("test_key") is True
        
        # Test expire
        result = cache.expire("test_key", 3600)
        assert result is True
        mock_redis.expire.assert_called()
        
        # Test ttl
        mock_redis.ttl.return_value = 1800
        ttl = cache.ttl("test_key")
        assert ttl == 1800
        
        # Test stats
        stats = cache.get_stats()
        assert "connected_clients" in stats
        assert stats["connected_clients"] == 1
    
    def test_error_handling(self, mock_redis):
        """Test error handling in cache operations"""
        cache = CacheManager()
        
        # Test set error
        mock_redis.setex.side_effect = Exception("Redis error")
        result = cache.set("test_key", "value")
        assert result is False
        
        # Test get error
        mock_redis.get.side_effect = Exception("Redis error")
        result = cache.get("test_key")
        assert result is None


class TestCachedDecorator:
    """Test cases for @cached decorator"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager for decorator testing"""
        with patch('src.utils.cache.cache_manager') as mock_cache:
            mock_cache.get.return_value = None
            mock_cache.set.return_value = True
            mock_cache.delete_pattern.return_value = 1
            mock_cache.get_stats.return_value = {"hits": 10, "misses": 5}
            yield mock_cache
    
    def test_cached_decorator_miss(self, mock_cache_manager):
        """Test cached decorator with cache miss"""
        @cached(ttl=1800, key_prefix="test")
        def test_function(x, y):
            return x + y
        
        result = test_function(1, 2)
        assert result == 3
        
        # Verify cache operations
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.set.assert_called_once()
    
    def test_cached_decorator_hit(self, mock_cache_manager):
        """Test cached decorator with cache hit"""
        mock_cache_manager.get.return_value = 42
        
        @cached(ttl=1800, key_prefix="test")
        def test_function(x, y):
            return x + y
        
        result = test_function(1, 2)
        assert result == 42  # Should return cached value
        
        # Verify only get was called, not set
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.set.assert_not_called()
    
    def test_cache_key_generator(self):
        """Test cache key generation"""
        key1 = cache_key_generator(1, 2, param="value")
        key2 = cache_key_generator(1, 2, param="value")
        key3 = cache_key_generator(1, 3, param="value")
        
        assert key1 == key2  # Same arguments should produce same key
        assert key1 != key3  # Different arguments should produce different keys


class TestMLModelCache:
    """Test cases for MLModelCache"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager for ML cache testing"""
        with patch('src.utils.cache.cache_manager') as mock_cache:
            mock_cache.set.return_value = True
            mock_cache.get.return_value = None
            mock_cache.delete_pattern.return_value = 5
            yield mock_cache
    
    def test_ml_model_cache_operations(self, mock_cache_manager):
        """Test ML model cache operations"""
        ml_cache_instance = MLModelCache(mock_cache_manager)
        
        # Test cache model result
        result = ml_cache_instance.cache_model_result("xgboost", "v1.0", "hash123", {"prediction": 0.85})
        assert result is True
        
        # Test get model result
        mock_cache_manager.get.return_value = {"prediction": 0.85}
        result = ml_cache_instance.get_model_result("xgboost", "v1.0", "hash123")
        assert result == {"prediction": 0.85}
        
        # Test cache analysis result
        result = ml_cache_instance.cache_analysis_result("delay", "BOM", "2024-01-01", {"avg_delay": 15})
        assert result is True
        
        # Test invalidate model cache
        deleted = ml_cache_instance.invalidate_model_cache("xgboost", "v1.0")
        assert deleted == 5


class TestCacheInvalidator:
    """Test cases for CacheInvalidator"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager for invalidator testing"""
        with patch('src.utils.cache.cache_manager') as mock_cache:
            mock_cache.delete_pattern.return_value = 10
            yield mock_cache
    
    def test_invalidate_flight_data_cache(self, mock_cache_manager):
        """Test flight data cache invalidation"""
        invalidator = CacheInvalidator(mock_cache_manager)
        
        # Test airport-specific invalidation
        deleted = invalidator.invalidate_flight_data_cache("BOM")
        assert deleted > 0
        
        # Test general invalidation
        deleted = invalidator.invalidate_flight_data_cache()
        assert deleted > 0
    
    def test_invalidate_model_cache(self, mock_cache_manager):
        """Test model cache invalidation"""
        invalidator = CacheInvalidator(mock_cache_manager)
        
        deleted = invalidator.invalidate_model_cache("xgboost")
        assert deleted == 10
        
        deleted = invalidator.invalidate_model_cache()
        assert deleted == 10
    
    def test_invalidate_schedule_cache(self, mock_cache_manager):
        """Test schedule cache invalidation"""
        invalidator = CacheInvalidator(mock_cache_manager)
        
        deleted = invalidator.invalidate_schedule_cache("BOM")
        assert deleted > 0


class TestCacheStrategies:
    """Test cases for cache strategies"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager for strategy testing"""
        with patch('src.utils.cache.cache_manager') as mock_cache:
            mock_cache.set.return_value = True
            mock_cache.get.return_value = None
            mock_cache.delete_pattern.return_value = 5
            yield mock_cache
    
    def test_flight_data_cache_strategy(self, mock_cache_manager):
        """Test flight data caching strategy"""
        strategy = FlightDataCacheStrategy()
        
        # Test cache flight data
        test_data = [{"flight": "AI101", "delay": 15}]
        result = strategy.cache_flight_data("BOM", "2024-01-01", test_data)
        assert result is True
        
        # Test get flight data
        mock_cache_manager.get.return_value = test_data
        result = strategy.get_flight_data("BOM", "2024-01-01")
        assert result == test_data
        
        # Test cache aggregated data
        agg_data = {"avg_delay": 12.5, "total_flights": 100}
        result = strategy.cache_aggregated_data("BOM", "daily", "2024-01-01", agg_data)
        assert result is True
    
    def test_analysis_cache_strategy(self, mock_cache_manager):
        """Test analysis caching strategy"""
        strategy = AnalysisCacheStrategy()
        
        # Test delay analysis caching
        params = {"time_window": "1h", "threshold": 15}
        result_data = {"avg_delay": 18.5, "delay_count": 25}
        
        result = strategy.cache_delay_analysis("BOM", "2024-01-01", params, result_data)
        assert result is True
        
        # Test congestion analysis caching
        result = strategy.cache_congestion_analysis("BOM", "2024-01-01", "peak_hours", {"congestion_score": 0.8})
        assert result is True
    
    def test_query_cache_strategy(self, mock_cache_manager):
        """Test query caching strategy"""
        strategy = QueryCacheStrategy()
        
        # Test query result caching
        query = "What's the average delay for Mumbai airport?"
        context_hash = "context123"
        result_data = {"answer": "15 minutes", "confidence": 0.9}
        
        result = strategy.cache_query_result(query, context_hash, result_data)
        assert result is True
        
        # Test query suggestions caching
        suggestions = ["What's the delay?", "What's the congestion?"]
        result = strategy.cache_query_suggestions("What's", suggestions)
        assert result is True


class TestPerformance:
    """Performance tests for caching functionality"""
    
    @pytest.fixture
    def mock_redis_performance(self):
        """Mock Redis for performance testing"""
        with patch('redis.Redis') as mock_redis_class:
            mock_client = Mock()
            mock_redis_class.return_value = mock_client
            mock_client.ping.return_value = True
            mock_client.setex.return_value = True
            mock_client.get.return_value = json.dumps({"test": "data"}).encode('utf-8')
            yield mock_client
    
    def test_cache_performance_bulk_operations(self, mock_redis_performance):
        """Test cache performance with bulk operations"""
        cache = CacheManager()
        
        # Test bulk set operations
        start_time = time.time()
        for i in range(100):
            cache.set(f"test_key_{i}", {"data": f"value_{i}"}, 3600)
        bulk_set_time = time.time() - start_time
        
        # Test bulk get operations
        start_time = time.time()
        for i in range(100):
            cache.get(f"test_key_{i}")
        bulk_get_time = time.time() - start_time
        
        # Performance assertions (should complete quickly)
        assert bulk_set_time < 1.0  # Should complete in less than 1 second
        assert bulk_get_time < 1.0  # Should complete in less than 1 second
    
    def test_cache_memory_usage(self, mock_redis_performance):
        """Test cache memory usage patterns"""
        cache = CacheManager()
        
        # Test with different data sizes
        small_data = {"key": "value"}
        medium_data = {"data": list(range(1000))}
        large_data = {"data": list(range(10000))}
        
        # All should succeed without memory issues
        assert cache.set("small", small_data) is True
        assert cache.set("medium", medium_data) is True
        assert cache.set("large", large_data) is True


class TestIntegration:
    """Integration tests for cache system"""
    
    @pytest.mark.integration
    def test_redis_connection_integration(self):
        """Test actual Redis connection (requires Redis server)"""
        try:
            from src.database.redis_client import check_redis_connection
            # This test requires actual Redis server
            # Skip if Redis is not available
            if not check_redis_connection():
                pytest.skip("Redis server not available")
            
            cache = CacheManager()
            assert cache.is_connected() is True
            
            # Test actual operations
            test_data = {"integration": "test", "timestamp": datetime.now().isoformat()}
            assert cache.set("integration_test", test_data, 60) is True
            
            retrieved_data = cache.get("integration_test")
            assert retrieved_data == test_data
            
            # Cleanup
            assert cache.delete("integration_test") is True
            
        except ImportError:
            pytest.skip("Redis client not available")
    
    @pytest.mark.integration
    def test_cache_strategies_integration(self):
        """Test cache strategies with actual Redis"""
        try:
            from src.database.redis_client import check_redis_connection
            if not check_redis_connection():
                pytest.skip("Redis server not available")
            
            # Test flight data strategy
            flight_strategy = FlightDataCacheStrategy()
            test_flights = [
                {"flight_id": "AI101", "delay": 15, "airport": "BOM"},
                {"flight_id": "6E202", "delay": 5, "airport": "BOM"}
            ]
            
            assert flight_strategy.cache_flight_data("BOM", "2024-01-01", test_flights) is True
            retrieved_flights = flight_strategy.get_flight_data("BOM", "2024-01-01")
            assert retrieved_flights == test_flights
            
            # Cleanup
            flight_strategy.invalidate_airport_data("BOM")
            
        except ImportError:
            pytest.skip("Redis client not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])