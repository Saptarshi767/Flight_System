"""
Redis caching utilities for the Flight Scheduling Analysis System
"""
import json
import pickle
import hashlib
from typing import Any, Optional, Union, Dict, List
from datetime import datetime, timedelta
import redis
from redis.exceptions import RedisError, ConnectionError
import logging
from functools import wraps

from src.config.settings import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis cache manager with connection management and error handling"""
    
    def __init__(self, redis_url: str = None):
        """Initialize Redis connection"""
        self.redis_url = redis_url or settings.redis_url
        self._redis_client = None
        self._connection_pool = None
        self.default_ttl = 3600  # 1 hour default TTL
        
    @property
    def redis_client(self) -> redis.Redis:
        """Get Redis client with connection pooling"""
        if self._redis_client is None:
            try:
                # Create connection pool for better performance
                self._connection_pool = redis.ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=20,
                    retry_on_timeout=True,
                    socket_keepalive=True,
                    socket_keepalive_options={}
                )
                self._redis_client = redis.Redis(
                    connection_pool=self._connection_pool,
                    decode_responses=False  # Keep binary for pickle support
                )
                # Test connection
                self._redis_client.ping()
                logger.info("Redis connection established successfully")
            except (ConnectionError, RedisError) as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._redis_client
    
    def is_connected(self) -> bool:
        """Check if Redis is connected and available"""
        try:
            self.redis_client.ping()
            return True
        except (ConnectionError, RedisError):
            return False
    
    def _serialize_key(self, key: str) -> str:
        """Create a consistent cache key with namespace"""
        namespace = "flight_analysis"
        return f"{namespace}:{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage"""
        try:
            # Try JSON first for simple types
            if isinstance(value, (str, int, float, bool, list, dict)):
                return json.dumps(value, default=str).encode('utf-8')
            else:
                # Use pickle for complex objects
                return pickle.dumps(value)
        except (TypeError, ValueError):
            # Fallback to pickle
            return pickle.dumps(value)
    
    def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize value from Redis storage"""
        try:
            # Try JSON first
            return json.loads(value.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fallback to pickle
            return pickle.loads(value)
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set a value in cache with optional TTL"""
        try:
            cache_key = self._serialize_key(key)
            serialized_value = self._serialize_value(value)
            ttl = ttl or self.default_ttl
            
            result = self.redis_client.setex(cache_key, ttl, serialized_value)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return result
        except (RedisError, Exception) as e:
            logger.error(f"Cache SET error for key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        try:
            cache_key = self._serialize_key(key)
            value = self.redis_client.get(cache_key)
            
            if value is None:
                logger.debug(f"Cache MISS: {key}")
                return None
            
            logger.debug(f"Cache HIT: {key}")
            return self._deserialize_value(value)
        except (RedisError, Exception) as e:
            logger.error(f"Cache GET error for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        try:
            cache_key = self._serialize_key(key)
            result = self.redis_client.delete(cache_key)
            logger.debug(f"Cache DELETE: {key}")
            return bool(result)
        except (RedisError, Exception) as e:
            logger.error(f"Cache DELETE error for key {key}: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern"""
        try:
            cache_pattern = self._serialize_key(pattern)
            keys = self.redis_client.keys(cache_pattern)
            if keys:
                result = self.redis_client.delete(*keys)
                logger.debug(f"Cache DELETE PATTERN: {pattern} ({len(keys)} keys)")
                return result
            return 0
        except (RedisError, Exception) as e:
            logger.error(f"Cache DELETE PATTERN error for pattern {pattern}: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache"""
        try:
            cache_key = self._serialize_key(key)
            return bool(self.redis_client.exists(cache_key))
        except (RedisError, Exception) as e:
            logger.error(f"Cache EXISTS error for key {key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for an existing key"""
        try:
            cache_key = self._serialize_key(key)
            result = self.redis_client.expire(cache_key, ttl)
            logger.debug(f"Cache EXPIRE: {key} (TTL: {ttl}s)")
            return bool(result)
        except (RedisError, Exception) as e:
            logger.error(f"Cache EXPIRE error for key {key}: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """Get TTL for a key (-1 if no TTL, -2 if key doesn't exist)"""
        try:
            cache_key = self._serialize_key(key)
            return self.redis_client.ttl(cache_key)
        except (RedisError, Exception) as e:
            logger.error(f"Cache TTL error for key {key}: {e}")
            return -2
    
    def flush_all(self) -> bool:
        """Clear all cache (use with caution)"""
        try:
            self.redis_client.flushdb()
            logger.warning("Cache FLUSH ALL executed")
            return True
        except (RedisError, Exception) as e:
            logger.error(f"Cache FLUSH ALL error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = self.redis_client.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'uptime_in_seconds': info.get('uptime_in_seconds', 0)
            }
        except (RedisError, Exception) as e:
            logger.error(f"Cache STATS error: {e}")
            return {}


# Global cache manager instance
cache_manager = CacheManager()


def cache_key_generator(*args, **kwargs) -> str:
    """Generate a consistent cache key from function arguments"""
    # Create a hash of the arguments
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items()) if kwargs else {}
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(ttl: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            func_name = f"{func.__module__}.{func.__name__}"
            arg_hash = cache_key_generator(*args, **kwargs)
            cache_key = f"{key_prefix}:{func_name}:{arg_hash}" if key_prefix else f"{func_name}:{arg_hash}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for function {func_name}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for function {func_name}, executing...")
            result = func(*args, **kwargs)
            
            # Cache the result
            cache_manager.set(cache_key, result, ttl)
            return result
        
        # Add cache management methods to the decorated function
        wrapper.cache_clear = lambda: cache_manager.delete_pattern(f"{key_prefix}:{func.__module__}.{func.__name__}:*")
        wrapper.cache_info = lambda: cache_manager.get_stats()
        
        return wrapper
    return decorator


class MLModelCache:
    """Specialized cache for ML model results and predictions"""
    
    def __init__(self, cache_manager: CacheManager = None):
        self.cache = cache_manager or cache_manager
        self.model_ttl = 7200  # 2 hours for model results
        self.prediction_ttl = 1800  # 30 minutes for predictions
    
    def cache_model_result(self, model_name: str, model_version: str, 
                          input_hash: str, result: Any) -> bool:
        """Cache ML model prediction result"""
        key = f"ml_model:{model_name}:{model_version}:{input_hash}"
        return self.cache.set(key, result, self.model_ttl)
    
    def get_model_result(self, model_name: str, model_version: str, 
                        input_hash: str) -> Optional[Any]:
        """Get cached ML model prediction result"""
        key = f"ml_model:{model_name}:{model_version}:{input_hash}"
        return self.cache.get(key)
    
    def cache_analysis_result(self, analysis_type: str, airport_code: str,
                            date_str: str, result: Any) -> bool:
        """Cache analysis result (delay, congestion, etc.)"""
        key = f"analysis:{analysis_type}:{airport_code}:{date_str}"
        return self.cache.set(key, result, self.model_ttl)
    
    def get_analysis_result(self, analysis_type: str, airport_code: str,
                           date_str: str) -> Optional[Any]:
        """Get cached analysis result"""
        key = f"analysis:{analysis_type}:{airport_code}:{date_str}"
        return self.cache.get(key)
    
    def invalidate_model_cache(self, model_name: str, model_version: str = None) -> int:
        """Invalidate all cached results for a specific model"""
        if model_version:
            pattern = f"ml_model:{model_name}:{model_version}:*"
        else:
            pattern = f"ml_model:{model_name}:*"
        return self.cache.delete_pattern(pattern)
    
    def invalidate_analysis_cache(self, analysis_type: str = None, 
                                 airport_code: str = None) -> int:
        """Invalidate analysis cache by type and/or airport"""
        if analysis_type and airport_code:
            pattern = f"analysis:{analysis_type}:{airport_code}:*"
        elif analysis_type:
            pattern = f"analysis:{analysis_type}:*"
        elif airport_code:
            pattern = f"analysis:*:{airport_code}:*"
        else:
            pattern = "analysis:*"
        return self.cache.delete_pattern(pattern)


# Global ML model cache instance
ml_cache = MLModelCache(cache_manager)


class CacheInvalidator:
    """Handles cache invalidation logic for data updates"""
    
    def __init__(self, cache_manager: CacheManager = None):
        self.cache = cache_manager or cache_manager
    
    def invalidate_flight_data_cache(self, airport_code: str = None, 
                                   date_range: tuple = None) -> int:
        """Invalidate cache when flight data is updated"""
        patterns = []
        
        if airport_code:
            patterns.extend([
                f"analysis:*:{airport_code}:*",
                f"flight_data:{airport_code}:*",
                f"congestion:{airport_code}:*",
                f"delay_analysis:{airport_code}:*"
            ])
        else:
            patterns.extend([
                "analysis:*",
                "flight_data:*",
                "congestion:*",
                "delay_analysis:*"
            ])
        
        total_deleted = 0
        for pattern in patterns:
            total_deleted += self.cache.delete_pattern(pattern)
        
        logger.info(f"Invalidated {total_deleted} cache entries for flight data update")
        return total_deleted
    
    def invalidate_model_cache(self, model_type: str = None) -> int:
        """Invalidate ML model cache when models are retrained"""
        if model_type:
            pattern = f"ml_model:{model_type}:*"
        else:
            pattern = "ml_model:*"
        
        deleted = self.cache.delete_pattern(pattern)
        logger.info(f"Invalidated {deleted} ML model cache entries")
        return deleted
    
    def invalidate_schedule_cache(self, airport_code: str = None) -> int:
        """Invalidate schedule-related cache"""
        patterns = [
            f"schedule:*:{airport_code}:*" if airport_code else "schedule:*",
            f"optimization:*:{airport_code}:*" if airport_code else "optimization:*"
        ]
        
        total_deleted = 0
        for pattern in patterns:
            total_deleted += self.cache.delete_pattern(pattern)
        
        logger.info(f"Invalidated {total_deleted} schedule cache entries")
        return total_deleted


# Global cache invalidator instance
cache_invalidator = CacheInvalidator(cache_manager)