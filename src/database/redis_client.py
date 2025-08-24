"""
Redis client configuration and connection management
"""
import redis
from redis.exceptions import RedisError, ConnectionError
import logging
from typing import Optional
from contextlib import contextmanager

from src.config.settings import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis client with connection management and health checks"""
    
    def __init__(self, redis_url: str = None):
        """Initialize Redis client"""
        self.redis_url = redis_url or settings.redis_url
        self._client = None
        self._connection_pool = None
    
    def connect(self) -> redis.Redis:
        """Establish Redis connection with connection pooling"""
        if self._client is None:
            try:
                # Create connection pool
                self._connection_pool = redis.ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=20,
                    retry_on_timeout=True,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=30
                )
                
                # Create Redis client
                self._client = redis.Redis(
                    connection_pool=self._connection_pool,
                    decode_responses=False,  # Keep binary for flexibility
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                
                # Test connection
                self._client.ping()
                logger.info(f"Redis connected successfully to {self.redis_url}")
                
            except (ConnectionError, RedisError) as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self._client
    
    def disconnect(self):
        """Close Redis connection"""
        if self._client:
            try:
                if self._connection_pool:
                    self._connection_pool.disconnect()
                self._client = None
                self._connection_pool = None
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
    
    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            if self._client is None:
                return False
            self._client.ping()
            return True
        except (ConnectionError, RedisError):
            logger.warning("Redis health check failed")
            return False
    
    def get_client(self) -> redis.Redis:
        """Get Redis client, connecting if necessary"""
        if self._client is None or not self.health_check():
            self.connect()
        return self._client
    
    def get_info(self) -> dict:
        """Get Redis server information"""
        try:
            client = self.get_client()
            return client.info()
        except (ConnectionError, RedisError) as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}
    
    @contextmanager
    def pipeline(self, transaction: bool = True):
        """Context manager for Redis pipeline operations"""
        client = self.get_client()
        pipe = client.pipeline(transaction=transaction)
        try:
            yield pipe
            pipe.execute()
        except Exception as e:
            logger.error(f"Redis pipeline error: {e}")
            raise
        finally:
            pipe.reset()


# Global Redis client instance
redis_client = RedisClient()


def get_redis_client() -> redis.Redis:
    """Get the global Redis client instance"""
    return redis_client.get_client()


def check_redis_connection() -> bool:
    """Check if Redis is available"""
    return redis_client.health_check()


def get_redis_stats() -> dict:
    """Get Redis connection statistics"""
    return redis_client.get_info()