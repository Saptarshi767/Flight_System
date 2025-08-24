"""
Cache configuration and settings management
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types"""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    LAZY_LOADING = "lazy_loading"


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live based


@dataclass
class CacheConfig:
    """Cache configuration settings"""
    
    # Connection settings
    redis_url: str = "redis://localhost:6379/0"
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    # Cache behavior
    default_ttl: int = 3600  # 1 hour
    max_key_length: int = 250
    max_value_size: int = 1024 * 1024  # 1MB
    namespace: str = "flight_analysis"
    
    # Performance settings
    enable_compression: bool = False
    compression_threshold: int = 1024  # Compress values larger than 1KB
    enable_monitoring: bool = True
    monitoring_sample_rate: float = 1.0  # Monitor 100% of operations
    
    # Strategy settings
    cache_strategy: CacheStrategy = CacheStrategy.LAZY_LOADING
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    
    # TTL settings for different data types
    ttl_settings: Dict[str, int] = None
    
    def __post_init__(self):
        """Initialize default TTL settings"""
        if self.ttl_settings is None:
            self.ttl_settings = {
                # Flight data TTLs
                "flight_data": 1800,  # 30 minutes
                "flight_data_historical": 86400,  # 24 hours
                "flight_aggregated": 3600,  # 1 hour
                
                # Analysis results TTLs
                "delay_analysis": 3600,  # 1 hour
                "congestion_analysis": 3600,  # 1 hour
                "schedule_impact": 1800,  # 30 minutes
                "cascading_impact": 3600,  # 1 hour
                
                # ML model TTLs
                "ml_predictions": 7200,  # 2 hours
                "ml_features": 1800,  # 30 minutes
                "ml_metrics": 7200,  # 2 hours
                
                # Query and report TTLs
                "nlp_queries": 1800,  # 30 minutes
                "popular_queries": 3600,  # 1 hour
                "reports": 7200,  # 2 hours
                "charts": 3600,  # 1 hour
                
                # System cache TTLs
                "system_config": 300,  # 5 minutes
                "user_sessions": 1800,  # 30 minutes
                "api_responses": 600,  # 10 minutes
            }
    
    def get_ttl(self, cache_type: str) -> int:
        """Get TTL for specific cache type"""
        return self.ttl_settings.get(cache_type, self.default_ttl)
    
    def validate(self) -> bool:
        """Validate cache configuration"""
        try:
            # Validate connection settings
            if not self.redis_url:
                raise ValueError("Redis URL is required")
            
            if self.max_connections <= 0:
                raise ValueError("Max connections must be positive")
            
            if self.socket_timeout <= 0:
                raise ValueError("Socket timeout must be positive")
            
            # Validate cache settings
            if self.default_ttl <= 0:
                raise ValueError("Default TTL must be positive")
            
            if self.max_key_length <= 0:
                raise ValueError("Max key length must be positive")
            
            if self.max_value_size <= 0:
                raise ValueError("Max value size must be positive")
            
            # Validate monitoring settings
            if not 0.0 <= self.monitoring_sample_rate <= 1.0:
                raise ValueError("Monitoring sample rate must be between 0.0 and 1.0")
            
            logger.info("Cache configuration validation passed")
            return True
            
        except ValueError as e:
            logger.error(f"Cache configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "redis_url": self.redis_url,
            "max_connections": self.max_connections,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "retry_on_timeout": self.retry_on_timeout,
            "health_check_interval": self.health_check_interval,
            "default_ttl": self.default_ttl,
            "max_key_length": self.max_key_length,
            "max_value_size": self.max_value_size,
            "namespace": self.namespace,
            "enable_compression": self.enable_compression,
            "compression_threshold": self.compression_threshold,
            "enable_monitoring": self.enable_monitoring,
            "monitoring_sample_rate": self.monitoring_sample_rate,
            "cache_strategy": self.cache_strategy.value,
            "eviction_policy": self.eviction_policy.value,
            "ttl_settings": self.ttl_settings
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CacheConfig':
        """Create configuration from dictionary"""
        # Handle enum conversions
        if "cache_strategy" in config_dict:
            config_dict["cache_strategy"] = CacheStrategy(config_dict["cache_strategy"])
        
        if "eviction_policy" in config_dict:
            config_dict["eviction_policy"] = EvictionPolicy(config_dict["eviction_policy"])
        
        return cls(**config_dict)


@dataclass
class CacheKeyConfig:
    """Configuration for cache key patterns"""
    
    # Key patterns for different data types
    patterns: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default key patterns"""
        if self.patterns is None:
            self.patterns = {
                # Flight data keys
                "flight_data": "flight_data:{airport}:{date}",
                "flight_aggregated": "flight_agg:{airport}:{type}:{period}",
                "flight_search": "flight_search:{query_hash}",
                
                # Analysis keys
                "delay_analysis": "delay_analysis:{airport}:{date}:{params_hash}",
                "congestion": "congestion:{airport}:{date}:{window}",
                "schedule_impact": "schedule_impact:{scenario_id}",
                "cascading_impact": "cascading:{flight_id}:{network_hash}",
                
                # ML model keys
                "ml_model": "ml_model:{model_name}:{version}:{input_hash}",
                "ml_features": "features:{feature_set}:{data_hash}",
                "ml_metrics": "model_metrics:{model_name}:{version}",
                
                # Query keys
                "nlp_query": "nlp_query:{query_hash}:{context_hash}",
                "query_suggestions": "query_suggestions:{partial_hash}",
                
                # Report keys
                "report": "report:{type}:{params_hash}",
                "chart": "chart:{type}:{params_hash}",
                
                # System keys
                "config": "config:{component}:{version}",
                "session": "session:{user_id}:{session_id}",
                "api_response": "api:{endpoint}:{params_hash}"
            }
    
    def get_pattern(self, key_type: str) -> str:
        """Get key pattern for specific type"""
        return self.patterns.get(key_type, "{key_type}:{id}")
    
    def format_key(self, key_type: str, **kwargs) -> str:
        """Format cache key using pattern and parameters"""
        pattern = self.get_pattern(key_type)
        try:
            return pattern.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing parameter for key pattern {key_type}: {e}")
            return f"{key_type}:{':'.join(str(v) for v in kwargs.values())}"


class CacheConfigManager:
    """Manages cache configuration and provides configuration access"""
    
    def __init__(self, config: CacheConfig = None, key_config: CacheKeyConfig = None):
        """Initialize configuration manager"""
        self.config = config or CacheConfig()
        self.key_config = key_config or CacheKeyConfig()
        
        # Validate configuration
        if not self.config.validate():
            logger.warning("Cache configuration validation failed, using defaults")
            self.config = CacheConfig()
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration"""
        return self.config
    
    def get_key_config(self) -> CacheKeyConfig:
        """Get key configuration"""
        return self.key_config
    
    def update_config(self, **kwargs):
        """Update cache configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated cache config: {key} = {value}")
            else:
                logger.warning(f"Unknown cache config parameter: {key}")
        
        # Re-validate after updates
        self.config.validate()
    
    def get_ttl(self, cache_type: str) -> int:
        """Get TTL for cache type"""
        return self.config.get_ttl(cache_type)
    
    def format_key(self, key_type: str, **kwargs) -> str:
        """Format cache key"""
        return self.key_config.format_key(key_type, **kwargs)
    
    def export_config(self) -> Dict[str, Any]:
        """Export complete configuration"""
        return {
            "cache_config": self.config.to_dict(),
            "key_patterns": self.key_config.patterns
        }
    
    def import_config(self, config_dict: Dict[str, Any]):
        """Import configuration from dictionary"""
        if "cache_config" in config_dict:
            self.config = CacheConfig.from_dict(config_dict["cache_config"])
        
        if "key_patterns" in config_dict:
            self.key_config.patterns = config_dict["key_patterns"]
        
        logger.info("Cache configuration imported successfully")


# Global configuration manager
cache_config_manager = CacheConfigManager()


def get_cache_config() -> CacheConfig:
    """Get global cache configuration"""
    return cache_config_manager.get_cache_config()


def get_cache_ttl(cache_type: str) -> int:
    """Get TTL for specific cache type"""
    return cache_config_manager.get_ttl(cache_type)


def format_cache_key(key_type: str, **kwargs) -> str:
    """Format cache key using global configuration"""
    return cache_config_manager.format_key(key_type, **kwargs)


def update_cache_config(**kwargs):
    """Update global cache configuration"""
    cache_config_manager.update_config(**kwargs)


# Environment-specific configurations
DEVELOPMENT_CONFIG = CacheConfig(
    default_ttl=300,  # 5 minutes for development
    enable_monitoring=True,
    monitoring_sample_rate=1.0,
    max_connections=5
)

PRODUCTION_CONFIG = CacheConfig(
    default_ttl=3600,  # 1 hour for production
    enable_monitoring=True,
    monitoring_sample_rate=0.1,  # Sample 10% in production
    max_connections=20,
    enable_compression=True
)

TESTING_CONFIG = CacheConfig(
    default_ttl=60,  # 1 minute for testing
    enable_monitoring=False,
    max_connections=2,
    namespace="test_flight_analysis"
)


def load_environment_config(environment: str = "development"):
    """Load configuration for specific environment"""
    configs = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "testing": TESTING_CONFIG
    }
    
    config = configs.get(environment, DEVELOPMENT_CONFIG)
    cache_config_manager.config = config
    logger.info(f"Loaded cache configuration for {environment} environment")
    
    return config