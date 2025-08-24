"""
Advanced caching strategies for the Flight Scheduling Analysis System
"""
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
import hashlib
import json
import asyncio
from functools import wraps
from dataclasses import dataclass
import logging

from src.utils.cache import cache_manager, CacheManager
from src.utils.logging import logger


@dataclass
class CacheStrategy:
    """Cache strategy configuration"""
    ttl: int
    key_pattern: str
    invalidation_triggers: List[str]
    refresh_threshold: float = 0.8  # Refresh when 80% of TTL has passed
    background_refresh: bool = False


class AdvancedCacheManager:
    """Advanced caching with multiple strategies and automatic refresh"""
    
    def __init__(self, cache_manager_instance: CacheManager = None):
        self.cache = cache_manager_instance or cache_manager
        self.logger = logger
        self.strategies: Dict[str, CacheStrategy] = {}
        self.refresh_tasks: Dict[str, asyncio.Task] = {}
        
        # Define cache strategies for different data types
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Setup default caching strategies"""
        
        # Flight data - medium TTL, refresh on data updates
        self.strategies['flight_data'] = CacheStrategy(
            ttl=1800,  # 30 minutes
            key_pattern="flight_data:{airport}:{date}",
            invalidation_triggers=['flight_data_update', 'schedule_change'],
            background_refresh=True
        )
        
        # Analysis results - longer TTL, refresh on model updates
        self.strategies['analysis_results'] = CacheStrategy(
            ttl=3600,  # 1 hour
            key_pattern="analysis:{type}:{airport}:{date}",
            invalidation_triggers=['model_update', 'data_refresh'],
            background_refresh=True
        )
        
        # ML predictions - shorter TTL, frequent refresh
        self.strategies['ml_predictions'] = CacheStrategy(
            ttl=900,  # 15 minutes
            key_pattern="ml_pred:{model}:{input_hash}",
            invalidation_triggers=['model_retrain', 'data_update'],
            background_refresh=False
        )
        
        # Static reference data - very long TTL
        self.strategies['reference_data'] = CacheStrategy(
            ttl=86400,  # 24 hours
            key_pattern="ref_data:{type}:{id}",
            invalidation_triggers=['reference_update'],
            background_refresh=False
        )
    
    def get_strategy(self, strategy_name: str) -> Optional[CacheStrategy]:
        """Get cache strategy by name"""
        return self.strategies.get(strategy_name)
    
    def set_with_strategy(self, strategy_name: str, key: str, value: Any, 
                         **key_params) -> bool:
        """Set cache value using specified strategy"""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            self.logger.warning(f"Unknown cache strategy: {strategy_name}")
            return False
        
        # Format key with parameters
        formatted_key = strategy.key_pattern.format(**key_params) if key_params else key
        
        # Set cache with strategy TTL
        success = self.cache.set(formatted_key, value, strategy.ttl)
        
        return success
    
    def get_with_strategy(self, strategy_name: str, key: str, 
                         **key_params) -> Optional[Any]:
        """Get cache value using specified strategy"""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return None
        
        # Format key with parameters
        formatted_key = strategy.key_pattern.format(**key_params) if key_params else key
        
        # Get value from cache
        value = self.cache.get(formatted_key)
        
        return value
    
    def invalidate_by_trigger(self, trigger: str) -> int:
        """Invalidate cache entries based on trigger"""
        total_invalidated = 0
        
        for strategy_name, strategy in self.strategies.items():
            if trigger in strategy.invalidation_triggers:
                # Create pattern to match all keys for this strategy
                pattern = strategy.key_pattern.replace('{', '*').replace('}', '*')
                invalidated = self.cache.delete_pattern(pattern)
                total_invalidated += invalidated
                
                self.logger.info(
                    f"Invalidated {invalidated} cache entries for strategy '{strategy_name}' "
                    f"due to trigger '{trigger}'"
                )
        
        return total_invalidated
    
    def get_cache_health(self) -> Dict[str, Any]:
        """Get cache health metrics"""
        stats = self.cache.get_stats()
        
        # Calculate hit rate
        hits = stats.get('keyspace_hits', 0)
        misses = stats.get('keyspace_misses', 0)
        total_requests = hits + misses
        hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_rate_percentage': round(hit_rate, 2),
            'total_requests': total_requests,
            'cache_hits': hits,
            'cache_misses': misses,
            'memory_usage': stats.get('used_memory_human', '0B'),
            'connected_clients': stats.get('connected_clients', 0),
            'strategies_configured': len(self.strategies)
        }


# Global instances
advanced_cache_manager = AdvancedCacheManager(cache_manager)