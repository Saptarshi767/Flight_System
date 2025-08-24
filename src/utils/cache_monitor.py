"""
Cache performance monitoring and metrics collection
"""
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from contextlib import contextmanager

from src.utils.cache import cache_manager
from src.database.redis_client import get_redis_client, get_redis_stats

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_operations: int = 0
    avg_response_time: float = 0.0
    hit_rate: float = 0.0
    error_rate: float = 0.0
    memory_usage: int = 0
    connected_clients: int = 0
    
    def calculate_rates(self):
        """Calculate hit rate and error rate"""
        if self.total_operations > 0:
            self.hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0
            self.error_rate = self.errors / self.total_operations
        else:
            self.hit_rate = 0.0
            self.error_rate = 0.0


@dataclass
class CacheOperation:
    """Individual cache operation record"""
    operation_type: str
    key: str
    timestamp: datetime
    duration: float
    success: bool
    error_message: Optional[str] = None


class CacheMonitor:
    """Cache performance monitor with metrics collection"""
    
    def __init__(self, max_history: int = 1000):
        """Initialize cache monitor"""
        self.max_history = max_history
        self.operations_history: deque = deque(maxlen=max_history)
        self.metrics = CacheMetrics()
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
        self.start_time = datetime.utcnow()
        
        # Performance thresholds
        self.slow_operation_threshold = 0.1  # 100ms
        self.high_error_rate_threshold = 0.05  # 5%
        self.low_hit_rate_threshold = 0.7  # 70%
    
    def record_operation(self, operation_type: str, key: str, 
                        duration: float, success: bool, 
                        error_message: Optional[str] = None):
        """Record a cache operation"""
        with self.lock:
            # Create operation record
            operation = CacheOperation(
                operation_type=operation_type,
                key=key,
                timestamp=datetime.utcnow(),
                duration=duration,
                success=success,
                error_message=error_message
            )
            
            # Add to history
            self.operations_history.append(operation)
            self.operation_times[operation_type].append(duration)
            
            # Update metrics
            self.metrics.total_operations += 1
            
            if operation_type == "get":
                if success:
                    self.metrics.hits += 1
                else:
                    self.metrics.misses += 1
            elif operation_type == "set":
                self.metrics.sets += 1
            elif operation_type == "delete":
                self.metrics.deletes += 1
            
            if not success and error_message:
                self.metrics.errors += 1
            
            # Update average response time
            total_time = sum(op.duration for op in self.operations_history)
            self.metrics.avg_response_time = total_time / len(self.operations_history)
            
            # Calculate rates
            self.metrics.calculate_rates()
    
    def get_current_metrics(self) -> CacheMetrics:
        """Get current cache metrics"""
        with self.lock:
            # Update Redis-specific metrics
            try:
                redis_stats = get_redis_stats()
                self.metrics.memory_usage = redis_stats.get('used_memory', 0)
                self.metrics.connected_clients = redis_stats.get('connected_clients', 0)
            except Exception as e:
                logger.warning(f"Failed to get Redis stats: {e}")
            
            return self.metrics
    
    def get_operation_stats(self, operation_type: str = None) -> Dict[str, Any]:
        """Get statistics for specific operation type"""
        with self.lock:
            if operation_type:
                times = list(self.operation_times[operation_type])
                if not times:
                    return {"count": 0, "avg_time": 0.0, "min_time": 0.0, "max_time": 0.0}
                
                return {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "slow_operations": len([t for t in times if t > self.slow_operation_threshold])
                }
            else:
                # Return stats for all operations
                stats = {}
                for op_type, times in self.operation_times.items():
                    if times:
                        stats[op_type] = {
                            "count": len(times),
                            "avg_time": sum(times) / len(times),
                            "min_time": min(times),
                            "max_time": max(times),
                            "slow_operations": len([t for t in times if t > self.slow_operation_threshold])
                        }
                return stats
    
    def get_recent_operations(self, limit: int = 50) -> List[CacheOperation]:
        """Get recent cache operations"""
        with self.lock:
            return list(self.operations_history)[-limit:]
    
    def get_slow_operations(self, threshold: float = None) -> List[CacheOperation]:
        """Get operations that exceeded the slow threshold"""
        threshold = threshold or self.slow_operation_threshold
        with self.lock:
            return [op for op in self.operations_history if op.duration > threshold]
    
    def get_error_operations(self) -> List[CacheOperation]:
        """Get failed operations"""
        with self.lock:
            return [op for op in self.operations_history if not op.success]
    
    def check_health(self) -> Dict[str, Any]:
        """Check cache health and return status"""
        metrics = self.get_current_metrics()
        
        health_status = {
            "status": "healthy",
            "issues": [],
            "metrics": metrics,
            "uptime": (datetime.utcnow() - self.start_time).total_seconds()
        }
        
        # Check for issues
        if metrics.error_rate > self.high_error_rate_threshold:
            health_status["status"] = "warning"
            health_status["issues"].append(f"High error rate: {metrics.error_rate:.2%}")
        
        if metrics.hit_rate < self.low_hit_rate_threshold and metrics.total_operations > 100:
            health_status["status"] = "warning"
            health_status["issues"].append(f"Low hit rate: {metrics.hit_rate:.2%}")
        
        if metrics.avg_response_time > self.slow_operation_threshold:
            health_status["status"] = "warning"
            health_status["issues"].append(f"Slow average response time: {metrics.avg_response_time:.3f}s")
        
        # Check Redis connection
        try:
            redis_client = get_redis_client()
            redis_client.ping()
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["issues"].append(f"Redis connection failed: {str(e)}")
        
        return health_status
    
    def reset_metrics(self):
        """Reset all metrics and history"""
        with self.lock:
            self.operations_history.clear()
            self.operation_times.clear()
            self.metrics = CacheMetrics()
            self.start_time = datetime.utcnow()
            logger.info("Cache metrics reset")
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external monitoring"""
        return {
            "current_metrics": self.get_current_metrics().__dict__,
            "operation_stats": self.get_operation_stats(),
            "health_check": self.check_health(),
            "recent_operations": [
                {
                    "type": op.operation_type,
                    "key": op.key,
                    "timestamp": op.timestamp.isoformat(),
                    "duration": op.duration,
                    "success": op.success,
                    "error": op.error_message
                }
                for op in self.get_recent_operations(20)
            ]
        }


# Global cache monitor instance
cache_monitor = CacheMonitor()


@contextmanager
def monitor_cache_operation(operation_type: str, key: str):
    """Context manager to monitor cache operations"""
    start_time = time.time()
    success = True
    error_message = None
    
    try:
        yield
    except Exception as e:
        success = False
        error_message = str(e)
        raise
    finally:
        duration = time.time() - start_time
        cache_monitor.record_operation(operation_type, key, duration, success, error_message)


class MonitoredCacheManager:
    """Cache manager wrapper with monitoring"""
    
    def __init__(self, cache_manager_instance=None):
        self.cache = cache_manager_instance or cache_manager
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set with monitoring"""
        with monitor_cache_operation("set", key):
            return self.cache.set(key, value, ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """Get with monitoring"""
        with monitor_cache_operation("get", key):
            result = self.cache.get(key)
            # Record as miss if result is None
            if result is None:
                raise Exception("Cache miss")
            return result
    
    def delete(self, key: str) -> bool:
        """Delete with monitoring"""
        with monitor_cache_operation("delete", key):
            return self.cache.delete(key)
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete pattern with monitoring"""
        with monitor_cache_operation("delete_pattern", pattern):
            return self.cache.delete_pattern(pattern)
    
    def exists(self, key: str) -> bool:
        """Exists check with monitoring"""
        with monitor_cache_operation("exists", key):
            return self.cache.exists(key)


# Global monitored cache manager
monitored_cache = MonitoredCacheManager()


def get_cache_health() -> Dict[str, Any]:
    """Get cache health status"""
    return cache_monitor.check_health()


def get_cache_metrics() -> Dict[str, Any]:
    """Get cache performance metrics"""
    return cache_monitor.export_metrics()


def reset_cache_monitoring():
    """Reset cache monitoring data"""
    cache_monitor.reset_metrics()


class CacheAlerts:
    """Cache alerting system"""
    
    def __init__(self, monitor: CacheMonitor = None):
        self.monitor = monitor or cache_monitor
        self.alert_callbacks = []
    
    def add_alert_callback(self, callback):
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def check_and_alert(self):
        """Check metrics and trigger alerts if needed"""
        health = self.monitor.check_health()
        
        if health["status"] != "healthy":
            alert_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "status": health["status"],
                "issues": health["issues"],
                "metrics": health["metrics"].__dict__ if hasattr(health["metrics"], '__dict__') else health["metrics"]
            }
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")


# Global cache alerts
cache_alerts = CacheAlerts()


def setup_cache_monitoring():
    """Setup cache monitoring with default configuration"""
    logger.info("Cache monitoring initialized")
    
    # Add default alert callback (logging)
    def log_alert(alert_data):
        logger.warning(f"Cache Alert: {alert_data['status']} - {alert_data['issues']}")
    
    cache_alerts.add_alert_callback(log_alert)
    
    return cache_monitor