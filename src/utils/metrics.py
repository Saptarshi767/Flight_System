"""
Metrics collection and monitoring for Flight Scheduling Analysis System
"""

import time
import psutil
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import json
import os

from .logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    unit: str = ""

@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: datetime

@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    active_requests: int
    total_requests: int
    failed_requests: int
    avg_response_time: float
    database_connections: int
    cache_hit_rate: float
    queue_size: int
    worker_count: int
    timestamp: datetime

class MetricsCollector:
    """Centralized metrics collection system"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        self.running = False
        self.collection_thread = None
        
        # System monitoring
        self.process = psutil.Process()
        self.system_metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        
        # Application metrics
        self.app_metrics_history = deque(maxlen=1440)
        
        logger.info("MetricsCollector initialized")
    
    def start(self):
        """Start metrics collection"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_application_metrics()
                self._cleanup_old_metrics()
                time.sleep(60)  # Collect every minute
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}", exc_info=True)
                time.sleep(60)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                timestamp=datetime.utcnow()
            )
            
            with self.lock:
                self.system_metrics_history.append(metrics)
            
            # Update gauges
            self.set_gauge('system.cpu.percent', cpu_percent)
            self.set_gauge('system.memory.percent', memory.percent)
            self.set_gauge('system.memory.used_mb', memory.used / (1024 * 1024))
            self.set_gauge('system.disk.percent', disk.percent)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # Get current application metrics
            metrics = ApplicationMetrics(
                active_requests=self.get_gauge('app.requests.active'),
                total_requests=self.get_counter('app.requests.total'),
                failed_requests=self.get_counter('app.requests.failed'),
                avg_response_time=self.get_average_timer('app.response_time'),
                database_connections=self.get_gauge('app.database.connections'),
                cache_hit_rate=self.get_gauge('app.cache.hit_rate'),
                queue_size=self.get_gauge('app.queue.size'),
                worker_count=self.get_gauge('app.workers.count'),
                timestamp=datetime.utcnow()
            )
            
            with self.lock:
                self.app_metrics_history.append(metrics)
                
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    def _cleanup_old_metrics(self):
        """Remove old metrics beyond retention period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        with self.lock:
            # Clean up metric points
            for metric_name, points in self.metrics.items():
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
            
            # Clean up histograms and timers (keep only recent data)
            for metric_name in list(self.histograms.keys()):
                if len(self.histograms[metric_name]) > 1000:
                    self.histograms[metric_name] = self.histograms[metric_name][-1000:]
            
            for metric_name in list(self.timers.keys()):
                if len(self.timers[metric_name]) > 1000:
                    self.timers[metric_name] = self.timers[metric_name][-1000:]
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self.lock:
            self.counters[name] += value
            
        self.record_metric(name, value, tags or {}, "count")
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        with self.lock:
            self.gauges[name] = value
            
        self.record_metric(name, value, tags or {}, "gauge")
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        with self.lock:
            self.histograms[name].append(value)
            
        self.record_metric(name, value, tags or {}, "histogram")
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer value"""
        with self.lock:
            self.timers[name].append(duration)
            
        self.record_metric(name, duration, tags or {}, "timer")
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str], unit: str = ""):
        """Record a generic metric point"""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags,
            unit=unit
        )
        
        with self.lock:
            self.metrics[name].append(point)
    
    def get_counter(self, name: str) -> int:
        """Get counter value"""
        with self.lock:
            return self.counters.get(name, 0)
    
    def get_gauge(self, name: str) -> float:
        """Get gauge value"""
        with self.lock:
            return self.gauges.get(name, 0.0)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics"""
        with self.lock:
            values = self.histograms.get(name, [])
            
        if not values:
            return {}
        
        values_sorted = sorted(values)
        count = len(values)
        
        return {
            'count': count,
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / count,
            'p50': values_sorted[int(count * 0.5)],
            'p90': values_sorted[int(count * 0.9)],
            'p95': values_sorted[int(count * 0.95)],
            'p99': values_sorted[int(count * 0.99)]
        }
    
    def get_average_timer(self, name: str) -> float:
        """Get average timer value"""
        with self.lock:
            values = self.timers.get(name, [])
            
        if not values:
            return 0.0
        
        return sum(values) / len(values)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self.lock:
            system_metrics = list(self.system_metrics_history)[-1] if self.system_metrics_history else None
            app_metrics = list(self.app_metrics_history)[-1] if self.app_metrics_history else None
            
            counters = dict(self.counters)
            gauges = dict(self.gauges)
        
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_metrics': asdict(system_metrics) if system_metrics else None,
            'application_metrics': asdict(app_metrics) if app_metrics else None,
            'counters': counters,
            'gauges': gauges,
            'histograms': {name: self.get_histogram_stats(name) for name in self.histograms.keys()},
            'timers': {name: self.get_average_timer(name) for name in self.timers.keys()}
        }
        
        return summary
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        timestamp = int(time.time() * 1000)
        
        with self.lock:
            # Export counters
            for name, value in self.counters.items():
                prometheus_name = name.replace('.', '_').replace('-', '_')
                lines.append(f'# TYPE {prometheus_name} counter')
                lines.append(f'{prometheus_name} {value} {timestamp}')
            
            # Export gauges
            for name, value in self.gauges.items():
                prometheus_name = name.replace('.', '_').replace('-', '_')
                lines.append(f'# TYPE {prometheus_name} gauge')
                lines.append(f'{prometheus_name} {value} {timestamp}')
            
            # Export histogram summaries
            for name, values in self.histograms.items():
                if values:
                    prometheus_name = name.replace('.', '_').replace('-', '_')
                    stats = self.get_histogram_stats(name)
                    
                    lines.append(f'# TYPE {prometheus_name} histogram')
                    lines.append(f'{prometheus_name}_count {stats["count"]} {timestamp}')
                    lines.append(f'{prometheus_name}_sum {sum(values)} {timestamp}')
                    
                    for percentile in ['p50', 'p90', 'p95', 'p99']:
                        if percentile in stats:
                            lines.append(f'{prometheus_name}{{quantile="{percentile[1:]}"}} {stats[percentile]} {timestamp}')
        
        return '\n'.join(lines)

# Global metrics collector instance
metrics_collector = MetricsCollector()

# Convenience functions
def increment_counter(name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
    """Increment a counter metric"""
    metrics_collector.increment_counter(name, value, tags)

def set_gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Set a gauge metric"""
    metrics_collector.set_gauge(name, value, tags)

def record_histogram(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Record a histogram value"""
    metrics_collector.record_histogram(name, value, tags)

def record_timer(name: str, duration: float, tags: Optional[Dict[str, str]] = None):
    """Record a timer value"""
    metrics_collector.record_timer(name, duration, tags)

def time_function(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to time function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                record_timer(metric_name, duration, tags)
                return result
            except Exception as e:
                duration = time.time() - start_time
                error_tags = (tags or {}).copy()
                error_tags['error'] = type(e).__name__
                record_timer(f"{metric_name}.error", duration, error_tags)
                raise
        return wrapper
    return decorator

def time_async_function(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to time async function execution"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                record_timer(metric_name, duration, tags)
                return result
            except Exception as e:
                duration = time.time() - start_time
                error_tags = (tags or {}).copy()
                error_tags['error'] = type(e).__name__
                record_timer(f"{metric_name}.error", duration, error_tags)
                raise
        return wrapper
    return decorator

class MetricsMiddleware:
    """Middleware for automatic request metrics collection"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        method = scope.get('method', 'UNKNOWN')
        path = scope.get('path', 'unknown')
        
        # Increment active requests
        increment_counter('app.requests.active', 1)
        increment_counter('app.requests.total', 1)
        
        status_code = 200
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            increment_counter('app.requests.failed', 1)
            increment_counter('app.requests.active', -1)
            
            # Record error metrics
            record_timer('app.response_time.error', time.time() - start_time, {
                'method': method,
                'path': path,
                'error': type(e).__name__
            })
            raise
        else:
            # Record success metrics
            duration = time.time() - start_time
            record_timer('app.response_time', duration, {
                'method': method,
                'path': path,
                'status_code': str(status_code)
            })
            
            # Record status code metrics
            increment_counter(f'app.requests.status.{status_code}', 1)
            
        finally:
            increment_counter('app.requests.active', -1)

def init_metrics():
    """Initialize metrics collection"""
    metrics_collector.start()
    logger.info("Metrics collection initialized")

def shutdown_metrics():
    """Shutdown metrics collection"""
    metrics_collector.stop()
    logger.info("Metrics collection shutdown")

# Health check metrics
def record_health_check(service: str, healthy: bool, response_time: float):
    """Record health check metrics"""
    tags = {'service': service, 'status': 'healthy' if healthy else 'unhealthy'}
    increment_counter('app.health_checks.total', 1, tags)
    record_timer('app.health_checks.response_time', response_time, tags)
    
    if healthy:
        increment_counter('app.health_checks.success', 1, tags)
    else:
        increment_counter('app.health_checks.failed', 1, tags)

# Database metrics
def record_database_query(query_type: str, duration: float, success: bool = True):
    """Record database query metrics"""
    tags = {'query_type': query_type, 'status': 'success' if success else 'error'}
    record_timer('app.database.query_time', duration, tags)
    increment_counter('app.database.queries.total', 1, tags)
    
    if not success:
        increment_counter('app.database.queries.failed', 1, tags)

# Cache metrics
def record_cache_operation(operation: str, hit: bool):
    """Record cache operation metrics"""
    tags = {'operation': operation, 'result': 'hit' if hit else 'miss'}
    increment_counter('app.cache.operations.total', 1, tags)
    
    if hit:
        increment_counter('app.cache.hits', 1, tags)
    else:
        increment_counter('app.cache.misses', 1, tags)
    
    # Update hit rate
    total_ops = metrics_collector.get_counter('app.cache.operations.total')
    hits = metrics_collector.get_counter('app.cache.hits')
    hit_rate = (hits / total_ops * 100) if total_ops > 0 else 0
    set_gauge('app.cache.hit_rate', hit_rate)

# ML model metrics
def record_model_prediction(model_name: str, duration: float, success: bool = True):
    """Record ML model prediction metrics"""
    tags = {'model': model_name, 'status': 'success' if success else 'error'}
    record_timer('app.ml.prediction_time', duration, tags)
    increment_counter('app.ml.predictions.total', 1, tags)
    
    if not success:
        increment_counter('app.ml.predictions.failed', 1, tags)