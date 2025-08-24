# Redis Caching Layer Documentation

## Overview

The Flight Scheduling Analysis System includes a comprehensive Redis caching layer designed to improve performance by caching frequently accessed data, ML model results, and analysis outputs. The caching system is built with multiple strategies, monitoring capabilities, and intelligent invalidation logic.

## Architecture

### Core Components

1. **CacheManager** - Core Redis connection and operation management
2. **Cache Strategies** - Specialized caching patterns for different data types
3. **Cache Monitor** - Performance monitoring and health checks
4. **Cache Configuration** - Flexible configuration management
5. **Cache Invalidation** - Intelligent cache invalidation logic

### Key Features

- **Connection Pooling** - Efficient Redis connection management
- **Serialization** - Automatic JSON/Pickle serialization for complex data types
- **TTL Management** - Configurable time-to-live for different cache types
- **Monitoring** - Real-time performance metrics and health checks
- **Error Handling** - Graceful degradation when Redis is unavailable
- **Invalidation** - Smart cache invalidation on data updates

## Usage Examples

### Basic Cache Operations

```python
from src.utils import cache_manager

# Set a value with TTL
cache_manager.set("flight_data", {"flight": "AI101", "delay": 15}, ttl=1800)

# Get a value
data = cache_manager.get("flight_data")

# Check if key exists
exists = cache_manager.exists("flight_data")

# Delete a key
cache_manager.delete("flight_data")

# Delete multiple keys by pattern
cache_manager.delete_pattern("flight_data:*")
```

### Using the @cached Decorator

```python
from src.utils import cached

@cached(ttl=3600, key_prefix="analysis")
def expensive_analysis(airport_code, date):
    # Expensive computation here
    return {"avg_delay": 15.5, "total_flights": 120}

# First call executes function and caches result
result = expensive_analysis("BOM", "2024-01-01")

# Second call returns cached result (much faster)
result = expensive_analysis("BOM", "2024-01-01")
```

### Flight Data Caching Strategy

```python
from src.utils import flight_data_cache

# Cache flight data for an airport and date
flight_data = [
    {"flight_id": "AI101", "delay": 15},
    {"flight_id": "6E202", "delay": 5}
]

flight_data_cache.cache_flight_data("BOM", "2024-01-01", flight_data)

# Retrieve cached flight data
cached_flights = flight_data_cache.get_flight_data("BOM", "2024-01-01")

# Cache aggregated statistics
agg_data = {"avg_delay": 10, "total_flights": 150}
flight_data_cache.cache_aggregated_data("BOM", "daily", "2024-01-01", agg_data)
```

### ML Model Result Caching

```python
from src.utils import ml_cache

# Cache ML model prediction
prediction = {"delay": 12.5, "confidence": 0.85}
ml_cache.cache_model_result("delay_predictor", "v1.0", "input_hash", prediction)

# Retrieve cached prediction
cached_prediction = ml_cache.get_model_result("delay_predictor", "v1.0", "input_hash")

# Cache analysis results
analysis_result = {"peak_hours": [8, 9, 18, 19], "avg_delay": 18.5}
ml_cache.cache_analysis_result("congestion", "BOM", "2024-01-01", analysis_result)
```

### Cache Monitoring

```python
from src.utils import get_cache_health, get_cache_metrics, setup_cache_monitoring

# Setup monitoring
setup_cache_monitoring()

# Check cache health
health = get_cache_health()
print(f"Status: {health['status']}")
print(f"Issues: {health['issues']}")

# Get performance metrics
metrics = get_cache_metrics()
current = metrics['current_metrics']
print(f"Hit rate: {current['hit_rate']:.2%}")
print(f"Average response time: {current['avg_response_time']:.3f}s")
```

### Cache Invalidation

```python
from src.utils import cache_invalidator

# Invalidate flight data cache when data is updated
cache_invalidator.invalidate_flight_data_cache("BOM")

# Invalidate ML model cache when models are retrained
cache_invalidator.invalidate_model_cache("delay_predictor")

# Invalidate schedule-related cache
cache_invalidator.invalidate_schedule_cache("BOM")
```

## Configuration

### Environment-Specific Configurations

```python
from src.utils import load_environment_config

# Load development configuration (short TTLs, full monitoring)
config = load_environment_config("development")

# Load production configuration (longer TTLs, sampled monitoring)
config = load_environment_config("production")

# Load testing configuration (very short TTLs, no monitoring)
config = load_environment_config("testing")
```

### Custom Configuration

```python
from src.utils import CacheConfig, update_cache_config

# Create custom configuration
config = CacheConfig(
    default_ttl=7200,  # 2 hours
    max_connections=30,
    enable_compression=True,
    monitoring_sample_rate=0.5  # Monitor 50% of operations
)

# Update global configuration
update_cache_config(
    default_ttl=3600,
    enable_monitoring=True
)
```

### TTL Settings by Data Type

The system includes predefined TTL settings for different types of cached data:

- **Flight Data**: 30 minutes (real-time), 24 hours (historical)
- **Analysis Results**: 1 hour
- **ML Predictions**: 2 hours
- **NLP Queries**: 30 minutes (regular), 1 hour (popular)
- **Reports**: 2 hours
- **Charts**: 1 hour

## Cache Strategies

### 1. Flight Data Strategy

Optimized for caching flight information with different TTLs for real-time vs historical data.

```python
# Real-time flight data (30 min TTL)
flight_data_cache.cache_flight_data("BOM", "2024-01-01", data, is_historical=False)

# Historical flight data (24 hour TTL)
flight_data_cache.cache_flight_data("BOM", "2023-01-01", data, is_historical=True)
```

### 2. Analysis Strategy

Caches analysis results with parameter-based keys to handle different analysis configurations.

```python
# Cache delay analysis with specific parameters
params = {"time_window": "1h", "threshold": 15}
analysis_cache.cache_delay_analysis("BOM", "2024-01-01", params, results)

# Cache congestion analysis
analysis_cache.cache_congestion_analysis("BOM", "2024-01-01", "peak_hours", results)
```

### 3. ML Model Strategy

Specialized for caching machine learning model results and features.

```python
# Cache model prediction with features
features = {"hour": 10, "weather": 0.8, "traffic": 0.6}
ml_model_cache.cache_model_prediction("delay_model", "v1.0", features, prediction)

# Cache engineered features
ml_model_cache.cache_feature_engineering("flight_features", "data_hash", features)
```

### 4. Query Strategy

Optimized for natural language query results with popularity-based TTLs.

```python
# Cache regular query (30 min TTL)
query_cache.cache_query_result(query, context_hash, result, is_popular=False)

# Cache popular query (1 hour TTL)
query_cache.cache_query_result(query, context_hash, result, is_popular=True)
```

## Monitoring and Health Checks

### Health Check Endpoints

The cache system provides health check functionality that can be integrated into monitoring systems:

```python
health = get_cache_health()
# Returns:
# {
#   "status": "healthy|warning|unhealthy",
#   "issues": ["list of issues"],
#   "metrics": {...},
#   "uptime": 3600
# }
```

### Performance Metrics

```python
metrics = get_cache_metrics()
# Returns comprehensive metrics including:
# - Hit/miss rates
# - Response times
# - Operation counts
# - Memory usage
# - Recent operations
```

### Alerting

```python
from src.utils import cache_alerts

# Add custom alert callback
def custom_alert_handler(alert_data):
    # Send alert to monitoring system
    send_to_monitoring_system(alert_data)

cache_alerts.add_alert_callback(custom_alert_handler)
```

## Performance Considerations

### Connection Pooling

The cache manager uses Redis connection pooling to handle concurrent requests efficiently:

```python
# Configured in CacheManager
connection_pool = redis.ConnectionPool.from_url(
    redis_url,
    max_connections=20,
    retry_on_timeout=True,
    socket_keepalive=True
)
```

### Serialization

The system automatically chooses the best serialization method:

- **JSON**: For simple data types (strings, numbers, lists, dicts)
- **Pickle**: For complex objects (DataFrames, custom classes)

### Memory Management

- Configurable max value size (default: 1MB)
- Optional compression for large values
- TTL-based automatic cleanup

## Error Handling

The caching system is designed to fail gracefully:

```python
# Cache operations return False on failure
success = cache_manager.set("key", "value")
if not success:
    # Handle cache failure, continue without caching
    pass

# Get operations return None on failure
data = cache_manager.get("key")
if data is None:
    # Cache miss or error, fetch from primary source
    data = fetch_from_database()
```

## Testing

### Unit Tests

Run unit tests that don't require Redis:

```bash
python -m pytest tests/test_cache_unit.py -v
```

### Integration Tests

Run integration tests with Redis server:

```bash
# Start Redis first
docker-compose up -d redis

# Run integration tests
python scripts/test_redis_integration.py
```

### Performance Tests

The test suite includes performance benchmarks:

```python
# Bulk operations test
def test_cache_performance_bulk_operations():
    # Tests 100 SET and GET operations
    # Should complete in < 1 second
```

## Deployment

### Docker Configuration

Redis is configured in `docker-compose.yml`:

```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Environment Variables

Configure Redis connection:

```bash
REDIS_URL=redis://localhost:6379/0
```

### Production Considerations

1. **Redis Persistence**: Configure RDB/AOF persistence
2. **Memory Limits**: Set appropriate maxmemory policies
3. **Monitoring**: Integrate with monitoring systems
4. **Security**: Use Redis AUTH and network security
5. **Clustering**: Consider Redis Cluster for high availability

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```
   Error 10061 connecting to localhost:6379
   ```
   - Solution: Start Redis server

2. **Memory Issues**
   ```
   OOM command not allowed when used memory > 'maxmemory'
   ```
   - Solution: Increase Redis memory or configure eviction policy

3. **Slow Performance**
   - Check network latency to Redis
   - Monitor Redis memory usage
   - Review TTL settings

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('src.utils.cache').setLevel(logging.DEBUG)
```

### Cache Statistics

Monitor cache performance:

```python
stats = cache_manager.get_stats()
print(f"Memory usage: {stats['used_memory_human']}")
print(f"Hit rate: {stats['keyspace_hits'] / (stats['keyspace_hits'] + stats['keyspace_misses']):.2%}")
```

## Best Practices

1. **Use Appropriate TTLs**: Set TTLs based on data freshness requirements
2. **Monitor Performance**: Regularly check hit rates and response times
3. **Handle Failures**: Always have fallback logic for cache failures
4. **Invalidate Smartly**: Use targeted invalidation instead of clearing all cache
5. **Test Thoroughly**: Test both with and without Redis available
6. **Size Limits**: Be mindful of value sizes and memory usage
7. **Key Naming**: Use consistent, descriptive key patterns

## API Reference

### Core Classes

- `CacheManager`: Main cache operations
- `MLModelCache`: ML-specific caching
- `CacheInvalidator`: Cache invalidation logic
- `CacheMonitor`: Performance monitoring
- `CacheConfig`: Configuration management

### Decorators

- `@cached(ttl, key_prefix)`: Function result caching

### Strategy Classes

- `FlightDataCacheStrategy`: Flight data caching
- `AnalysisCacheStrategy`: Analysis result caching
- `MLModelCacheStrategy`: ML model caching
- `QueryCacheStrategy`: NLP query caching
- `ReportCacheStrategy`: Report caching

### Utility Functions

- `get_cache_health()`: Health status
- `get_cache_metrics()`: Performance metrics
- `setup_cache_monitoring()`: Initialize monitoring
- `load_environment_config()`: Load environment config