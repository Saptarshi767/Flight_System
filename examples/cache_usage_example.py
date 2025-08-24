"""
Example usage of the Redis caching layer for Flight Scheduling Analysis System
"""
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import cache utilities
from src.utils import (
    cache_manager, cached, ml_cache, cache_invalidator,
    flight_data_cache, analysis_cache, ml_model_cache,
    get_cache_health, get_cache_metrics, setup_cache_monitoring
)


def example_basic_cache_operations():
    """Demonstrate basic cache operations"""
    print("=== Basic Cache Operations ===")
    
    # Set a value in cache
    test_data = {
        "flight_id": "AI101",
        "departure": "2024-01-01T10:00:00",
        "arrival": "2024-01-01T12:30:00",
        "delay_minutes": 15
    }
    
    success = cache_manager.set("example_flight", test_data, ttl=300)
    print(f"Cache SET result: {success}")
    
    # Get value from cache
    cached_data = cache_manager.get("example_flight")
    print(f"Cache GET result: {cached_data}")
    
    # Check if key exists
    exists = cache_manager.exists("example_flight")
    print(f"Key exists: {exists}")
    
    # Get TTL
    ttl = cache_manager.ttl("example_flight")
    print(f"TTL remaining: {ttl} seconds")
    
    # Delete key
    deleted = cache_manager.delete("example_flight")
    print(f"Cache DELETE result: {deleted}")


def example_cached_decorator():
    """Demonstrate the @cached decorator"""
    print("\n=== Cached Decorator Example ===")
    
    @cached(ttl=300, key_prefix="flight_analysis")
    def expensive_flight_calculation(airport_code: str, date: str) -> Dict[str, Any]:
        """Simulate an expensive calculation"""
        print(f"Performing expensive calculation for {airport_code} on {date}")
        time.sleep(1)  # Simulate processing time
        
        return {
            "airport": airport_code,
            "date": date,
            "avg_delay": 18.5,
            "total_flights": 150,
            "calculated_at": datetime.now().isoformat()
        }
    
    # First call - will execute function
    start_time = time.time()
    result1 = expensive_flight_calculation("BOM", "2024-01-01")
    time1 = time.time() - start_time
    print(f"First call took {time1:.3f} seconds")
    print(f"Result: {result1}")
    
    # Second call - will use cache
    start_time = time.time()
    result2 = expensive_flight_calculation("BOM", "2024-01-01")
    time2 = time.time() - start_time
    print(f"Second call took {time2:.3f} seconds")
    print(f"Result: {result2}")
    
    print(f"Speed improvement: {time1/time2:.1f}x faster")


def example_flight_data_caching():
    """Demonstrate flight data caching strategy"""
    print("\n=== Flight Data Caching Strategy ===")
    
    # Sample flight data
    flight_data = [
        {
            "flight_id": "AI101",
            "airline": "Air India",
            "origin": "BOM",
            "destination": "DEL",
            "scheduled_departure": "2024-01-01T10:00:00",
            "actual_departure": "2024-01-01T10:15:00",
            "delay_minutes": 15
        },
        {
            "flight_id": "6E202",
            "airline": "IndiGo",
            "origin": "BOM",
            "destination": "BLR",
            "scheduled_departure": "2024-01-01T11:00:00",
            "actual_departure": "2024-01-01T11:05:00",
            "delay_minutes": 5
        }
    ]
    
    # Cache flight data
    success = flight_data_cache.cache_flight_data("BOM", "2024-01-01", flight_data)
    print(f"Flight data cached: {success}")
    
    # Retrieve flight data
    cached_flights = flight_data_cache.get_flight_data("BOM", "2024-01-01")
    print(f"Retrieved {len(cached_flights) if cached_flights else 0} flights from cache")
    
    # Cache aggregated data
    agg_data = {
        "total_flights": len(flight_data),
        "avg_delay": sum(f["delay_minutes"] for f in flight_data) / len(flight_data),
        "on_time_percentage": len([f for f in flight_data if f["delay_minutes"] <= 5]) / len(flight_data) * 100
    }
    
    success = flight_data_cache.cache_aggregated_data("BOM", "daily", "2024-01-01", agg_data)
    print(f"Aggregated data cached: {success}")
    
    # Retrieve aggregated data
    cached_agg = flight_data_cache.get_aggregated_data("BOM", "daily", "2024-01-01")
    print(f"Cached aggregated data: {cached_agg}")


def example_ml_model_caching():
    """Demonstrate ML model result caching"""
    print("\n=== ML Model Caching ===")
    
    # Simulate ML model prediction
    model_name = "delay_predictor"
    model_version = "v1.2"
    
    features = {
        "hour_of_day": 10,
        "day_of_week": 1,
        "weather_score": 0.8,
        "traffic_density": 0.6,
        "airline": "AI",
        "aircraft_type": "A320"
    }
    
    prediction_result = {
        "predicted_delay": 12.5,
        "confidence": 0.85,
        "probability_on_time": 0.72
    }
    
    # Cache ML prediction
    success = ml_cache.cache_model_result(model_name, model_version, 
                                        str(hash(str(features))), prediction_result)
    print(f"ML prediction cached: {success}")
    
    # Retrieve ML prediction
    cached_prediction = ml_cache.get_model_result(model_name, model_version, 
                                                str(hash(str(features))))
    print(f"Cached prediction: {cached_prediction}")
    
    # Cache analysis result
    analysis_result = {
        "analysis_type": "delay_pattern",
        "airport": "BOM",
        "peak_delay_hours": [8, 9, 18, 19, 20],
        "avg_delay_by_hour": {str(h): h * 2.5 for h in range(24)},
        "recommendations": [
            "Avoid scheduling during peak hours (8-9 AM, 6-8 PM)",
            "Consider alternative time slots with lower delay probability"
        ]
    }
    
    success = ml_cache.cache_analysis_result("delay_pattern", "BOM", "2024-01-01", analysis_result)
    print(f"Analysis result cached: {success}")


def example_cache_invalidation():
    """Demonstrate cache invalidation strategies"""
    print("\n=== Cache Invalidation ===")
    
    # Cache some test data first
    test_keys = []
    for i in range(5):
        key = f"test_flight_{i}"
        cache_manager.set(key, {"flight": f"AI{100+i}", "delay": i * 5}, 3600)
        test_keys.append(key)
    
    print(f"Cached {len(test_keys)} test entries")
    
    # Invalidate flight data cache for specific airport
    deleted = cache_invalidator.invalidate_flight_data_cache("BOM")
    print(f"Invalidated {deleted} flight data cache entries for BOM")
    
    # Invalidate ML model cache
    deleted = cache_invalidator.invalidate_model_cache("delay_predictor")
    print(f"Invalidated {deleted} ML model cache entries")
    
    # Clean up test data
    for key in test_keys:
        cache_manager.delete(key)
    print("Cleaned up test data")


def example_cache_monitoring():
    """Demonstrate cache monitoring and health checks"""
    print("\n=== Cache Monitoring ===")
    
    # Setup monitoring
    setup_cache_monitoring()
    
    # Perform some cache operations to generate metrics
    for i in range(10):
        cache_manager.set(f"monitor_test_{i}", {"value": i}, 300)
        cache_manager.get(f"monitor_test_{i}")
    
    # Get cache health
    health = get_cache_health()
    print(f"Cache health status: {health['status']}")
    if health['issues']:
        print(f"Issues: {health['issues']}")
    
    # Get cache metrics
    metrics = get_cache_metrics()
    current_metrics = metrics['current_metrics']
    print(f"Cache hit rate: {current_metrics['hit_rate']:.2%}")
    print(f"Total operations: {current_metrics['total_operations']}")
    print(f"Average response time: {current_metrics['avg_response_time']:.3f}s")
    
    # Clean up test data
    for i in range(10):
        cache_manager.delete(f"monitor_test_{i}")


def example_performance_comparison():
    """Compare performance with and without caching"""
    print("\n=== Performance Comparison ===")
    
    def simulate_database_query(query_id: str) -> Dict[str, Any]:
        """Simulate a slow database query"""
        time.sleep(0.1)  # Simulate 100ms database query
        return {
            "query_id": query_id,
            "results": [f"result_{i}" for i in range(10)],
            "timestamp": datetime.now().isoformat()
        }
    
    @cached(ttl=300, key_prefix="db_query")
    def cached_database_query(query_id: str) -> Dict[str, Any]:
        """Cached version of database query"""
        return simulate_database_query(query_id)
    
    # Test without caching
    start_time = time.time()
    for i in range(5):
        result = simulate_database_query(f"query_{i % 2}")  # Repeat queries
    uncached_time = time.time() - start_time
    
    # Test with caching
    start_time = time.time()
    for i in range(5):
        result = cached_database_query(f"query_{i % 2}")  # Repeat queries
    cached_time = time.time() - start_time
    
    print(f"Without caching: {uncached_time:.3f} seconds")
    print(f"With caching: {cached_time:.3f} seconds")
    print(f"Performance improvement: {uncached_time/cached_time:.1f}x faster")
    
    # Clean up
    cache_manager.delete_pattern("db_query:*")


def main():
    """Run all cache examples"""
    print("Flight Scheduling Analysis System - Redis Cache Examples")
    print("=" * 60)
    
    try:
        # Check if Redis is available
        if not cache_manager.is_connected():
            print("Error: Redis is not available. Please start Redis server.")
            return
        
        print("Redis connection successful!")
        
        # Run examples
        example_basic_cache_operations()
        example_cached_decorator()
        example_flight_data_caching()
        example_ml_model_caching()
        example_cache_invalidation()
        example_cache_monitoring()
        example_performance_comparison()
        
        print("\n" + "=" * 60)
        print("All cache examples completed successfully!")
        
    except Exception as e:
        print(f"Error running cache examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()