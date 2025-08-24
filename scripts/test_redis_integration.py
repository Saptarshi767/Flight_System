"""
Manual integration test for Redis caching functionality
Run this script when Redis is available to test the complete caching system
"""
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.cache import cache_manager, cached, ml_cache, cache_invalidator
from utils.cache_strategies import flight_data_cache, analysis_cache
from utils.cache_monitor import setup_cache_monitoring, get_cache_health, get_cache_metrics
from database.redis_client import check_redis_connection


def test_redis_connection():
    """Test Redis connection"""
    print("Testing Redis connection...")
    try:
        if check_redis_connection():
            print("✅ Redis connection successful")
            return True
        else:
            print("❌ Redis connection failed")
            return False
    except Exception as e:
        print(f"❌ Redis connection error: {e}")
        return False


def test_basic_cache_operations():
    """Test basic cache operations"""
    print("\nTesting basic cache operations...")
    
    try:
        # Test set
        test_data = {"flight": "AI101", "delay": 15, "timestamp": datetime.now().isoformat()}
        success = cache_manager.set("test_flight", test_data, 300)
        print(f"Cache SET: {'✅' if success else '❌'}")
        
        # Test get
        retrieved_data = cache_manager.get("test_flight")
        if retrieved_data == test_data:
            print("Cache GET: ✅")
        else:
            print(f"Cache GET: ❌ (Expected: {test_data}, Got: {retrieved_data})")
        
        # Test exists
        exists = cache_manager.exists("test_flight")
        print(f"Cache EXISTS: {'✅' if exists else '❌'}")
        
        # Test TTL
        ttl = cache_manager.ttl("test_flight")
        print(f"Cache TTL: {'✅' if ttl > 0 else '❌'} ({ttl} seconds)")
        
        # Test delete
        deleted = cache_manager.delete("test_flight")
        print(f"Cache DELETE: {'✅' if deleted else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic cache operations failed: {e}")
        return False


def test_cached_decorator():
    """Test @cached decorator"""
    print("\nTesting @cached decorator...")
    
    try:
        @cached(ttl=300, key_prefix="test")
        def expensive_calculation(x, y):
            time.sleep(0.1)  # Simulate expensive operation
            return {"result": x + y, "timestamp": datetime.now().isoformat()}
        
        # First call (cache miss)
        start_time = time.time()
        result1 = expensive_calculation(5, 10)
        time1 = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = expensive_calculation(5, 10)
        time2 = time.time() - start_time
        
        if result1 == result2 and time2 < time1 / 2:
            print("✅ Cached decorator working correctly")
            print(f"   First call: {time1:.3f}s, Second call: {time2:.3f}s")
            return True
        else:
            print("❌ Cached decorator not working correctly")
            return False
            
    except Exception as e:
        print(f"❌ Cached decorator test failed: {e}")
        return False


def test_flight_data_caching():
    """Test flight data caching strategy"""
    print("\nTesting flight data caching strategy...")
    
    try:
        # Sample flight data
        flight_data = [
            {
                "flight_id": "AI101",
                "airline": "Air India",
                "origin": "BOM",
                "destination": "DEL",
                "delay_minutes": 15
            },
            {
                "flight_id": "6E202",
                "airline": "IndiGo",
                "origin": "BOM",
                "destination": "BLR",
                "delay_minutes": 5
            }
        ]
        
        # Cache flight data
        success = flight_data_cache.cache_flight_data("BOM", "2024-01-01", flight_data)
        if not success:
            print("❌ Failed to cache flight data")
            return False
        
        # Retrieve flight data
        cached_flights = flight_data_cache.get_flight_data("BOM", "2024-01-01")
        if cached_flights == flight_data:
            print("✅ Flight data caching working correctly")
        else:
            print("❌ Flight data caching failed")
            return False
        
        # Test aggregated data
        agg_data = {
            "total_flights": len(flight_data),
            "avg_delay": sum(f["delay_minutes"] for f in flight_data) / len(flight_data)
        }
        
        success = flight_data_cache.cache_aggregated_data("BOM", "daily", "2024-01-01", agg_data)
        cached_agg = flight_data_cache.get_aggregated_data("BOM", "daily", "2024-01-01")
        
        if success and cached_agg == agg_data:
            print("✅ Aggregated data caching working correctly")
            return True
        else:
            print("❌ Aggregated data caching failed")
            return False
            
    except Exception as e:
        print(f"❌ Flight data caching test failed: {e}")
        return False


def test_ml_model_caching():
    """Test ML model caching"""
    print("\nTesting ML model caching...")
    
    try:
        # Test ML model result caching
        model_name = "delay_predictor"
        model_version = "v1.0"
        input_hash = "test_hash_123"
        
        prediction_result = {
            "predicted_delay": 12.5,
            "confidence": 0.85,
            "model_version": model_version
        }
        
        # Cache ML result
        success = ml_cache.cache_model_result(model_name, model_version, input_hash, prediction_result)
        if not success:
            print("❌ Failed to cache ML model result")
            return False
        
        # Retrieve ML result
        cached_result = ml_cache.get_model_result(model_name, model_version, input_hash)
        if cached_result == prediction_result:
            print("✅ ML model caching working correctly")
            return True
        else:
            print("❌ ML model caching failed")
            return False
            
    except Exception as e:
        print(f"❌ ML model caching test failed: {e}")
        return False


def test_cache_invalidation():
    """Test cache invalidation"""
    print("\nTesting cache invalidation...")
    
    try:
        # Cache some test data
        for i in range(5):
            cache_manager.set(f"test_invalidation_{i}", {"data": f"value_{i}"}, 3600)
        
        # Test pattern deletion
        deleted = cache_manager.delete_pattern("test_invalidation_*")
        if deleted >= 5:
            print("✅ Cache invalidation working correctly")
            return True
        else:
            print(f"❌ Cache invalidation failed (deleted {deleted} items)")
            return False
            
    except Exception as e:
        print(f"❌ Cache invalidation test failed: {e}")
        return False


def test_cache_monitoring():
    """Test cache monitoring"""
    print("\nTesting cache monitoring...")
    
    try:
        # Setup monitoring
        setup_cache_monitoring()
        
        # Perform some operations to generate metrics
        for i in range(10):
            cache_manager.set(f"monitor_test_{i}", {"value": i}, 300)
            cache_manager.get(f"monitor_test_{i}")
        
        # Get health status
        health = get_cache_health()
        print(f"Cache health status: {health['status']}")
        
        # Get metrics
        metrics = get_cache_metrics()
        current_metrics = metrics['current_metrics']
        print(f"Total operations: {current_metrics['total_operations']}")
        print(f"Hit rate: {current_metrics['hit_rate']:.2%}")
        
        # Cleanup
        for i in range(10):
            cache_manager.delete(f"monitor_test_{i}")
        
        if health['status'] in ['healthy', 'warning']:
            print("✅ Cache monitoring working correctly")
            return True
        else:
            print("❌ Cache monitoring failed")
            return False
            
    except Exception as e:
        print(f"❌ Cache monitoring test failed: {e}")
        return False


def test_performance():
    """Test cache performance"""
    print("\nTesting cache performance...")
    
    try:
        # Test bulk operations
        start_time = time.time()
        for i in range(100):
            cache_manager.set(f"perf_test_{i}", {"data": f"value_{i}"}, 300)
        set_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(100):
            cache_manager.get(f"perf_test_{i}")
        get_time = time.time() - start_time
        
        print(f"Bulk SET (100 items): {set_time:.3f}s ({100/set_time:.0f} ops/sec)")
        print(f"Bulk GET (100 items): {get_time:.3f}s ({100/get_time:.0f} ops/sec)")
        
        # Cleanup
        cache_manager.delete_pattern("perf_test_*")
        
        if set_time < 1.0 and get_time < 1.0:
            print("✅ Cache performance acceptable")
            return True
        else:
            print("❌ Cache performance too slow")
            return False
            
    except Exception as e:
        print(f"❌ Cache performance test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("Redis Cache Integration Tests")
    print("=" * 50)
    
    # Check Redis connection first
    if not test_redis_connection():
        print("\n❌ Redis is not available. Please start Redis server:")
        print("   docker-compose up -d redis")
        print("   OR")
        print("   redis-server")
        return False
    
    # Run all tests
    tests = [
        test_basic_cache_operations,
        test_cached_decorator,
        test_flight_data_caching,
        test_ml_model_caching,
        test_cache_invalidation,
        test_cache_monitoring,
        test_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Redis cache integration tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)