#!/usr/bin/env python3
"""
Test script for InfluxDB setup and functionality.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.database.setup_influxdb import setup_influxdb, InfluxDBSetup
from src.database.influxdb_client import get_influxdb_manager
from src.database.time_series_ingestion import get_ingestion_pipeline
from src.database.time_series_utils import get_query_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_influxdb_functionality():
    """Test InfluxDB functionality end-to-end."""
    
    print("=" * 60)
    print("InfluxDB Setup and Functionality Test")
    print("=" * 60)
    
    try:
        # Test 1: Setup InfluxDB
        print("\n1. Testing InfluxDB Setup...")
        setup_success = await setup_influxdb()
        
        if setup_success:
            print("✅ InfluxDB setup completed successfully")
        else:
            print("❌ InfluxDB setup failed")
            return False
        
        # Test 2: Test connection
        print("\n2. Testing InfluxDB Connection...")
        manager = get_influxdb_manager()
        
        if manager.health_check():
            print("✅ InfluxDB connection healthy")
        else:
            print("❌ InfluxDB connection failed")
            return False
        
        # Test 3: Test data ingestion
        print("\n3. Testing Data Ingestion...")
        pipeline = get_ingestion_pipeline()
        
        # Create sample flight data
        sample_flight_data = [
            {
                "flight_id": "TEST001",
                "airline": "AI",
                "aircraft_type": "A320",
                "origin_airport": "BOM",
                "destination_airport": "DEL",
                "runway": "RW01",
                "scheduled_departure": datetime.now() - timedelta(minutes=30),
                "actual_departure": datetime.now() - timedelta(minutes=15),
                "delay_minutes": 15.0,
                "passenger_count": 150,
                "weather_condition": "clear"
            },
            {
                "flight_id": "TEST002",
                "airline": "6E",
                "aircraft_type": "A321",
                "origin_airport": "DEL",
                "destination_airport": "BOM",
                "runway": "RW02",
                "scheduled_departure": datetime.now() - timedelta(minutes=45),
                "actual_departure": datetime.now() - timedelta(minutes=40),
                "delay_minutes": 5.0,
                "passenger_count": 180,
                "weather_condition": "cloudy"
            }
        ]
        
        ingestion_success = await pipeline.ingest_flight_data_batch(
            flight_data=sample_flight_data,
            calculate_delays=True,
            include_congestion=True
        )
        
        if ingestion_success:
            print("✅ Data ingestion successful")
            stats = pipeline.get_ingestion_stats()
            print(f"   - Processed: {stats['total_processed']} records")
            print(f"   - Successful writes: {stats['successful_writes']}")
            print(f"   - Processing time: {stats['processing_time']:.2f}s")
        else:
            print("❌ Data ingestion failed")
            return False
        
        # Test 4: Test querying
        print("\n4. Testing Data Querying...")
        query_utils = get_query_utils()
        
        # Query delay statistics
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        delay_stats = await query_utils.get_delay_statistics(
            airport_code="BOM",
            start_time=start_time,
            end_time=end_time
        )
        
        print(f"✅ Delay statistics query completed")
        print(f"   - Records found: {delay_stats.record_count}")
        print(f"   - Execution time: {delay_stats.execution_time:.3f}s")
        
        if delay_stats.data:
            print(f"   - Sample data: {delay_stats.data[0]}")
        
        # Query congestion trends
        congestion_trends = await query_utils.get_congestion_trends(
            airport_code="BOM",
            start_time=start_time,
            end_time=end_time
        )
        
        print(f"✅ Congestion trends query completed")
        print(f"   - Records found: {congestion_trends.record_count}")
        print(f"   - Execution time: {congestion_trends.execution_time:.3f}s")
        
        # Test 5: Test performance metrics
        print("\n5. Testing Performance Metrics...")
        
        performance_success = await pipeline.ingest_performance_metrics(
            component="test_component",
            metrics_data={
                "response_time_ms": 45.2,
                "requests_per_second": 150.0,
                "error_rate": 0.02
            }
        )
        
        if performance_success:
            print("✅ Performance metrics ingestion successful")
        else:
            print("❌ Performance metrics ingestion failed")
        
        # Test 6: Verify setup
        print("\n6. Verifying Complete Setup...")
        setup = InfluxDBSetup()
        verification = await setup.verify_setup()
        
        print(f"✅ Setup verification completed")
        print(f"   - Connection healthy: {verification['connection_healthy']}")
        print(f"   - Bucket exists: {verification['bucket_exists']}")
        print(f"   - Retention policies: {verification['retention_policies']}")
        print(f"   - Sample data count: {verification['sample_data_count']}")
        print(f"   - Measurements: {len(verification['measurements'])}")
        
        if verification["errors"]:
            print(f"   - Errors: {verification['errors']}")
        
        for measurement in verification["measurements"]:
            print(f"     * {measurement['name']}: {measurement['record_count']} records")
        
        setup.close()
        
        print("\n" + "=" * 60)
        print("🎉 All InfluxDB tests completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Test execution failed")
        return False
    
    finally:
        # Clean up connections
        try:
            from src.database.influxdb_client import close_influxdb_connection
            close_influxdb_connection()
            print("\n🔌 InfluxDB connections closed")
        except Exception as e:
            logger.warning(f"Error closing connections: {e}")


async def test_basic_connection():
    """Test basic InfluxDB connection without full setup."""
    
    print("Testing basic InfluxDB connection...")
    
    try:
        manager = get_influxdb_manager()
        
        if manager.health_check():
            print("✅ Basic InfluxDB connection successful")
            
            # Get bucket info
            bucket_info = manager.get_bucket_info()
            if bucket_info:
                print(f"✅ Bucket info retrieved: {bucket_info['name']}")
            else:
                print("⚠️  Could not retrieve bucket info")
            
            return True
        else:
            print("❌ InfluxDB connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def main():
    """Main test function."""
    
    print("InfluxDB Test Script")
    print("Choose test mode:")
    print("1. Basic connection test")
    print("2. Full functionality test")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            success = asyncio.run(test_basic_connection())
        elif choice == "2":
            success = asyncio.run(test_influxdb_functionality())
        else:
            print("Invalid choice. Running basic connection test...")
            success = asyncio.run(test_basic_connection())
        
        if success:
            print("\n✅ Test completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Test failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()