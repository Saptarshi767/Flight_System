"""
Database performance tests for the Flight Scheduling Analysis System
"""
import pytest
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database.connection import db_session_scope
from src.database.models import Flight, Airport, Airline, Aircraft
from src.database.query_optimizer import query_optimizer
from src.utils.cache import cache_manager
from tests.conftest import sample_flight_data


class DatabasePerformanceTest:
    """Database performance testing utilities"""
    
    def __init__(self):
        self.performance_results = []
    
    def measure_query_time(self, query_func, iterations: int = 10) -> Dict[str, float]:
        """Measure query execution time over multiple iterations"""
        execution_times = []
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                result = query_func()
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            except Exception as e:
                pytest.fail(f"Query failed: {e}")
        
        return {
            'avg_time': statistics.mean(execution_times),
            'min_time': min(execution_times),
            'max_time': max(execution_times),
            'median_time': statistics.median(execution_times),
            'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        }
    
    def benchmark_flight_queries(self, session: Session) -> Dict[str, Any]:
        """Benchmark common flight data queries"""
        benchmarks = {}
        
        # Test 1: Simple flight lookup by ID
        def simple_lookup():
            return session.query(Flight).filter(Flight.flight_id == 'TEST001').first()
        
        benchmarks['simple_lookup'] = self.measure_query_time(simple_lookup)
        
        # Test 2: Route-based query
        def route_query():
            return session.query(Flight).filter(
                Flight.origin_airport == 'BOM',
                Flight.destination_airport == 'DEL'
            ).limit(100).all()
        
        benchmarks['route_query'] = self.measure_query_time(route_query)
        
        # Test 3: Date range query
        def date_range_query():
            start_date = datetime.now() - timedelta(days=7)
            end_date = datetime.now()
            return session.query(Flight).filter(
                Flight.scheduled_departure >= start_date,
                Flight.scheduled_departure <= end_date
            ).limit(100).all()
        
        benchmarks['date_range_query'] = self.measure_query_time(date_range_query)
        
        # Test 4: Complex join query
        def complex_join_query():
            return session.query(Flight, Airport, Airline).join(
                Airport, Flight.origin_airport == Airport.code
            ).join(
                Airline, Flight.airline_code == Airline.code
            ).limit(50).all()
        
        benchmarks['complex_join_query'] = self.measure_query_time(complex_join_query)
        
        # Test 5: Aggregation query
        def aggregation_query():
            return session.query(
                Flight.airline_code,
                Flight.origin_airport,
                func.avg(Flight.departure_delay_minutes),
                func.count(Flight.id)
            ).group_by(
                Flight.airline_code,
                Flight.origin_airport
            ).limit(20).all()
        
        benchmarks['aggregation_query'] = self.measure_query_time(aggregation_query)
        
        return benchmarks
    
    def benchmark_optimized_queries(self, session: Session) -> Dict[str, Any]:
        """Benchmark optimized queries using query optimizer"""
        benchmarks = {}
        
        # Test optimized delay analysis
        def optimized_delay_analysis():
            return query_optimizer.get_delay_analysis_data(session, 'BOM', 30)
        
        benchmarks['optimized_delay_analysis'] = self.measure_query_time(optimized_delay_analysis)
        
        # Test optimized congestion analysis
        def optimized_congestion_analysis():
            return query_optimizer.get_congestion_analysis_data(session, 'BOM', 30)
        
        benchmarks['optimized_congestion_analysis'] = self.measure_query_time(optimized_congestion_analysis)
        
        # Test optimized route analysis
        def optimized_route_analysis():
            return query_optimizer.get_route_performance_data(session, 'BOM', 'DEL', 90)
        
        benchmarks['optimized_route_analysis'] = self.measure_query_time(optimized_route_analysis)
        
        return benchmarks


@pytest.fixture
def performance_tester():
    """Fixture for database performance tester"""
    return DatabasePerformanceTest()


@pytest.fixture
def populated_database():
    """Fixture to populate database with test data"""
    with db_session_scope() as session:
        # Create test airports
        airports = [
            Airport(code='BOM', name='Mumbai Airport', city='Mumbai', country='India', 
                   runway_count=2, runway_capacity=30),
            Airport(code='DEL', name='Delhi Airport', city='Delhi', country='India',
                   runway_count=3, runway_capacity=45),
            Airport(code='BLR', name='Bangalore Airport', city='Bangalore', country='India',
                   runway_count=2, runway_capacity=25)
        ]
        
        for airport in airports:
            existing = session.query(Airport).filter(Airport.code == airport.code).first()
            if not existing:
                session.add(airport)
        
        # Create test airlines
        airlines = [
            Airline(code='AI', name='Air India', country='India'),
            Airline(code='6E', name='IndiGo', country='India'),
            Airline(code='SG', name='SpiceJet', country='India')
        ]
        
        for airline in airlines:
            existing = session.query(Airline).filter(Airline.code == airline.code).first()
            if not existing:
                session.add(airline)
        
        # Create test aircraft
        aircraft = [
            Aircraft(type_code='A320', manufacturer='Airbus', model='A320',
                    typical_seating=180, max_seating=200),
            Aircraft(type_code='B737', manufacturer='Boeing', model='737-800',
                    typical_seating=160, max_seating=189)
        ]
        
        for ac in aircraft:
            existing = session.query(Aircraft).filter(Aircraft.type_code == ac.type_code).first()
            if not existing:
                session.add(ac)
        
        session.commit()
        
        # Create test flights
        base_date = datetime.now() - timedelta(days=30)
        flights = []
        
        for i in range(1000):  # Create 1000 test flights
            flight_date = base_date + timedelta(hours=i * 0.5)  # Every 30 minutes
            
            flight = Flight(
                flight_id=f'TEST{i:04d}',
                flight_number=f'AI{100 + i % 900}',
                airline_code='AI',
                origin_airport='BOM' if i % 2 == 0 else 'DEL',
                destination_airport='DEL' if i % 2 == 0 else 'BOM',
                aircraft_type='A320' if i % 2 == 0 else 'B737',
                scheduled_departure=flight_date,
                scheduled_arrival=flight_date + timedelta(hours=2),
                actual_departure=flight_date + timedelta(minutes=i % 60),  # Variable delays
                actual_arrival=flight_date + timedelta(hours=2, minutes=i % 60),
                departure_delay_minutes=i % 60,
                arrival_delay_minutes=i % 60,
                delay_category='operational' if i % 3 == 0 else 'weather',
                status='completed',
                passenger_count=150 + (i % 50),
                data_source='test'
            )
            flights.append(flight)
        
        session.add_all(flights)
        session.commit()
        
        yield session


class TestDatabasePerformance:
    """Database performance test cases"""
    
    def test_basic_query_performance(self, performance_tester, populated_database):
        """Test basic query performance"""
        session = populated_database
        
        benchmarks = performance_tester.benchmark_flight_queries(session)
        
        # Assert performance thresholds
        assert benchmarks['simple_lookup']['avg_time'] < 0.1, "Simple lookup too slow"
        assert benchmarks['route_query']['avg_time'] < 0.5, "Route query too slow"
        assert benchmarks['date_range_query']['avg_time'] < 0.5, "Date range query too slow"
        assert benchmarks['complex_join_query']['avg_time'] < 1.0, "Complex join too slow"
        assert benchmarks['aggregation_query']['avg_time'] < 1.0, "Aggregation query too slow"
        
        print("\nBasic Query Performance Results:")
        for query_type, metrics in benchmarks.items():
            print(f"{query_type}: {metrics['avg_time']:.3f}s avg, {metrics['max_time']:.3f}s max")
    
    def test_optimized_query_performance(self, performance_tester, populated_database):
        """Test optimized query performance"""
        session = populated_database
        
        # First, optimize the database
        query_optimizer.optimize_flight_queries(session)
        
        benchmarks = performance_tester.benchmark_optimized_queries(session)
        
        # Assert performance thresholds for optimized queries
        assert benchmarks['optimized_delay_analysis']['avg_time'] < 2.0, "Optimized delay analysis too slow"
        assert benchmarks['optimized_congestion_analysis']['avg_time'] < 2.0, "Optimized congestion analysis too slow"
        assert benchmarks['optimized_route_analysis']['avg_time'] < 2.0, "Optimized route analysis too slow"
        
        print("\nOptimized Query Performance Results:")
        for query_type, metrics in benchmarks.items():
            print(f"{query_type}: {metrics['avg_time']:.3f}s avg, {metrics['max_time']:.3f}s max")
    
    def test_cache_performance(self, performance_tester, populated_database):
        """Test cache performance impact"""
        session = populated_database
        
        # Test without cache
        def uncached_query():
            return query_optimizer.get_delay_analysis_data.__wrapped__(
                query_optimizer, session, 'BOM', 30
            )
        
        uncached_metrics = performance_tester.measure_query_time(uncached_query, 5)
        
        # Test with cache (first call will populate cache)
        def cached_query():
            return query_optimizer.get_delay_analysis_data(session, 'BOM', 30)
        
        # First call to populate cache
        cached_query()
        
        # Now measure cached performance
        cached_metrics = performance_tester.measure_query_time(cached_query, 5)
        
        # Cache should be significantly faster
        cache_speedup = uncached_metrics['avg_time'] / cached_metrics['avg_time']
        assert cache_speedup > 2, f"Cache speedup insufficient: {cache_speedup:.2f}x"
        
        print(f"\nCache Performance:")
        print(f"Uncached: {uncached_metrics['avg_time']:.3f}s avg")
        print(f"Cached: {cached_metrics['avg_time']:.3f}s avg")
        print(f"Speedup: {cache_speedup:.2f}x")
    
    def test_connection_pool_performance(self, performance_tester):
        """Test connection pool performance"""
        
        def connection_test():
            with db_session_scope() as session:
                return session.execute(text("SELECT 1")).scalar()
        
        # Test connection acquisition performance
        connection_metrics = performance_tester.measure_query_time(connection_test, 20)
        
        # Connection acquisition should be fast
        assert connection_metrics['avg_time'] < 0.1, "Connection acquisition too slow"
        assert connection_metrics['max_time'] < 0.5, "Maximum connection time too high"
        
        print(f"\nConnection Pool Performance:")
        print(f"Avg connection time: {connection_metrics['avg_time']:.3f}s")
        print(f"Max connection time: {connection_metrics['max_time']:.3f}s")
    
    def test_bulk_insert_performance(self, performance_tester):
        """Test bulk insert performance"""
        
        def bulk_insert_test():
            with db_session_scope() as session:
                flights = []
                base_time = datetime.now()
                
                for i in range(100):
                    flight = Flight(
                        flight_id=f'BULK{i:04d}',
                        flight_number=f'BK{i:03d}',
                        airline_code='AI',
                        origin_airport='BOM',
                        destination_airport='DEL',
                        scheduled_departure=base_time + timedelta(hours=i),
                        scheduled_arrival=base_time + timedelta(hours=i+2),
                        status='scheduled',
                        data_source='bulk_test'
                    )
                    flights.append(flight)
                
                session.add_all(flights)
                session.commit()
                
                # Clean up
                session.query(Flight).filter(Flight.data_source == 'bulk_test').delete()
                session.commit()
        
        bulk_metrics = performance_tester.measure_query_time(bulk_insert_test, 3)
        
        # Bulk insert should complete within reasonable time
        assert bulk_metrics['avg_time'] < 5.0, "Bulk insert too slow"
        
        print(f"\nBulk Insert Performance:")
        print(f"100 records: {bulk_metrics['avg_time']:.3f}s avg")
        print(f"Rate: {100 / bulk_metrics['avg_time']:.1f} records/second")
    
    def test_database_statistics(self, populated_database):
        """Test database statistics collection"""
        session = populated_database
        
        stats = query_optimizer.get_database_statistics(session)
        
        assert 'table_statistics' in stats
        assert 'index_usage' in stats
        assert 'connection_stats' in stats
        
        print("\nDatabase Statistics:")
        print(f"Tables analyzed: {len(stats.get('table_statistics', []))}")
        print(f"Indexes tracked: {len(stats.get('index_usage', []))}")
        print(f"Connection states: {len(stats.get('connection_stats', []))}")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])