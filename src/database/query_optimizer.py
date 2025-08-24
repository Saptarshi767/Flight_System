"""
Database query optimization utilities for the Flight Scheduling Analysis System
"""
from typing import Dict, List, Any, Optional, Tuple, Union
from sqlalchemy import text, func, and_, or_, Index
from sqlalchemy.orm import Session, Query, joinedload, selectinload
from sqlalchemy.sql import select
from datetime import datetime, timedelta
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

from src.database.models import Flight, Airport, Airline, Aircraft, AnalysisResult
from src.utils.cache import cache_manager, cached
from src.utils.logging import logger


@dataclass
class QueryPerformanceMetrics:
    """Query performance metrics"""
    query_hash: str
    execution_time: float
    rows_returned: int
    cache_hit: bool
    timestamp: datetime


class QueryOptimizer:
    """Database query optimization and performance monitoring"""
    
    def __init__(self):
        self.performance_metrics: List[QueryPerformanceMetrics] = []
        self.slow_query_threshold = 1.0  # seconds
        self.logger = logger
    
    @contextmanager
    def monitor_query(self, query_description: str):
        """Context manager to monitor query performance"""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            if execution_time > self.slow_query_threshold:
                self.logger.warning(
                    f"Slow query detected: {query_description} took {execution_time:.2f}s"
                )
    
    def optimize_flight_queries(self, session: Session) -> Dict[str, Any]:
        """Optimize common flight data queries"""
        optimizations = {}
        
        # Create composite indexes for common query patterns
        try:
            # Index for route and date queries
            session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_flights_route_date_optimized 
                ON flights (origin_airport, destination_airport, scheduled_departure DESC)
            """))
            
            # Index for delay analysis queries
            session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_flights_delay_analysis_optimized 
                ON flights (departure_delay_minutes, delay_category, scheduled_departure DESC)
                WHERE departure_delay_minutes IS NOT NULL
            """))
            
            # Index for airline performance queries
            session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_flights_airline_performance 
                ON flights (airline_code, scheduled_departure DESC, departure_delay_minutes)
            """))
            
            # Index for airport congestion queries
            session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_flights_airport_congestion 
                ON flights (origin_airport, scheduled_departure)
                INCLUDE (destination_airport, departure_delay_minutes)
            """))
            
            # Partial index for recent flights (last 30 days)
            session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_flights_recent 
                ON flights (scheduled_departure DESC, origin_airport, destination_airport)
                WHERE scheduled_departure >= CURRENT_DATE - INTERVAL '30 days'
            """))
            
            session.commit()
            optimizations['indexes_created'] = True
            self.logger.info("Database indexes optimized successfully")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error creating optimized indexes: {e}")
            optimizations['indexes_created'] = False
        
        return optimizations
    
    def get_optimized_flight_query(self, session: Session, filters: Dict[str, Any]) -> Query:
        """Get optimized flight query with proper joins and filtering"""
        
        # Base query with optimized joins
        query = session.query(Flight).options(
            # Use selectinload for one-to-many relationships
            selectinload(Flight.analysis_results),
            # Use joinedload for many-to-one relationships
            joinedload(Flight.airline_info),
            joinedload(Flight.origin),
            joinedload(Flight.destination),
            joinedload(Flight.aircraft_info)
        )
        
        # Apply filters in order of selectivity (most selective first)
        if 'flight_id' in filters:
            query = query.filter(Flight.flight_id == filters['flight_id'])
        
        if 'date_range' in filters:
            start_date, end_date = filters['date_range']
            query = query.filter(
                and_(
                    Flight.scheduled_departure >= start_date,
                    Flight.scheduled_departure <= end_date
                )
            )
        
        if 'route' in filters:
            origin, destination = filters['route']
            query = query.filter(
                and_(
                    Flight.origin_airport == origin,
                    Flight.destination_airport == destination
                )
            )
        
        if 'airline' in filters:
            query = query.filter(Flight.airline_code == filters['airline'])
        
        if 'delay_threshold' in filters:
            threshold = filters['delay_threshold']
            query = query.filter(
                or_(
                    Flight.departure_delay_minutes >= threshold,
                    Flight.arrival_delay_minutes >= threshold
                )
            )
        
        if 'status' in filters:
            query = query.filter(Flight.status == filters['status'])
        
        return query
    
    @cached(ttl=1800, key_prefix="optimized_delay_analysis")
    def get_delay_analysis_data(self, session: Session, airport_code: str, 
                               days_back: int = 30) -> Dict[str, Any]:
        """Optimized query for delay analysis data"""
        
        with self.monitor_query(f"delay_analysis_{airport_code}_{days_back}d"):
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Use raw SQL for better performance on aggregations
            query = text("""
                SELECT 
                    DATE_TRUNC('hour', scheduled_departure) as hour_bucket,
                    AVG(departure_delay_minutes) as avg_delay,
                    COUNT(*) as flight_count,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY departure_delay_minutes) as median_delay,
                    MAX(departure_delay_minutes) as max_delay,
                    delay_category,
                    airline_code
                FROM flights 
                WHERE origin_airport = :airport_code 
                    AND scheduled_departure >= :cutoff_date
                    AND departure_delay_minutes IS NOT NULL
                GROUP BY DATE_TRUNC('hour', scheduled_departure), delay_category, airline_code
                ORDER BY hour_bucket DESC
            """)
            
            result = session.execute(query, {
                'airport_code': airport_code,
                'cutoff_date': cutoff_date
            }).fetchall()
            
            # Process results
            delay_data = {
                'hourly_patterns': [],
                'delay_categories': {},
                'airline_performance': {},
                'summary_stats': {}
            }
            
            for row in result:
                delay_data['hourly_patterns'].append({
                    'hour': row.hour_bucket,
                    'avg_delay': float(row.avg_delay or 0),
                    'flight_count': row.flight_count,
                    'median_delay': float(row.median_delay or 0),
                    'max_delay': row.max_delay or 0,
                    'delay_category': row.delay_category,
                    'airline': row.airline_code
                })
            
            return delay_data
    
    @cached(ttl=1800, key_prefix="optimized_congestion_analysis")
    def get_congestion_analysis_data(self, session: Session, airport_code: str,
                                   days_back: int = 30) -> Dict[str, Any]:
        """Optimized query for congestion analysis data"""
        
        with self.monitor_query(f"congestion_analysis_{airport_code}_{days_back}d"):
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Optimized query for congestion patterns
            query = text("""
                WITH hourly_traffic AS (
                    SELECT 
                        DATE_TRUNC('hour', scheduled_departure) as hour_bucket,
                        COUNT(*) as departures,
                        COUNT(*) FILTER (WHERE destination_airport = :airport_code) as arrivals
                    FROM flights 
                    WHERE (origin_airport = :airport_code OR destination_airport = :airport_code)
                        AND scheduled_departure >= :cutoff_date
                    GROUP BY DATE_TRUNC('hour', scheduled_departure)
                ),
                runway_capacity AS (
                    SELECT runway_capacity 
                    FROM airports 
                    WHERE code = :airport_code
                )
                SELECT 
                    ht.hour_bucket,
                    ht.departures,
                    ht.arrivals,
                    (ht.departures + ht.arrivals) as total_movements,
                    rc.runway_capacity,
                    ROUND(((ht.departures + ht.arrivals)::numeric / rc.runway_capacity) * 100, 2) as capacity_utilization
                FROM hourly_traffic ht
                CROSS JOIN runway_capacity rc
                ORDER BY ht.hour_bucket DESC
            """)
            
            result = session.execute(query, {
                'airport_code': airport_code,
                'cutoff_date': cutoff_date
            }).fetchall()
            
            congestion_data = {
                'hourly_traffic': [],
                'peak_hours': [],
                'capacity_analysis': {}
            }
            
            for row in result:
                congestion_data['hourly_traffic'].append({
                    'hour': row.hour_bucket,
                    'departures': row.departures,
                    'arrivals': row.arrivals,
                    'total_movements': row.total_movements,
                    'capacity_utilization': float(row.capacity_utilization or 0)
                })
            
            return congestion_data
    
    @cached(ttl=3600, key_prefix="optimized_route_analysis")
    def get_route_performance_data(self, session: Session, origin: str, 
                                 destination: str, days_back: int = 90) -> Dict[str, Any]:
        """Optimized query for route performance analysis"""
        
        with self.monitor_query(f"route_analysis_{origin}_{destination}_{days_back}d"):
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            query = text("""
                SELECT 
                    airline_code,
                    aircraft_type,
                    COUNT(*) as total_flights,
                    AVG(departure_delay_minutes) as avg_departure_delay,
                    AVG(arrival_delay_minutes) as avg_arrival_delay,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY departure_delay_minutes) as p95_departure_delay,
                    COUNT(*) FILTER (WHERE departure_delay_minutes > 15) as delayed_flights,
                    AVG(EXTRACT(EPOCH FROM (actual_arrival - actual_departure))/60) as avg_flight_time
                FROM flights 
                WHERE origin_airport = :origin 
                    AND destination_airport = :destination
                    AND scheduled_departure >= :cutoff_date
                    AND actual_departure IS NOT NULL
                    AND actual_arrival IS NOT NULL
                GROUP BY airline_code, aircraft_type
                ORDER BY total_flights DESC
            """)
            
            result = session.execute(query, {
                'origin': origin,
                'destination': destination,
                'cutoff_date': cutoff_date
            }).fetchall()
            
            route_data = {
                'airline_performance': [],
                'aircraft_performance': {},
                'route_summary': {}
            }
            
            for row in result:
                route_data['airline_performance'].append({
                    'airline': row.airline_code,
                    'aircraft_type': row.aircraft_type,
                    'total_flights': row.total_flights,
                    'avg_departure_delay': float(row.avg_departure_delay or 0),
                    'avg_arrival_delay': float(row.avg_arrival_delay or 0),
                    'p95_departure_delay': float(row.p95_departure_delay or 0),
                    'delayed_flights': row.delayed_flights,
                    'on_time_percentage': ((row.total_flights - row.delayed_flights) / row.total_flights * 100) if row.total_flights > 0 else 0,
                    'avg_flight_time': float(row.avg_flight_time or 0)
                })
            
            return route_data
    
    def analyze_query_performance(self) -> Dict[str, Any]:
        """Analyze query performance metrics"""
        if not self.performance_metrics:
            return {'message': 'No performance metrics available'}
        
        total_queries = len(self.performance_metrics)
        slow_queries = [m for m in self.performance_metrics if m.execution_time > self.slow_query_threshold]
        cache_hits = [m for m in self.performance_metrics if m.cache_hit]
        
        avg_execution_time = sum(m.execution_time for m in self.performance_metrics) / total_queries
        
        return {
            'total_queries': total_queries,
            'slow_queries': len(slow_queries),
            'cache_hit_rate': len(cache_hits) / total_queries * 100,
            'avg_execution_time': avg_execution_time,
            'slowest_queries': sorted(slow_queries, key=lambda x: x.execution_time, reverse=True)[:5]
        }
    
    def get_database_statistics(self, session: Session) -> Dict[str, Any]:
        """Get database performance statistics"""
        try:
            # Table sizes
            table_stats = session.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    correlation
                FROM pg_stats 
                WHERE schemaname = 'public' 
                    AND tablename IN ('flights', 'airports', 'airlines', 'aircraft')
                ORDER BY tablename, attname
            """)).fetchall()
            
            # Index usage
            index_stats = session.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes 
                WHERE schemaname = 'public'
                ORDER BY idx_tup_read DESC
            """)).fetchall()
            
            # Connection stats
            connection_stats = session.execute(text("""
                SELECT 
                    state,
                    COUNT(*) as connection_count
                FROM pg_stat_activity 
                WHERE datname = current_database()
                GROUP BY state
            """)).fetchall()
            
            return {
                'table_statistics': [dict(row._mapping) for row in table_stats],
                'index_usage': [dict(row._mapping) for row in index_stats],
                'connection_stats': [dict(row._mapping) for row in connection_stats]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting database statistics: {e}")
            return {'error': str(e)}


class ConnectionPoolOptimizer:
    """Optimize database connection pooling"""
    
    def __init__(self):
        self.logger = logger
    
    def optimize_pool_settings(self, engine) -> Dict[str, Any]:
        """Optimize connection pool settings based on usage patterns"""
        try:
            pool = engine.pool
            
            # Get current pool statistics
            current_stats = {
                'pool_size': pool.size(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'checked_in': pool.checkedin()
            }
            
            # Calculate optimal settings based on usage
            optimal_settings = {
                'pool_size': max(10, current_stats['checked_out'] * 2),
                'max_overflow': max(20, current_stats['overflow'] * 1.5),
                'pool_recycle': 3600,  # 1 hour
                'pool_pre_ping': True
            }
            
            self.logger.info(f"Current pool stats: {current_stats}")
            self.logger.info(f"Recommended pool settings: {optimal_settings}")
            
            return {
                'current_stats': current_stats,
                'recommended_settings': optimal_settings
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing pool settings: {e}")
            return {'error': str(e)}


# Global instances
query_optimizer = QueryOptimizer()
pool_optimizer = ConnectionPoolOptimizer()