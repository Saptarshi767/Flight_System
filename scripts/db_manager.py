#!/usr/bin/env python3
"""
Database Management Script for Flight Scheduling Analysis System

This script provides comprehensive database management capabilities including:
- Database initialization and schema management
- Data migration and backup/restore
- Health monitoring and performance checks
- Sample data management
- CRUD operations testing
"""
import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import db_manager, db_session_scope
from src.database.utils import db_utils
from src.database.operations import (
    flight_repo, airport_repo, airline_repo, aircraft_repo,
    analysis_result_repo, ingestion_log_repo, get_flights_dataframe
)
from src.utils.logging import logger


class DatabaseManager:
    """Comprehensive database management utility"""
    
    def __init__(self):
        self.logger = logger
    
    def init_database(self, create_sample_data: bool = False, reset: bool = False):
        """Initialize database with schema and optional sample data"""
        try:
            if reset:
                self.logger.warning("Resetting database...")
                if not db_utils.reset_database(confirm=True):
                    return False
            
            # Initialize tables
            if not db_utils.initialize_database(create_sample_data):
                return False
            
            # Apply migrations
            if not db_utils.apply_migrations():
                self.logger.warning("Migration issues (may be normal)")
            
            self.logger.info("Database initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return False
    
    def health_check(self):
        """Perform comprehensive database health check"""
        try:
            health_info = db_utils.check_database_health()
            
            print("\n" + "="*50)
            print("DATABASE HEALTH CHECK REPORT")
            print("="*50)
            print(f"Timestamp: {health_info.get('timestamp', 'N/A')}")
            
            # Connection test
            conn_status = health_info.get('connection_test', False)
            print(f"\n🔗 Connection Test: {'✅ PASS' if conn_status else '❌ FAIL'}")
            
            # Table integrity
            if 'table_integrity' in health_info:
                print("\n📊 Table Integrity:")
                total_records = 0
                for table, info in health_info['table_integrity'].items():
                    accessible = info.get('accessible', False)
                    row_count = info.get('row_count', 'N/A')
                    status_icon = '✅' if accessible else '❌'
                    
                    if isinstance(row_count, int):
                        total_records += row_count
                    
                    print(f"  {status_icon} {table:<25} {row_count:>10} rows")
                
                print(f"\n📈 Total Records: {total_records:,}")
            
            # Performance metrics
            if 'performance_metrics' in health_info:
                print("\n⚡ Performance Metrics:")
                metrics = health_info['performance_metrics']
                
                if 'database_size' in metrics:
                    print(f"  💾 Database Size: {metrics['database_size']}")
                if 'active_connections' in metrics:
                    print(f"  🔌 Active Connections: {metrics['active_connections']}")
            
            # Connection info
            conn_info = db_manager.get_connection_info()
            if conn_info:
                print(f"\n🌐 Connection Details:")
                print(f"  Host: {conn_info.get('host', 'N/A')}")
                print(f"  Port: {conn_info.get('port', 'N/A')}")
                print(f"  Database: {conn_info.get('database', 'N/A')}")
                print(f"  Pool Size: {conn_info.get('pool_size', 'N/A')}")
                print(f"  Checked Out: {conn_info.get('checked_out_connections', 'N/A')}")
            
            print("\n" + "="*50)
            return conn_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def backup_database(self, backup_file: str = None):
        """Create database backup"""
        try:
            if not backup_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f"backup_flight_db_{timestamp}.sql"
            
            if db_utils.backup_database(backup_file):
                print(f"✅ Database backup created: {backup_file}")
                return True
            else:
                print(f"❌ Database backup failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    def restore_database(self, backup_file: str):
        """Restore database from backup"""
        try:
            if not os.path.exists(backup_file):
                print(f"❌ Backup file not found: {backup_file}")
                return False
            
            confirm = input(f"⚠️  This will restore database from {backup_file}. Continue? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Restore cancelled")
                return False
            
            if db_utils.restore_database(backup_file):
                print(f"✅ Database restored from: {backup_file}")
                return True
            else:
                print(f"❌ Database restore failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def migration_status(self):
        """Show migration status and history"""
        try:
            print("\n" + "="*40)
            print("MIGRATION STATUS")
            print("="*40)
            
            history = db_utils.get_migration_history()
            if history:
                print(history)
            else:
                print("❌ Could not retrieve migration history")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Migration status check failed: {e}")
            return False
    
    def create_migration(self, message: str):
        """Create new migration"""
        try:
            if db_utils.create_migration(message):
                print(f"✅ Migration created: {message}")
                return True
            else:
                print(f"❌ Migration creation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Migration creation failed: {e}")
            return False
    
    def test_crud_operations(self):
        """Test CRUD operations on all tables"""
        try:
            print("\n" + "="*40)
            print("TESTING CRUD OPERATIONS")
            print("="*40)
            
            with db_session_scope() as session:
                # Test Airport operations
                print("\n🛫 Testing Airport operations...")
                test_airport = {
                    'code': 'TST',
                    'name': 'Test Airport',
                    'city': 'Test City',
                    'country': 'Test Country',
                    'timezone': 'UTC',
                    'runway_count': 2,
                    'runway_capacity': 30,
                    'latitude': 0.0,
                    'longitude': 0.0
                }
                
                # Create
                airport = airport_repo.create_airport(session, test_airport)
                print(f"  ✅ Created airport: {airport.code}")
                
                # Read
                retrieved = airport_repo.get_airport_by_code(session, 'TST')
                print(f"  ✅ Retrieved airport: {retrieved.name}")
                
                # Update
                updated = airport_repo.update_airport(session, 'TST', {'runway_count': 3})
                print(f"  ✅ Updated airport: {updated.runway_count} runways")
                
                # Test Airline operations
                print("\n✈️  Testing Airline operations...")
                test_airline = {
                    'code': 'TS',
                    'name': 'Test Airlines',
                    'country': 'Test Country'
                }
                
                airline = airline_repo.create_airline(session, test_airline)
                print(f"  ✅ Created airline: {airline.name}")
                
                # Test Aircraft operations
                print("\n🛩️  Testing Aircraft operations...")
                test_aircraft = {
                    'type_code': 'TST1',
                    'manufacturer': 'Test Manufacturer',
                    'model': 'Test Model',
                    'typical_seating': 150,
                    'max_seating': 180
                }
                
                aircraft = aircraft_repo.create_aircraft(session, test_aircraft)
                print(f"  ✅ Created aircraft: {aircraft.type_code}")
                
                # Test Flight operations
                print("\n🛫 Testing Flight operations...")
                test_flight = {
                    'flight_id': 'TST001_20240101',
                    'flight_number': 'TST001',
                    'airline_code': 'TS',
                    'origin_airport': 'TST',
                    'destination_airport': 'TST',
                    'aircraft_type': 'TST1',
                    'scheduled_departure': datetime.now(),
                    'scheduled_arrival': datetime.now() + timedelta(hours=2),
                    'data_source': 'test'
                }
                
                flight = flight_repo.create_flight(session, test_flight)
                print(f"  ✅ Created flight: {flight.flight_number}")
                
                # Test bulk operations
                print("\n📦 Testing bulk operations...")
                bulk_flights = []
                for i in range(5):
                    bulk_flights.append({
                        'flight_id': f'BULK{i:03d}_20240101',
                        'flight_number': f'BULK{i:03d}',
                        'airline_code': 'TS',
                        'origin_airport': 'TST',
                        'destination_airport': 'TST',
                        'scheduled_departure': datetime.now() + timedelta(hours=i),
                        'scheduled_arrival': datetime.now() + timedelta(hours=i+2),
                        'data_source': 'test'
                    })
                
                count = flight_repo.bulk_insert_flights(session, bulk_flights)
                print(f"  ✅ Bulk inserted {count} flights")
                
                # Test statistics
                print("\n📊 Testing statistics...")
                stats = flight_repo.get_flight_statistics(session, airport_code='TST')
                print(f"  ✅ Statistics: {stats['total_flights']} flights")
                
                # Test DataFrame export
                print("\n📈 Testing DataFrame export...")
                df = get_flights_dataframe(airport_code='TST')
                print(f"  ✅ DataFrame: {len(df)} rows, {len(df.columns)} columns")
                
                # Cleanup test data
                print("\n🧹 Cleaning up test data...")
                from sqlalchemy import text
                session.execute(text("DELETE FROM flights WHERE airline_code = 'TS'"))
                session.execute(text("DELETE FROM aircraft WHERE type_code = 'TST1'"))
                session.execute(text("DELETE FROM airlines WHERE code = 'TS'"))
                session.execute(text("DELETE FROM airports WHERE code = 'TST'"))
                print("  ✅ Test data cleaned up")
            
            print("\n✅ All CRUD operations completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"CRUD operations test failed: {e}")
            return False
    
    def show_statistics(self):
        """Show database statistics and insights"""
        try:
            print("\n" + "="*50)
            print("DATABASE STATISTICS")
            print("="*50)
            
            with db_session_scope() as session:
                # Get table counts
                info = db_utils.get_database_info()
                if 'table_counts' in info:
                    print("\n📊 Table Statistics:")
                    total_records = 0
                    for table, count in info['table_counts'].items():
                        if isinstance(count, int):
                            total_records += count
                        print(f"  {table:<25} {count:>10}")
                    print(f"  {'TOTAL':<25} {total_records:>10}")
                
                # Flight statistics if flights exist
                if info.get('table_counts', {}).get('flights', 0) > 0:
                    print("\n✈️  Flight Analysis:")
                    
                    # Overall statistics
                    stats = flight_repo.get_flight_statistics(session)
                    if stats:
                        print(f"  Total Flights: {stats.get('total_flights', 0):,}")
                        print(f"  Delayed Flights: {stats.get('delayed_flights', 0):,}")
                        print(f"  Delay Rate: {stats.get('delay_percentage', 0):.1f}%")
                        print(f"  Avg Departure Delay: {stats.get('avg_departure_delay_minutes', 0):.1f} min")
                        print(f"  Avg Arrival Delay: {stats.get('avg_arrival_delay_minutes', 0):.1f} min")
                    
                    # Airport-specific statistics
                    airports = airport_repo.get_all_airports(session)
                    if airports:
                        print(f"\n🛫 Airport Statistics:")
                        for airport in airports[:5]:  # Show top 5
                            airport_stats = flight_repo.get_flight_statistics(session, airport.code)
                            if airport_stats.get('total_flights', 0) > 0:
                                print(f"  {airport.code} ({airport.city}):")
                                print(f"    Flights: {airport_stats.get('total_flights', 0):,}")
                                print(f"    Delay Rate: {airport_stats.get('delay_percentage', 0):.1f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Statistics generation failed: {e}")
            return False


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Flight Scheduling Database Manager")
    
    # Main operations
    parser.add_argument("--init", action="store_true", help="Initialize database")
    parser.add_argument("--reset", action="store_true", help="Reset database (WARNING: destroys data)")
    parser.add_argument("--sample-data", action="store_true", help="Include sample data")
    
    # Health and monitoring
    parser.add_argument("--health", action="store_true", help="Perform health check")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--test-crud", action="store_true", help="Test CRUD operations")
    
    # Backup and restore
    parser.add_argument("--backup", type=str, help="Create backup (optional filename)")
    parser.add_argument("--restore", type=str, help="Restore from backup file")
    
    # Migration management
    parser.add_argument("--migration-status", action="store_true", help="Show migration status")
    parser.add_argument("--create-migration", type=str, help="Create new migration")
    parser.add_argument("--apply-migrations", action="store_true", help="Apply pending migrations")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    db_mgr = DatabaseManager()
    success = True
    
    try:
        # Initialize database
        if args.init:
            success &= db_mgr.init_database(
                create_sample_data=args.sample_data,
                reset=args.reset
            )
        
        # Health check
        if args.health:
            success &= db_mgr.health_check()
        
        # Statistics
        if args.stats:
            success &= db_mgr.show_statistics()
        
        # Test CRUD operations
        if args.test_crud:
            success &= db_mgr.test_crud_operations()
        
        # Backup
        if args.backup is not None:
            backup_file = args.backup if args.backup else None
            success &= db_mgr.backup_database(backup_file)
        
        # Restore
        if args.restore:
            success &= db_mgr.restore_database(args.restore)
        
        # Migration operations
        if args.migration_status:
            success &= db_mgr.migration_status()
        
        if args.create_migration:
            success &= db_mgr.create_migration(args.create_migration)
        
        if args.apply_migrations:
            success &= db_utils.apply_migrations()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Database manager failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)