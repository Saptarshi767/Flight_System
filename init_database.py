#!/usr/bin/env python3
"""
Database initialization script for Flight Scheduling Analysis System

This script initializes the PostgreSQL database with the required schema,
applies migrations, and optionally inserts sample data for testing.
"""
import sys
import os
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database.connection import db_manager, init_database, test_database_connection
from src.database.utils import db_utils, init_database as utils_init_database
from src.database.operations import insert_sample_data
from src.utils.logging import logger


def main():
    """Main initialization function"""
    parser = argparse.ArgumentParser(description="Initialize Flight Scheduling Analysis Database")
    parser.add_argument("--create-db", action="store_true", help="Create database if it doesn't exist")
    parser.add_argument("--sample-data", action="store_true", help="Insert sample data")
    parser.add_argument("--reset", action="store_true", help="Reset database (WARNING: destroys all data)")
    parser.add_argument("--test-only", action="store_true", help="Only test database connection")
    parser.add_argument("--health-check", action="store_true", help="Perform database health check")
    parser.add_argument("--info", action="store_true", help="Show database information")
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting database initialization...")
        
        # Test connection only
        if args.test_only:
            logger.info("Testing database connection...")
            if test_database_connection():
                logger.info("✅ Database connection successful!")
                return 0
            else:
                logger.error("❌ Database connection failed!")
                return 1
        
        # Health check
        if args.health_check:
            logger.info("Performing database health check...")
            health_info = db_utils.check_database_health()
            
            print("\n=== Database Health Check ===")
            print(f"Connection Test: {'✅ PASS' if health_info.get('connection_test') else '❌ FAIL'}")
            
            if 'table_integrity' in health_info:
                print("\nTable Integrity:")
                for table, info in health_info['table_integrity'].items():
                    status = '✅ OK' if info.get('accessible') else '❌ ERROR'
                    count = info.get('row_count', 'N/A')
                    print(f"  {table}: {status} ({count} rows)")
            
            if 'performance_metrics' in health_info:
                print("\nPerformance Metrics:")
                metrics = health_info['performance_metrics']
                if 'database_size' in metrics:
                    print(f"  Database Size: {metrics['database_size']}")
                if 'active_connections' in metrics:
                    print(f"  Active Connections: {metrics['active_connections']}")
            
            return 0
        
        # Show database info
        if args.info:
            logger.info("Getting database information...")
            info = db_utils.get_database_info()
            
            print("\n=== Database Information ===")
            if 'connection_info' in info:
                conn_info = info['connection_info']
                print(f"Host: {conn_info.get('host')}")
                print(f"Port: {conn_info.get('port')}")
                print(f"Database: {conn_info.get('database')}")
                print(f"Username: {conn_info.get('username')}")
                print(f"Pool Size: {conn_info.get('pool_size')}")
                print(f"Checked Out Connections: {conn_info.get('checked_out_connections')}")
            
            if 'tables' in info:
                print(f"\nTables: {', '.join(info['tables'])}")
            
            if 'table_counts' in info:
                print("\nTable Row Counts:")
                for table, count in info['table_counts'].items():
                    print(f"  {table}: {count}")
            
            return 0
        
        # Create database if requested
        if args.create_db:
            logger.info("Creating database if it doesn't exist...")
            from src.config.settings import settings
            from urllib.parse import urlparse
            
            parsed_url = urlparse(settings.database_url)
            db_name = parsed_url.path.lstrip('/')
            
            if db_utils.create_database_if_not_exists(db_name):
                logger.info(f"✅ Database '{db_name}' is ready")
            else:
                logger.error(f"❌ Failed to create database '{db_name}'")
                return 1
        
        # Reset database if requested
        if args.reset:
            logger.warning("⚠️  RESETTING DATABASE - ALL DATA WILL BE LOST!")
            confirm = input("Are you sure you want to reset the database? (yes/no): ")
            if confirm.lower() != 'yes':
                logger.info("Database reset cancelled")
                return 0
            
            if db_utils.reset_database(confirm=True):
                logger.info("✅ Database reset successfully")
            else:
                logger.error("❌ Failed to reset database")
                return 1
        
        # Initialize database tables
        logger.info("Initializing database tables...")
        if utils_init_database(create_sample_data=False):
            logger.info("✅ Database tables initialized successfully")
        else:
            logger.error("❌ Failed to initialize database tables")
            return 1
        
        # Apply migrations
        logger.info("Applying database migrations...")
        if db_utils.apply_migrations():
            logger.info("✅ Database migrations applied successfully")
        else:
            logger.warning("⚠️  Migration application had issues (may be normal if already up to date)")
        
        # Insert sample data if requested
        if args.sample_data:
            logger.info("Inserting sample data...")
            try:
                insert_sample_data()
                logger.info("✅ Sample data inserted successfully")
            except Exception as e:
                logger.error(f"❌ Failed to insert sample data: {e}")
                return 1
        
        # Final connection test
        logger.info("Performing final connection test...")
        if test_database_connection():
            logger.info("✅ Database initialization completed successfully!")
            
            # Show final status
            info = db_utils.get_database_info()
            if 'table_counts' in info:
                print("\n=== Final Database Status ===")
                total_records = 0
                for table, count in info['table_counts'].items():
                    if isinstance(count, int):
                        total_records += count
                    print(f"  {table}: {count} records")
                print(f"\nTotal records: {total_records}")
            
            return 0
        else:
            logger.error("❌ Final connection test failed!")
            return 1
    
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)