#!/usr/bin/env python3
"""
Local database setup script for development without Docker

This script sets up a local SQLite database for development and testing
when PostgreSQL is not available.
"""
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import settings
from src.utils.logging import logger


def setup_sqlite_for_development():
    """Set up SQLite database for local development"""
    try:
        # Update database URL to use SQLite
        sqlite_path = Path("data/flight_analysis.db")
        sqlite_path.parent.mkdir(exist_ok=True)
        
        # Create a temporary settings override
        import os
        os.environ['DATABASE_URL'] = f"sqlite:///{sqlite_path.absolute()}"
        
        logger.info(f"Using SQLite database: {sqlite_path.absolute()}")
        
        # Import database components after setting the URL
        from src.database.connection import db_manager
        from src.database.utils import db_utils
        from src.database.operations import insert_sample_data
        
        # Initialize database
        logger.info("Creating database tables...")
        db_manager.create_tables()
        
        # Test connection
        if db_manager.test_connection():
            logger.info("✅ Database connection successful!")
            
            # Insert sample data
            logger.info("Inserting sample data...")
            insert_sample_data()
            
            # Show database info
            info = db_utils.get_database_info()
            if 'table_counts' in info:
                print("\n=== Database Setup Complete ===")
                for table, count in info['table_counts'].items():
                    print(f"  {table}: {count} records")
            
            return True
        else:
            logger.error("❌ Database connection failed!")
            return False
            
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def main():
    """Main function"""
    print("Setting up local development database...")
    
    if setup_sqlite_for_development():
        print("\n✅ Local database setup completed successfully!")
        print("\nTo use this database, set the environment variable:")
        print("DATABASE_URL=sqlite:///data/flight_analysis.db")
        return 0
    else:
        print("\n❌ Local database setup failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)