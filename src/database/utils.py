"""
Database utility functions for flight scheduling analysis
"""
import os
import subprocess
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import db_manager, db_session_scope
from src.database.operations import insert_sample_data
from src.utils.logging import logger


class DatabaseUtils:
    """Utility class for database operations"""
    
    def __init__(self):
        self.logger = logger
    
    def create_database_if_not_exists(self, database_name: str) -> bool:
        """Create database if it doesn't exist"""
        try:
            # Connect to postgres database to create the target database
            from src.config.settings import settings
            from urllib.parse import urlparse
            
            parsed_url = urlparse(settings.database_url)
            admin_url = f"{parsed_url.scheme}://{parsed_url.username}:{parsed_url.password}@{parsed_url.hostname}:{parsed_url.port}/postgres"
            
            from sqlalchemy import create_engine
            admin_engine = create_engine(admin_url)
            
            with admin_engine.connect() as conn:
                # Check if database exists
                result = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                    {"db_name": database_name}
                )
                
                if not result.fetchone():
                    # Create database
                    conn.execute(text("COMMIT"))  # End any existing transaction
                    conn.execute(text(f"CREATE DATABASE {database_name}"))
                    self.logger.info(f"Created database: {database_name}")
                else:
                    self.logger.info(f"Database {database_name} already exists")
            
            admin_engine.dispose()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create database {database_name}: {e}")
            return False
    
    def initialize_database(self, create_sample_data: bool = False) -> bool:
        """Initialize database with tables and optional sample data"""
        try:
            # Create tables
            db_manager.create_tables()
            self.logger.info("Database tables created successfully")
            
            # Insert sample data if requested
            if create_sample_data:
                insert_sample_data()
                self.logger.info("Sample data inserted successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            return False
    
    def reset_database(self, confirm: bool = False) -> bool:
        """Reset database by dropping and recreating all tables"""
        if not confirm:
            self.logger.warning("Database reset requires confirmation. Set confirm=True")
            return False
        
        try:
            # Drop all tables
            db_manager.drop_tables()
            self.logger.warning("All database tables dropped")
            
            # Recreate tables
            db_manager.create_tables()
            self.logger.info("Database tables recreated")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset database: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics"""
        try:
            with db_session_scope() as session:
                # Get table information
                inspector = inspect(session.bind)
                tables = inspector.get_table_names()
                
                info = {
                    'connection_info': db_manager.get_connection_info(),
                    'tables': tables,
                    'table_counts': {}
                }
                
                # Get row counts for each table
                for table in tables:
                    try:
                        result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        info['table_counts'][table] = count
                    except Exception as e:
                        info['table_counts'][table] = f"Error: {e}"
                
                return info
                
        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            return {}
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check database health and performance"""
        try:
            health_info = {
                'connection_test': False,
                'table_integrity': {},
                'performance_metrics': {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Test connection
            health_info['connection_test'] = db_manager.test_connection()
            
            with db_session_scope() as session:
                # Check table integrity
                inspector = inspect(session.bind)
                tables = inspector.get_table_names()
                
                for table in tables:
                    try:
                        # Simple count query to test table access
                        result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        health_info['table_integrity'][table] = {
                            'accessible': True,
                            'row_count': count
                        }
                    except Exception as e:
                        health_info['table_integrity'][table] = {
                            'accessible': False,
                            'error': str(e)
                        }
                
                # Get performance metrics
                try:
                    # Database size
                    result = session.execute(text("""
                        SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                    """))
                    db_size = result.scalar()
                    health_info['performance_metrics']['database_size'] = db_size
                    
                    # Active connections
                    result = session.execute(text("""
                        SELECT count(*) FROM pg_stat_activity 
                        WHERE state = 'active'
                    """))
                    active_connections = result.scalar()
                    health_info['performance_metrics']['active_connections'] = active_connections
                    
                except Exception as e:
                    health_info['performance_metrics']['error'] = str(e)
            
            return health_info
            
        except Exception as e:
            self.logger.error(f"Failed to check database health: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    def run_alembic_migration(self, command: str = "upgrade head") -> bool:
        """Run Alembic migration command"""
        try:
            # Change to project root directory
            original_dir = os.getcwd()
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            os.chdir(project_root)
            
            # Run alembic command
            cmd = f"alembic {command}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # Return to original directory
            os.chdir(original_dir)
            
            if result.returncode == 0:
                self.logger.info(f"Alembic command '{command}' completed successfully")
                if result.stdout:
                    self.logger.info(f"Output: {result.stdout}")
                return True
            else:
                self.logger.error(f"Alembic command '{command}' failed")
                if result.stderr:
                    self.logger.error(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to run Alembic migration: {e}")
            return False
    
    def create_migration(self, message: str) -> bool:
        """Create a new Alembic migration"""
        return self.run_alembic_migration(f'revision --autogenerate -m "{message}"')
    
    def apply_migrations(self) -> bool:
        """Apply all pending migrations"""
        return self.run_alembic_migration("upgrade head")
    
    def rollback_migration(self, revision: str = "-1") -> bool:
        """Rollback to a specific migration"""
        return self.run_alembic_migration(f"downgrade {revision}")
    
    def get_migration_history(self) -> Optional[str]:
        """Get migration history"""
        try:
            original_dir = os.getcwd()
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            os.chdir(project_root)
            
            result = subprocess.run("alembic history", shell=True, capture_output=True, text=True)
            os.chdir(original_dir)
            
            if result.returncode == 0:
                return result.stdout
            else:
                self.logger.error(f"Failed to get migration history: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get migration history: {e}")
            return None
    
    def backup_database(self, backup_file: str) -> bool:
        """Create a database backup using pg_dump"""
        try:
            from src.config.settings import settings
            from urllib.parse import urlparse
            
            parsed_url = urlparse(settings.database_url)
            
            # Build pg_dump command
            cmd = [
                "pg_dump",
                f"--host={parsed_url.hostname}",
                f"--port={parsed_url.port or 5432}",
                f"--username={parsed_url.username}",
                f"--dbname={parsed_url.path.lstrip('/')}",
                "--verbose",
                "--clean",
                "--no-owner",
                "--no-privileges",
                f"--file={backup_file}"
            ]
            
            # Set password environment variable
            env = os.environ.copy()
            if parsed_url.password:
                env['PGPASSWORD'] = parsed_url.password
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Database backup created: {backup_file}")
                return True
            else:
                self.logger.error(f"Database backup failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False
    
    def restore_database(self, backup_file: str) -> bool:
        """Restore database from backup using psql"""
        try:
            from src.config.settings import settings
            from urllib.parse import urlparse
            
            parsed_url = urlparse(settings.database_url)
            
            # Build psql command
            cmd = [
                "psql",
                f"--host={parsed_url.hostname}",
                f"--port={parsed_url.port or 5432}",
                f"--username={parsed_url.username}",
                f"--dbname={parsed_url.path.lstrip('/')}",
                f"--file={backup_file}"
            ]
            
            # Set password environment variable
            env = os.environ.copy()
            if parsed_url.password:
                env['PGPASSWORD'] = parsed_url.password
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Database restored from: {backup_file}")
                return True
            else:
                self.logger.error(f"Database restore failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to restore database: {e}")
            return False


# Global database utilities instance
db_utils = DatabaseUtils()


# Convenience functions
def init_database(create_sample_data: bool = False) -> bool:
    """Initialize database with tables and optional sample data"""
    return db_utils.initialize_database(create_sample_data)


def reset_database(confirm: bool = False) -> bool:
    """Reset database (requires confirmation)"""
    return db_utils.reset_database(confirm)


def check_database_health() -> Dict[str, Any]:
    """Check database health"""
    return db_utils.check_database_health()


def create_migration(message: str) -> bool:
    """Create a new migration"""
    return db_utils.create_migration(message)


def apply_migrations() -> bool:
    """Apply all pending migrations"""
    return db_utils.apply_migrations()