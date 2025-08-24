"""
Database connection management for PostgreSQL
"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator, Optional
import logging
from urllib.parse import urlparse

from src.config.settings import settings
from src.database.models import Base
from src.utils.logging import logger


class DatabaseManager:
    """
    Database connection and session management
    """
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        self.engine = None
        self.SessionLocal = None
        self.logger = logger
        
        # Parse database URL for validation
        self._validate_database_url()
        
        # Initialize connection
        self._initialize_engine()
        self._initialize_session_factory()
    
    def _validate_database_url(self):
        """Validate database URL format"""
        try:
            parsed = urlparse(self.database_url)
            if parsed.scheme.startswith('postgresql'):
                if not parsed.hostname:
                    raise ValueError("PostgreSQL URL must include hostname")
                if not parsed.path or parsed.path == '/':
                    raise ValueError("PostgreSQL URL must include database name")
            elif parsed.scheme.startswith('sqlite'):
                # SQLite URLs are valid as long as they have the sqlite scheme
                pass
            else:
                raise ValueError(f"Unsupported database scheme: {parsed.scheme}")
        except Exception as e:
            self.logger.error(f"Invalid database URL: {e}")
            raise
    
    def _initialize_engine(self):
        """Initialize SQLAlchemy engine with connection pooling"""
        try:
            # Configure engine based on database type
            parsed = urlparse(self.database_url)
            
            if parsed.scheme.startswith('sqlite'):
                # SQLite configuration
                self.engine = create_engine(
                    self.database_url,
                    echo=settings.debug,  # Log SQL queries in debug mode
                    connect_args={"check_same_thread": False}  # Allow SQLite to be used across threads
                )
            else:
                # PostgreSQL configuration
                self.engine = create_engine(
                    self.database_url,
                    poolclass=QueuePool,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600,  # Recycle connections after 1 hour
                    echo=settings.debug,  # Log SQL queries in debug mode
                    connect_args={
                        "options": "-c timezone=utc"  # Set timezone to UTC
                    }
                )
            
            # Add connection event listeners
            self._setup_connection_events()
            
            self.logger.info("Database engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    def _initialize_session_factory(self):
        """Initialize session factory"""
        try:
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            self.logger.info("Database session factory initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize session factory: {e}")
            raise
    
    def _setup_connection_events(self):
        """Setup connection event listeners for monitoring"""
        
        @event.listens_for(self.engine, "connect")
        def set_connection_params(dbapi_connection, connection_record):
            """Set connection parameters on connect"""
            parsed = urlparse(self.database_url)
            
            if parsed.scheme.startswith('sqlite'):
                # SQLite-specific settings
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()
            elif parsed.scheme.startswith('postgresql') and hasattr(dbapi_connection, 'execute'):
                # PostgreSQL-specific settings
                dbapi_connection.execute("SET statement_timeout = '30s'")
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Log connection checkout"""
            self.logger.debug("Connection checked out from pool")
        
        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Log connection checkin"""
            self.logger.debug("Connection checked in to pool")
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            self.logger.warning("All database tables dropped")
        except Exception as e:
            self.logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session"""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope around a series of operations
        
        Usage:
            with db_manager.session_scope() as session:
                # perform database operations
                session.add(obj)
                # session will be committed automatically
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            from sqlalchemy import text
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            self.logger.info("Database connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """Get database connection information"""
        try:
            parsed = urlparse(self.database_url)
            return {
                'host': parsed.hostname,
                'port': parsed.port or 5432,
                'database': parsed.path.lstrip('/'),
                'username': parsed.username,
                'scheme': parsed.scheme,
                'pool_size': self.engine.pool.size() if self.engine else None,
                'checked_out_connections': self.engine.pool.checkedout() if self.engine else None
            }
        except Exception as e:
            self.logger.error(f"Failed to get connection info: {e}")
            return {}
    
    def close(self):
        """Close database connections"""
        try:
            if self.engine:
                self.engine.dispose()
                self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
def get_db_session() -> Session:
    """Get a database session"""
    return db_manager.get_session()


@contextmanager
def db_session_scope() -> Generator[Session, None, None]:
    """Get a database session with automatic transaction management"""
    with db_manager.session_scope() as session:
        yield session


def init_database():
    """Initialize database tables"""
    db_manager.create_tables()


def test_database_connection() -> bool:
    """Test database connection"""
    return db_manager.test_connection()