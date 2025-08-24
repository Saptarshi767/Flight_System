"""
Structured logging configuration for Flight Scheduling Analysis System
"""

import logging
import logging.config
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar
import os
import sys

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def filter(self, record):
        record.correlation_id = correlation_id.get() or str(uuid.uuid4())[:8]
        return True

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', 'unknown'),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': os.getpid(),
            'thread_id': record.thread,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info', 'correlation_id']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records"""
    
    def filter(self, record):
        if hasattr(record, 'duration'):
            record.performance = {
                'duration_ms': record.duration * 1000,
                'slow_query': record.duration > 1.0  # Mark queries > 1s as slow
            }
        return True

def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> None:
    """
    Setup structured logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'text')
        log_file: Optional log file path
        enable_console: Whether to enable console logging
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Base configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': JSONFormatter,
            },
            'text': {
                'format': '%(asctime)s [%(correlation_id)s] %(levelname)-8s %(name)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'filters': {
            'correlation_id': {
                '()': CorrelationIdFilter,
            },
            'performance': {
                '()': PerformanceFilter,
            }
        },
        'handlers': {},
        'loggers': {
            '': {  # Root logger
                'level': log_level,
                'handlers': [],
                'propagate': False
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': [],
                'propagate': False
            },
            'uvicorn.access': {
                'level': 'INFO',
                'handlers': [],
                'propagate': False
            },
            'sqlalchemy.engine': {
                'level': 'WARNING',
                'handlers': [],
                'propagate': False
            },
            'celery': {
                'level': 'INFO',
                'handlers': [],
                'propagate': False
            }
        }
    }
    
    # Console handler
    if enable_console:
        config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
            'formatter': log_format,
            'filters': ['correlation_id', 'performance'],
            'level': log_level
        }
        
        # Add console handler to all loggers
        for logger_name in config['loggers']:
            config['loggers'][logger_name]['handlers'].append('console')
    
    # File handler
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': log_file,
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
            'filters': ['correlation_id', 'performance'],
            'level': log_level
        }
        
        # Add file handler to all loggers
        for logger_name in config['loggers']:
            config['loggers'][logger_name]['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(config)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)

def set_correlation_id(corr_id: str) -> None:
    """Set correlation ID for current context"""
    correlation_id.set(corr_id)

def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return correlation_id.get()

def log_performance(logger: logging.Logger, operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics"""
    logger.info(
        f"Performance: {operation}",
        extra={
            'operation': operation,
            'duration': duration,
            'performance_log': True,
            **kwargs
        }
    )

def log_business_event(logger: logging.Logger, event: str, **kwargs) -> None:
    """Log business events for analytics"""
    logger.info(
        f"Business Event: {event}",
        extra={
            'event_type': 'business',
            'event_name': event,
            'business_log': True,
            **kwargs
        }
    )

def log_security_event(logger: logging.Logger, event: str, **kwargs) -> None:
    """Log security events"""
    logger.warning(
        f"Security Event: {event}",
        extra={
            'event_type': 'security',
            'event_name': event,
            'security_log': True,
            **kwargs
        }
    )

def log_error_with_context(logger: logging.Logger, error: Exception, context: Dict[str, Any]) -> None:
    """Log error with additional context"""
    logger.error(
        f"Error occurred: {str(error)}",
        extra={
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'error_log': True
        },
        exc_info=True
    )

class LoggingMiddleware:
    """Middleware for request/response logging"""
    
    def __init__(self, app):
        self.app = app
        self.logger = get_logger(__name__)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Generate correlation ID for request
        corr_id = str(uuid.uuid4())[:8]
        set_correlation_id(corr_id)
        
        start_time = time.time()
        
        # Log request
        self.logger.info(
            "Request started",
            extra={
                'method': scope.get('method'),
                'path': scope.get('path'),
                'query_string': scope.get('query_string', b'').decode(),
                'client': scope.get('client'),
                'request_log': True
            }
        )
        
        # Process request
        try:
            await self.app(scope, receive, send)
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "Request failed",
                extra={
                    'method': scope.get('method'),
                    'path': scope.get('path'),
                    'duration': duration,
                    'error': str(e),
                    'request_log': True
                },
                exc_info=True
            )
            raise
        finally:
            duration = time.time() - start_time
            self.logger.info(
                "Request completed",
                extra={
                    'method': scope.get('method'),
                    'path': scope.get('path'),
                    'duration': duration,
                    'request_log': True
                }
            )

# Initialize logging based on environment
def init_logging():
    """Initialize logging based on environment variables"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    environment = os.getenv('ENVIRONMENT', 'development')
    
    # Determine log format based on environment
    log_format = 'json' if environment == 'production' else 'text'
    
    # Determine log file path
    log_file = None
    if environment in ['production', 'staging']:
        log_file = f'logs/app_{environment}.log'
    
    setup_logging(
        log_level=log_level,
        log_format=log_format,
        log_file=log_file,
        enable_console=True
    )

# Performance monitoring decorator
def monitor_performance(operation_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                log_performance(logger, operation_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                log_performance(logger, f"{operation_name}_failed", duration, error=str(e))
                raise
        
        return wrapper
    return decorator

# Async performance monitoring decorator
def monitor_async_performance(operation_name: str):
    """Decorator to monitor async function performance"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                log_performance(logger, operation_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                log_performance(logger, f"{operation_name}_failed", duration, error=str(e))
                raise
        
        return wrapper
    return decorator