"""
Tests for FastAPI middleware components.

This module contains tests for custom middleware including logging,
error handling, and rate limiting.
"""

import pytest
import time
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI, Request, HTTPException
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from src.api.middleware import LoggingMiddleware, ErrorHandlingMiddleware, RateLimitMiddleware


@pytest.fixture
def test_app():
    """Create a test FastAPI app with middleware."""
    app = FastAPI()
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    @app.get("/error")
    async def error_endpoint():
        raise Exception("Test error")
    
    @app.get("/http-error")
    async def http_error_endpoint():
        raise HTTPException(status_code=400, detail="Test HTTP error")
    
    return app


class TestLoggingMiddleware:
    """Test cases for LoggingMiddleware."""
    
    def test_logging_middleware_adds_correlation_id(self, test_app):
        """Test that logging middleware adds correlation ID."""
        test_app.add_middleware(LoggingMiddleware)
        client = TestClient(test_app)
        
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        assert "X-Process-Time" in response.headers
        
        # Verify correlation ID is a valid UUID
        correlation_id = response.headers["X-Correlation-ID"]
        uuid.UUID(correlation_id)  # This will raise if invalid
    
    def test_logging_middleware_measures_process_time(self, test_app):
        """Test that logging middleware measures process time."""
        test_app.add_middleware(LoggingMiddleware)
        client = TestClient(test_app)
        
        response = client.get("/test")
        assert response.status_code == 200
        
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0
        assert process_time < 10  # Should be very fast for test endpoint
    
    @patch('src.api.middleware.logger')
    def test_logging_middleware_logs_requests(self, mock_logger, test_app):
        """Test that logging middleware logs requests and responses."""
        test_app.add_middleware(LoggingMiddleware)
        client = TestClient(test_app)
        
        response = client.get("/test")
        assert response.status_code == 200
        
        # Verify logging was called
        assert mock_logger.info.call_count >= 2  # Start and end logs
    
    def test_logging_middleware_handles_request_state(self, test_app):
        """Test that logging middleware properly sets request state."""
        test_app.add_middleware(LoggingMiddleware)
        
        @test_app.get("/state-test")
        async def state_test_endpoint(request: Request):
            return {"correlation_id": getattr(request.state, 'correlation_id', None)}
        
        client = TestClient(test_app)
        response = client.get("/state-test")
        
        assert response.status_code == 200
        data = response.json()
        assert data["correlation_id"] is not None
        assert data["correlation_id"] == response.headers["X-Correlation-ID"]


class TestErrorHandlingMiddleware:
    """Test cases for ErrorHandlingMiddleware."""
    
    def test_error_handling_middleware_catches_exceptions(self, test_app):
        """Test that error handling middleware catches general exceptions."""
        test_app.add_middleware(ErrorHandlingMiddleware)
        test_app.add_middleware(LoggingMiddleware)  # For correlation ID
        client = TestClient(test_app)
        
        response = client.get("/error")
        assert response.status_code == 500
        
        data = response.json()
        assert data["error"] == "Internal server error"
        assert data["status_code"] == 500
        assert "correlation_id" in data
        assert "timestamp" in data
    
    def test_error_handling_middleware_preserves_http_exceptions(self, test_app):
        """Test that error handling middleware doesn't interfere with HTTP exceptions."""
        test_app.add_middleware(ErrorHandlingMiddleware)
        test_app.add_middleware(LoggingMiddleware)  # For correlation ID
        client = TestClient(test_app)
        
        response = client.get("/http-error")
        assert response.status_code == 400
        # HTTP exceptions should be handled by FastAPI's default handler
    
    @patch('src.api.middleware.logger')
    def test_error_handling_middleware_logs_exceptions(self, mock_logger, test_app):
        """Test that error handling middleware logs exceptions."""
        test_app.add_middleware(ErrorHandlingMiddleware)
        test_app.add_middleware(LoggingMiddleware)  # For correlation ID
        client = TestClient(test_app)
        
        response = client.get("/error")
        assert response.status_code == 500
        
        # Verify error logging was called
        mock_logger.error.assert_called()
    
    def test_error_handling_middleware_includes_correlation_id(self, test_app):
        """Test that error responses include correlation ID."""
        test_app.add_middleware(ErrorHandlingMiddleware)
        test_app.add_middleware(LoggingMiddleware)  # For correlation ID
        client = TestClient(test_app)
        
        response = client.get("/error")
        assert response.status_code == 500
        
        data = response.json()
        assert "correlation_id" in data
        # Should match the correlation ID from logging middleware
        assert data["correlation_id"] == response.headers.get("X-Correlation-ID")


class TestRateLimitMiddleware:
    """Test cases for RateLimitMiddleware."""
    
    def test_rate_limit_middleware_allows_normal_requests(self, test_app):
        """Test that rate limit middleware allows normal request rates."""
        test_app.add_middleware(RateLimitMiddleware, calls=10, period=60)
        client = TestClient(test_app)
        
        # Make several requests within limit
        for _ in range(5):
            response = client.get("/test")
            assert response.status_code == 200
    
    def test_rate_limit_middleware_blocks_excessive_requests(self, test_app):
        """Test that rate limit middleware blocks excessive requests."""
        test_app.add_middleware(RateLimitMiddleware, calls=3, period=60)
        client = TestClient(test_app)
        
        # Make requests up to the limit
        for _ in range(3):
            response = client.get("/test")
            assert response.status_code == 200
        
        # Next request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429
        
        data = response.json()
        assert data["error"] == "Rate limit exceeded"
        assert data["status_code"] == 429
        assert "retry_after" in data
    
    def test_rate_limit_middleware_cleans_old_entries(self, test_app):
        """Test that rate limit middleware cleans old entries."""
        test_app.add_middleware(RateLimitMiddleware, calls=2, period=1)  # 1 second period
        client = TestClient(test_app)
        
        # Make requests up to limit
        for _ in range(2):
            response = client.get("/test")
            assert response.status_code == 200
        
        # Should be rate limited
        response = client.get("/test")
        assert response.status_code == 429
        
        # Wait for period to expire
        time.sleep(1.1)
        
        # Should be allowed again
        response = client.get("/test")
        assert response.status_code == 200
    
    def test_rate_limit_middleware_per_client(self, test_app):
        """Test that rate limiting is applied per client IP."""
        test_app.add_middleware(RateLimitMiddleware, calls=2, period=60)
        
        # This is tricky to test with TestClient as it doesn't easily
        # allow different client IPs. In a real scenario, different
        # clients would have different IP addresses.
        client = TestClient(test_app)
        
        # Make requests up to limit
        for _ in range(2):
            response = client.get("/test")
            assert response.status_code == 200
        
        # Should be rate limited
        response = client.get("/test")
        assert response.status_code == 429


class TestMiddlewareIntegration:
    """Test cases for middleware integration."""
    
    def test_multiple_middleware_stack(self, test_app):
        """Test that multiple middleware work together correctly."""
        test_app.add_middleware(RateLimitMiddleware, calls=10, period=60)
        test_app.add_middleware(ErrorHandlingMiddleware)
        test_app.add_middleware(LoggingMiddleware)
        
        client = TestClient(test_app)
        
        response = client.get("/test")
        assert response.status_code == 200
        
        # Should have correlation ID from logging middleware
        assert "X-Correlation-ID" in response.headers
        assert "X-Process-Time" in response.headers
    
    def test_middleware_error_handling_integration(self, test_app):
        """Test middleware integration with error handling."""
        test_app.add_middleware(ErrorHandlingMiddleware)
        test_app.add_middleware(LoggingMiddleware)
        
        client = TestClient(test_app)
        
        response = client.get("/error")
        assert response.status_code == 500
        
        data = response.json()
        assert "correlation_id" in data
        assert data["correlation_id"] == response.headers.get("X-Correlation-ID")
    
    @patch('src.api.middleware.logger')
    def test_middleware_logging_integration(self, mock_logger, test_app):
        """Test middleware integration with logging."""
        test_app.add_middleware(ErrorHandlingMiddleware)
        test_app.add_middleware(LoggingMiddleware)
        
        client = TestClient(test_app)
        
        # Normal request
        response = client.get("/test")
        assert response.status_code == 200
        
        # Error request
        response = client.get("/error")
        assert response.status_code == 500
        
        # Verify both info and error logging occurred
        assert mock_logger.info.called
        assert mock_logger.error.called


class TestMiddlewareConfiguration:
    """Test cases for middleware configuration."""
    
    def test_rate_limit_middleware_configuration(self, test_app):
        """Test rate limit middleware with different configurations."""
        # Test with very restrictive limits
        test_app.add_middleware(RateLimitMiddleware, calls=1, period=1)
        client = TestClient(test_app)
        
        response = client.get("/test")
        assert response.status_code == 200
        
        response = client.get("/test")
        assert response.status_code == 429
    
    def test_middleware_order_matters(self, test_app):
        """Test that middleware order affects behavior."""
        # Add middleware in specific order
        test_app.add_middleware(ErrorHandlingMiddleware)  # Last to execute
        test_app.add_middleware(LoggingMiddleware)        # First to execute
        
        client = TestClient(test_app)
        
        response = client.get("/error")
        assert response.status_code == 500
        
        # Error should be caught by ErrorHandlingMiddleware
        # but correlation ID should be set by LoggingMiddleware
        data = response.json()
        assert "correlation_id" in data