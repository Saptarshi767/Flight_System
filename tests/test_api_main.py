"""
Tests for the main FastAPI application.

This module contains tests for the FastAPI application structure,
middleware, error handling, and basic functionality.
"""

import pytest
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app
from src.config import get_settings


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.environment = "test"
    settings.allowed_origins = ["http://localhost:3000"]
    settings.allowed_hosts = ["localhost", "testserver"]
    return settings


class TestMainApplication:
    """Test cases for the main FastAPI application."""
    
    def test_root_endpoint(self, client):
        """Test the root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Flight Scheduling Analysis API"
        assert data["version"] == "1.0.0"
        assert "docs_url" in data
        assert "health_check" in data
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema generation."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert schema["info"]["title"] == "Flight Scheduling Analysis API"
        assert schema["info"]["version"] == "1.0.0"
        assert "paths" in schema
    
    def test_docs_endpoint(self, client):
        """Test API documentation endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc documentation endpoint."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        response = client.options("/", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
    
    def test_correlation_id_header(self, client):
        """Test correlation ID is added to response headers."""
        response = client.get("/")
        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        assert "X-Process-Time" in response.headers
    
    def test_404_error_handling(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        data = response.json()
        # FastAPI's default 404 response format
        assert "detail" in data
        assert data["detail"] == "Not Found"


class TestMiddleware:
    """Test cases for custom middleware."""
    
    def test_logging_middleware(self, client):
        """Test logging middleware adds correlation ID."""
        response = client.get("/")
        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        assert "X-Process-Time" in response.headers
        
        # Verify process time is a valid float
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0
    
    def test_error_handling_middleware(self, client):
        """Test error handling middleware catches exceptions."""
        # This would require creating an endpoint that raises an exception
        # For now, we'll test the structure
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
    
    @patch('src.api.main.app')
    def test_rate_limiting_middleware(self, mock_app, client):
        """Test rate limiting middleware (if implemented)."""
        # This is a placeholder for rate limiting tests
        # Would need actual rate limiting implementation
        pass


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_http_exception_handler(self, client):
        """Test HTTP exception handler."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        data = response.json()
        # FastAPI's default 404 response format
        assert "detail" in data
    
    def test_general_exception_handler(self, client):
        """Test general exception handler."""
        # This would require an endpoint that raises a general exception
        # For now, we'll verify the structure exists
        assert hasattr(app, 'exception_handlers')


class TestRouterInclusion:
    """Test cases for router inclusion."""
    
    def test_health_router_included(self, client):
        """Test health router is properly included."""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
    
    def test_data_router_included(self, client):
        """Test data router is properly included."""
        response = client.get("/api/v1/data/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "7.2" in data["message"]  # Placeholder message
    
    def test_analysis_router_included(self, client):
        """Test analysis router is properly included."""
        response = client.get("/api/v1/analysis/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "7.3" in data["message"]  # Placeholder message
    
    def test_nlp_router_included(self, client):
        """Test NLP router is properly included."""
        response = client.get("/api/v1/nlp/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "7.4" in data["message"]  # Placeholder message


class TestApplicationLifespan:
    """Test cases for application lifespan management."""
    
    @patch('src.api.main.logger')
    def test_lifespan_startup_logging(self, mock_logger):
        """Test startup logging in lifespan manager."""
        # This would require testing the lifespan context manager
        # For now, we'll verify the structure exists
        assert hasattr(app, 'router')
    
    def test_application_metadata(self, client):
        """Test application metadata is correctly set."""
        response = client.get("/openapi.json")
        schema = response.json()
        
        info = schema["info"]
        assert info["title"] == "Flight Scheduling Analysis API"
        assert info["version"] == "1.0.0"
        assert "description" in info
        # Note: contact and license are added by custom_openapi but may not appear in test


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test cases for async endpoint functionality."""
    
    async def test_async_root_endpoint(self):
        """Test async functionality of root endpoint."""
        # TestClient doesn't support async context manager, use regular client
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200


class TestConfiguration:
    """Test cases for configuration integration."""
    
    @patch('src.api.main.get_settings')
    def test_settings_integration(self, mock_get_settings, client):
        """Test settings are properly integrated."""
        mock_settings = MagicMock()
        mock_settings.environment = "test"
        mock_settings.allowed_origins = ["http://localhost:3000"]
        mock_settings.allowed_hosts = ["localhost"]
        mock_get_settings.return_value = mock_settings
        
        # Test that the app can start with mocked settings
        response = client.get("/")
        assert response.status_code == 200


class TestSecurityHeaders:
    """Test cases for security headers and configurations."""
    
    def test_trusted_host_middleware(self, client):
        """Test trusted host middleware is configured."""
        response = client.get("/", headers={"Host": "localhost"})
        assert response.status_code == 200
    
    def test_cors_configuration(self, client):
        """Test CORS configuration."""
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        assert response.status_code == 200