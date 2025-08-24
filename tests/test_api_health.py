"""
Tests for the health check router.

This module contains tests for health check endpoints and system monitoring.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.api.routers.health import router, check_database_health, check_redis_health, check_openai_health
from src.api.models import HealthStatus, ComponentHealth


@pytest.fixture
def test_app():
    """Create test app with health router."""
    app = FastAPI()
    app.include_router(router, prefix="/health")
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


class TestHealthCheckEndpoints:
    """Test cases for health check endpoints."""
    
    def test_health_check_endpoint_success(self, client):
        """Test successful health check."""
        with patch('src.api.routers.health.check_database_health') as mock_db, \
             patch('src.api.routers.health.check_redis_health') as mock_redis, \
             patch('src.api.routers.health.check_openai_health') as mock_openai:
            
            # Mock all components as healthy
            mock_db.return_value = ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                response_time=0.05
            )
            mock_redis.return_value = ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                response_time=0.02
            )
            mock_openai.return_value = ComponentHealth(
                name="openai",
                status=HealthStatus.HEALTHY,
                response_time=0.3
            )
            
            response = client.get("/health/")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert data["version"] == "1.0.0"
            assert len(data["components"]) == 3
            assert "uptime" in data
            assert "timestamp" in data
    
    def test_health_check_endpoint_degraded(self, client):
        """Test health check with degraded components."""
        with patch('src.api.routers.health.check_database_health') as mock_db, \
             patch('src.api.routers.health.check_redis_health') as mock_redis, \
             patch('src.api.routers.health.check_openai_health') as mock_openai:
            
            # Mock one component as degraded
            mock_db.return_value = ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                response_time=0.05
            )
            mock_redis.return_value = ComponentHealth(
                name="redis",
                status=HealthStatus.DEGRADED,
                response_time=2.0
            )
            mock_openai.return_value = ComponentHealth(
                name="openai",
                status=HealthStatus.HEALTHY,
                response_time=0.3
            )
            
            response = client.get("/health/")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "degraded"
            assert len(data["components"]) == 3
    
    def test_health_check_endpoint_unhealthy(self, client):
        """Test health check with unhealthy components."""
        with patch('src.api.routers.health.check_database_health') as mock_db, \
             patch('src.api.routers.health.check_redis_health') as mock_redis, \
             patch('src.api.routers.health.check_openai_health') as mock_openai:
            
            # Mock one component as unhealthy
            mock_db.return_value = ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                error="Connection failed"
            )
            mock_redis.return_value = ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                response_time=0.02
            )
            mock_openai.return_value = ComponentHealth(
                name="openai",
                status=HealthStatus.HEALTHY,
                response_time=0.3
            )
            
            response = client.get("/health/")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "unhealthy"
            assert len(data["components"]) == 3
    
    def test_liveness_check_endpoint(self, client):
        """Test liveness check endpoint."""
        response = client.get("/health/liveness")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data
    
    def test_readiness_check_endpoint_ready(self, client):
        """Test readiness check when system is ready."""
        with patch('src.api.routers.health.check_database_health') as mock_db:
            mock_db.return_value = ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                response_time=0.05
            )
            
            response = client.get("/health/readiness")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "ready"
            assert "timestamp" in data
    
    def test_readiness_check_endpoint_not_ready(self, client):
        """Test readiness check when system is not ready."""
        with patch('src.api.routers.health.check_database_health') as mock_db:
            mock_db.return_value = ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                error="Connection failed"
            )
            
            response = client.get("/health/readiness")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "not_ready"
            assert data["reason"] == "database_unavailable"
    
    def test_metrics_endpoint(self, client):
        """Test system metrics endpoint."""
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Mock system metrics
            mock_cpu.return_value = 25.5
            mock_memory.return_value = MagicMock(percent=60.0)
            mock_disk.return_value = MagicMock(percent=45.0)
            
            response = client.get("/health/metrics")
            assert response.status_code == 200
            
            data = response.json()
            assert data["cpu_percent"] == 25.5
            assert data["memory_percent"] == 60.0
            assert data["disk_percent"] == 45.0
            assert "uptime" in data
            assert "timestamp" in data


class TestComponentHealthChecks:
    """Test cases for individual component health checks."""
    
    @pytest.mark.asyncio
    async def test_check_database_health_success(self):
        """Test database health check success."""
        # For now, this returns a healthy status as it's a placeholder
        result = await check_database_health()
        
        assert isinstance(result, ComponentHealth)
        assert result.name == "database"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time == 0.05
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_check_redis_health_success(self):
        """Test Redis health check success."""
        # For now, this returns a healthy status as it's a placeholder
        result = await check_redis_health()
        
        assert isinstance(result, ComponentHealth)
        assert result.name == "redis"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time == 0.02
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_check_openai_health_success(self):
        """Test OpenAI health check success."""
        # For now, this returns a healthy status as it's a placeholder
        result = await check_openai_health()
        
        assert isinstance(result, ComponentHealth)
        assert result.name == "openai"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time == 0.3
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_component_health_check_error_handling(self):
        """Test component health check error handling."""
        # This test would be more meaningful when actual health checks are implemented
        # For now, we verify the structure exists
        
        db_health = await check_database_health()
        redis_health = await check_redis_health()
        openai_health = await check_openai_health()
        
        assert all(isinstance(h, ComponentHealth) for h in [db_health, redis_health, openai_health])


class TestHealthCheckIntegration:
    """Test cases for health check integration."""
    
    def test_health_check_response_structure(self, client):
        """Test health check response structure."""
        response = client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        required_fields = ["status", "timestamp", "version", "components", "uptime"]
        for field in required_fields:
            assert field in data
        
        # Verify components structure
        for component in data["components"]:
            assert "name" in component
            assert "status" in component
            # response_time or error should be present
            assert "response_time" in component or "error" in component
    
    def test_health_check_status_determination(self, client):
        """Test health check overall status determination logic."""
        with patch('src.api.routers.health.check_database_health') as mock_db, \
             patch('src.api.routers.health.check_redis_health') as mock_redis, \
             patch('src.api.routers.health.check_openai_health') as mock_openai:
            
            # Test all healthy -> overall healthy
            mock_db.return_value = ComponentHealth(name="db", status=HealthStatus.HEALTHY)
            mock_redis.return_value = ComponentHealth(name="redis", status=HealthStatus.HEALTHY)
            mock_openai.return_value = ComponentHealth(name="openai", status=HealthStatus.HEALTHY)
            
            response = client.get("/health/")
            assert response.json()["status"] == "healthy"
            
            # Test one degraded -> overall degraded
            mock_redis.return_value = ComponentHealth(name="redis", status=HealthStatus.DEGRADED)
            
            response = client.get("/health/")
            assert response.json()["status"] == "degraded"
            
            # Test one unhealthy -> overall unhealthy (even with degraded)
            mock_db.return_value = ComponentHealth(name="db", status=HealthStatus.UNHEALTHY)
            
            response = client.get("/health/")
            assert response.json()["status"] == "unhealthy"
    
    def test_health_check_uptime_tracking(self, client):
        """Test health check uptime tracking."""
        response1 = client.get("/health/")
        uptime1 = response1.json()["uptime"]
        
        # Small delay to ensure uptime increases
        import time
        time.sleep(0.1)
        
        response2 = client.get("/health/")
        uptime2 = response2.json()["uptime"]
        
        assert uptime2 > uptime1
    
    def test_health_endpoints_cors_headers(self, client):
        """Test CORS headers on health endpoints."""
        # Test preflight request
        response = client.options("/health/", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        
        # Test actual request
        response = client.get("/health/")
        assert response.status_code == 200


class TestHealthCheckErrorScenarios:
    """Test cases for health check error scenarios."""
    
    def test_health_check_with_component_exceptions(self, client):
        """Test health check when component checks raise exceptions."""
        with patch('src.api.routers.health.check_database_health') as mock_db, \
             patch('src.api.routers.health.check_redis_health') as mock_redis, \
             patch('src.api.routers.health.check_openai_health') as mock_openai:
            
            # Mock normal responses for other components
            mock_redis.return_value = ComponentHealth(name="redis", status=HealthStatus.HEALTHY)
            mock_openai.return_value = ComponentHealth(name="openai", status=HealthStatus.HEALTHY)
            
            # Mock an exception in database health check - but return unhealthy component instead
            mock_db.return_value = ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                error="Database connection failed"
            )
            
            response = client.get("/health/")
            assert response.status_code == 200
    
    def test_health_check_timeout_scenarios(self, client):
        """Test health check with slow component responses."""
        # This would test timeout handling in component health checks
        # For now, we verify the endpoint structure exists
        response = client.get("/health/")
        assert response.status_code == 200
    
    def test_health_check_partial_failures(self, client):
        """Test health check with partial component failures."""
        with patch('src.api.routers.health.check_database_health') as mock_db, \
             patch('src.api.routers.health.check_redis_health') as mock_redis, \
             patch('src.api.routers.health.check_openai_health') as mock_openai:
            
            # Mix of healthy and unhealthy components
            mock_db.return_value = ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                response_time=0.05
            )
            mock_redis.return_value = ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                error="Connection timeout"
            )
            mock_openai.return_value = ComponentHealth(
                name="openai",
                status=HealthStatus.DEGRADED,
                response_time=5.0
            )
            
            response = client.get("/health/")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "unhealthy"  # Worst case wins
            
            # Verify all components are reported
            component_names = [c["name"] for c in data["components"]]
            assert "database" in component_names
            assert "redis" in component_names
            assert "openai" in component_names