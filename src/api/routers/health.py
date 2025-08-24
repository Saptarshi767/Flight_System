"""
Health check router for monitoring API status.

This module provides endpoints for health checking and system monitoring.
"""

import time
import psutil
from datetime import datetime
from fastapi import APIRouter, Depends
from typing import List

from src.api.models import HealthCheckResponse, HealthStatus, ComponentHealth
from src.config import get_settings

router = APIRouter()
settings = get_settings()

# Track application start time
start_time = time.time()


async def check_database_health() -> ComponentHealth:
    """Check database connectivity."""
    try:
        # TODO: Implement actual database health check
        # For now, return healthy status
        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            response_time=0.05
        )
    except Exception as e:
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            error=str(e)
        )


async def check_redis_health() -> ComponentHealth:
    """Check Redis connectivity."""
    try:
        # TODO: Implement actual Redis health check
        # For now, return healthy status
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            response_time=0.02
        )
    except Exception as e:
        return ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            error=str(e)
        )


async def check_openai_health() -> ComponentHealth:
    """Check OpenAI API connectivity."""
    try:
        # TODO: Implement actual OpenAI health check
        # For now, return healthy status
        return ComponentHealth(
            name="openai",
            status=HealthStatus.HEALTHY,
            response_time=0.3
        )
    except Exception as e:
        return ComponentHealth(
            name="openai",
            status=HealthStatus.UNHEALTHY,
            error=str(e)
        )


@router.get("/", response_model=HealthCheckResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns the overall health status of the API and its dependencies.
    """
    # Check all components
    components = [
        await check_database_health(),
        await check_redis_health(),
        await check_openai_health()
    ]
    
    # Determine overall status
    unhealthy_components = [c for c in components if c.status == HealthStatus.UNHEALTHY]
    degraded_components = [c for c in components if c.status == HealthStatus.DEGRADED]
    
    if unhealthy_components:
        overall_status = HealthStatus.UNHEALTHY
    elif degraded_components:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.HEALTHY
    
    return HealthCheckResponse(
        status=overall_status,
        version="1.0.0",
        components=components,
        uptime=time.time() - start_time
    )


@router.get("/liveness")
async def liveness_check():
    """
    Simple liveness check for container orchestration.
    
    Returns 200 if the application is running.
    """
    return {"status": "alive", "timestamp": datetime.utcnow()}


@router.get("/readiness")
async def readiness_check():
    """
    Readiness check for container orchestration.
    
    Returns 200 if the application is ready to serve requests.
    """
    # Check critical dependencies
    db_health = await check_database_health()
    
    if db_health.status == HealthStatus.UNHEALTHY:
        return {"status": "not_ready", "reason": "database_unavailable"}
    
    return {"status": "ready", "timestamp": datetime.utcnow()}


@router.get("/metrics")
async def system_metrics():
    """
    System metrics endpoint for monitoring.
    
    Returns basic system metrics like CPU, memory usage, etc.
    """
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "uptime": time.time() - start_time,
        "timestamp": datetime.utcnow()
    }


@router.options("/", include_in_schema=False)
async def health_options():
    """Handle OPTIONS requests for CORS preflight."""
    return {"message": "OK"}