"""
FastAPI main application for Flight Scheduling Analysis System.

This module sets up the FastAPI application with proper routing, middleware,
and configuration for the flight scheduling analysis system.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

from src.api.routers import data, analysis, nlp, health, reports
from src.api.middleware import LoggingMiddleware, ErrorHandlingMiddleware
from src.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Flight Scheduling Analysis API")
    # Startup logic here
    yield
    # Shutdown logic here
    logger.info("Shutting down Flight Scheduling Analysis API")


# Create FastAPI application
app = FastAPI(
    title="Flight Scheduling Analysis API",
    description="""
    AI-powered flight scheduling analysis system for optimizing airport operations.
    
    This API provides endpoints for:
    - Flight data management and processing
    - Delay analysis and optimization recommendations
    - Congestion analysis and peak hour identification
    - Schedule impact modeling and what-if analysis
    - Cascading impact analysis for critical flight identification
    - Natural language query processing for flight insights
    
    ## Features
    
    - **Open Source AI Tools**: Leverages scikit-learn, XGBoost, Prophet, NetworkX, and spaCy
    - **Natural Language Interface**: OpenAI-powered query processing
    - **Real-time Analysis**: Live flight data processing and insights
    - **Comprehensive Reporting**: Interactive visualizations and export capabilities
    """,
    version="1.0.0",
    contact={
        "name": "Flight Scheduling Analysis Team",
        "email": "support@flightanalysis.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom schema extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Include routers
app.include_router(
    health.router,
    prefix="/api/v1/health",
    tags=["Health Check"]
)

app.include_router(
    data.router,
    prefix="/api/v1/data",
    tags=["Data Management"]
)

app.include_router(
    analysis.router,
    prefix="/api/v1/analysis",
    tags=["Flight Analysis"]
)

app.include_router(
    nlp.router,
    prefix="/api/v1/nlp",
    tags=["Natural Language Processing"]
)

app.include_router(
    reports.router,
    prefix="/api/v1/reports",
    tags=["Reports and Export"]
)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Flight Scheduling Analysis API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }


@app.options("/", include_in_schema=False)
async def root_options():
    """Handle OPTIONS requests for CORS preflight."""
    return {"message": "OK"}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))
    
    logger.error(
        f"HTTP Exception: {exc.status_code} - {exc.detail}",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "correlation_id": correlation_id,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with proper logging."""
    correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))
    
    logger.error(
        f"Unhandled Exception: {str(exc)}",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
            "method": request.method
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "correlation_id": correlation_id,
            "timestamp": time.time()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.environment == "development" else False,
        log_level="info"
    )