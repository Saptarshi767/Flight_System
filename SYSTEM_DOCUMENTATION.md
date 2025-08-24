# Flight Scheduling Analysis System - Complete Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [API Documentation](#api-documentation)
5. [Database Schema](#database-schema)
6. [Performance Optimization](#performance-optimization)
7. [Security Features](#security-features)
8. [Monitoring & Logging](#monitoring--logging)
9. [Deployment Guide](#deployment-guide)
10. [User Guide](#user-guide)
11. [Developer Guide](#developer-guide)
12. [Troubleshooting](#troubleshooting)

## System Overview

The Flight Scheduling Analysis System is a comprehensive AI-powered platform designed to optimize flight scheduling at busy airports, particularly Mumbai (BOM) and Delhi (DEL). The system integrates multiple data sources, processes flight information using open-source machine learning algorithms, and provides actionable insights through a natural language interface powered by OpenAI.

### Key Features

- **Multi-source Data Integration**: Excel, CSV, and web scraping from FlightRadar24 and FlightAware
- **AI-Powered Analysis**: Uses scikit-learn, XGBoost, Prophet, and NetworkX for comprehensive analysis
- **Natural Language Interface**: OpenAI GPT-4 integration for intuitive querying
- **Real-time Processing**: Live data ingestion and analysis capabilities
- **Scalable Architecture**: Horizontal scaling with load balancing and auto-scaling
- **Comprehensive Reporting**: Interactive dashboards and automated report generation

### Project Expectations Fulfillment

#### ✅ EXPECTATION 1: Open Source AI Tools + NLP Interface
- **Open Source Tools**: scikit-learn, XGBoost, Prophet, NetworkX, spaCy
- **NLP Interface**: OpenAI GPT-4 integration with LangChain orchestration
- **Natural Language Queries**: Chat-like interface for flight data analysis

#### ✅ EXPECTATION 2: Best Takeoff/Landing Times
- **Scheduled vs Actual Analysis**: Comprehensive delay pattern analysis
- **Optimal Time Identification**: ML-powered recommendations for best scheduling windows
- **Historical Pattern Recognition**: Time-of-day analysis for minimal delays

#### ✅ EXPECTATION 3: Busiest Time Slots to Avoid
- **Peak Hour Identification**: Time series analysis of flight density
- **Congestion Forecasting**: Prophet-based prediction of busy periods
- **Alternative Slot Recommendations**: Smart scheduling suggestions

#### ✅ EXPECTATION 4: Schedule Tuning Model
- **What-if Analysis**: Impact simulation for schedule changes
- **Delay Impact Prediction**: ML models for cascading effect analysis
- **Optimization Algorithms**: Schedule adjustment recommendations

#### ✅ EXPECTATION 5: Cascading Impact Analysis
- **Network Graph Modeling**: NetworkX-based flight connection analysis
- **Critical Flight Identification**: Ranking by cascading impact potential
- **Delay Propagation Tracking**: Real-time impact monitoring

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Load Balancer  │    │   Web Client    │
│                 │    │    (HAProxy)    │    │   (Browser)     │
│ • FlightRadar24 │    │                 │    │                 │
│ • FlightAware   │    └─────────────────┘    └─────────────────┘
│ • Excel/CSV     │             │                       │
└─────────────────┘             │                       │
         │                      │                       │
         │              ┌───────▼───────┐       ┌───────▼───────┐
         │              │  API Gateway  │       │   Dashboard   │
         │              │   (FastAPI)   │       │  (Streamlit)  │
         │              └───────────────┘       └───────────────┘
         │                      │
         │              ┌───────▼───────┐
         │              │ Analysis Layer│
         │              │               │
         │              │ • Delay       │
         │              │ • Congestion  │
         │              │ • Cascading   │
         │              │ • ML Models   │
         │              └───────────────┘
         │                      │
         │              ┌───────▼───────┐
         │              │  NLP Layer    │
         │              │               │
         │              │ • OpenAI GPT  │
         │              │ • LangChain   │
         │              │ • spaCy       │
         │              └───────────────┘
         │                      │
         ▼              ┌───────▼───────┐
┌─────────────────┐     │ Data Storage  │
│ Data Processing │     │               │
│                 │     │ • PostgreSQL  │
│ • ETL Pipeline  │◄────┤ • InfluxDB    │
│ • Validation    │     │ • Redis Cache │
│ • Cleaning      │     └───────────────┘
└─────────────────┘
```

### Component Architecture

#### 1. Data Ingestion Layer
- **ExcelDataProcessor**: Handles Excel file processing and conversion
- **WebScrapers**: FlightRadar24 and FlightAware data extraction
- **DataProcessor**: Unified data processing and validation pipeline
- **Real-time Ingestion**: Streaming data processing capabilities

#### 2. Storage Layer
- **PostgreSQL**: Primary relational database for structured data
- **InfluxDB**: Time series database for metrics and analytics
- **Redis**: Caching layer and session storage
- **File Storage**: Document and report storage

#### 3. Analysis Layer
- **DelayAnalyzer**: Scheduled vs actual time analysis
- **CongestionAnalyzer**: Peak hour and traffic density analysis
- **CascadingImpactAnalyzer**: Network effect and critical flight identification
- **ScheduleImpactModeler**: What-if analysis and optimization

#### 4. ML/AI Layer
- **Prediction Models**: XGBoost and scikit-learn based models
- **Time Series Forecasting**: Prophet for congestion and delay prediction
- **Anomaly Detection**: Isolation Forest for unusual pattern detection
- **Network Analysis**: NetworkX for flight connection modeling

#### 5. NLP Layer
- **QueryProcessor**: Natural language query understanding
- **LangChainOrchestrator**: Complex query processing and context management
- **ResponseGenerator**: Intelligent response formatting and visualization suggestions
- **OpenAI Integration**: GPT-4 powered natural language interface

#### 6. API Layer
- **FastAPI Application**: RESTful API endpoints
- **Authentication**: JWT-based security
- **Rate Limiting**: Request throttling and abuse prevention
- **WebSocket Support**: Real-time data streaming

#### 7. Presentation Layer
- **Streamlit Dashboard**: Interactive web interface
- **Report Generator**: Automated report creation
- **Visualization Engine**: Plotly-based interactive charts
- **Export Functionality**: Multiple format support

## Installation & Setup

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+
- InfluxDB 2.7+

### Environment Setup

1. **Clone the Repository**
```bash
git clone <repository-url>
cd flight-scheduling-analysis
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Database Setup**
```bash
# Initialize PostgreSQL database
python init_database.py

# Run migrations
alembic upgrade head
```

6. **Start Services**
```bash
# Development mode
docker-compose up -d

# Production mode
docker-compose -f docker-compose.prod.yml up -d

# Scaled deployment
docker-compose -f docker-compose.scale.yml up -d
```

### Configuration

#### Environment Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://flight_user:flight_pass@localhost:5432/flight_analysis
REDIS_URL=redis://localhost:6379/0
INFLUXDB_URL=http://localhost:8086

# API Keys
OPENAI_API_KEY=your_openai_api_key_here

# Security
JWT_SECRET_KEY=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key

# Data Sources
FLIGHTRADAR24_URL=https://www.flightradar24.com/
FLIGHTAWARE_URL=https://www.flightaware.com/
MUMBAI_AIRPORT_URL=https://www.flightradar24.com/data/airports/bom
DELHI_AIRPORT_URL=https://www.flightradar24.com/data/airports/del

# Performance
CACHE_TTL=3600
MAX_WORKERS=10
RATE_LIMIT_REQUESTS=100
```

## API Documentation

### Authentication

All API endpoints require authentication via JWT tokens or API keys.

```bash
# Get JWT token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'

# Use token in requests
curl -H "Authorization: Bearer <token>" \
  "http://localhost:8000/api/v1/data/flights"
```

### Core Endpoints

#### Data Management

```bash
# Upload flight data
POST /api/v1/data/upload
Content-Type: multipart/form-data
Body: file (Excel/CSV)

# Get flight data
GET /api/v1/data/flights
Query Parameters:
  - airline: string (optional)
  - origin_airport: string (optional)
  - destination_airport: string (optional)
  - date_from: date (optional)
  - date_to: date (optional)
  - limit: integer (default: 100)
  - offset: integer (default: 0)

# Get specific flight
GET /api/v1/data/flights/{flight_id}

# Update flight
PUT /api/v1/data/flights/{flight_id}
Content-Type: application/json
Body: FlightDataUpdate object

# Delete flight
DELETE /api/v1/data/flights/{flight_id}
```

#### Analysis Endpoints

```bash
# Delay analysis
GET /api/v1/analysis/delays
Query Parameters:
  - airport_code: string (required)
  - days_back: integer (default: 30)
  - airline: string (optional)

# Congestion analysis
GET /api/v1/analysis/congestion
Query Parameters:
  - airport_code: string (required)
  - days_back: integer (default: 30)

# Schedule impact analysis
POST /api/v1/analysis/schedule-impact
Content-Type: application/json
Body: {
  "flight_id": "string",
  "proposed_changes": {
    "new_departure_time": "datetime",
    "new_arrival_time": "datetime"
  }
}

# Cascading impact analysis
GET /api/v1/analysis/cascading-impact
Query Parameters:
  - airport_code: string (required)
  - days_back: integer (default: 30)
```

#### NLP Endpoints

```bash
# Process natural language query
POST /api/v1/nlp/query
Content-Type: application/json
Body: {
  "query": "What are the busiest hours at Mumbai airport?",
  "context": {},
  "session_id": "optional_session_id"
}

# Get query suggestions
GET /api/v1/nlp/suggestions

# Submit query feedback
POST /api/v1/nlp/feedback
Content-Type: application/json
Body: {
  "query_id": "string",
  "rating": 1-5,
  "feedback": "string"
}

# Get query history
GET /api/v1/nlp/history
Query Parameters:
  - session_id: string (optional)
  - limit: integer (default: 20)
```

#### Reporting Endpoints

```bash
# Get dashboard data
GET /api/v1/reports/dashboard
Query Parameters:
  - airport_code: string (optional)
  - date_range: string (optional)

# Generate report
POST /api/v1/reports/generate
Content-Type: application/json
Body: {
  "report_type": "delay_analysis|congestion_analysis|route_analysis",
  "airport_code": "string",
  "date_range": {
    "start_date": "date",
    "end_date": "date"
  },
  "format": "pdf|json|csv"
}

# Export data
GET /api/v1/reports/export/{format}
Query Parameters:
  - data_type: string
  - filters: object (optional)
```

### Response Format

All API responses follow a consistent format:

```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": {
    // Response data
  },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "uuid",
    "processing_time": 0.123
  }
}
```

Error responses:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      // Additional error details
    }
  },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "uuid"
  }
}
```

## Database Schema

### Core Tables

#### Flights Table
```sql
CREATE TABLE flights (
    id UUID PRIMARY KEY,
    flight_id VARCHAR(50) UNIQUE NOT NULL,
    flight_number VARCHAR(10) NOT NULL,
    airline_code VARCHAR(3) NOT NULL,
    origin_airport VARCHAR(3) NOT NULL,
    destination_airport VARCHAR(3) NOT NULL,
    aircraft_type VARCHAR(10),
    scheduled_departure TIMESTAMP NOT NULL,
    scheduled_arrival TIMESTAMP NOT NULL,
    actual_departure TIMESTAMP,
    actual_arrival TIMESTAMP,
    departure_delay_minutes INTEGER DEFAULT 0,
    arrival_delay_minutes INTEGER DEFAULT 0,
    delay_category VARCHAR(20),
    status VARCHAR(20) DEFAULT 'scheduled',
    passenger_count INTEGER,
    runway_used VARCHAR(10),
    gate VARCHAR(10),
    data_source VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### Airports Table
```sql
CREATE TABLE airports (
    code VARCHAR(3) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    city VARCHAR(50) NOT NULL,
    country VARCHAR(50) NOT NULL,
    runway_count INTEGER NOT NULL,
    runway_capacity INTEGER NOT NULL,
    timezone VARCHAR(50) NOT NULL,
    latitude FLOAT,
    longitude FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### Airlines Table
```sql
CREATE TABLE airlines (
    code VARCHAR(3) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    country VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Indexes for Performance

```sql
-- Flight data indexes
CREATE INDEX idx_flights_route_date ON flights (origin_airport, destination_airport, scheduled_departure DESC);
CREATE INDEX idx_flights_delay_analysis ON flights (departure_delay_minutes, delay_category, scheduled_departure DESC);
CREATE INDEX idx_flights_airline_performance ON flights (airline_code, scheduled_departure DESC, departure_delay_minutes);
CREATE INDEX idx_flights_airport_congestion ON flights (origin_airport, scheduled_departure) INCLUDE (destination_airport, departure_delay_minutes);

-- Partial indexes for recent data
CREATE INDEX idx_flights_recent ON flights (scheduled_departure DESC, origin_airport, destination_airport) 
WHERE scheduled_departure >= CURRENT_DATE - INTERVAL '30 days';
```

## Performance Optimization

### Database Optimization

1. **Query Optimization**
   - Composite indexes for common query patterns
   - Partial indexes for recent data
   - Query result caching with Redis
   - Connection pooling with optimized settings

2. **Caching Strategy**
   - Multi-level caching (L1: Application, L2: Redis)
   - Cache invalidation on data updates
   - Background cache warming
   - Smart cache key generation

3. **Connection Management**
   - Connection pooling with SQLAlchemy
   - Read replicas for query optimization
   - Connection health monitoring

### Application Optimization

1. **Load Balancing**
   - HAProxy for request distribution
   - Health check monitoring
   - Automatic failover
   - Session affinity for NLP queries

2. **Auto-scaling**
   - Horizontal scaling based on load metrics
   - Container orchestration with Docker Compose
   - Resource monitoring and alerting
   - Graceful scaling up/down

3. **Asynchronous Processing**
   - Background task processing with Celery
   - Queue-based request handling
   - Non-blocking I/O operations
   - Batch processing for large datasets

### Performance Monitoring

```python
# Performance metrics collection
from src.utils.metrics import metrics_collector

# Database query monitoring
@metrics_collector.time_function
def get_flight_data(filters):
    # Query implementation
    pass

# API endpoint monitoring
@metrics_collector.track_requests
async def api_endpoint(request):
    # Endpoint implementation
    pass
```

## Security Features

### Authentication & Authorization

1. **JWT Token Authentication**
   - Secure token generation and validation
   - Configurable expiration times
   - Token refresh mechanism
   - Role-based access control

2. **API Key Management**
   - Secure API key generation
   - Key rotation capabilities
   - Usage tracking and limits
   - Scope-based permissions

### Data Protection

1. **Encryption**
   - Data at rest encryption
   - Sensitive field encryption
   - Secure key management
   - TLS/SSL for data in transit

2. **Input Validation**
   - SQL injection prevention
   - XSS protection
   - Input sanitization
   - Parameter validation

### Security Monitoring

1. **Rate Limiting**
   - Request throttling per IP/user
   - Configurable limits and windows
   - Abuse detection and blocking
   - Whitelist/blacklist management

2. **Audit Logging**
   - Comprehensive access logging
   - Security event monitoring
   - Failed authentication tracking
   - Data modification auditing

### Security Headers

```python
# Security headers automatically added
{
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}
```

## Monitoring & Logging

### Application Monitoring

1. **Metrics Collection**
   - Request/response metrics
   - Database performance metrics
   - Cache hit/miss ratios
   - Error rates and types

2. **Health Checks**
   - Service availability monitoring
   - Database connectivity checks
   - External service health
   - Resource utilization tracking

### Logging Strategy

1. **Structured Logging**
   - JSON formatted logs
   - Correlation ID tracking
   - Log level management
   - Centralized log aggregation

2. **Log Analysis**
   - Error pattern detection
   - Performance bottleneck identification
   - User behavior analysis
   - Security event monitoring

### Alerting

1. **Threshold-based Alerts**
   - Response time alerts
   - Error rate alerts
   - Resource utilization alerts
   - Service availability alerts

2. **Notification Channels**
   - Email notifications
   - Slack integration
   - SMS alerts for critical issues
   - Dashboard notifications

## Deployment Guide

### Development Deployment

```bash
# Start development environment
docker-compose up -d

# Initialize database
python init_database.py

# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start dashboard
streamlit run src/dashboard/main.py --server.port 8501
```

### Production Deployment

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with scaling
docker-compose -f docker-compose.scale.yml up -d

# Verify deployment
docker-compose ps
curl http://localhost/health
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flight-analysis-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: flight-analysis-api
  template:
    metadata:
      labels:
        app: flight-analysis-api
    spec:
      containers:
      - name: api
        image: flight-analysis:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: flight-analysis-secrets
              key: database-url
```

### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Run tests
      run: |
        python -m pytest tests/
        python -m pytest tests/performance/
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      run: |
        docker-compose -f docker-compose.prod.yml up -d
```

## User Guide

### Getting Started

1. **Access the System**
   - Web Dashboard: http://localhost:8501
   - API Documentation: http://localhost:8000/docs
   - Monitoring: http://localhost:3000 (Grafana)

2. **Upload Flight Data**
   - Navigate to Data Upload section
   - Select Excel or CSV file
   - Review data validation results
   - Confirm upload

3. **Run Analysis**
   - Select analysis type (Delay, Congestion, Cascading Impact)
   - Choose airport and date range
   - Review results and recommendations

### Natural Language Queries

The system supports natural language queries for intuitive data exploration:

**Example Queries:**
- "What are the busiest hours at Mumbai airport?"
- "Show me delay patterns for flights last week"
- "Which flights cause the most cascading delays?"
- "What's the best time to schedule a flight from Delhi to Mumbai?"
- "Compare congestion between Mumbai and Delhi airports"

### Dashboard Features

1. **Overview Dashboard**
   - Key performance indicators
   - Recent analysis summaries
   - System health status
   - Quick action buttons

2. **Analysis Dashboards**
   - Interactive charts and graphs
   - Drill-down capabilities
   - Export functionality
   - Real-time updates

3. **Reporting**
   - Automated report generation
   - Multiple export formats
   - Scheduled reports
   - Custom report templates

### Best Practices

1. **Data Quality**
   - Ensure complete flight data
   - Validate data before upload
   - Regular data quality checks
   - Monitor data freshness

2. **Analysis Interpretation**
   - Consider seasonal patterns
   - Account for external factors
   - Validate recommendations
   - Monitor implementation results

## Developer Guide

### Project Structure

```
flight-scheduling-analysis/
├── src/
│   ├── api/                 # FastAPI application
│   ├── analysis/           # Analysis engines
│   ├── data/               # Data processing
│   ├── database/           # Database models and operations
│   ├── ml/                 # Machine learning models
│   ├── nlp/                # Natural language processing
│   ├── reporting/          # Report generation
│   ├── security/           # Security features
│   ├── utils/              # Utility functions
│   └── dashboard/          # Streamlit dashboard
├── tests/                  # Test suites
├── config/                 # Configuration files
├── docs/                   # Documentation
├── scripts/                # Utility scripts
└── docker-compose*.yml     # Docker configurations
```

### Development Setup

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use type hints
   - Write comprehensive docstrings
   - Maintain test coverage >90%

2. **Testing**
   ```bash
   # Run all tests
   pytest
   
   # Run specific test suite
   pytest tests/unit/
   pytest tests/integration/
   pytest tests/performance/
   
   # Generate coverage report
   pytest --cov=src --cov-report=html
   ```

3. **Code Quality**
   ```bash
   # Linting
   flake8 src/
   black src/
   isort src/
   
   # Type checking
   mypy src/
   ```

### Adding New Features

1. **Analysis Engine**
   ```python
   # src/analysis/new_analyzer.py
   from src.analysis.base import BaseAnalyzer
   
   class NewAnalyzer(BaseAnalyzer):
       def analyze(self, data):
           # Implementation
           pass
   ```

2. **API Endpoint**
   ```python
   # src/api/routers/new_endpoint.py
   from fastapi import APIRouter
   
   router = APIRouter(prefix="/api/v1/new", tags=["new"])
   
   @router.get("/")
   async def get_new_data():
       # Implementation
       pass
   ```

3. **ML Model**
   ```python
   # src/ml/new_model.py
   from src.ml.base import BaseModel
   
   class NewModel(BaseModel):
       def train(self, data):
           # Implementation
           pass
       
       def predict(self, features):
           # Implementation
           pass
   ```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Performance Testing

```bash
# Database performance tests
pytest tests/performance/test_database_performance.py

# Load testing
python tests/performance/load_test.py --users 50 --requests 1000

# Stress testing
python tests/performance/load_test.py --stress-test --max-users 200
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   ```bash
   # Check database status
   docker-compose ps postgres
   
   # View database logs
   docker-compose logs postgres
   
   # Test connection
   python -c "from src.database.connection import test_database_connection; print(test_database_connection())"
   ```

2. **Cache Issues**
   ```bash
   # Check Redis status
   docker-compose ps redis
   
   # Clear cache
   python -c "from src.utils.cache import cache_manager; cache_manager.flush_all()"
   
   # Monitor cache performance
   python -c "from src.utils.cache import cache_manager; print(cache_manager.get_stats())"
   ```

3. **API Performance Issues**
   ```bash
   # Check API health
   curl http://localhost:8000/health
   
   # Monitor response times
   curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/v1/data/flights
   
   # Check rate limiting
   curl -I http://localhost:8000/api/v1/data/flights
   ```

### Debugging

1. **Enable Debug Logging**
   ```python
   # Set environment variable
   export LOG_LEVEL=DEBUG
   
   # Or in code
   import logging
   logging.getLogger().setLevel(logging.DEBUG)
   ```

2. **Database Query Debugging**
   ```python
   # Enable SQL logging
   export DATABASE_ECHO=true
   
   # Or in code
   engine = create_engine(database_url, echo=True)
   ```

3. **Performance Profiling**
   ```python
   # Profile API endpoints
   from src.utils.profiler import profile_endpoint
   
   @profile_endpoint
   async def slow_endpoint():
       # Implementation
       pass
   ```

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| DB001 | Database connection failed | Check database service and credentials |
| CACHE001 | Redis connection failed | Check Redis service and configuration |
| API001 | Authentication failed | Verify JWT token or API key |
| RATE001 | Rate limit exceeded | Reduce request frequency or increase limits |
| DATA001 | Data validation failed | Check data format and required fields |
| ML001 | Model prediction failed | Retrain model or check input data |
| NLP001 | Query processing failed | Check OpenAI API key and quota |

### Support

For additional support:
- Check the [GitHub Issues](https://github.com/your-repo/issues)
- Review the [API Documentation](http://localhost:8000/docs)
- Monitor system health at [Grafana Dashboard](http://localhost:3000)
- Contact the development team

---

*This documentation is automatically updated with each release. Last updated: 2024-01-01*