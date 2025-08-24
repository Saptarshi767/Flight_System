# Database Module

This module provides comprehensive database functionality for the Flight Scheduling Analysis System, including PostgreSQL schema management, CRUD operations, and database utilities.

## Overview

The database module is organized into several key components:

- **Models**: SQLAlchemy ORM models for all database tables
- **Connection**: Database connection management and session handling
- **Operations**: Repository pattern for CRUD operations
- **Utils**: Database utilities, migrations, and management tools

## Database Schema

### Core Tables

#### Airports
- **Purpose**: Store airport information and operational data
- **Key Fields**: IATA code, name, location, runway capacity
- **Indexes**: City, country, operational status

#### Airlines
- **Purpose**: Store airline information and operational status
- **Key Fields**: IATA code, name, country, active status
- **Indexes**: Name, country, active status

#### Aircraft
- **Purpose**: Store aircraft type information and specifications
- **Key Fields**: Type code, manufacturer, seating capacity, performance data
- **Indexes**: Type code, manufacturer

#### Flights
- **Purpose**: Core flight data with scheduling and delay information
- **Key Fields**: Flight ID, route, times, delays, operational data
- **Indexes**: Multiple indexes for performance optimization
- **Relationships**: Links to airports, airlines, and aircraft

#### Analysis Results
- **Purpose**: Store results from various analysis engines
- **Key Fields**: Analysis type, metrics, recommendations, confidence scores
- **Indexes**: Analysis type, date, airport

#### Data Ingestion Logs
- **Purpose**: Track data ingestion processes and quality
- **Key Fields**: Source type, processing statistics, error information
- **Indexes**: Source type, status, timestamps

## Usage Examples

### Basic Database Operations

```python
from src.database import (
    db_session_scope, 
    flight_repo, 
    airport_repo,
    get_flights_dataframe
)

# Create a new flight record
with db_session_scope() as session:
    flight_data = {
        'flight_id': 'AI101-20240101-BOM-DEL',
        'flight_number': 'AI101',
        'airline_code': 'AI',
        'origin_airport': 'BOM',
        'destination_airport': 'DEL',
        'scheduled_departure': datetime(2024, 1, 1, 10, 0),
        'scheduled_arrival': datetime(2024, 1, 1, 12, 30),
        'data_source': 'api'
    }
    flight = flight_repo.create_flight(session, flight_data)

# Query flights for analysis
with db_session_scope() as session:
    delayed_flights = flight_repo.get_delayed_flights(
        session, 
        min_delay_minutes=15,
        airport_code='BOM'
    )

# Get flights as pandas DataFrame
df = get_flights_dataframe(
    airport_code='BOM',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)
```

### Database Management

```python
from src.database.utils import (
    init_database,
    check_database_health,
    create_migration,
    apply_migrations
)

# Initialize database with tables
init_database(create_sample_data=True)

# Check database health
health_info = check_database_health()
print(f"Connection: {health_info['connection_test']}")

# Create and apply migrations
create_migration("Add new analysis table")
apply_migrations()
```

### Using the CLI Tool

The database module includes a comprehensive CLI tool for management:

```bash
# Initialize database
python scripts/db_manager.py init --sample-data

# Check database health
python scripts/db_manager.py health

# Show database information
python scripts/db_manager.py info

# Create migration
python scripts/db_manager.py create-migration "Add new feature"

# Apply migrations
python scripts/db_manager.py migrate

# Backup database
python scripts/db_manager.py backup --file backup.sql

# Reset database (with confirmation)
python scripts/db_manager.py reset --confirm
```

## Configuration

### Environment Variables

```bash
# Database connection
DATABASE_URL=postgresql://user:password@localhost:5432/flightdb

# Additional database settings
DEBUG=true
LOG_LEVEL=INFO
```

### Docker Setup

The module works seamlessly with Docker Compose:

```bash
# Start PostgreSQL
docker-compose up postgres

# Start all services
docker-compose up
```

## Performance Optimization

### Indexes

The schema includes comprehensive indexing for optimal query performance:

- **Single column indexes**: For frequently filtered columns
- **Composite indexes**: For common query patterns
- **Partial indexes**: For conditional queries

### Query Optimization

- Connection pooling with SQLAlchemy
- Query result caching with Redis integration
- Bulk operations for large datasets
- Optimized joins with proper foreign key relationships

## Data Quality and Integrity

### Constraints

- **Check constraints**: Validate data ranges and business rules
- **Foreign key constraints**: Ensure referential integrity
- **Unique constraints**: Prevent duplicate records
- **Not null constraints**: Ensure required fields

### Data Validation

- Schema validation on data ingestion
- Data quality scoring for imported records
- Anomaly detection for unusual patterns
- Comprehensive error logging and reporting

## Migration Management

The module uses Alembic for database schema migrations:

### Creating Migrations

```bash
# Auto-generate migration from model changes
python scripts/db_manager.py create-migration "Description of changes"

# Apply pending migrations
python scripts/db_manager.py migrate

# View migration history
python scripts/db_manager.py history
```

### Migration Best Practices

- Always review auto-generated migrations
- Test migrations on development data
- Create rollback procedures for production
- Document significant schema changes

## Testing

The module includes comprehensive tests:

```bash
# Run database tests
python -m pytest tests/test_database.py -v

# Run with coverage
python -m pytest tests/test_database.py --cov=src.database
```

## Monitoring and Maintenance

### Health Checks

Regular health checks monitor:
- Database connectivity
- Table accessibility
- Performance metrics
- Data quality indicators

### Backup and Recovery

```bash
# Create backup
python scripts/db_manager.py backup --file backup_$(date +%Y%m%d).sql

# Restore from backup
python scripts/db_manager.py restore backup_20240101.sql --confirm
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure PostgreSQL is running
   - Check DATABASE_URL configuration
   - Verify network connectivity

2. **Migration Failures**
   - Check for conflicting schema changes
   - Verify database permissions
   - Review migration logs

3. **Performance Issues**
   - Analyze slow queries
   - Check index usage
   - Monitor connection pool

### Debug Mode

Enable debug mode for detailed SQL logging:

```python
# In settings
DEBUG=true

# Or programmatically
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

## Security Considerations

- Database credentials stored in environment variables
- Connection encryption in production
- SQL injection prevention through parameterized queries
- Access control through database roles and permissions

## Future Enhancements

- Read replicas for query scaling
- Partitioning for large historical data
- Advanced monitoring and alerting
- Automated backup scheduling
- Performance analytics and optimization