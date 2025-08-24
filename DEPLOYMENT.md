# Flight Scheduling Analysis System - Deployment Guide

## Overview

This document provides comprehensive instructions for deploying the Flight Scheduling Analysis System using Docker containers. The system supports multiple deployment environments: development, staging, and production.

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 8GB, Recommended 16GB
- **Storage**: Minimum 20GB free space
- **CPU**: Multi-core processor recommended

### Required Software

1. **Docker Desktop** (Windows/macOS) or **Docker Engine** (Linux)
   - Version 20.10 or higher
   - Download: https://www.docker.com/products/docker-desktop

2. **Docker Compose**
   - Version 2.0 or higher
   - Usually included with Docker Desktop

3. **Git** (for cloning the repository)
   - Download: https://git-scm.com/downloads

### Optional Tools

- **OpenSSL** (for SSL certificate generation)
- **curl** (for health checks)
- **PowerShell** (Windows users)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd flight-scheduling-analysis
```

### 2. Environment Setup

Copy the example environment file and configure it:

```bash
# Linux/macOS
cp .env.example .env

# Windows
copy .env.example .env
```

Edit the `.env` file with your configuration:

```env
# Database Configuration
POSTGRES_DB=flightdb
POSTGRES_USER=user
POSTGRES_PASSWORD=your_secure_password

# Redis Configuration
REDIS_PASSWORD=your_redis_password

# InfluxDB Configuration
INFLUXDB_USERNAME=admin
INFLUXDB_PASSWORD=your_influxdb_password
INFLUXDB_ORG=flight-analysis
INFLUXDB_BUCKET=flight-metrics
INFLUXDB_TOKEN=your_influxdb_token

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Monitoring (Production)
GRAFANA_PASSWORD=your_grafana_password
```

### 3. Deploy the System

#### Linux/macOS
```bash
./scripts/deploy.sh development
```

#### Windows (PowerShell)
```powershell
.\scripts\deploy.ps1 development
```

### 4. Verify Deployment

Access the following URLs to verify the deployment:

- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8000/health

## Deployment Environments

### Development Environment

**Purpose**: Local development and testing

**Features**:
- Hot reloading enabled
- Debug logging
- Development SSL certificates
- Volume mounts for code changes

**Command**:
```bash
# Linux/macOS
./scripts/deploy.sh development

# Windows
.\scripts\deploy.ps1 development
```

**Services**:
- PostgreSQL (port 5432)
- Redis (port 6379)
- InfluxDB (port 8086)
- FastAPI (port 8000)
- Streamlit Dashboard (port 8501)
- Celery Worker
- Celery Scheduler

### Staging Environment

**Purpose**: Pre-production testing with monitoring

**Features**:
- Production-like configuration
- Monitoring enabled (Prometheus + Grafana)
- SSL certificates
- Performance testing

**Command**:
```bash
# Linux/macOS
./scripts/deploy.sh staging

# Windows
.\scripts\deploy.ps1 staging
```

**Additional Services**:
- Prometheus (port 9090)
- Grafana (port 3000)

### Production Environment

**Purpose**: Live production deployment

**Features**:
- Optimized Docker images
- Security hardening
- SSL/TLS encryption
- Load balancing with Nginx
- Comprehensive monitoring
- Resource limits
- Auto-restart policies

**Command**:
```bash
# Linux/macOS
./scripts/deploy.sh production

# Windows
.\scripts\deploy.ps1 production
```

**Additional Services**:
- Nginx Reverse Proxy (ports 80, 443)
- Prometheus (monitoring)
- Grafana (visualization)

## Docker Architecture

### Multi-Stage Builds

The system uses multi-stage Docker builds for optimization:

1. **Base Stage**: Common dependencies and system setup
2. **Development Stage**: Development tools and hot reloading
3. **Production Stage**: Optimized for performance and security

### Container Security

- Non-root user execution
- Minimal base images (Alpine Linux)
- Security headers in Nginx
- Environment variable encryption
- Network isolation

### Health Checks

All containers include comprehensive health checks:

- **API**: HTTP endpoint monitoring
- **Database**: Connection testing
- **Cache**: Redis ping
- **Worker**: Celery inspection
- **Dashboard**: Streamlit health endpoint

## Service Configuration

### PostgreSQL Database

**Configuration**: `config/postgres/`
- Optimized for time-series data
- Automatic backups
- Connection pooling
- Performance tuning

### Redis Cache

**Configuration**: `config/redis.conf`
- Memory optimization
- Persistence configuration
- Security settings
- Performance tuning

### InfluxDB Time Series

**Configuration**: Environment variables
- Retention policies
- Bucket organization
- Token-based authentication
- Performance optimization

### Nginx Reverse Proxy

**Configuration**: `config/nginx.conf`
- SSL/TLS termination
- Load balancing
- Rate limiting
- Security headers
- Gzip compression

## Monitoring and Logging

### Prometheus Metrics

**Configuration**: `config/prometheus.yml`

**Monitored Metrics**:
- Application performance
- Database connections
- Cache hit rates
- API response times
- System resources

### Grafana Dashboards

**Access**: http://localhost:3000
**Default Credentials**: admin/admin123

**Available Dashboards**:
- System Overview
- API Performance
- Database Metrics
- Cache Performance
- Business Metrics

### Centralized Logging

**Log Locations**:
- Application logs: `./logs/`
- Container logs: `docker-compose logs`
- System logs: Docker Desktop logs

**Log Levels**:
- Development: DEBUG
- Staging: INFO
- Production: WARNING

## Backup and Recovery

### Automated Backups

```bash
# Create backup
./scripts/backup.sh

# Create named backup
./scripts/backup.sh "pre_deployment_backup"
```

**Backup Contents**:
- PostgreSQL database dump
- InfluxDB time-series data
- Redis cache data
- Application data files
- Configuration files

### Backup Schedule

Production environments should implement automated backups:

```bash
# Add to crontab for daily backups at 2 AM
0 2 * * * /path/to/project/scripts/backup.sh
```

### Recovery Process

1. Stop all services
2. Extract backup archive
3. Restore database dumps
4. Copy application data
5. Restart services
6. Verify system health

## Scaling and Performance

### Horizontal Scaling

Scale individual services based on load:

```bash
# Scale API instances
docker-compose up -d --scale api=3

# Scale workers
docker-compose up -d --scale worker=5

# Scale with load balancer
docker-compose -f docker-compose.prod.yml up -d
```

### Performance Optimization

**Database Optimization**:
- Connection pooling
- Query optimization
- Index management
- Partitioning for large datasets

**Cache Optimization**:
- Redis clustering
- Cache warming strategies
- TTL optimization
- Memory management

**Application Optimization**:
- Async processing
- Background tasks
- API rate limiting
- Response caching

## Security Considerations

### Network Security

- Internal Docker network isolation
- Firewall configuration
- VPN access for production
- SSL/TLS encryption

### Application Security

- Environment variable encryption
- API authentication
- Input validation
- SQL injection prevention
- XSS protection

### Infrastructure Security

- Regular security updates
- Container scanning
- Secrets management
- Access logging
- Intrusion detection

## Troubleshooting

### Common Issues

#### Services Not Starting

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs [service_name]

# Restart specific service
docker-compose restart [service_name]
```

#### Database Connection Issues

```bash
# Check database health
docker-compose exec postgres pg_isready -U user -d flightdb

# Reset database
docker-compose down
docker volume rm flight_postgres_data
docker-compose up -d
```

#### Memory Issues

```bash
# Check resource usage
docker stats

# Increase Docker memory limits
# Docker Desktop -> Settings -> Resources -> Memory
```

#### Port Conflicts

```bash
# Check port usage
netstat -tulpn | grep :8000

# Change ports in docker-compose.yml
# Restart services
```

### Health Check Script

Run comprehensive health checks:

```bash
# Linux/macOS
./scripts/health-check.sh

# Windows
.\scripts\health-check.ps1
```

### Log Analysis

```bash
# View recent errors
docker-compose logs --tail=100 | grep -i error

# Follow logs in real-time
docker-compose logs -f [service_name]

# Export logs
docker-compose logs > system_logs.txt
```

## Maintenance

### Regular Maintenance Tasks

1. **Weekly**:
   - Review system logs
   - Check disk space
   - Verify backups
   - Update dependencies

2. **Monthly**:
   - Security updates
   - Performance review
   - Backup testing
   - Capacity planning

3. **Quarterly**:
   - Full system audit
   - Disaster recovery testing
   - Performance optimization
   - Security assessment

### Update Process

1. Create system backup
2. Test updates in staging
3. Schedule maintenance window
4. Deploy updates to production
5. Verify system functionality
6. Monitor for issues

## Support and Documentation

### Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **System Metrics**: http://localhost:9090 (Prometheus)
- **Dashboards**: http://localhost:3000 (Grafana)
- **Health Status**: http://localhost:8000/health

### Getting Help

1. Check this documentation
2. Review system logs
3. Run health check script
4. Check GitHub issues
5. Contact system administrators

### Contributing

1. Fork the repository
2. Create feature branch
3. Test changes locally
4. Submit pull request
5. Update documentation

## Appendix

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POSTGRES_DB` | Database name | `flightdb` | Yes |
| `POSTGRES_USER` | Database user | `user` | Yes |
| `POSTGRES_PASSWORD` | Database password | - | Yes |
| `REDIS_PASSWORD` | Redis password | - | Production |
| `OPENAI_API_KEY` | OpenAI API key | - | Yes |
| `INFLUXDB_TOKEN` | InfluxDB token | - | Yes |
| `LOG_LEVEL` | Logging level | `INFO` | No |

### Port Reference

| Service | Development | Production |
|---------|-------------|------------|
| PostgreSQL | 5432 | Internal |
| Redis | 6379 | Internal |
| InfluxDB | 8086 | Internal |
| FastAPI | 8000 | Internal |
| Streamlit | 8501 | Internal |
| Nginx | - | 80, 443 |
| Prometheus | 9090 | Internal |
| Grafana | 3000 | Internal |

### Docker Commands Reference

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service]

# Execute commands
docker-compose exec [service] [command]

# Scale services
docker-compose up -d --scale [service]=[count]

# Update images
docker-compose pull
docker-compose up -d

# Clean up
docker system prune -a
```