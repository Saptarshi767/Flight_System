#!/bin/bash

# Health check script for Flight Scheduling Analysis System
# Usage: ./scripts/health-check.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check service health
check_service_health() {
    local service_name=$1
    local health_url=$2
    local timeout=${3:-30}
    
    print_status "Checking $service_name health..."
    
    if curl -f -s --max-time $timeout "$health_url" > /dev/null; then
        print_success "$service_name is healthy"
        return 0
    else
        print_error "$service_name is not healthy"
        return 1
    fi
}

# Check Docker container status
check_container_status() {
    local container_name=$1
    
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$container_name.*Up"; then
        local status=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$container_name" | awk '{print $2, $3}')
        print_success "$container_name: $status"
        return 0
    else
        print_error "$container_name is not running"
        return 1
    fi
}

# Check database connectivity
check_database() {
    print_status "Checking database connectivity..."
    
    if docker-compose exec -T postgres pg_isready -U user -d flightdb > /dev/null 2>&1; then
        print_success "PostgreSQL database is accessible"
    else
        print_error "PostgreSQL database is not accessible"
        return 1
    fi
    
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        print_success "Redis cache is accessible"
    else
        print_error "Redis cache is not accessible"
        return 1
    fi
    
    if curl -f -s http://localhost:8086/ping > /dev/null; then
        print_success "InfluxDB is accessible"
    else
        print_error "InfluxDB is not accessible"
        return 1
    fi
}

# Check API endpoints
check_api_endpoints() {
    print_status "Checking API endpoints..."
    
    local base_url="http://localhost:8000"
    
    # Health endpoint
    if check_service_health "API Health" "$base_url/health"; then
        echo "  ✓ Health endpoint working"
    fi
    
    # API docs
    if curl -f -s "$base_url/docs" > /dev/null; then
        echo "  ✓ API documentation accessible"
    else
        echo "  ✗ API documentation not accessible"
    fi
    
    # API endpoints
    local endpoints=("/api/v1/data/flights" "/api/v1/analysis/delays" "/api/v1/nlp/query")
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f -s "$base_url$endpoint" > /dev/null; then
            echo "  ✓ $endpoint accessible"
        else
            echo "  ✗ $endpoint not accessible"
        fi
    done
}

# Check dashboard
check_dashboard() {
    print_status "Checking dashboard..."
    
    if check_service_health "Streamlit Dashboard" "http://localhost:8501/_stcore/health"; then
        echo "  ✓ Dashboard is accessible"
    fi
}

# Check system resources
check_system_resources() {
    print_status "Checking system resources..."
    
    # Docker system info
    echo "Docker system usage:"
    docker system df
    
    echo ""
    echo "Container resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

# Check logs for errors
check_logs() {
    print_status "Checking recent logs for errors..."
    
    local services=("api" "worker" "scheduler" "dashboard")
    
    for service in "${services[@]}"; do
        echo "Recent errors in $service:"
        docker-compose logs --tail=10 "$service" 2>/dev/null | grep -i error || echo "  No recent errors found"
        echo ""
    done
}

# Main health check
main() {
    echo "🏥 Flight Scheduling Analysis System Health Check"
    echo "================================================"
    
    cd "$PROJECT_DIR"
    
    # Check container status
    echo ""
    print_status "Container Status:"
    local containers=("flight_postgres" "flight_redis" "flight_influxdb" "flight_api" "flight_worker" "flight_scheduler" "flight_dashboard")
    
    local healthy_containers=0
    for container in "${containers[@]}"; do
        if check_container_status "$container"; then
            ((healthy_containers++))
        fi
    done
    
    echo ""
    echo "Healthy containers: $healthy_containers/${#containers[@]}"
    
    # Check services
    echo ""
    check_database
    echo ""
    check_api_endpoints
    echo ""
    check_dashboard
    echo ""
    check_system_resources
    echo ""
    check_logs
    
    echo ""
    if [ $healthy_containers -eq ${#containers[@]} ]; then
        print_success "🎉 All systems are healthy!"
        exit 0
    else
        print_warning "⚠️  Some systems need attention"
        exit 1
    fi
}

main "$@"