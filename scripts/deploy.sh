#!/bin/bash

# Flight Scheduling Analysis System Deployment Script
# Usage: ./scripts/deploy.sh [environment]
# Environment: development (default), production, staging

set -e

ENVIRONMENT=${1:-development}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 Deploying Flight Scheduling Analysis System - Environment: $ENVIRONMENT"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Check if Docker Compose is installed
check_docker_compose() {
    print_status "Checking Docker Compose installation..."
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker Compose is installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p "$PROJECT_DIR/logs/postgres"
    mkdir -p "$PROJECT_DIR/logs/influxdb"
    mkdir -p "$PROJECT_DIR/logs/nginx"
    mkdir -p "$PROJECT_DIR/logs/api"
    mkdir -p "$PROJECT_DIR/logs/worker"
    mkdir -p "$PROJECT_DIR/logs/scheduler"
    mkdir -p "$PROJECT_DIR/logs/dashboard"
    mkdir -p "$PROJECT_DIR/data"
    mkdir -p "$PROJECT_DIR/exports"
    mkdir -p "$PROJECT_DIR/reports"
    mkdir -p "$PROJECT_DIR/config/ssl"
    mkdir -p "$PROJECT_DIR/config/grafana"
    print_success "Directories created"
}

# Generate SSL certificates for development
generate_ssl_certs() {
    if [ "$ENVIRONMENT" = "development" ]; then
        print_status "Generating self-signed SSL certificates for development..."
        if [ ! -f "$PROJECT_DIR/config/ssl/cert.pem" ]; then
            openssl req -x509 -newkey rsa:4096 -keyout "$PROJECT_DIR/config/ssl/key.pem" \
                -out "$PROJECT_DIR/config/ssl/cert.pem" -days 365 -nodes \
                -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
            print_success "SSL certificates generated"
        else
            print_warning "SSL certificates already exist"
        fi
    fi
}

# Check environment file
check_env_file() {
    print_status "Checking environment configuration..."
    if [ "$ENVIRONMENT" = "production" ]; then
        if [ ! -f "$PROJECT_DIR/.env.prod" ]; then
            print_error "Production environment file (.env.prod) not found!"
            print_error "Please create .env.prod with production configuration"
            exit 1
        fi
        cp "$PROJECT_DIR/.env.prod" "$PROJECT_DIR/.env"
    else
        if [ ! -f "$PROJECT_DIR/.env" ]; then
            if [ -f "$PROJECT_DIR/.env.example" ]; then
                cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
                print_warning "Created .env from .env.example. Please review and update configuration."
            else
                print_error "No environment file found. Please create .env file."
                exit 1
            fi
        fi
    fi
    print_success "Environment configuration checked"
}

# Build and start services
deploy_services() {
    print_status "Building and starting services..."
    
    cd "$PROJECT_DIR"
    
    case $ENVIRONMENT in
        "production")
            print_status "Deploying production environment..."
            docker-compose -f docker-compose.prod.yml down --remove-orphans
            docker-compose -f docker-compose.prod.yml build --no-cache
            docker-compose -f docker-compose.prod.yml up -d
            ;;
        "staging")
            print_status "Deploying staging environment..."
            docker-compose -f docker-compose.yml down --remove-orphans
            docker-compose -f docker-compose.yml build
            docker-compose -f docker-compose.yml up -d --profile monitoring
            ;;
        *)
            print_status "Deploying development environment..."
            docker-compose down --remove-orphans
            docker-compose build
            docker-compose up -d
            ;;
    esac
    
    print_success "Services started"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        print_status "Health check attempt $attempt/$max_attempts..."
        
        if docker-compose ps | grep -q "Up (healthy)"; then
            print_success "Services are healthy"
            return 0
        fi
        
        sleep 10
        ((attempt++))
    done
    
    print_error "Services failed to become healthy within timeout"
    docker-compose logs
    exit 1
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose -f docker-compose.prod.yml exec -T api alembic upgrade head
    else
        docker-compose exec -T api alembic upgrade head
    fi
    
    print_success "Database migrations completed"
}

# Display deployment information
show_deployment_info() {
    print_success "🎉 Deployment completed successfully!"
    echo ""
    echo "📊 Service URLs:"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • Dashboard: http://localhost:8501"
    echo "  • Health Check: http://localhost:8000/health"
    
    if [ "$ENVIRONMENT" != "development" ]; then
        echo "  • Prometheus: http://localhost:9090"
        echo "  • Grafana: http://localhost:3000 (admin/admin123)"
    fi
    
    echo ""
    echo "🔧 Useful commands:"
    echo "  • View logs: docker-compose logs -f [service_name]"
    echo "  • Stop services: docker-compose down"
    echo "  • Restart service: docker-compose restart [service_name]"
    echo "  • Scale workers: docker-compose up -d --scale worker=3"
    echo ""
    echo "📁 Important directories:"
    echo "  • Logs: ./logs/"
    echo "  • Data: ./data/"
    echo "  • Exports: ./exports/"
    echo "  • Reports: ./reports/"
}

# Main deployment flow
main() {
    print_status "Starting deployment for environment: $ENVIRONMENT"
    
    check_docker
    check_docker_compose
    create_directories
    generate_ssl_certs
    check_env_file
    deploy_services
    wait_for_services
    run_migrations
    show_deployment_info
}

# Handle script interruption
trap 'print_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"