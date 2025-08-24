#!/bin/bash

# Backup script for Flight Scheduling Analysis System
# Usage: ./scripts/backup.sh [backup_name]

set -e

BACKUP_NAME=${1:-"backup_$(date +%Y%m%d_%H%M%S)"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$PROJECT_DIR/backups"

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

# Create backup directory
create_backup_dir() {
    print_status "Creating backup directory..."
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME"
    print_success "Backup directory created: $BACKUP_DIR/$BACKUP_NAME"
}

# Backup PostgreSQL database
backup_postgres() {
    print_status "Backing up PostgreSQL database..."
    
    docker-compose exec -T postgres pg_dump -U user -d flightdb > "$BACKUP_DIR/$BACKUP_NAME/postgres_backup.sql"
    
    if [ $? -eq 0 ]; then
        print_success "PostgreSQL backup completed"
    else
        print_error "PostgreSQL backup failed"
        return 1
    fi
}

# Backup InfluxDB
backup_influxdb() {
    print_status "Backing up InfluxDB..."
    
    # Create InfluxDB backup
    docker-compose exec -T influxdb influx backup /tmp/influxdb_backup
    docker cp $(docker-compose ps -q influxdb):/tmp/influxdb_backup "$BACKUP_DIR/$BACKUP_NAME/"
    
    if [ $? -eq 0 ]; then
        print_success "InfluxDB backup completed"
    else
        print_error "InfluxDB backup failed"
        return 1
    fi
}

# Backup Redis data
backup_redis() {
    print_status "Backing up Redis data..."
    
    # Force Redis to save current state
    docker-compose exec -T redis redis-cli BGSAVE
    
    # Wait for background save to complete
    sleep 5
    
    # Copy Redis dump file
    docker cp $(docker-compose ps -q redis):/data/dump.rdb "$BACKUP_DIR/$BACKUP_NAME/"
    
    if [ $? -eq 0 ]; then
        print_success "Redis backup completed"
    else
        print_error "Redis backup failed"
        return 1
    fi
}

# Backup application data
backup_app_data() {
    print_status "Backing up application data..."
    
    # Copy data directories
    cp -r "$PROJECT_DIR/data" "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || true
    cp -r "$PROJECT_DIR/exports" "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || true
    cp -r "$PROJECT_DIR/reports" "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || true
    cp -r "$PROJECT_DIR/logs" "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || true
    
    print_success "Application data backup completed"
}

# Backup configuration files
backup_config() {
    print_status "Backing up configuration files..."
    
    # Copy configuration files
    cp "$PROJECT_DIR/.env" "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || true
    cp "$PROJECT_DIR/docker-compose.yml" "$BACKUP_DIR/$BACKUP_NAME/"
    cp "$PROJECT_DIR/docker-compose.prod.yml" "$BACKUP_DIR/$BACKUP_NAME/"
    cp -r "$PROJECT_DIR/config" "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || true
    
    print_success "Configuration backup completed"
}

# Create backup manifest
create_manifest() {
    print_status "Creating backup manifest..."
    
    cat > "$BACKUP_DIR/$BACKUP_NAME/manifest.txt" << EOF
Flight Scheduling Analysis System Backup
========================================

Backup Name: $BACKUP_NAME
Backup Date: $(date)
Backup Location: $BACKUP_DIR/$BACKUP_NAME

Contents:
- PostgreSQL database dump (postgres_backup.sql)
- InfluxDB backup (influxdb_backup/)
- Redis data dump (dump.rdb)
- Application data (data/, exports/, reports/, logs/)
- Configuration files (.env, docker-compose files, config/)

Restore Instructions:
1. Stop all services: docker-compose down
2. Restore PostgreSQL: docker-compose exec -T postgres psql -U user -d flightdb < postgres_backup.sql
3. Restore InfluxDB: docker cp influxdb_backup/ container:/tmp/ && docker-compose exec influxdb influx restore /tmp/influxdb_backup
4. Restore Redis: docker cp dump.rdb container:/data/
5. Restore application data: copy data/, exports/, reports/ directories
6. Restore configuration: copy .env and config files
7. Start services: docker-compose up -d

EOF

    print_success "Backup manifest created"
}

# Compress backup
compress_backup() {
    print_status "Compressing backup..."
    
    cd "$BACKUP_DIR"
    tar -czf "$BACKUP_NAME.tar.gz" "$BACKUP_NAME"
    
    if [ $? -eq 0 ]; then
        rm -rf "$BACKUP_NAME"
        print_success "Backup compressed: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
    else
        print_error "Backup compression failed"
        return 1
    fi
}

# Clean old backups
clean_old_backups() {
    print_status "Cleaning old backups (keeping last 7)..."
    
    cd "$BACKUP_DIR"
    ls -t *.tar.gz 2>/dev/null | tail -n +8 | xargs rm -f
    
    print_success "Old backups cleaned"
}

# Main backup function
main() {
    echo "💾 Flight Scheduling Analysis System Backup"
    echo "==========================================="
    
    cd "$PROJECT_DIR"
    
    # Check if services are running
    if ! docker-compose ps | grep -q "Up"; then
        print_error "Services are not running. Please start services first."
        exit 1
    fi
    
    create_backup_dir
    backup_postgres
    backup_influxdb
    backup_redis
    backup_app_data
    backup_config
    create_manifest
    compress_backup
    clean_old_backups
    
    print_success "🎉 Backup completed successfully!"
    echo "Backup location: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
    echo "Backup size: $(du -h "$BACKUP_DIR/$BACKUP_NAME.tar.gz" | cut -f1)"
}

# Handle script interruption
trap 'print_error "Backup interrupted"; exit 1' INT TERM

main "$@"