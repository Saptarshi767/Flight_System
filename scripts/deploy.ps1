# Flight Scheduling Analysis System Deployment Script (PowerShell)
# Usage: .\scripts\deploy.ps1 [environment]
# Environment: development (default), production, staging

param(
    [string]$Environment = "development"
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Deploying Flight Scheduling Analysis System - Environment: $Environment" -ForegroundColor Blue

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if Docker is installed and running
function Test-Docker {
    Write-Status "Checking Docker installation..."
    
    try {
        $dockerVersion = docker --version
        Write-Status "Docker version: $dockerVersion"
    }
    catch {
        Write-Error "Docker is not installed. Please install Docker Desktop first."
        exit 1
    }
    
    try {
        docker info | Out-Null
        Write-Success "Docker is installed and running"
    }
    catch {
        Write-Error "Docker is not running. Please start Docker Desktop first."
        exit 1
    }
}

# Check if Docker Compose is installed
function Test-DockerCompose {
    Write-Status "Checking Docker Compose installation..."
    
    try {
        $composeVersion = docker-compose --version
        Write-Status "Docker Compose version: $composeVersion"
        Write-Success "Docker Compose is installed"
    }
    catch {
        Write-Error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    }
}

# Create necessary directories
function New-ProjectDirectories {
    Write-Status "Creating necessary directories..."
    
    $directories = @(
        "logs\postgres",
        "logs\influxdb", 
        "logs\nginx",
        "logs\api",
        "logs\worker",
        "logs\scheduler",
        "logs\dashboard",
        "data",
        "exports",
        "reports",
        "config\ssl",
        "config\grafana"
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Success "Directories created"
}

# Generate SSL certificates for development
function New-SSLCertificates {
    if ($Environment -eq "development") {
        Write-Status "Generating self-signed SSL certificates for development..."
        
        if (!(Test-Path "config\ssl\cert.pem")) {
            try {
                # Try using OpenSSL if available
                openssl req -x509 -newkey rsa:4096 -keyout "config\ssl\key.pem" -out "config\ssl\cert.pem" -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
                Write-Success "SSL certificates generated using OpenSSL"
            }
            catch {
                Write-Warning "OpenSSL not found. Creating placeholder SSL certificates."
                Write-Warning "For production, please generate proper SSL certificates."
                
                # Create placeholder files
                "# Placeholder SSL certificate" | Out-File -FilePath "config\ssl\cert.pem" -Encoding UTF8
                "# Placeholder SSL key" | Out-File -FilePath "config\ssl\key.pem" -Encoding UTF8
            }
        }
        else {
            Write-Warning "SSL certificates already exist"
        }
    }
}

# Check environment file
function Test-EnvironmentFile {
    Write-Status "Checking environment configuration..."
    
    if ($Environment -eq "production") {
        if (!(Test-Path ".env.prod")) {
            Write-Error "Production environment file (.env.prod) not found!"
            Write-Error "Please create .env.prod with production configuration"
            exit 1
        }
        Copy-Item ".env.prod" ".env" -Force
    }
    else {
        if (!(Test-Path ".env")) {
            if (Test-Path ".env.example") {
                Copy-Item ".env.example" ".env" -Force
                Write-Warning "Created .env from .env.example. Please review and update configuration."
            }
            else {
                Write-Error "No environment file found. Please create .env file."
                exit 1
            }
        }
    }
    
    Write-Success "Environment configuration checked"
}

# Build and start services
function Start-Services {
    Write-Status "Building and starting services..."
    
    try {
        switch ($Environment) {
            "production" {
                Write-Status "Deploying production environment..."
                docker-compose -f docker-compose.prod.yml down --remove-orphans
                docker-compose -f docker-compose.prod.yml build --no-cache
                docker-compose -f docker-compose.prod.yml up -d
            }
            "staging" {
                Write-Status "Deploying staging environment..."
                docker-compose -f docker-compose.yml down --remove-orphans
                docker-compose -f docker-compose.yml build
                docker-compose -f docker-compose.yml --profile monitoring up -d
            }
            default {
                Write-Status "Deploying development environment..."
                docker-compose down --remove-orphans
                docker-compose build
                docker-compose up -d
            }
        }
        
        Write-Success "Services started"
    }
    catch {
        Write-Error "Failed to start services: $_"
        exit 1
    }
}

# Wait for services to be healthy
function Wait-ForServices {
    Write-Status "Waiting for services to be healthy..."
    
    $maxAttempts = 30
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        Write-Status "Health check attempt $attempt/$maxAttempts..."
        
        $healthyServices = docker-compose ps | Select-String "Up \(healthy\)"
        if ($healthyServices) {
            Write-Success "Services are healthy"
            return
        }
        
        Start-Sleep -Seconds 10
        $attempt++
    }
    
    Write-Error "Services failed to become healthy within timeout"
    docker-compose logs
    exit 1
}

# Run database migrations
function Invoke-DatabaseMigrations {
    Write-Status "Running database migrations..."
    
    try {
        if ($Environment -eq "production") {
            docker-compose -f docker-compose.prod.yml exec -T api alembic upgrade head
        }
        else {
            docker-compose exec -T api alembic upgrade head
        }
        
        Write-Success "Database migrations completed"
    }
    catch {
        Write-Warning "Database migrations failed or not needed: $_"
    }
}

# Display deployment information
function Show-DeploymentInfo {
    Write-Success "🎉 Deployment completed successfully!"
    Write-Host ""
    Write-Host "📊 Service URLs:" -ForegroundColor Yellow
    Write-Host "  • API Documentation: http://localhost:8000/docs"
    Write-Host "  • Dashboard: http://localhost:8501"
    Write-Host "  • Health Check: http://localhost:8000/health"
    
    if ($Environment -ne "development") {
        Write-Host "  • Prometheus: http://localhost:9090"
        Write-Host "  • Grafana: http://localhost:3000 (admin/admin123)"
    }
    
    Write-Host ""
    Write-Host "🔧 Useful commands:" -ForegroundColor Yellow
    Write-Host "  • View logs: docker-compose logs -f [service_name]"
    Write-Host "  • Stop services: docker-compose down"
    Write-Host "  • Restart service: docker-compose restart [service_name]"
    Write-Host "  • Scale workers: docker-compose up -d --scale worker=3"
    Write-Host ""
    Write-Host "📁 Important directories:" -ForegroundColor Yellow
    Write-Host "  • Logs: .\logs\"
    Write-Host "  • Data: .\data\"
    Write-Host "  • Exports: .\exports\"
    Write-Host "  • Reports: .\reports\"
}

# Main deployment flow
function Main {
    Write-Status "Starting deployment for environment: $Environment"
    
    Test-Docker
    Test-DockerCompose
    New-ProjectDirectories
    New-SSLCertificates
    Test-EnvironmentFile
    Start-Services
    Wait-ForServices
    Invoke-DatabaseMigrations
    Show-DeploymentInfo
}

# Handle script interruption
trap {
    Write-Error "Deployment interrupted"
    exit 1
}

# Run main function
try {
    Main
}
catch {
    Write-Error "Deployment failed: $_"
    exit 1
}