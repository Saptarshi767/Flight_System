"""
Configuration settings for the Flight Scheduling Analysis System
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///data/flight_analysis.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    influxdb_url: str = Field(default="http://localhost:8086", env="INFLUXDB_URL")
    influxdb_token: Optional[str] = Field(default=None, env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="flight-analysis", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="flight-metrics", env="INFLUXDB_BUCKET")
    
    # Data Source URLs
    flightradar24_url: str = Field(default="https://www.flightradar24.com/", env="FLIGHTRADAR24_URL")
    flightaware_url: str = Field(default="https://www.flightaware.com/", env="FLIGHTAWARE_URL")
    mumbai_airport_url: str = Field(default="https://www.flightradar24.com/data/airports/bom", env="MUMBAI_AIRPORT_URL")
    delhi_airport_url: str = Field(default="https://www.flightradar24.com/data/airports/del", env="DELHI_AIRPORT_URL")
    
    # File Paths
    flight_data_excel: str = Field(default="Flight_Data.xlsx", env="FLIGHT_DATA_EXCEL")
    flight_data_csv: str = Field(default="flight_data.csv", env="FLIGHT_DATA_CSV")
    
    # Application Configuration
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # CORS and Security
    allowed_origins: list = Field(default=["http://localhost:3000", "http://localhost:8501", "http://localhost:8000"], env="ALLOWED_ORIGINS")
    allowed_hosts: list = Field(default=["localhost", "127.0.0.1", "*"], env="ALLOWED_HOSTS")
    
    # Rate Limiting
    scraping_delay: float = Field(default=1.0, env="SCRAPING_DELAY")
    max_requests_per_minute: int = Field(default=60, env="MAX_REQUESTS_PER_MINUTE")
    
    # Data directories
    data_dir: str = Field(default="data/", env="DATA_DIR")
    logs_dir: str = Field(default="logs/", env="LOGS_DIR")
    
    # Security
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra='ignore'  # Ignore extra environment variables
    )


from pathlib import Path
from cryptography.fernet import Fernet


class SecureConfig:
    """Secure configuration management with encryption."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.encryption_key and len(settings.encryption_key) > 0:
            try:
                self.cipher_suite = Fernet(settings.encryption_key.encode())
            except ValueError:
                # Invalid key, generate a new one
                key = Fernet.generate_key()
                self.cipher_suite = Fernet(key)
        else:
            self.cipher_suite = None
    
    def get_openai_key(self) -> str:
        """Get OpenAI API key (decrypt if encrypted)."""
        return self.settings.openai_api_key
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a value."""
        if self.cipher_suite:
            return self.cipher_suite.encrypt(value.encode()).decode()
        return value
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a value."""
        if self.cipher_suite:
            return self.cipher_suite.decrypt(encrypted_value.encode()).decode()
        return encrypted_value


# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get application settings (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
        # Ensure directories exist
        Path(_settings.data_dir).mkdir(exist_ok=True)
        Path(_settings.logs_dir).mkdir(exist_ok=True)
    return _settings

# Initialize settings
settings = get_settings()
secure_config = SecureConfig(settings)