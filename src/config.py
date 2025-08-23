"""Configuration management for the Flight Scheduling Analysis System."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field
from cryptography.fernet import Fernet


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # Database Configuration
    database_url: str = Field("postgresql://user:password@localhost:5432/flightdb", env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    influxdb_url: str = Field("http://localhost:8086", env="INFLUXDB_URL")
    influxdb_token: Optional[str] = Field(None, env="INFLUXDB_TOKEN")
    influxdb_org: str = Field("flight-analysis", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field("flight-metrics", env="INFLUXDB_BUCKET")
    
    # Data Source URLs
    flightradar24_url: str = Field("https://www.flightradar24.com/", env="FLIGHTRADAR24_URL")
    flightaware_url: str = Field("https://www.flightaware.com/", env="FLIGHTAWARE_URL")
    mumbai_airport_url: str = Field("https://www.flightradar24.com/data/airports/bom", env="MUMBAI_AIRPORT_URL")
    delhi_airport_url: str = Field("https://www.flightradar24.com/data/airports/del", env="DELHI_AIRPORT_URL")
    
    # File Paths
    flight_data_excel: str = Field("Flight_Data.xlsx", env="FLIGHT_DATA_EXCEL")
    flight_data_csv: str = Field("flight_data.csv", env="FLIGHT_DATA_CSV")
    data_dir: str = Field("data/", env="DATA_DIR")
    logs_dir: str = Field("logs/", env="LOGS_DIR")
    
    # Security
    encryption_key: Optional[str] = Field(None, env="ENCRYPTION_KEY")
    secret_key: str = Field("your-secret-key-for-sessions", env="SECRET_KEY")
    
    # Application Configuration
    debug: bool = Field(True, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    streamlit_port: int = Field(8501, env="STREAMLIT_PORT")
    
    # Rate Limiting
    scraping_delay: float = Field(1.0, env="SCRAPING_DELAY")
    max_requests_per_minute: int = Field(60, env="MAX_REQUESTS_PER_MINUTE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
        self._setup_encryption()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        Path(self.data_dir).mkdir(exist_ok=True)
        Path(self.logs_dir).mkdir(exist_ok=True)
    
    def _setup_encryption(self):
        """Set up encryption key if not provided."""
        if not self.encryption_key:
            # Generate a new encryption key
            key = Fernet.generate_key()
            self.encryption_key = key.decode()
            
            # Update .env file with the generated key
            env_path = Path(".env")
            if env_path.exists():
                content = env_path.read_text()
                if "ENCRYPTION_KEY=fernet-key-will-be-generated" in content:
                    content = content.replace(
                        "ENCRYPTION_KEY=fernet-key-will-be-generated",
                        f"ENCRYPTION_KEY={self.encryption_key}"
                    )
                    env_path.write_text(content)


class SecureConfig:
    """Secure configuration management with encryption."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.encryption_key:
            self.cipher_suite = Fernet(settings.encryption_key.encode())
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
settings = Settings()
secure_config = SecureConfig(settings)