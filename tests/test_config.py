"""Test configuration management."""

import pytest
from src.config import Settings, SecureConfig


def test_settings_creation():
    """Test that settings can be created with defaults."""
    settings = Settings()
    
    # Check that required fields have values
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.debug is True
    assert settings.log_level == "INFO"
    
    # Check data source URLs
    assert "flightradar24.com" in settings.flightradar24_url
    assert "flightaware.com" in settings.flightaware_url
    assert "bom" in settings.mumbai_airport_url
    assert "del" in settings.delhi_airport_url


def test_secure_config():
    """Test secure configuration management."""
    settings = Settings()
    secure_config = SecureConfig(settings)
    
    # Test OpenAI key retrieval
    api_key = secure_config.get_openai_key()
    assert api_key is not None
    assert isinstance(api_key, str)


def test_directory_creation():
    """Test that necessary directories are created."""
    from pathlib import Path
    
    settings = Settings()
    
    # Check that directories exist
    assert Path(settings.data_dir).exists()
    assert Path(settings.logs_dir).exists()