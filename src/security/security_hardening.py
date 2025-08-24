"""
Security hardening measures for the Flight Scheduling Analysis System
"""
import os
import secrets
import hashlib
import hmac
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from functools import wraps
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import re

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import redis

from src.config.settings import settings
from src.utils.logging import logger


class SecurityConfig:
    """Security configuration and constants"""
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
    JWT_ALGORITHM = 'HS256'
    JWT_EXPIRATION_HOURS = 24
    
    # API Rate Limiting
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 3600  # 1 hour
    
    # Password Requirements
    MIN_PASSWORD_LENGTH = 12
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGITS = True
    REQUIRE_SPECIAL_CHARS = True
    
    # Session Security
    SESSION_TIMEOUT_MINUTES = 30
    MAX_CONCURRENT_SESSIONS = 5
    
    # API Key Security
    API_KEY_LENGTH = 32
    API_KEY_PREFIX = 'fsa_'
    
    # Encryption
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode())


class PasswordValidator:
    """Password validation and security"""
    
    @staticmethod
    def validate_password(password: str) -> Dict[str, Any]:
        """Validate password against security requirements"""
        errors = []
        
        if len(password) < SecurityConfig.MIN_PASSWORD_LENGTH:
            errors.append(f"Password must be at least {SecurityConfig.MIN_PASSWORD_LENGTH} characters long")
        
        if SecurityConfig.REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if SecurityConfig.REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if SecurityConfig.REQUIRE_DIGITS and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if SecurityConfig.REQUIRE_SPECIAL_CHARS and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Check for common weak passwords
        weak_patterns = [
            r'password', r'123456', r'qwerty', r'admin', r'user',
            r'login', r'welcome', r'default', r'guest'
        ]
        
        for pattern in weak_patterns:
            if re.search(pattern, password.lower()):
                errors.append("Password contains common weak patterns")
                break
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'strength_score': PasswordValidator._calculate_strength(password)
        }
    
    @staticmethod
    def _calculate_strength(password: str) -> int:
        """Calculate password strength score (0-100)"""
        score = 0
        
        # Length bonus
        score += min(25, len(password) * 2)
        
        # Character variety bonus
        if re.search(r'[a-z]', password):
            score += 10
        if re.search(r'[A-Z]', password):
            score += 10
        if re.search(r'\d', password):
            score += 10
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 15
        
        # Uniqueness bonus
        unique_chars = len(set(password))
        score += min(20, unique_chars * 2)
        
        # Pattern penalty
        if re.search(r'(.)\1{2,}', password):  # Repeated characters
            score -= 10
        if re.search(r'(012|123|234|345|456|567|678|789|890)', password):  # Sequential numbers
            score -= 10
        if re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)', password.lower()):  # Sequential letters
            score -= 10
        
        return max(0, min(100, score))
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using PBKDF2"""
        salt = os.urandom(32)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return base64.urlsafe_b64encode(salt + key).decode()
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            decoded = base64.urlsafe_b64decode(hashed.encode())
            salt = decoded[:32]
            stored_key = decoded[32:]
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = kdf.derive(password.encode())
            
            return hmac.compare_digest(stored_key, key)
        except Exception:
            return False


class JWTManager:
    """JWT token management"""
    
    @staticmethod
    def create_token(user_id: str, permissions: List[str] = None) -> str:
        """Create JWT token"""
        payload = {
            'user_id': user_id,
            'permissions': permissions or [],
            'exp': datetime.utcnow() + timedelta(hours=SecurityConfig.JWT_EXPIRATION_HOURS),
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)  # JWT ID for revocation
        }
        
        return jwt.encode(payload, SecurityConfig.JWT_SECRET_KEY, algorithm=SecurityConfig.JWT_ALGORITHM)
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SecurityConfig.JWT_SECRET_KEY, algorithms=[SecurityConfig.JWT_ALGORITHM])
            return {'valid': True, 'payload': payload}
        except jwt.ExpiredSignatureError:
            return {'valid': False, 'error': 'Token has expired'}
        except jwt.InvalidTokenError:
            return {'valid': False, 'error': 'Invalid token'}
    
    @staticmethod
    def refresh_token(token: str) -> Optional[str]:
        """Refresh JWT token if valid and not expired"""
        verification = JWTManager.verify_token(token)
        if verification['valid']:
            payload = verification['payload']
            return JWTManager.create_token(payload['user_id'], payload['permissions'])
        return None


class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self, redis_client: redis.Redis = None):
        self.redis_client = redis_client or redis.Redis.from_url(settings.redis_url)
        self.logger = logger
    
    def is_allowed(self, identifier: str, limit: int = None, window: int = None) -> Dict[str, Any]:
        """Check if request is allowed under rate limit"""
        limit = limit or SecurityConfig.RATE_LIMIT_REQUESTS
        window = window or SecurityConfig.RATE_LIMIT_WINDOW
        
        key = f"rate_limit:{identifier}"
        
        try:
            current = self.redis_client.get(key)
            
            if current is None:
                # First request
                self.redis_client.setex(key, window, 1)
                return {
                    'allowed': True,
                    'requests_made': 1,
                    'requests_remaining': limit - 1,
                    'reset_time': datetime.utcnow() + timedelta(seconds=window)
                }
            
            current_count = int(current)
            
            if current_count >= limit:
                # Rate limit exceeded
                ttl = self.redis_client.ttl(key)
                return {
                    'allowed': False,
                    'requests_made': current_count,
                    'requests_remaining': 0,
                    'reset_time': datetime.utcnow() + timedelta(seconds=ttl)
                }
            
            # Increment counter
            new_count = self.redis_client.incr(key)
            ttl = self.redis_client.ttl(key)
            
            return {
                'allowed': True,
                'requests_made': new_count,
                'requests_remaining': limit - new_count,
                'reset_time': datetime.utcnow() + timedelta(seconds=ttl)
            }
            
        except Exception as e:
            self.logger.error(f"Rate limiting error: {e}")
            # Allow request if rate limiting fails
            return {
                'allowed': True,
                'requests_made': 0,
                'requests_remaining': limit,
                'reset_time': datetime.utcnow() + timedelta(seconds=window)
            }


class APIKeyManager:
    """API key management"""
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate secure API key"""
        key = secrets.token_urlsafe(SecurityConfig.API_KEY_LENGTH)
        return f"{SecurityConfig.API_KEY_PREFIX}{key}"
    
    @staticmethod
    def validate_api_key_format(api_key: str) -> bool:
        """Validate API key format"""
        if not api_key.startswith(SecurityConfig.API_KEY_PREFIX):
            return False
        
        key_part = api_key[len(SecurityConfig.API_KEY_PREFIX):]
        return len(key_part) >= SecurityConfig.API_KEY_LENGTH
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()


class DataEncryption:
    """Data encryption utilities"""
    
    def __init__(self):
        self.cipher_suite = Fernet(SecurityConfig.ENCRYPTION_KEY.encode())
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_dict(self, data: Dict[str, Any], sensitive_keys: List[str]) -> Dict[str, Any]:
        """Encrypt sensitive keys in dictionary"""
        encrypted_data = data.copy()
        for key in sensitive_keys:
            if key in encrypted_data:
                encrypted_data[key] = self.encrypt(str(encrypted_data[key]))
        return encrypted_data
    
    def decrypt_dict(self, data: Dict[str, Any], sensitive_keys: List[str]) -> Dict[str, Any]:
        """Decrypt sensitive keys in dictionary"""
        decrypted_data = data.copy()
        for key in sensitive_keys:
            if key in decrypted_data:
                try:
                    decrypted_data[key] = self.decrypt(decrypted_data[key])
                except Exception:
                    # If decryption fails, leave as is (might not be encrypted)
                    pass
        return decrypted_data


class SecurityMiddleware:
    """Security middleware for FastAPI"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.logger = logger
    
    async def security_headers(self, request: Request, call_next):
        """Add security headers to responses"""
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response
    
    async def rate_limiting(self, request: Request, call_next):
        """Apply rate limiting"""
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "unknown")
        identifier = f"{client_ip}:{hashlib.md5(user_agent.encode()).hexdigest()[:8]}"
        
        rate_limit_result = self.rate_limiter.is_allowed(identifier)
        
        if not rate_limit_result['allowed']:
            self.logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(SecurityConfig.RATE_LIMIT_REQUESTS),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(rate_limit_result['reset_time'].timestamp()))
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(SecurityConfig.RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_result['requests_remaining'])
        response.headers["X-RateLimit-Reset"] = str(int(rate_limit_result['reset_time'].timestamp()))
        
        return response


class AuthenticationManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.security = HTTPBearer()
        self.logger = logger
    
    async def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Verify JWT token"""
        token = credentials.credentials
        verification = JWTManager.verify_token(token)
        
        if not verification['valid']:
            raise HTTPException(
                status_code=401,
                detail=verification['error'],
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return verification['payload']
    
    def require_permission(self, required_permission: str):
        """Decorator to require specific permission"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract token payload from kwargs or dependencies
                token_payload = kwargs.get('token_payload')
                if not token_payload:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                permissions = token_payload.get('permissions', [])
                if required_permission not in permissions and 'admin' not in permissions:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator


class InputSanitizer:
    """Input sanitization and validation"""
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            return ""
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in input_str if ord(char) >= 32 or char in '\t\n\r')
        
        # Limit length
        sanitized = sanitized[:max_length]
        
        # Remove potential SQL injection patterns
        dangerous_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
            r'(--|/\*|\*/)',
            r'(\bOR\b.*=.*\bOR\b)',
            r'(\bAND\b.*=.*\bAND\b)'
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_airport_code(code: str) -> bool:
        """Validate airport code format"""
        return re.match(r'^[A-Z]{3}$', code) is not None
    
    @staticmethod
    def validate_flight_number(flight_number: str) -> bool:
        """Validate flight number format"""
        return re.match(r'^[A-Z0-9]{2,3}[0-9]{1,4}$', flight_number) is not None


# Global instances
password_validator = PasswordValidator()
jwt_manager = JWTManager()
rate_limiter = RateLimiter()
api_key_manager = APIKeyManager()
data_encryption = DataEncryption()
security_middleware = SecurityMiddleware()
auth_manager = AuthenticationManager()
input_sanitizer = InputSanitizer()


# Security decorators
def require_api_key(func):
    """Decorator to require valid API key"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = kwargs.get('request') or args[0] if args else None
        if not request:
            raise HTTPException(status_code=500, detail="Request object not found")
        
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")
        
        if not api_key_manager.validate_api_key_format(api_key):
            raise HTTPException(status_code=401, detail="Invalid API key format")
        
        # Here you would validate against stored API keys
        # For now, we'll just check the format
        
        return await func(*args, **kwargs)
    return wrapper


def sanitize_inputs(func):
    """Decorator to sanitize function inputs"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Sanitize string arguments
        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                sanitized_kwargs[key] = input_sanitizer.sanitize_string(value)
            else:
                sanitized_kwargs[key] = value
        
        return await func(*args, **sanitized_kwargs)
    return wrapper