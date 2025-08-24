"""
Alerting system for Flight Scheduling Analysis System
"""

import asyncio
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
import os
import requests
from collections import defaultdict, deque

from .logging_config import get_logger
from .metrics import metrics_collector

logger = get_logger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        return data

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    description: str
    condition: Callable[[], bool]
    severity: AlertSeverity
    cooldown_minutes: int = 15
    enabled: bool = True
    metadata: Dict[str, Any] = None

class AlertManager:
    """Centralized alert management system"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.last_triggered: Dict[str, datetime] = {}
        self.notification_handlers: List[Callable] = []
        self.running = False
        self.check_interval = 60  # seconds
        
        # Load configuration
        self.smtp_config = self._load_smtp_config()
        self.webhook_config = self._load_webhook_config()
        
        logger.info("AlertManager initialized")
    
    def _load_smtp_config(self) -> Dict[str, str]:
        """Load SMTP configuration from environment"""
        return {
            'host': os.getenv('SMTP_HOST', 'localhost'),
            'port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('SMTP_USERNAME', ''),
            'password': os.getenv('SMTP_PASSWORD', ''),
            'from_email': os.getenv('SMTP_FROM_EMAIL', 'alerts@flightanalysis.com'),
            'to_emails': os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(',')
        }
    
    def _load_webhook_config(self) -> Dict[str, str]:
        """Load webhook configuration from environment"""
        return {
            'slack_webhook': os.getenv('SLACK_WEBHOOK_URL', ''),
            'teams_webhook': os.getenv('TEAMS_WEBHOOK_URL', ''),
            'discord_webhook': os.getenv('DISCORD_WEBHOOK_URL', '')
        }
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def add_notification_handler(self, handler: Callable):
        """Add a custom notification handler"""
        self.notification_handlers.append(handler)
    
    async def start(self):
        """Start the alert monitoring system"""
        if self.running:
            return
        
        self.running = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("Alert monitoring started")
    
    def stop(self):
        """Stop the alert monitoring system"""
        self.running = False
        logger.info("Alert monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._check_rules()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)
    
    async def _check_rules(self):
        """Check all alert rules"""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check cooldown period
                last_triggered = self.last_triggered.get(rule_name)
                if last_triggered:
                    cooldown_end = last_triggered + timedelta(minutes=rule.cooldown_minutes)
                    if datetime.utcnow() < cooldown_end:
                        continue
                
                # Evaluate rule condition
                if rule.condition():
                    await self._trigger_alert(rule)
                else:
                    # Check if we need to resolve an active alert
                    if rule_name in self.active_alerts:
                        await self._resolve_alert(rule_name)
                        
            except Exception as e:
                logger.error(f"Error checking rule {rule_name}: {e}", exc_info=True)
    
    async def _trigger_alert(self, rule: AlertRule):
        """Trigger an alert"""
        alert_id = f"{rule.name}_{int(datetime.utcnow().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            name=rule.name,
            description=rule.description,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=rule.metadata or {}
        )
        
        # Store alert
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        self.last_triggered[rule.name] = datetime.utcnow()
        
        # Send notifications
        await self._send_notifications(alert)
        
        logger.warning(f"Alert triggered: {rule.name} - {rule.description}")
    
    async def _resolve_alert(self, rule_name: str):
        """Resolve an active alert"""
        if rule_name not in self.active_alerts:
            return
        
        alert = self.active_alerts[rule_name]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.updated_at = datetime.utcnow()
        
        # Remove from active alerts
        del self.active_alerts[rule_name]
        
        # Send resolution notification
        await self._send_resolution_notification(alert)
        
        logger.info(f"Alert resolved: {rule_name}")
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        # Email notification
        if self.smtp_config['to_emails'] and self.smtp_config['to_emails'][0]:
            await self._send_email_notification(alert)
        
        # Slack notification
        if self.webhook_config['slack_webhook']:
            await self._send_slack_notification(alert)
        
        # Teams notification
        if self.webhook_config['teams_webhook']:
            await self._send_teams_notification(alert)
        
        # Discord notification
        if self.webhook_config['discord_webhook']:
            await self._send_discord_notification(alert)
        
        # Custom handlers
        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in custom notification handler: {e}")
    
    async def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notification"""
        # Create a copy with resolved status for notification
        resolved_alert = Alert(
            id=alert.id,
            name=f"RESOLVED: {alert.name}",
            description=f"Alert resolved: {alert.description}",
            severity=AlertSeverity.LOW,
            status=AlertStatus.RESOLVED,
            created_at=alert.created_at,
            updated_at=alert.updated_at,
            resolved_at=alert.resolved_at,
            metadata=alert.metadata
        )
        
        await self._send_notifications(resolved_alert)
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = ', '.join(self.smtp_config['to_emails'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] Flight Analysis Alert: {alert.name}"
            
            body = f"""
            Alert Details:
            
            Name: {alert.name}
            Description: {alert.description}
            Severity: {alert.severity.value.upper()}
            Status: {alert.status.value.upper()}
            Created: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
            
            Metadata:
            {json.dumps(alert.metadata or {}, indent=2)}
            
            Please investigate this alert promptly.
            
            Flight Scheduling Analysis System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            server.starttls()
            if self.smtp_config['username']:
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert: {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        try:
            color_map = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning", 
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger"
            }
            
            payload = {
                "text": f"Flight Analysis Alert: {alert.name}",
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "fields": [
                        {"title": "Alert", "value": alert.name, "short": True},
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Status", "value": alert.status.value.upper(), "short": True},
                        {"title": "Time", "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": True},
                        {"title": "Description", "value": alert.description, "short": False}
                    ]
                }]
            }
            
            response = requests.post(self.webhook_config['slack_webhook'], json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert: {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    async def _send_teams_notification(self, alert: Alert):
        """Send Microsoft Teams notification"""
        try:
            color_map = {
                AlertSeverity.LOW: "00FF00",
                AlertSeverity.MEDIUM: "FFFF00",
                AlertSeverity.HIGH: "FF8000", 
                AlertSeverity.CRITICAL: "FF0000"
            }
            
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color_map.get(alert.severity, "FFFF00"),
                "summary": f"Flight Analysis Alert: {alert.name}",
                "sections": [{
                    "activityTitle": f"Flight Analysis Alert: {alert.name}",
                    "activitySubtitle": alert.description,
                    "facts": [
                        {"name": "Severity", "value": alert.severity.value.upper()},
                        {"name": "Status", "value": alert.status.value.upper()},
                        {"name": "Time", "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
                    ]
                }]
            }
            
            response = requests.post(self.webhook_config['teams_webhook'], json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Teams notification sent for alert: {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send Teams notification: {e}")
    
    async def _send_discord_notification(self, alert: Alert):
        """Send Discord notification"""
        try:
            color_map = {
                AlertSeverity.LOW: 0x00FF00,
                AlertSeverity.MEDIUM: 0xFFFF00,
                AlertSeverity.HIGH: 0xFF8000,
                AlertSeverity.CRITICAL: 0xFF0000
            }
            
            payload = {
                "embeds": [{
                    "title": f"Flight Analysis Alert: {alert.name}",
                    "description": alert.description,
                    "color": color_map.get(alert.severity, 0xFFFF00),
                    "fields": [
                        {"name": "Severity", "value": alert.severity.value.upper(), "inline": True},
                        {"name": "Status", "value": alert.status.value.upper(), "inline": True},
                        {"name": "Time", "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'), "inline": False}
                    ],
                    "timestamp": alert.created_at.isoformat()
                }]
            }
            
            response = requests.post(self.webhook_config['discord_webhook'], json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Discord notification sent for alert: {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
    
    def acknowledge_alert(self, rule_name: str) -> bool:
        """Acknowledge an active alert"""
        if rule_name not in self.active_alerts:
            return False
        
        alert = self.active_alerts[rule_name]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.updated_at = datetime.utcnow()
        
        logger.info(f"Alert acknowledged: {rule_name}")
        return True
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        history = list(self.alert_history)[-limit:]
        return [alert.to_dict() for alert in history]

# Global alert manager instance
alert_manager = AlertManager()

# Predefined alert rules
def create_system_alert_rules():
    """Create system monitoring alert rules"""
    
    # High CPU usage
    def high_cpu_condition():
        cpu_percent = metrics_collector.get_gauge('system.cpu.percent')
        return cpu_percent > 80
    
    alert_manager.add_rule(AlertRule(
        name="high_cpu_usage",
        description="CPU usage is above 80%",
        condition=high_cpu_condition,
        severity=AlertSeverity.HIGH,
        cooldown_minutes=10
    ))
    
    # High memory usage
    def high_memory_condition():
        memory_percent = metrics_collector.get_gauge('system.memory.percent')
        return memory_percent > 85
    
    alert_manager.add_rule(AlertRule(
        name="high_memory_usage",
        description="Memory usage is above 85%",
        condition=high_memory_condition,
        severity=AlertSeverity.HIGH,
        cooldown_minutes=10
    ))
    
    # High disk usage
    def high_disk_condition():
        disk_percent = metrics_collector.get_gauge('system.disk.percent')
        return disk_percent > 90
    
    alert_manager.add_rule(AlertRule(
        name="high_disk_usage",
        description="Disk usage is above 90%",
        condition=high_disk_condition,
        severity=AlertSeverity.CRITICAL,
        cooldown_minutes=15
    ))
    
    # High error rate
    def high_error_rate_condition():
        total_requests = metrics_collector.get_counter('app.requests.total')
        failed_requests = metrics_collector.get_counter('app.requests.failed')
        if total_requests < 100:  # Need minimum requests for meaningful rate
            return False
        error_rate = (failed_requests / total_requests) * 100
        return error_rate > 5  # 5% error rate
    
    alert_manager.add_rule(AlertRule(
        name="high_error_rate",
        description="Application error rate is above 5%",
        condition=high_error_rate_condition,
        severity=AlertSeverity.HIGH,
        cooldown_minutes=5
    ))
    
    # Slow response time
    def slow_response_condition():
        avg_response_time = metrics_collector.get_average_timer('app.response_time')
        return avg_response_time > 5.0  # 5 seconds
    
    alert_manager.add_rule(AlertRule(
        name="slow_response_time",
        description="Average response time is above 5 seconds",
        condition=slow_response_condition,
        severity=AlertSeverity.MEDIUM,
        cooldown_minutes=10
    ))
    
    # Database connection issues
    def database_connection_condition():
        db_connections = metrics_collector.get_gauge('app.database.connections')
        return db_connections == 0
    
    alert_manager.add_rule(AlertRule(
        name="database_connection_lost",
        description="Database connections are zero",
        condition=database_connection_condition,
        severity=AlertSeverity.CRITICAL,
        cooldown_minutes=5
    ))
    
    # Low cache hit rate
    def low_cache_hit_rate_condition():
        hit_rate = metrics_collector.get_gauge('app.cache.hit_rate')
        return hit_rate < 50  # Below 50% hit rate
    
    alert_manager.add_rule(AlertRule(
        name="low_cache_hit_rate",
        description="Cache hit rate is below 50%",
        condition=low_cache_hit_rate_condition,
        severity=AlertSeverity.MEDIUM,
        cooldown_minutes=15
    ))

async def init_alerts():
    """Initialize the alerting system"""
    create_system_alert_rules()
    await alert_manager.start()
    logger.info("Alerting system initialized")

def shutdown_alerts():
    """Shutdown the alerting system"""
    alert_manager.stop()
    logger.info("Alerting system shutdown")

# Convenience functions
def trigger_custom_alert(name: str, description: str, severity: AlertSeverity, metadata: Dict[str, Any] = None):
    """Trigger a custom alert"""
    alert_id = f"{name}_{int(datetime.utcnow().timestamp())}"
    
    alert = Alert(
        id=alert_id,
        name=name,
        description=description,
        severity=severity,
        status=AlertStatus.ACTIVE,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        metadata=metadata or {}
    )
    
    alert_manager.active_alerts[name] = alert
    alert_manager.alert_history.append(alert)
    
    # Send notifications asynchronously
    asyncio.create_task(alert_manager._send_notifications(alert))
    
    logger.warning(f"Custom alert triggered: {name} - {description}")

def resolve_custom_alert(name: str):
    """Resolve a custom alert"""
    if name in alert_manager.active_alerts:
        asyncio.create_task(alert_manager._resolve_alert(name))