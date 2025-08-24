"""
Tests for monitoring and logging functionality
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.utils.logging_config import (
    setup_logging, get_logger, set_correlation_id, get_correlation_id,
    log_performance, log_business_event, log_security_event,
    monitor_performance, monitor_async_performance
)
from src.utils.metrics import (
    MetricsCollector, metrics_collector, increment_counter, set_gauge,
    record_histogram, record_timer, time_function, time_async_function
)
from src.utils.alerts import (
    Alert, AlertRule, AlertManager, AlertSeverity, AlertStatus,
    alert_manager, trigger_custom_alert
)
from src.utils.log_analyzer import LogAnalyzer, LogEntry, analyze_recent_logs

class TestLoggingConfig:
    """Test logging configuration"""
    
    def test_setup_logging_json_format(self):
        """Test JSON logging setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            setup_logging(
                log_level="INFO",
                log_format="json",
                log_file=log_file,
                enable_console=False
            )
            
            logger = get_logger("test")
            logger.info("Test message")
            
            # Check log file exists and contains JSON
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert content.strip()
                # Should be valid JSON
                log_entry = json.loads(content.strip())
                assert log_entry['message'] == "Test message"
                assert log_entry['level'] == "INFO"
    
    def test_setup_logging_text_format(self):
        """Test text logging setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            setup_logging(
                log_level="DEBUG",
                log_format="text",
                log_file=log_file,
                enable_console=False
            )
            
            logger = get_logger("test")
            logger.debug("Debug message")
            
            # Check log file exists and contains text format
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Debug message" in content
                assert "DEBUG" in content
    
    def test_correlation_id(self):
        """Test correlation ID functionality"""
        # Initially no correlation ID
        assert get_correlation_id() is None
        
        # Set correlation ID
        test_id = "test-123"
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id
    
    def test_performance_logging(self):
        """Test performance logging"""
        logger = get_logger("test")
        
        with patch.object(logger, 'info') as mock_info:
            log_performance(logger, "test_operation", 1.5, param1="value1")
            
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            assert "Performance: test_operation" in args[0]
            assert kwargs['extra']['operation'] == "test_operation"
            assert kwargs['extra']['duration'] == 1.5
            assert kwargs['extra']['param1'] == "value1"
    
    def test_business_event_logging(self):
        """Test business event logging"""
        logger = get_logger("test")
        
        with patch.object(logger, 'info') as mock_info:
            log_business_event(logger, "user_login", user_id="123")
            
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            assert "Business Event: user_login" in args[0]
            assert kwargs['extra']['event_type'] == "business"
            assert kwargs['extra']['event_name'] == "user_login"
            assert kwargs['extra']['user_id'] == "123"
    
    def test_security_event_logging(self):
        """Test security event logging"""
        logger = get_logger("test")
        
        with patch.object(logger, 'warning') as mock_warning:
            log_security_event(logger, "failed_login", ip="192.168.1.1")
            
            mock_warning.assert_called_once()
            args, kwargs = mock_warning.call_args
            assert "Security Event: failed_login" in args[0]
            assert kwargs['extra']['event_type'] == "security"
            assert kwargs['extra']['event_name'] == "failed_login"
            assert kwargs['extra']['ip'] == "192.168.1.1"
    
    def test_performance_decorator(self):
        """Test performance monitoring decorator"""
        logger = get_logger("test")
        
        @monitor_performance("test_function")
        def test_func():
            time.sleep(0.1)
            return "result"
        
        with patch.object(logger, 'info') as mock_info:
            result = test_func()
            assert result == "result"
            
            # Should have logged performance
            mock_info.assert_called()
    
    @pytest.mark.asyncio
    async def test_async_performance_decorator(self):
        """Test async performance monitoring decorator"""
        logger = get_logger("test")
        
        @monitor_async_performance("test_async_function")
        async def test_async_func():
            await asyncio.sleep(0.1)
            return "async_result"
        
        with patch.object(logger, 'info') as mock_info:
            result = await test_async_func()
            assert result == "async_result"
            
            # Should have logged performance
            mock_info.assert_called()

class TestMetrics:
    """Test metrics collection"""
    
    def setup_method(self):
        """Setup for each test"""
        self.collector = MetricsCollector(retention_hours=1)
    
    def teardown_method(self):
        """Cleanup after each test"""
        if self.collector.running:
            self.collector.stop()
    
    def test_counter_operations(self):
        """Test counter metrics"""
        self.collector.increment_counter("test.counter", 5)
        assert self.collector.get_counter("test.counter") == 5
        
        self.collector.increment_counter("test.counter", 3)
        assert self.collector.get_counter("test.counter") == 8
    
    def test_gauge_operations(self):
        """Test gauge metrics"""
        self.collector.set_gauge("test.gauge", 42.5)
        assert self.collector.get_gauge("test.gauge") == 42.5
        
        self.collector.set_gauge("test.gauge", 100.0)
        assert self.collector.get_gauge("test.gauge") == 100.0
    
    def test_histogram_operations(self):
        """Test histogram metrics"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            self.collector.record_histogram("test.histogram", value)
        
        stats = self.collector.get_histogram_stats("test.histogram")
        assert stats['count'] == 5
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['mean'] == 3.0
    
    def test_timer_operations(self):
        """Test timer metrics"""
        durations = [0.1, 0.2, 0.3, 0.4, 0.5]
        for duration in durations:
            self.collector.record_timer("test.timer", duration)
        
        avg_duration = self.collector.get_average_timer("test.timer")
        assert avg_duration == 0.3
    
    def test_metrics_summary(self):
        """Test metrics summary"""
        self.collector.increment_counter("test.requests", 100)
        self.collector.set_gauge("test.cpu", 75.5)
        self.collector.record_histogram("test.response_time", 1.5)
        
        summary = self.collector.get_metrics_summary()
        
        assert summary['counters']['test.requests'] == 100
        assert summary['gauges']['test.cpu'] == 75.5
        assert 'test.response_time' in summary['histograms']
        assert 'timestamp' in summary
    
    def test_prometheus_export(self):
        """Test Prometheus format export"""
        self.collector.increment_counter("test.requests", 50)
        self.collector.set_gauge("test.memory", 80.0)
        
        prometheus_output = self.collector.export_prometheus_format()
        
        assert "test_requests 50" in prometheus_output
        assert "test_memory 80.0" in prometheus_output
        assert "# TYPE test_requests counter" in prometheus_output
        assert "# TYPE test_memory gauge" in prometheus_output
    
    def test_time_function_decorator(self):
        """Test function timing decorator"""
        @time_function("test.decorated_function")
        def test_func():
            time.sleep(0.1)
            return "success"
        
        result = test_func()
        assert result == "success"
        
        # Check that timer was recorded
        avg_time = metrics_collector.get_average_timer("test.decorated_function")
        assert avg_time > 0.05  # Should be at least 50ms
    
    @pytest.mark.asyncio
    async def test_time_async_function_decorator(self):
        """Test async function timing decorator"""
        @time_async_function("test.async_decorated_function")
        async def test_async_func():
            await asyncio.sleep(0.1)
            return "async_success"
        
        result = await test_async_func()
        assert result == "async_success"
        
        # Check that timer was recorded
        avg_time = metrics_collector.get_average_timer("test.async_decorated_function")
        assert avg_time > 0.05  # Should be at least 50ms

class TestAlerts:
    """Test alerting system"""
    
    def setup_method(self):
        """Setup for each test"""
        self.alert_manager = AlertManager()
    
    def teardown_method(self):
        """Cleanup after each test"""
        if self.alert_manager.running:
            self.alert_manager.stop()
    
    def test_add_remove_rule(self):
        """Test adding and removing alert rules"""
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            condition=lambda: True,
            severity=AlertSeverity.MEDIUM
        )
        
        self.alert_manager.add_rule(rule)
        assert "test_rule" in self.alert_manager.rules
        
        self.alert_manager.remove_rule("test_rule")
        assert "test_rule" not in self.alert_manager.rules
    
    @pytest.mark.asyncio
    async def test_alert_triggering(self):
        """Test alert triggering"""
        triggered = False
        
        def test_condition():
            return triggered
        
        rule = AlertRule(
            name="test_alert",
            description="Test alert description",
            condition=test_condition,
            severity=AlertSeverity.HIGH,
            cooldown_minutes=1
        )
        
        self.alert_manager.add_rule(rule)
        
        # Mock notification sending
        with patch.object(self.alert_manager, '_send_notifications') as mock_send:
            # Trigger condition
            triggered = True
            await self.alert_manager._check_rules()
            
            # Should have triggered alert
            assert "test_alert" in self.alert_manager.active_alerts
            mock_send.assert_called_once()
            
            # Reset condition
            triggered = False
            await self.alert_manager._check_rules()
            
            # Should have resolved alert
            assert "test_alert" not in self.alert_manager.active_alerts
    
    def test_alert_cooldown(self):
        """Test alert cooldown functionality"""
        rule = AlertRule(
            name="cooldown_test",
            description="Cooldown test",
            condition=lambda: True,
            severity=AlertSeverity.LOW,
            cooldown_minutes=5
        )
        
        self.alert_manager.add_rule(rule)
        
        # Set last triggered time to recent
        self.alert_manager.last_triggered["cooldown_test"] = datetime.utcnow() - timedelta(minutes=2)
        
        # Should not trigger due to cooldown
        with patch.object(self.alert_manager, '_trigger_alert') as mock_trigger:
            asyncio.run(self.alert_manager._check_rules())
            mock_trigger.assert_not_called()
    
    def test_acknowledge_alert(self):
        """Test alert acknowledgment"""
        alert = Alert(
            id="test_alert_1",
            name="test_alert",
            description="Test alert",
            severity=AlertSeverity.MEDIUM,
            status=AlertStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.alert_manager.active_alerts["test_alert"] = alert
        
        # Acknowledge alert
        result = self.alert_manager.acknowledge_alert("test_alert")
        assert result is True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_at is not None
    
    def test_get_active_alerts(self):
        """Test getting active alerts"""
        alert1 = Alert(
            id="alert_1",
            name="alert_1",
            description="First alert",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        alert2 = Alert(
            id="alert_2",
            name="alert_2",
            description="Second alert",
            severity=AlertSeverity.MEDIUM,
            status=AlertStatus.ACKNOWLEDGED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.alert_manager.active_alerts["alert_1"] = alert1
        self.alert_manager.active_alerts["alert_2"] = alert2
        
        active_alerts = self.alert_manager.get_active_alerts()
        assert len(active_alerts) == 2
        assert any(alert['name'] == 'alert_1' for alert in active_alerts)
        assert any(alert['name'] == 'alert_2' for alert in active_alerts)
    
    def test_trigger_custom_alert(self):
        """Test triggering custom alerts"""
        with patch.object(alert_manager, '_send_notifications') as mock_send:
            trigger_custom_alert(
                name="custom_test",
                description="Custom test alert",
                severity=AlertSeverity.CRITICAL,
                metadata={"source": "test"}
            )
            
            # Should have created active alert
            assert "custom_test" in alert_manager.active_alerts
            alert = alert_manager.active_alerts["custom_test"]
            assert alert.severity == AlertSeverity.CRITICAL
            assert alert.metadata["source"] == "test"

class TestLogAnalyzer:
    """Test log analysis functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = LogAnalyzer(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_parse_json_log_entry(self):
        """Test parsing JSON log entries"""
        json_line = json.dumps({
            "timestamp": "2024-01-01T12:00:00Z",
            "level": "INFO",
            "logger": "test.logger",
            "message": "Test message",
            "correlation_id": "test-123",
            "module": "test_module",
            "function": "test_function",
            "line": 42,
            "process_id": 1234,
            "thread_id": 5678
        })
        
        entry = self.analyzer.parse_json_log_entry(json_line)
        
        assert entry is not None
        assert entry.level == "INFO"
        assert entry.logger == "test.logger"
        assert entry.message == "Test message"
        assert entry.correlation_id == "test-123"
        assert entry.line == 42
    
    def test_parse_text_log_entry(self):
        """Test parsing text log entries"""
        text_line = "2024-01-01 12:00:00 [test-123] INFO test.logger:42 - Test message"
        
        entry = self.analyzer.parse_text_log_entry(text_line)
        
        assert entry is not None
        assert entry.level == "INFO"
        assert entry.logger == "test.logger"
        assert entry.message == "Test message"
        assert entry.correlation_id == "test-123"
        assert entry.line == 42
    
    def test_analyze_logs(self):
        """Test log analysis"""
        # Create test log file
        log_file = os.path.join(self.temp_dir, "test.log")
        
        log_entries = [
            json.dumps({
                "timestamp": "2024-01-01T12:00:00Z",
                "level": "INFO",
                "logger": "test.logger",
                "message": "Info message",
                "correlation_id": "test-123"
            }),
            json.dumps({
                "timestamp": "2024-01-01T12:01:00Z",
                "level": "ERROR",
                "logger": "test.logger",
                "message": "Error message",
                "correlation_id": "test-456"
            }),
            json.dumps({
                "timestamp": "2024-01-01T12:02:00Z",
                "level": "WARNING",
                "logger": "other.logger",
                "message": "Warning message",
                "correlation_id": "test-789"
            })
        ]
        
        with open(log_file, 'w') as f:
            f.write('\n'.join(log_entries))
        
        # Analyze logs
        start_time = datetime(2024, 1, 1, 11, 0, 0)
        end_time = datetime(2024, 1, 1, 13, 0, 0)
        
        analysis = self.analyzer.analyze_logs(start_time, end_time)
        
        assert analysis.total_entries == 3
        assert analysis.entries_by_level['INFO'] == 1
        assert analysis.entries_by_level['ERROR'] == 1
        assert analysis.entries_by_level['WARNING'] == 1
        assert analysis.entries_by_logger['test.logger'] == 2
        assert analysis.entries_by_logger['other.logger'] == 1
    
    def test_trace_correlation_id(self):
        """Test correlation ID tracing"""
        # Create test log file
        log_file = os.path.join(self.temp_dir, "test.log")
        
        log_entries = [
            json.dumps({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": "INFO",
                "logger": "test.logger",
                "message": "Request started",
                "correlation_id": "trace-123"
            }),
            json.dumps({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": "DEBUG",
                "logger": "test.logger",
                "message": "Processing request",
                "correlation_id": "trace-123"
            }),
            json.dumps({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": "INFO",
                "logger": "test.logger",
                "message": "Request completed",
                "correlation_id": "trace-123"
            }),
            json.dumps({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": "INFO",
                "logger": "test.logger",
                "message": "Other request",
                "correlation_id": "other-456"
            })
        ]
        
        with open(log_file, 'w') as f:
            f.write('\n'.join(log_entries))
        
        # Trace specific correlation ID
        trace_entries = self.analyzer.trace_correlation_id("trace-123")
        
        assert len(trace_entries) == 3
        assert all(entry.correlation_id == "trace-123" for entry in trace_entries)
        assert trace_entries[0].message == "Request started"
        assert trace_entries[-1].message == "Request completed"
    
    def test_generate_report(self):
        """Test report generation"""
        from src.utils.log_analyzer import LogAnalysisResult
        
        analysis = LogAnalysisResult(
            total_entries=100,
            entries_by_level={'INFO': 70, 'ERROR': 20, 'WARNING': 10},
            entries_by_logger={'app.main': 60, 'app.db': 40},
            error_patterns=[('.*database.*', 15), ('.*timeout.*', 5)],
            performance_issues=[],
            security_events=[],
            top_errors=[('Database connection failed', 10), ('Request timeout', 5)],
            correlation_traces={},
            time_range=(datetime.utcnow() - timedelta(hours=1), datetime.utcnow())
        )
        
        report = self.analyzer.generate_report(analysis)
        
        assert "LOG ANALYSIS REPORT" in report
        assert "Total Entries: 100" in report
        assert "INFO" in report
        assert "ERROR" in report
        assert "app.main" in report
        assert "Database connection failed" in report

@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for monitoring components"""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        # Setup
        collector = MetricsCollector(retention_hours=1)
        alert_mgr = AlertManager()
        
        try:
            # Start systems
            collector.start()
            await alert_mgr.start()
            
            # Generate some metrics
            collector.increment_counter("test.requests", 100)
            collector.set_gauge("test.cpu", 85.0)  # High CPU
            collector.record_timer("test.response_time", 2.5)
            
            # Create alert rule for high CPU
            def high_cpu_condition():
                return collector.get_gauge("test.cpu") > 80
            
            rule = AlertRule(
                name="high_cpu_test",
                description="High CPU usage detected",
                condition=high_cpu_condition,
                severity=AlertSeverity.HIGH,
                cooldown_minutes=1
            )
            
            alert_mgr.add_rule(rule)
            
            # Wait for alert check
            await asyncio.sleep(0.1)
            await alert_mgr._check_rules()
            
            # Verify alert was triggered
            assert "high_cpu_test" in alert_mgr.active_alerts
            
            # Get metrics summary
            summary = collector.get_metrics_summary()
            assert summary['counters']['test.requests'] == 100
            assert summary['gauges']['test.cpu'] == 85.0
            
            # Resolve condition
            collector.set_gauge("test.cpu", 50.0)
            await alert_mgr._check_rules()
            
            # Verify alert was resolved
            assert "high_cpu_test" not in alert_mgr.active_alerts
            
        finally:
            # Cleanup
            collector.stop()
            alert_mgr.stop()

if __name__ == "__main__":
    pytest.main([__file__])