#!/usr/bin/env python3
"""
Command Line Interface for Flight Scheduling Analysis System
"""

import argparse
import asyncio
import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logging_config import init_logging, get_logger
from utils.metrics import metrics_collector, init_metrics, shutdown_metrics
from utils.alerts import alert_manager, init_alerts, shutdown_alerts
from utils.log_analyzer import LogAnalyzer, analyze_recent_logs, generate_daily_report

logger = get_logger(__name__)

class FlightAnalysisCLI:
    """Command Line Interface for Flight Analysis System"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Flight Scheduling Analysis System CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s logs analyze --hours 24
  %(prog)s metrics summary
  %(prog)s alerts list
  %(prog)s health check
  %(prog)s backup create
            """
        )
        
        parser.add_argument(
            '--config',
            type=str,
            help='Configuration file path'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Logs commands
        logs_parser = subparsers.add_parser('logs', help='Log analysis commands')
        logs_subparsers = logs_parser.add_subparsers(dest='logs_action')
        
        analyze_parser = logs_subparsers.add_parser('analyze', help='Analyze logs')
        analyze_parser.add_argument('--hours', type=int, default=24, help='Hours to analyze')
        analyze_parser.add_argument('--pattern', type=str, default='*.log', help='Log file pattern')
        analyze_parser.add_argument('--output', type=str, help='Output file for report')
        
        trace_parser = logs_subparsers.add_parser('trace', help='Trace correlation ID')
        trace_parser.add_argument('correlation_id', help='Correlation ID to trace')
        trace_parser.add_argument('--hours', type=int, default=24, help='Hours to search')
        
        report_parser = logs_subparsers.add_parser('report', help='Generate daily report')
        report_parser.add_argument('--output-dir', type=str, default='reports', help='Output directory')
        
        # Metrics commands
        metrics_parser = subparsers.add_parser('metrics', help='Metrics commands')
        metrics_subparsers = metrics_parser.add_subparsers(dest='metrics_action')
        
        metrics_subparsers.add_parser('summary', help='Show metrics summary')
        metrics_subparsers.add_parser('prometheus', help='Export Prometheus format')
        
        reset_parser = metrics_subparsers.add_parser('reset', help='Reset metrics')
        reset_parser.add_argument('--confirm', action='store_true', help='Confirm reset')
        
        # Alerts commands
        alerts_parser = subparsers.add_parser('alerts', help='Alerts commands')
        alerts_subparsers = alerts_parser.add_subparsers(dest='alerts_action')
        
        alerts_subparsers.add_parser('list', help='List active alerts')
        alerts_subparsers.add_parser('history', help='Show alert history')
        
        ack_parser = alerts_subparsers.add_parser('acknowledge', help='Acknowledge alert')
        ack_parser.add_argument('alert_name', help='Alert name to acknowledge')
        
        test_parser = alerts_subparsers.add_parser('test', help='Test alert system')
        test_parser.add_argument('--severity', choices=['low', 'medium', 'high', 'critical'], 
                               default='medium', help='Test alert severity')
        
        # Health commands
        health_parser = subparsers.add_parser('health', help='Health check commands')
        health_subparsers = health_parser.add_subparsers(dest='health_action')
        
        health_subparsers.add_parser('check', help='Run health checks')
        health_subparsers.add_parser('status', help='Show system status')
        
        # Backup commands
        backup_parser = subparsers.add_parser('backup', help='Backup commands')
        backup_subparsers = backup_parser.add_subparsers(dest='backup_action')
        
        create_backup_parser = backup_subparsers.add_parser('create', help='Create backup')
        create_backup_parser.add_argument('--name', type=str, help='Backup name')
        
        list_backup_parser = backup_subparsers.add_parser('list', help='List backups')
        
        restore_parser = backup_subparsers.add_parser('restore', help='Restore backup')
        restore_parser.add_argument('backup_name', help='Backup name to restore')
        restore_parser.add_argument('--confirm', action='store_true', help='Confirm restore')
        
        return parser
    
    async def run(self, args: Optional[list] = None) -> int:
        """Run CLI command"""
        parsed_args = self.parser.parse_args(args)
        
        # Initialize logging
        log_level = 'DEBUG' if parsed_args.verbose else 'INFO'
        init_logging()
        
        try:
            if parsed_args.command == 'logs':
                return await self._handle_logs_command(parsed_args)
            elif parsed_args.command == 'metrics':
                return await self._handle_metrics_command(parsed_args)
            elif parsed_args.command == 'alerts':
                return await self._handle_alerts_command(parsed_args)
            elif parsed_args.command == 'health':
                return await self._handle_health_command(parsed_args)
            elif parsed_args.command == 'backup':
                return await self._handle_backup_command(parsed_args)
            else:
                self.parser.print_help()
                return 1
                
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            return 130
        except Exception as e:
            logger.error(f"Command failed: {e}", exc_info=parsed_args.verbose)
            return 1
    
    async def _handle_logs_command(self, args) -> int:
        """Handle logs commands"""
        if args.logs_action == 'analyze':
            return await self._analyze_logs(args)
        elif args.logs_action == 'trace':
            return await self._trace_correlation_id(args)
        elif args.logs_action == 'report':
            return await self._generate_log_report(args)
        else:
            logger.error("Unknown logs action")
            return 1
    
    async def _analyze_logs(self, args) -> int:
        """Analyze logs"""
        logger.info(f"Analyzing logs for the last {args.hours} hours...")
        
        try:
            analysis = analyze_recent_logs(args.hours, args.pattern)
            
            analyzer = LogAnalyzer()
            report = analyzer.generate_report(analysis)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                logger.info(f"Report saved to {args.output}")
            else:
                print(report)
            
            return 0
            
        except Exception as e:
            logger.error(f"Log analysis failed: {e}")
            return 1
    
    async def _trace_correlation_id(self, args) -> int:
        """Trace correlation ID"""
        logger.info(f"Tracing correlation ID: {args.correlation_id}")
        
        try:
            analyzer = LogAnalyzer()
            entries = analyzer.trace_correlation_id(args.correlation_id, args.hours)
            
            if not entries:
                print(f"No entries found for correlation ID: {args.correlation_id}")
                return 0
            
            print(f"Found {len(entries)} entries for correlation ID: {args.correlation_id}")
            print("=" * 80)
            
            for entry in entries:
                print(f"[{entry.timestamp}] {entry.level:8} {entry.logger:20} - {entry.message}")
                if entry.exception:
                    print(f"  Exception: {entry.exception}")
                print()
            
            return 0
            
        except Exception as e:
            logger.error(f"Trace failed: {e}")
            return 1
    
    async def _generate_log_report(self, args) -> int:
        """Generate daily log report"""
        logger.info("Generating daily log report...")
        
        try:
            report_file = generate_daily_report(args.output_dir)
            logger.info(f"Daily report generated: {report_file}")
            return 0
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return 1
    
    async def _handle_metrics_command(self, args) -> int:
        """Handle metrics commands"""
        init_metrics()
        
        try:
            if args.metrics_action == 'summary':
                return await self._show_metrics_summary()
            elif args.metrics_action == 'prometheus':
                return await self._export_prometheus_metrics()
            elif args.metrics_action == 'reset':
                return await self._reset_metrics(args)
            else:
                logger.error("Unknown metrics action")
                return 1
        finally:
            shutdown_metrics()
    
    async def _show_metrics_summary(self) -> int:
        """Show metrics summary"""
        try:
            summary = metrics_collector.get_metrics_summary()
            
            print("METRICS SUMMARY")
            print("=" * 50)
            print(f"Timestamp: {summary['timestamp']}")
            print()
            
            if summary['counters']:
                print("COUNTERS:")
                for name, value in summary['counters'].items():
                    print(f"  {name:40}: {value:>10,}")
                print()
            
            if summary['gauges']:
                print("GAUGES:")
                for name, value in summary['gauges'].items():
                    print(f"  {name:40}: {value:>10.2f}")
                print()
            
            if summary['histograms']:
                print("HISTOGRAMS:")
                for name, stats in summary['histograms'].items():
                    if stats:
                        print(f"  {name}:")
                        print(f"    Count: {stats['count']:>10,}")
                        print(f"    Mean:  {stats['mean']:>10.3f}")
                        print(f"    P95:   {stats['p95']:>10.3f}")
                print()
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to show metrics summary: {e}")
            return 1
    
    async def _export_prometheus_metrics(self) -> int:
        """Export Prometheus metrics"""
        try:
            prometheus_output = metrics_collector.export_prometheus_format()
            print(prometheus_output)
            return 0
            
        except Exception as e:
            logger.error(f"Failed to export Prometheus metrics: {e}")
            return 1
    
    async def _reset_metrics(self, args) -> int:
        """Reset metrics"""
        if not args.confirm:
            print("This will reset all metrics. Use --confirm to proceed.")
            return 1
        
        try:
            # Reset metrics collector
            metrics_collector.counters.clear()
            metrics_collector.gauges.clear()
            metrics_collector.histograms.clear()
            metrics_collector.timers.clear()
            
            logger.info("Metrics reset successfully")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to reset metrics: {e}")
            return 1
    
    async def _handle_alerts_command(self, args) -> int:
        """Handle alerts commands"""
        await init_alerts()
        
        try:
            if args.alerts_action == 'list':
                return await self._list_alerts()
            elif args.alerts_action == 'history':
                return await self._show_alert_history()
            elif args.alerts_action == 'acknowledge':
                return await self._acknowledge_alert(args)
            elif args.alerts_action == 'test':
                return await self._test_alerts(args)
            else:
                logger.error("Unknown alerts action")
                return 1
        finally:
            shutdown_alerts()
    
    async def _list_alerts(self) -> int:
        """List active alerts"""
        try:
            alerts = alert_manager.get_active_alerts()
            
            if not alerts:
                print("No active alerts")
                return 0
            
            print(f"ACTIVE ALERTS ({len(alerts)})")
            print("=" * 80)
            
            for alert in alerts:
                print(f"Name: {alert['name']}")
                print(f"Severity: {alert['severity'].upper()}")
                print(f"Status: {alert['status'].upper()}")
                print(f"Created: {alert['created_at']}")
                print(f"Description: {alert['description']}")
                print("-" * 40)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to list alerts: {e}")
            return 1
    
    async def _show_alert_history(self) -> int:
        """Show alert history"""
        try:
            history = alert_manager.get_alert_history(50)
            
            if not history:
                print("No alert history")
                return 0
            
            print(f"ALERT HISTORY ({len(history)} recent alerts)")
            print("=" * 80)
            
            for alert in history:
                status_icon = "🔴" if alert['status'] == 'active' else "✅" if alert['status'] == 'resolved' else "⚠️"
                print(f"{status_icon} [{alert['created_at']}] {alert['name']} - {alert['severity'].upper()}")
                print(f"   {alert['description']}")
                print()
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to show alert history: {e}")
            return 1
    
    async def _acknowledge_alert(self, args) -> int:
        """Acknowledge alert"""
        try:
            success = alert_manager.acknowledge_alert(args.alert_name)
            
            if success:
                logger.info(f"Alert '{args.alert_name}' acknowledged")
                return 0
            else:
                logger.error(f"Alert '{args.alert_name}' not found or already resolved")
                return 1
                
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return 1
    
    async def _test_alerts(self, args) -> int:
        """Test alert system"""
        try:
            from utils.alerts import trigger_custom_alert, AlertSeverity
            
            severity_map = {
                'low': AlertSeverity.LOW,
                'medium': AlertSeverity.MEDIUM,
                'high': AlertSeverity.HIGH,
                'critical': AlertSeverity.CRITICAL
            }
            
            trigger_custom_alert(
                name="cli_test_alert",
                description=f"Test alert triggered from CLI with {args.severity} severity",
                severity=severity_map[args.severity],
                metadata={"source": "cli", "test": True}
            )
            
            logger.info(f"Test alert triggered with {args.severity} severity")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to trigger test alert: {e}")
            return 1
    
    async def _handle_health_command(self, args) -> int:
        """Handle health commands"""
        if args.health_action == 'check':
            return await self._run_health_checks()
        elif args.health_action == 'status':
            return await self._show_system_status()
        else:
            logger.error("Unknown health action")
            return 1
    
    async def _run_health_checks(self) -> int:
        """Run health checks"""
        logger.info("Running health checks...")
        
        # This would typically run the health check script
        try:
            import subprocess
            result = subprocess.run(['./scripts/health-check.sh'], 
                                  capture_output=True, text=True, timeout=60)
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            return result.returncode
            
        except subprocess.TimeoutExpired:
            logger.error("Health check timed out")
            return 1
        except FileNotFoundError:
            logger.error("Health check script not found")
            return 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return 1
    
    async def _show_system_status(self) -> int:
        """Show system status"""
        try:
            import psutil
            
            print("SYSTEM STATUS")
            print("=" * 50)
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            print(f"CPU Usage: {cpu_percent:.1f}%")
            
            # Memory
            memory = psutil.virtual_memory()
            print(f"Memory Usage: {memory.percent:.1f}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
            
            # Disk
            disk = psutil.disk_usage('/')
            print(f"Disk Usage: {disk.percent:.1f}% ({disk.used // (1024**3):.1f}GB / {disk.total // (1024**3):.1f}GB)")
            
            # Network
            network = psutil.net_io_counters()
            print(f"Network: {network.bytes_sent // (1024**2):.1f}MB sent, {network.bytes_recv // (1024**2):.1f}MB received")
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to show system status: {e}")
            return 1
    
    async def _handle_backup_command(self, args) -> int:
        """Handle backup commands"""
        if args.backup_action == 'create':
            return await self._create_backup(args)
        elif args.backup_action == 'list':
            return await self._list_backups()
        elif args.backup_action == 'restore':
            return await self._restore_backup(args)
        else:
            logger.error("Unknown backup action")
            return 1
    
    async def _create_backup(self, args) -> int:
        """Create backup"""
        logger.info("Creating backup...")
        
        try:
            import subprocess
            
            cmd = ['./scripts/backup.sh']
            if args.name:
                cmd.append(args.name)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            return result.returncode
            
        except subprocess.TimeoutExpired:
            logger.error("Backup timed out")
            return 1
        except FileNotFoundError:
            logger.error("Backup script not found")
            return 1
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return 1
    
    async def _list_backups(self) -> int:
        """List backups"""
        try:
            backup_dir = Path("backups")
            if not backup_dir.exists():
                print("No backups directory found")
                return 0
            
            backups = list(backup_dir.glob("*.tar.gz"))
            backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if not backups:
                print("No backups found")
                return 0
            
            print(f"AVAILABLE BACKUPS ({len(backups)})")
            print("=" * 60)
            
            for backup in backups:
                stat = backup.stat()
                size_mb = stat.st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(stat.st_mtime)
                print(f"{backup.name:40} {size_mb:8.1f}MB {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return 1
    
    async def _restore_backup(self, args) -> int:
        """Restore backup"""
        if not args.confirm:
            print("This will restore the backup and may overwrite existing data.")
            print("Use --confirm to proceed.")
            return 1
        
        logger.info(f"Restoring backup: {args.backup_name}")
        
        try:
            # This would typically run a restore script
            logger.info("Backup restore functionality would be implemented here")
            return 0
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return 1

def main():
    """Main entry point"""
    cli = FlightAnalysisCLI()
    return asyncio.run(cli.run())

if __name__ == '__main__':
    sys.exit(main())