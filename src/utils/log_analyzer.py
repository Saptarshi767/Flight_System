"""
Log analysis and aggregation utilities for Flight Scheduling Analysis System
"""

import json
import re
import os
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Iterator, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import glob
from pathlib import Path

from .logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: str
    logger: str
    message: str
    correlation_id: str
    module: str
    function: str
    line: int
    process_id: int
    thread_id: int
    exception: Optional[str] = None
    extra_fields: Dict[str, Any] = None

@dataclass
class LogAnalysisResult:
    """Log analysis results"""
    total_entries: int
    entries_by_level: Dict[str, int]
    entries_by_logger: Dict[str, int]
    error_patterns: List[Tuple[str, int]]
    performance_issues: List[Dict[str, Any]]
    security_events: List[Dict[str, Any]]
    top_errors: List[Tuple[str, int]]
    correlation_traces: Dict[str, List[LogEntry]]
    time_range: Tuple[datetime, datetime]

class LogAnalyzer:
    """Log analysis and aggregation system"""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = Path(log_directory)
        self.error_patterns = [
            r"ERROR.*database.*connection",
            r"ERROR.*timeout",
            r"ERROR.*memory",
            r"ERROR.*permission",
            r"CRITICAL.*",
            r".*exception.*",
            r".*failed.*",
            r".*error.*"
        ]
        self.performance_thresholds = {
            'slow_query': 1.0,  # seconds
            'slow_request': 5.0,  # seconds
            'high_memory': 1000,  # MB
            'high_cpu': 80  # percent
        }
        
    def parse_json_log_entry(self, line: str) -> Optional[LogEntry]:
        """Parse a JSON log entry"""
        try:
            data = json.loads(line.strip())
            
            return LogEntry(
                timestamp=datetime.fromisoformat(data.get('timestamp', '').replace('Z', '+00:00')),
                level=data.get('level', 'UNKNOWN'),
                logger=data.get('logger', 'unknown'),
                message=data.get('message', ''),
                correlation_id=data.get('correlation_id', 'unknown'),
                module=data.get('module', 'unknown'),
                function=data.get('function', 'unknown'),
                line=data.get('line', 0),
                process_id=data.get('process_id', 0),
                thread_id=data.get('thread_id', 0),
                exception=data.get('exception'),
                extra_fields={k: v for k, v in data.items() if k not in [
                    'timestamp', 'level', 'logger', 'message', 'correlation_id',
                    'module', 'function', 'line', 'process_id', 'thread_id', 'exception'
                ]}
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse log entry: {e}")
            return None
    
    def parse_text_log_entry(self, line: str) -> Optional[LogEntry]:
        """Parse a text log entry"""
        # Pattern for text logs: timestamp [correlation_id] level logger:line - message
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[([^\]]+)\] (\w+)\s+([^:]+):(\d+) - (.+)'
        match = re.match(pattern, line.strip())
        
        if not match:
            return None
        
        try:
            timestamp_str, correlation_id, level, logger_info, line_num, message = match.groups()
            
            return LogEntry(
                timestamp=datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S'),
                level=level,
                logger=logger_info,
                message=message,
                correlation_id=correlation_id,
                module='unknown',
                function='unknown',
                line=int(line_num),
                process_id=0,
                thread_id=0
            )
        except ValueError as e:
            logger.warning(f"Failed to parse text log entry: {e}")
            return None
    
    def read_log_file(self, file_path: Path) -> Iterator[LogEntry]:
        """Read and parse log file"""
        try:
            # Handle gzipped files
            if file_path.suffix == '.gz':
                open_func = gzip.open
                mode = 'rt'
            else:
                open_func = open
                mode = 'r'
            
            with open_func(file_path, mode, encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    # Try JSON format first
                    entry = self.parse_json_log_entry(line)
                    if entry is None:
                        # Try text format
                        entry = self.parse_text_log_entry(line)
                    
                    if entry:
                        yield entry
                    
        except Exception as e:
            logger.error(f"Error reading log file {file_path}: {e}")
    
    def get_log_files(self, pattern: str = "*.log", include_rotated: bool = True) -> List[Path]:
        """Get list of log files to analyze"""
        files = []
        
        # Current log files
        files.extend(self.log_directory.glob(pattern))
        
        # Rotated log files
        if include_rotated:
            files.extend(self.log_directory.glob(f"{pattern}.*"))
            files.extend(self.log_directory.glob(f"{pattern}.gz"))
        
        # Sort by modification time (newest first)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        return files
    
    def analyze_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        log_pattern: str = "*.log",
        max_files: int = 10
    ) -> LogAnalysisResult:
        """Analyze logs within time range"""
        
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.utcnow()
        
        logger.info(f"Analyzing logs from {start_time} to {end_time}")
        
        # Initialize counters
        total_entries = 0
        entries_by_level = Counter()
        entries_by_logger = Counter()
        error_messages = Counter()
        performance_issues = []
        security_events = []
        correlation_traces = defaultdict(list)
        
        min_timestamp = None
        max_timestamp = None
        
        # Process log files
        log_files = self.get_log_files(log_pattern)[:max_files]
        
        for log_file in log_files:
            logger.info(f"Processing log file: {log_file}")
            
            for entry in self.read_log_file(log_file):
                # Filter by time range
                if entry.timestamp < start_time or entry.timestamp > end_time:
                    continue
                
                total_entries += 1
                entries_by_level[entry.level] += 1
                entries_by_logger[entry.logger] += 1
                
                # Track time range
                if min_timestamp is None or entry.timestamp < min_timestamp:
                    min_timestamp = entry.timestamp
                if max_timestamp is None or entry.timestamp > max_timestamp:
                    max_timestamp = entry.timestamp
                
                # Collect correlation traces
                correlation_traces[entry.correlation_id].append(entry)
                
                # Analyze errors
                if entry.level in ['ERROR', 'CRITICAL']:
                    error_messages[entry.message] += 1
                
                # Analyze performance issues
                if entry.extra_fields:
                    self._analyze_performance_issues(entry, performance_issues)
                
                # Analyze security events
                if entry.extra_fields and entry.extra_fields.get('security_log'):
                    security_events.append({
                        'timestamp': entry.timestamp,
                        'event': entry.extra_fields.get('event_name'),
                        'message': entry.message,
                        'correlation_id': entry.correlation_id
                    })
        
        # Find error patterns
        error_patterns = self._find_error_patterns(error_messages)
        
        # Get top errors
        top_errors = error_messages.most_common(10)
        
        return LogAnalysisResult(
            total_entries=total_entries,
            entries_by_level=dict(entries_by_level),
            entries_by_logger=dict(entries_by_logger),
            error_patterns=error_patterns,
            performance_issues=performance_issues,
            security_events=security_events,
            top_errors=top_errors,
            correlation_traces=dict(correlation_traces),
            time_range=(min_timestamp or start_time, max_timestamp or end_time)
        )
    
    def _analyze_performance_issues(self, entry: LogEntry, issues: List[Dict[str, Any]]):
        """Analyze performance-related log entries"""
        if not entry.extra_fields:
            return
        
        # Check for slow queries
        if entry.extra_fields.get('performance_log') and 'duration' in entry.extra_fields:
            duration = entry.extra_fields['duration']
            operation = entry.extra_fields.get('operation', 'unknown')
            
            if duration > self.performance_thresholds['slow_query']:
                issues.append({
                    'type': 'slow_operation',
                    'timestamp': entry.timestamp,
                    'operation': operation,
                    'duration': duration,
                    'correlation_id': entry.correlation_id,
                    'message': entry.message
                })
        
        # Check for memory issues
        if 'memory_used_mb' in entry.extra_fields:
            memory_used = entry.extra_fields['memory_used_mb']
            if memory_used > self.performance_thresholds['high_memory']:
                issues.append({
                    'type': 'high_memory',
                    'timestamp': entry.timestamp,
                    'memory_used_mb': memory_used,
                    'correlation_id': entry.correlation_id,
                    'message': entry.message
                })
    
    def _find_error_patterns(self, error_messages: Counter) -> List[Tuple[str, int]]:
        """Find common error patterns"""
        patterns = []
        
        for pattern in self.error_patterns:
            count = 0
            for message, msg_count in error_messages.items():
                if re.search(pattern, message, re.IGNORECASE):
                    count += msg_count
            
            if count > 0:
                patterns.append((pattern, count))
        
        # Sort by count
        patterns.sort(key=lambda x: x[1], reverse=True)
        return patterns
    
    def trace_correlation_id(self, correlation_id: str, hours_back: int = 24) -> List[LogEntry]:
        """Trace all log entries for a specific correlation ID"""
        start_time = datetime.utcnow() - timedelta(hours=hours_back)
        end_time = datetime.utcnow()
        
        entries = []
        log_files = self.get_log_files()
        
        for log_file in log_files:
            for entry in self.read_log_file(log_file):
                if (entry.correlation_id == correlation_id and 
                    start_time <= entry.timestamp <= end_time):
                    entries.append(entry)
        
        # Sort by timestamp
        entries.sort(key=lambda e: e.timestamp)
        return entries
    
    def generate_report(self, analysis: LogAnalysisResult) -> str:
        """Generate a human-readable analysis report"""
        report = []
        report.append("=" * 60)
        report.append("LOG ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Time Range: {analysis.time_range[0]} to {analysis.time_range[1]}")
        report.append(f"Total Entries: {analysis.total_entries:,}")
        report.append("")
        
        # Entries by level
        report.append("ENTRIES BY LOG LEVEL:")
        report.append("-" * 30)
        for level, count in sorted(analysis.entries_by_level.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / analysis.total_entries) * 100 if analysis.total_entries > 0 else 0
            report.append(f"  {level:10}: {count:8,} ({percentage:5.1f}%)")
        report.append("")
        
        # Top loggers
        report.append("TOP LOGGERS:")
        report.append("-" * 30)
        for logger, count in sorted(analysis.entries_by_logger.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / analysis.total_entries) * 100 if analysis.total_entries > 0 else 0
            report.append(f"  {logger:30}: {count:8,} ({percentage:5.1f}%)")
        report.append("")
        
        # Error patterns
        if analysis.error_patterns:
            report.append("ERROR PATTERNS:")
            report.append("-" * 30)
            for pattern, count in analysis.error_patterns[:10]:
                report.append(f"  {pattern:50}: {count:6,}")
            report.append("")
        
        # Top errors
        if analysis.top_errors:
            report.append("TOP ERROR MESSAGES:")
            report.append("-" * 30)
            for message, count in analysis.top_errors[:5]:
                truncated_message = message[:80] + "..." if len(message) > 80 else message
                report.append(f"  [{count:4}] {truncated_message}")
            report.append("")
        
        # Performance issues
        if analysis.performance_issues:
            report.append("PERFORMANCE ISSUES:")
            report.append("-" * 30)
            slow_ops = [issue for issue in analysis.performance_issues if issue['type'] == 'slow_operation']
            memory_issues = [issue for issue in analysis.performance_issues if issue['type'] == 'high_memory']
            
            if slow_ops:
                report.append(f"  Slow Operations: {len(slow_ops)}")
                avg_duration = sum(issue['duration'] for issue in slow_ops) / len(slow_ops)
                report.append(f"  Average Duration: {avg_duration:.2f}s")
            
            if memory_issues:
                report.append(f"  High Memory Events: {len(memory_issues)}")
                max_memory = max(issue['memory_used_mb'] for issue in memory_issues)
                report.append(f"  Peak Memory Usage: {max_memory:.1f}MB")
            report.append("")
        
        # Security events
        if analysis.security_events:
            report.append("SECURITY EVENTS:")
            report.append("-" * 30)
            event_types = Counter(event['event'] for event in analysis.security_events)
            for event_type, count in event_types.most_common():
                report.append(f"  {event_type:30}: {count:6,}")
            report.append("")
        
        # Correlation traces summary
        if analysis.correlation_traces:
            report.append("CORRELATION TRACES:")
            report.append("-" * 30)
            trace_lengths = [len(entries) for entries in analysis.correlation_traces.values()]
            avg_trace_length = sum(trace_lengths) / len(trace_lengths)
            max_trace_length = max(trace_lengths)
            report.append(f"  Total Traces: {len(analysis.correlation_traces):,}")
            report.append(f"  Average Trace Length: {avg_trace_length:.1f}")
            report.append(f"  Longest Trace: {max_trace_length}")
            report.append("")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    def export_analysis_json(self, analysis: LogAnalysisResult, output_file: str):
        """Export analysis results to JSON file"""
        data = {
            'total_entries': analysis.total_entries,
            'entries_by_level': analysis.entries_by_level,
            'entries_by_logger': analysis.entries_by_logger,
            'error_patterns': analysis.error_patterns,
            'performance_issues': analysis.performance_issues,
            'security_events': analysis.security_events,
            'top_errors': analysis.top_errors,
            'time_range': [
                analysis.time_range[0].isoformat(),
                analysis.time_range[1].isoformat()
            ],
            'correlation_trace_count': len(analysis.correlation_traces)
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Analysis results exported to {output_file}")

# Convenience functions
def analyze_recent_logs(hours: int = 24, log_pattern: str = "*.log") -> LogAnalysisResult:
    """Analyze recent logs"""
    analyzer = LogAnalyzer()
    start_time = datetime.utcnow() - timedelta(hours=hours)
    return analyzer.analyze_logs(start_time=start_time, log_pattern=log_pattern)

def generate_daily_report(output_dir: str = "reports") -> str:
    """Generate daily log analysis report"""
    analyzer = LogAnalyzer()
    
    # Analyze last 24 hours
    analysis = analyze_recent_logs(24)
    
    # Generate report
    report_content = analyzer.generate_report(analysis)
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f"log_analysis_{datetime.utcnow().strftime('%Y%m%d')}.txt")
    
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    # Export JSON
    json_file = os.path.join(output_dir, f"log_analysis_{datetime.utcnow().strftime('%Y%m%d')}.json")
    analyzer.export_analysis_json(analysis, json_file)
    
    logger.info(f"Daily log report generated: {report_file}")
    return report_file

def trace_request(correlation_id: str) -> List[LogEntry]:
    """Trace a specific request by correlation ID"""
    analyzer = LogAnalyzer()
    return analyzer.trace_correlation_id(correlation_id)