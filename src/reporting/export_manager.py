"""
Export Manager for handling different data export formats
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .models import ExportFormat, ExportResult

logger = logging.getLogger(__name__)


class ExportManager:
    """Manages data export in various formats"""
    
    def __init__(self, export_directory: str = "exports"):
        self.export_directory = Path(export_directory)
        self.export_directory.mkdir(exist_ok=True)
        
    def export_data(
        self,
        data: Any,
        export_format: ExportFormat,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export data in the specified format
        
        Args:
            data: Data to export (DataFrame, dict, list, etc.)
            export_format: Format to export to
            filename: Base filename (without extension)
            metadata: Additional metadata to include
            
        Returns:
            ExportResult with success status and file information
        """
        try:
            if export_format == ExportFormat.CSV:
                return self._export_csv(data, filename, metadata)
            elif export_format == ExportFormat.EXCEL:
                return self._export_excel(data, filename, metadata)
            elif export_format == ExportFormat.JSON:
                return self._export_json(data, filename, metadata)
            else:
                return ExportResult(
                    success=False,
                    export_format=export_format,
                    error_message=f"Unsupported export format: {export_format}"
                )
                
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return ExportResult(
                success=False,
                export_format=export_format,
                error_message=str(e)
            )
    
    def _export_csv(self, data: Any, filename: str, metadata: Optional[Dict[str, Any]]) -> ExportResult:
        """Export data to CSV format"""
        file_path = self.export_directory / f"{filename}.csv"
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        elif isinstance(data, list) and len(data) > 0:
            # Convert list of dicts to DataFrame
            if isinstance(data[0], dict):
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
            else:
                # Simple list
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for item in data:
                        writer.writerow([item])
        elif isinstance(data, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame([data])
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Cannot export data type {type(data)} to CSV")
        
        return ExportResult(
            success=True,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            export_format=ExportFormat.CSV,
            metadata=metadata
        )
    
    def _export_excel(self, data: Any, filename: str, metadata: Optional[Dict[str, Any]]) -> ExportResult:
        """Export data to Excel format"""
        file_path = self.export_directory / f"{filename}.xlsx"
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name='Data', index=False)
            elif isinstance(data, dict):
                # Multiple sheets from dict
                for sheet_name, sheet_data in data.items():
                    if isinstance(sheet_data, pd.DataFrame):
                        sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
                    elif isinstance(sheet_data, list) and len(sheet_data) > 0:
                        if isinstance(sheet_data[0], dict):
                            df = pd.DataFrame(sheet_data)
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
            elif isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                    df.to_excel(writer, sheet_name='Data', index=False)
            
            # Add metadata sheet if provided
            if metadata:
                metadata_df = pd.DataFrame([metadata])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        return ExportResult(
            success=True,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            export_format=ExportFormat.EXCEL,
            metadata=metadata
        )
    
    def _export_json(self, data: Any, filename: str, metadata: Optional[Dict[str, Any]]) -> ExportResult:
        """Export data to JSON format"""
        file_path = self.export_directory / f"{filename}.json"
        
        export_data = {
            "data": self._serialize_for_json(data),
            "exported_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return ExportResult(
            success=True,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            export_format=ExportFormat.JSON,
            metadata=metadata
        )
    
    def _serialize_for_json(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif hasattr(data, 'to_dict'):
            return data.to_dict()
        elif hasattr(data, '__dict__'):
            return data.__dict__
        else:
            return data
    
    def export_flight_data(
        self,
        flights_df: pd.DataFrame,
        export_format: ExportFormat,
        filename_prefix: str = "flight_data"
    ) -> ExportResult:
        """Export flight data with proper formatting"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}"
        
        metadata = {
            "total_flights": len(flights_df),
            "date_range": {
                "start": flights_df['scheduled_departure'].min().isoformat() if 'scheduled_departure' in flights_df.columns else None,
                "end": flights_df['scheduled_departure'].max().isoformat() if 'scheduled_departure' in flights_df.columns else None
            },
            "airports": flights_df['origin_airport'].unique().tolist() if 'origin_airport' in flights_df.columns else [],
            "export_timestamp": datetime.now().isoformat()
        }
        
        return self.export_data(flights_df, export_format, filename, metadata)
    
    def export_analysis_results(
        self,
        analysis_results: Dict[str, Any],
        export_format: ExportFormat,
        analysis_type: str,
        filename_prefix: str = "analysis_results"
    ) -> ExportResult:
        """Export analysis results with proper structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{analysis_type}_{timestamp}"
        
        metadata = {
            "analysis_type": analysis_type,
            "generated_at": datetime.now().isoformat(),
            "metrics_included": list(analysis_results.keys()) if isinstance(analysis_results, dict) else []
        }
        
        return self.export_data(analysis_results, export_format, filename, metadata)
    
    def cleanup_old_exports(self, days_to_keep: int = 30):
        """Clean up old export files"""
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        
        for file_path in self.export_directory.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    logger.info(f"Deleted old export file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {str(e)}")