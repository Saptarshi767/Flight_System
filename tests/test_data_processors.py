"""
Unit tests for data processors
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from src.data.processors import ExcelDataProcessor, DataValidator
from src.data.models import DelayCategory


class TestExcelDataProcessor:
    """Test cases for ExcelDataProcessor"""
    
    @pytest.fixture
    def sample_excel_data(self):
        """Create sample Excel data for testing"""
        data = {
            'Flight': ['AI101', 'SG102', 'UK103', 'AI104'],
            'Airline': ['AI', 'SG', 'UK', 'AI'],
            'From': ['BOM', 'DEL', 'BOM', 'DEL'],
            'To': ['DEL', 'BOM', 'DEL', 'BOM'],
            'Departure': [
                datetime(2023, 12, 1, 12, 0),
                datetime(2023, 12, 1, 13, 0),
                datetime(2023, 12, 1, 14, 0),
                datetime(2023, 12, 1, 15, 0)
            ],
            'Arrival': [
                datetime(2023, 12, 1, 14, 30),
                datetime(2023, 12, 1, 15, 30),
                datetime(2023, 12, 1, 16, 30),
                datetime(2023, 12, 1, 17, 30)
            ],
            'Aircraft': ['A320', 'B737', 'A320', 'B737'],
            'Passengers': [180, 150, 175, 160]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_excel_file(self, sample_excel_data):
        """Create temporary Excel file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            sample_excel_data.to_excel(tmp_file.name, index=False)
            yield tmp_file.name
        # Cleanup
        os.unlink(tmp_file.name)
    
    def test_init(self):
        """Test ExcelDataProcessor initialization"""
        processor = ExcelDataProcessor("test_file.xlsx")
        assert processor.file_path == Path("test_file.xlsx")
        assert processor.logger is not None
    
    def test_load_flight_data_success(self, sample_excel_file):
        """Test successful loading of flight data"""
        processor = ExcelDataProcessor(sample_excel_file)
        df = processor.load_flight_data()
        
        assert not df.empty
        assert len(df) == 4
        assert 'flight_number' in df.columns
        assert 'origin_airport' in df.columns
        assert 'destination_airport' in df.columns
        assert 'scheduled_departure' in df.columns
        assert 'scheduled_arrival' in df.columns
    
    def test_load_flight_data_file_not_found(self):
        """Test loading when file doesn't exist"""
        processor = ExcelDataProcessor("nonexistent_file.xlsx")
        df = processor.load_flight_data()
        
        assert df.empty
    
    def test_convert_to_csv_success(self, sample_excel_file):
        """Test successful conversion to CSV"""
        processor = ExcelDataProcessor(sample_excel_file)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_csv:
            csv_path = tmp_csv.name
        
        try:
            success = processor.convert_to_csv(csv_path)
            
            assert success
            assert os.path.exists(csv_path)
            
            # Verify CSV content
            csv_df = pd.read_csv(csv_path)
            assert not csv_df.empty
            assert len(csv_df) == 4
            
        finally:
            # Cleanup with error handling
            try:
                if os.path.exists(csv_path):
                    os.unlink(csv_path)
            except PermissionError:
                pass  # Ignore permission errors on Windows
    
    def test_convert_to_csv_no_data(self):
        """Test CSV conversion when no data available"""
        processor = ExcelDataProcessor("nonexistent_file.xlsx")
        success = processor.convert_to_csv("output.csv")
        
        assert not success
    
    def test_validate_data_schema_valid(self, sample_excel_data):
        """Test schema validation with valid data"""
        processor = ExcelDataProcessor()
        
        # Clean the data first to get proper column names
        cleaned_df = processor._clean_flight_data(sample_excel_data)
        is_valid, errors = processor.validate_data_schema(cleaned_df)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_data_schema_missing_columns(self):
        """Test schema validation with missing columns"""
        processor = ExcelDataProcessor()
        
        # Create DataFrame with missing required columns
        df = pd.DataFrame({'some_column': [1, 2, 3]})
        is_valid, errors = processor.validate_data_schema(df)
        
        assert not is_valid
        assert len(errors) > 0
        assert "Missing required columns" in errors[0]
    
    def test_clean_flight_data_column_mapping(self, sample_excel_data):
        """Test column name standardization"""
        processor = ExcelDataProcessor()
        cleaned_df = processor._clean_flight_data(sample_excel_data)
        
        # Check that columns were renamed correctly
        assert 'flight_number' in cleaned_df.columns
        assert 'origin_airport' in cleaned_df.columns
        assert 'destination_airport' in cleaned_df.columns
        assert 'scheduled_departure' in cleaned_df.columns
        assert 'scheduled_arrival' in cleaned_df.columns
        assert 'airline' in cleaned_df.columns
        assert 'aircraft_type' in cleaned_df.columns
        assert 'passenger_count' in cleaned_df.columns
    
    def test_clean_flight_data_airport_codes(self, sample_excel_data):
        """Test airport code cleaning"""
        processor = ExcelDataProcessor()
        cleaned_df = processor._clean_flight_data(sample_excel_data)
        
        # Check that airport codes are uppercase and 3 characters
        for airport in cleaned_df['origin_airport']:
            assert len(airport) == 3
            assert airport.isupper()
        
        for airport in cleaned_df['destination_airport']:
            assert len(airport) == 3
            assert airport.isupper()
    
    def test_clean_flight_data_datetime_conversion(self, sample_excel_data):
        """Test datetime column conversion"""
        processor = ExcelDataProcessor()
        cleaned_df = processor._clean_flight_data(sample_excel_data)
        
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df['scheduled_departure'])
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df['scheduled_arrival'])
    
    def test_add_derived_fields(self, sample_excel_data):
        """Test addition of derived fields"""
        processor = ExcelDataProcessor()
        cleaned_df = processor._clean_flight_data(sample_excel_data)
        
        # Check derived fields
        assert 'departure_hour' in cleaned_df.columns
        assert 'departure_day_of_week' in cleaned_df.columns
        assert 'departure_month' in cleaned_df.columns
        assert 'route' in cleaned_df.columns
        assert 'scheduled_duration_minutes' in cleaned_df.columns
        
        # Check route format
        assert cleaned_df['route'].iloc[0] == 'BOM-DEL'
        assert cleaned_df['route'].iloc[1] == 'DEL-BOM'
    
    def test_generate_flight_ids(self, sample_excel_data):
        """Test flight ID generation"""
        processor = ExcelDataProcessor()
        cleaned_df = processor._clean_flight_data(sample_excel_data)
        
        assert 'flight_id' in cleaned_df.columns
        
        # Check flight ID format
        flight_id = cleaned_df['flight_id'].iloc[0]
        assert 'AI101_' in flight_id
        assert len(flight_id.split('_')) == 3  # FLIGHT_YYYYMMDD_HHMM
    
    def test_categorize_delays(self):
        """Test delay categorization"""
        processor = ExcelDataProcessor()
        
        # Create test data with different delay amounts
        data = pd.DataFrame({
            'delay_minutes': [0, 10, 30, 90, 150]
        })
        
        result_df = processor._categorize_delays(data)
        
        assert result_df['delay_category'].iloc[0] is None  # No delay
        assert result_df['delay_category'].iloc[1] == DelayCategory.OTHER.value  # 10 min
        assert result_df['delay_category'].iloc[2] == DelayCategory.OPERATIONAL.value  # 30 min
        assert result_df['delay_category'].iloc[3] == DelayCategory.TRAFFIC.value  # 90 min
        assert result_df['delay_category'].iloc[4] == DelayCategory.WEATHER.value  # 150 min
    
    def test_get_data_summary(self, sample_excel_data):
        """Test data summary generation"""
        processor = ExcelDataProcessor()
        cleaned_df = processor._clean_flight_data(sample_excel_data)
        
        summary = processor.get_data_summary(cleaned_df)
        
        assert summary['total_flights'] == 4
        assert summary['unique_airlines'] == 3  # AI, SG, UK
        assert summary['unique_routes'] == 2  # BOM-DEL, DEL-BOM
        assert 'date_range' in summary
        assert 'delay_stats' in summary
        assert 'airports' in summary
    
    def test_get_data_summary_empty_dataframe(self):
        """Test data summary with empty DataFrame"""
        processor = ExcelDataProcessor()
        summary = processor.get_data_summary(pd.DataFrame())
        
        assert summary == {}


class TestDataValidator:
    """Test cases for DataValidator"""
    
    @pytest.fixture
    def valid_flight_data(self):
        """Create valid flight data for testing"""
        return pd.DataFrame({
            'flight_id': ['AI101_20231201_1200', 'SG102_20231201_1300'],
            'flight_number': ['AI101', 'SG102'],
            'airline': ['AI', 'SG'],
            'origin_airport': ['BOM', 'DEL'],
            'destination_airport': ['DEL', 'BOM'],
            'scheduled_departure': [
                datetime(2023, 12, 1, 12, 0),
                datetime(2023, 12, 1, 13, 0)
            ],
            'scheduled_arrival': [
                datetime(2023, 12, 1, 14, 30),
                datetime(2023, 12, 1, 15, 30)
            ],
            'delay_minutes': [15, 5],
            'scheduled_duration_minutes': [150, 150]
        })
    
    def test_validate_flight_data_valid(self, valid_flight_data):
        """Test validation with valid data"""
        validator = DataValidator()
        results = validator.validate_flight_data(valid_flight_data)
        
        assert results['is_valid']
        assert len(results['errors']) == 0
        assert 'statistics' in results
    
    def test_validate_flight_data_empty(self):
        """Test validation with empty DataFrame"""
        validator = DataValidator()
        results = validator.validate_flight_data(pd.DataFrame())
        
        assert not results['is_valid']
        assert "DataFrame is empty" in results['errors']
    
    def test_validate_flight_data_missing_columns(self):
        """Test validation with missing required columns"""
        validator = DataValidator()
        
        # Create DataFrame missing required columns
        df = pd.DataFrame({'some_column': [1, 2, 3]})
        results = validator.validate_flight_data(df)
        
        assert not results['is_valid']
        assert any("Missing required columns" in error for error in results['errors'])
    
    def test_validate_flight_data_quality_warnings(self):
        """Test data quality warnings"""
        validator = DataValidator()
        
        # Create data with quality issues
        df = pd.DataFrame({
            'flight_id': ['AI101_20231201_1200', 'SG102_20231201_1300'],
            'flight_number': ['AI101', 'SG102'],
            'origin_airport': ['BOM', 'DEL'],
            'destination_airport': ['DEL', 'BOM'],
            'airline': [None, 'SG'],  # Missing airline for first row
            'delay_minutes': [15, 1500]  # Extreme delay
        })
        
        results = validator.validate_flight_data(df)
        
        # Should have warnings but still be valid
        assert len(results['warnings']) > 0
    
    def test_validate_business_logic_extreme_delays(self):
        """Test business logic validation for extreme delays"""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'flight_id': ['AI101_20231201_1200'],
            'flight_number': ['AI101'],
            'origin_airport': ['BOM'],
            'destination_airport': ['DEL'],
            'delay_minutes': [1500]  # 25 hours delay
        })
        
        warnings = validator._validate_business_logic(df)
        
        assert len(warnings) > 0
        assert any("delays > 24 hours" in warning for warning in warnings)
    
    def test_validate_business_logic_flight_duration(self):
        """Test business logic validation for flight duration"""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'flight_id': ['AI101_20231201_1200', 'SG102_20231201_1300'],
            'flight_number': ['AI101', 'SG102'],
            'origin_airport': ['BOM', 'DEL'],
            'destination_airport': ['DEL', 'BOM'],
            'scheduled_duration_minutes': [15, 1500]  # Too short and too long
        })
        
        warnings = validator._validate_business_logic(df)
        
        assert len(warnings) >= 2
        assert any("duration < 30 minutes" in warning for warning in warnings)
        assert any("duration > 20 hours" in warning for warning in warnings)
    
    def test_generate_validation_stats(self, valid_flight_data):
        """Test validation statistics generation"""
        validator = DataValidator()
        stats = validator._generate_validation_stats(valid_flight_data)
        
        assert stats['total_rows'] == 2
        assert stats['total_columns'] == len(valid_flight_data.columns)
        assert 'memory_usage_mb' in stats
        
        # Check missing percentage stats for each column
        for col in valid_flight_data.columns:
            assert f'{col}_missing_pct' in stats


class TestIntegration:
    """Integration tests for data processing pipeline"""
    
    def test_full_processing_pipeline(self):
        """Test complete data processing pipeline"""
        # Create sample data
        data = {
            'Flight': ['AI101', 'SG102'],
            'From': ['bom', 'del'],  # lowercase to test cleaning
            'To': ['del', 'bom'],
            'Departure': [
                datetime(2023, 12, 1, 12, 0),
                datetime(2023, 12, 1, 13, 0)
            ],
            'Arrival': [
                datetime(2023, 12, 1, 14, 30),
                datetime(2023, 12, 1, 15, 30)
            ],
            'Airline': ['AI', 'SG']
        }
        df = pd.DataFrame(data)
        
        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            excel_path = tmp_file.name
            df.to_excel(excel_path, index=False)
            
        try:
            # Process the data
            processor = ExcelDataProcessor(excel_path)
            processed_df = processor.load_flight_data()
            
            # Validate the processed data
            validator = DataValidator()
            validation_results = validator.validate_flight_data(processed_df)
            
            # Assertions
            assert not processed_df.empty
            assert validation_results['is_valid']
            assert 'flight_id' in processed_df.columns
            assert processed_df['origin_airport'].iloc[0] == 'BOM'  # Should be uppercase
            assert processed_df['destination_airport'].iloc[0] == 'DEL'
            
            # Test summary
            summary = processor.get_data_summary(processed_df)
            assert summary['total_flights'] == 2
            
        finally:
            # Cleanup with error handling
            try:
                if os.path.exists(excel_path):
                    os.unlink(excel_path)
            except PermissionError:
                pass  # Ignore permission errors on Windows