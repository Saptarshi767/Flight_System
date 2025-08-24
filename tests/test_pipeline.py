"""
Unit tests for unified data processing pipeline
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.data.pipeline import (
    DataProcessor, 
    ProcessingConfig, 
    DataSource
)


class TestProcessingConfig:
    """Test cases for ProcessingConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = ProcessingConfig()
        
        assert config.excel_file_path == "Flight_Data.xlsx"
        assert config.enable_web_scraping is True
        assert config.airports_to_scrape == ['BOM', 'DEL']
        assert config.deduplication_threshold == 0.8
        assert config.data_quality_threshold == 0.7
        assert config.max_processing_time_minutes == 30
        assert config.enable_anomaly_detection is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ProcessingConfig(
            excel_file_path="custom.xlsx",
            enable_web_scraping=False,
            airports_to_scrape=['CCU', 'MAA'],
            deduplication_threshold=0.9
        )
        
        assert config.excel_file_path == "custom.xlsx"
        assert config.enable_web_scraping is False
        assert config.airports_to_scrape == ['CCU', 'MAA']
        assert config.deduplication_threshold == 0.9


class TestDataProcessor:
    """Test cases for DataProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Create processor for testing"""
        config = ProcessingConfig(enable_web_scraping=False)  # Disable scraping for tests
        return DataProcessor(config)
    
    @pytest.fixture
    def sample_excel_data(self):
        """Sample Excel data for testing"""
        return pd.DataFrame({
            'flight_number': ['AI101', 'SG102', 'UK103'],
            'airline': ['AI', 'SG', 'UK'],
            'origin_airport': ['BOM', 'DEL', 'BOM'],
            'destination_airport': ['DEL', 'BOM', 'DEL'],
            'scheduled_departure': [
                datetime(2023, 12, 1, 12, 0),
                datetime(2023, 12, 1, 13, 0),
                datetime(2023, 12, 1, 14, 0)
            ],
            'scheduled_arrival': [
                datetime(2023, 12, 1, 14, 30),
                datetime(2023, 12, 1, 15, 30),
                datetime(2023, 12, 1, 16, 30)
            ],
            'data_source': ['excel', 'excel', 'excel'],
            'source_priority': [1, 1, 1]
        })
    
    @pytest.fixture
    def sample_scraped_data(self):
        """Sample scraped data for testing"""
        return pd.DataFrame({
            'flight_number': ['AI104', 'SG105'],
            'origin_airport': ['DEL', 'BOM'],
            'destination_airport': ['BOM', 'DEL'],
            'scheduled_departure': [
                datetime(2023, 12, 1, 15, 0),
                datetime(2023, 12, 1, 16, 0)
            ],
            'scheduled_arrival': [
                datetime(2023, 12, 1, 17, 30),
                datetime(2023, 12, 1, 18, 30)
            ],
            'data_source': ['web_scraping', 'web_scraping'],
            'source_priority': [2, 2]
        })
    
    def test_init(self, processor):
        """Test processor initialization"""
        assert processor.config is not None
        assert processor.excel_processor is not None
        assert processor.scraping_manager is not None
        assert processor.validator is not None
        assert processor.logger is not None
        assert processor.processing_stats is not None
    
    @patch.object(DataProcessor, '_load_excel_data')
    @patch.object(DataProcessor, '_load_scraped_data')
    def test_process_all_data_success(self, mock_scraped, mock_excel, processor, sample_excel_data):
        """Test successful data processing"""
        mock_excel.return_value = sample_excel_data
        mock_scraped.return_value = pd.DataFrame()  # Empty scraped data
        
        result = processor.process_all_data()
        
        assert not result.empty
        assert len(result) == 3
        assert 'processed_at' in result.columns
        assert 'data_quality_score' in result.columns
        
        # Check processing stats
        assert processor.processing_stats['start_time'] is not None
        assert processor.processing_stats['end_time'] is not None
    
    @patch.object(DataProcessor, '_load_excel_data')
    def test_process_all_data_failure(self, mock_excel, processor):
        """Test data processing failure handling"""
        mock_excel.side_effect = Exception("Test error")
        
        with pytest.raises(Exception):
            processor.process_all_data()
        
        # Check that end time is set even on failure
        assert processor.processing_stats['end_time'] is not None
    
    @patch('src.data.processors.ExcelDataProcessor.load_flight_data')
    def test_load_excel_data_success(self, mock_load, processor, sample_excel_data):
        """Test successful Excel data loading"""
        mock_load.return_value = sample_excel_data
        
        result = processor._load_excel_data()
        
        assert not result.empty
        assert len(result) == 3
        assert 'data_source' in result.columns
        assert 'source_priority' in result.columns
        assert processor.processing_stats['excel_flights'] == 3
    
    @patch('src.data.processors.ExcelDataProcessor.load_flight_data')
    def test_load_excel_data_empty(self, mock_load, processor):
        """Test Excel data loading with empty result"""
        mock_load.return_value = pd.DataFrame()
        
        result = processor._load_excel_data()
        
        assert result.empty
        assert processor.processing_stats['excel_flights'] == 0
    
    @patch('src.data.scrapers.FlightScrapingManager.scrape_all_sources')
    def test_load_scraped_data_success(self, mock_scrape, sample_scraped_data):
        """Test successful scraped data loading"""
        config = ProcessingConfig(enable_web_scraping=True)
        processor = DataProcessor(config)
        
        mock_scrape.return_value = sample_scraped_data
        
        result = processor._load_scraped_data()
        
        assert not result.empty
        assert len(result) == 2
        assert 'data_source' in result.columns
        assert 'source_priority' in result.columns
        assert processor.processing_stats['scraped_flights'] == 2
    
    def test_load_scraped_data_disabled(self, processor):
        """Test scraped data loading when disabled"""
        result = processor._load_scraped_data()
        
        assert result.empty
    
    @patch('src.data.scrapers.FlightScrapingManager.scrape_all_sources')
    def test_load_scraped_data_failure(self, mock_scrape):
        """Test scraped data loading failure"""
        config = ProcessingConfig(enable_web_scraping=True)
        processor = DataProcessor(config)
        
        mock_scrape.side_effect = Exception("Scraping failed")
        
        result = processor._load_scraped_data()
        
        assert result.empty
    
    def test_combine_data_sources_both(self, processor, sample_excel_data, sample_scraped_data):
        """Test combining data from both sources"""
        result = processor._combine_data_sources(sample_excel_data, sample_scraped_data)
        
        assert not result.empty
        assert len(result) == 5  # 3 + 2
        assert 'processed_at' in result.columns
        assert processor.processing_stats['combined_flights'] == 5
    
    def test_combine_data_sources_excel_only(self, processor, sample_excel_data):
        """Test combining data with only Excel source"""
        result = processor._combine_data_sources(sample_excel_data, pd.DataFrame())
        
        assert not result.empty
        assert len(result) == 3
    
    def test_combine_data_sources_empty(self, processor):
        """Test combining empty data sources"""
        result = processor._combine_data_sources(pd.DataFrame(), pd.DataFrame())
        
        assert result.empty
    
    def test_standardize_combined_data(self, processor, sample_excel_data):
        """Test data standardization"""
        # Add some messy data
        messy_data = sample_excel_data.copy()
        messy_data.loc[0, 'origin_airport'] = 'bom'  # lowercase
        messy_data.loc[1, 'flight_number'] = ' sg102 '  # spaces
        
        result = processor._standardize_combined_data(messy_data)
        
        assert result['origin_airport'].iloc[0] == 'BOM'  # Should be uppercase
        assert result['flight_number'].iloc[1] == 'SG102'  # Should be cleaned
        assert 'processed_at' in result.columns
    
    def test_deduplicate_data_exact_duplicates(self, processor):
        """Test deduplication with exact duplicates"""
        # Create data with exact duplicates
        data = pd.DataFrame({
            'flight_number': ['AI101', 'AI101', 'SG102'],
            'origin_airport': ['BOM', 'BOM', 'DEL'],
            'destination_airport': ['DEL', 'DEL', 'BOM'],
            'scheduled_departure': [
                datetime(2023, 12, 1, 12, 0),
                datetime(2023, 12, 1, 12, 0),  # Exact duplicate
                datetime(2023, 12, 1, 13, 0)
            ],
            'source_priority': [1, 2, 1]  # First has higher priority
        })
        
        result = processor._deduplicate_data(data)
        
        assert len(result) == 2  # One duplicate removed
        assert processor.processing_stats['duplicates_removed'] == 1
    
    def test_deduplicate_data_no_duplicates(self, processor, sample_excel_data):
        """Test deduplication with no duplicates"""
        result = processor._deduplicate_data(sample_excel_data)
        
        assert len(result) == len(sample_excel_data)
        assert processor.processing_stats['duplicates_removed'] == 0
    
    def test_calculate_flight_similarity(self, processor):
        """Test flight similarity calculation"""
        flight1 = pd.Series({
            'flight_number': 'AI101',
            'scheduled_departure': datetime(2023, 12, 1, 12, 0),
            'origin_airport': 'BOM',
            'destination_airport': 'DEL'
        })
        
        flight2 = pd.Series({
            'flight_number': 'AI101',
            'scheduled_departure': datetime(2023, 12, 1, 12, 5),  # 5 minutes later
            'origin_airport': 'BOM',
            'destination_airport': 'DEL'
        })
        
        similarity = processor._calculate_flight_similarity(flight1, flight2)
        
        assert similarity > 0.8  # Should be highly similar
    
    @patch('src.data.processors.DataValidator.validate_flight_data')
    def test_validate_and_clean_data(self, mock_validate, processor, sample_excel_data):
        """Test data validation and cleaning"""
        mock_validate.return_value = {
            'is_valid': True,
            'errors': [],
            'warnings': ['Some warning']
        }
        
        result = processor._validate_and_clean_data(sample_excel_data)
        
        assert not result.empty
        assert processor.processing_stats['validation_errors'] == 0
        mock_validate.assert_called_once()
    
    def test_apply_data_cleaning_rules(self, processor):
        """Test data cleaning rules"""
        # Create data with issues
        dirty_data = pd.DataFrame({
            'flight_number': ['AI101', None, 'SG102'],  # Missing flight number
            'origin_airport': ['BOM', 'DEL', 'XX'],  # Invalid airport code
            'destination_airport': ['DEL', 'BOM', 'YY'],
            'scheduled_departure': [
                datetime(2023, 12, 1, 12, 0),
                datetime(2023, 12, 1, 13, 0),
                datetime(2023, 12, 1, 14, 0)
            ],
            'scheduled_arrival': [
                datetime(2023, 12, 1, 14, 30),
                datetime(2023, 12, 1, 15, 30),
                datetime(2023, 12, 1, 14, 5)  # Too short duration
            ]
        })
        
        result = processor._apply_data_cleaning_rules(dirty_data)
        
        assert len(result) < len(dirty_data)  # Some rows should be removed
    
    def test_enrich_data(self, processor, sample_excel_data):
        """Test data enrichment"""
        result = processor._enrich_data(sample_excel_data)
        
        # Check for derived fields
        assert 'departure_hour' in result.columns
        assert 'departure_day_of_week' in result.columns
        assert 'route' in result.columns
        assert 'is_domestic' in result.columns
        assert 'scheduled_duration_minutes' in result.columns
        assert 'flight_id' in result.columns
        assert 'data_quality_score' in result.columns
    
    def test_classify_domestic_routes(self, processor):
        """Test domestic route classification"""
        data = pd.DataFrame({
            'origin_airport': ['BOM', 'DEL', 'BOM'],
            'destination_airport': ['DEL', 'LHR', 'CCU']  # DEL-LHR is international
        })
        
        is_domestic = processor._classify_domestic_routes(data)
        
        assert is_domestic.iloc[0] == True  # BOM-DEL
        assert is_domestic.iloc[1] == False  # DEL-LHR
        assert is_domestic.iloc[2] == True  # BOM-CCU
    
    def test_generate_flight_ids(self, processor, sample_excel_data):
        """Test flight ID generation"""
        flight_ids = processor._generate_flight_ids(sample_excel_data)
        
        assert len(flight_ids) == len(sample_excel_data)
        assert all('AI101_20231201_1200' in fid or 'SG102_20231201_1300' in fid or 'UK103_20231201_1400' in fid for fid in flight_ids)
    
    def test_calculate_data_quality_scores(self, processor, sample_excel_data):
        """Test data quality score calculation"""
        scores = processor._calculate_data_quality_scores(sample_excel_data)
        
        assert len(scores) == len(sample_excel_data)
        assert all(0 <= score <= 1 for score in scores)
        assert all(score > 0.8 for score in scores)  # Should be high quality
    
    def test_perform_quality_checks(self, processor, sample_excel_data):
        """Test quality checks"""
        # Add quality scores
        sample_excel_data['data_quality_score'] = [0.9, 0.8, 0.6]  # Last one below threshold
        
        result = processor._perform_quality_checks(sample_excel_data)
        
        assert len(result) == 2  # One row should be filtered out
        assert processor.processing_stats['data_quality_score'] > 0
    
    def test_detect_and_remove_anomalies(self, processor):
        """Test anomaly detection"""
        data = pd.DataFrame({
            'flight_number': ['AI101', 'SG102', 'UK103'],
            'scheduled_duration_minutes': [150, 160, 2000],  # Last one is anomaly
            'departure_delay_minutes': [10, 15, 500]  # Last one is anomaly
        })
        
        result = processor._detect_and_remove_anomalies(data)
        
        assert len(result) < len(data)  # Anomalies should be removed
    
    def test_export_processed_data_csv(self, processor, sample_excel_data):
        """Test CSV export"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            csv_path = tmp_file.name
        
        try:
            success = processor.export_processed_data(sample_excel_data, csv_path, 'csv')
            
            assert success
            assert os.path.exists(csv_path)
            
            # Verify content
            exported_df = pd.read_csv(csv_path)
            assert len(exported_df) == len(sample_excel_data)
            
        finally:
            # Cleanup with error handling
            try:
                if os.path.exists(csv_path):
                    os.unlink(csv_path)
            except PermissionError:
                pass  # Ignore permission errors on Windows
    
    def test_export_processed_data_excel(self, processor, sample_excel_data):
        """Test Excel export"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            excel_path = tmp_file.name
        
        try:
            success = processor.export_processed_data(sample_excel_data, excel_path, 'excel')
            
            assert success
            assert os.path.exists(excel_path)
            
        finally:
            # Cleanup with error handling
            try:
                if os.path.exists(excel_path):
                    os.unlink(excel_path)
            except PermissionError:
                pass  # Ignore permission errors on Windows
    
    def test_export_processed_data_json(self, processor, sample_excel_data):
        """Test JSON export"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            json_path = tmp_file.name
        
        try:
            success = processor.export_processed_data(sample_excel_data, json_path, 'json')
            
            assert success
            assert os.path.exists(json_path)
            
        finally:
            # Cleanup with error handling
            try:
                if os.path.exists(json_path):
                    os.unlink(json_path)
            except PermissionError:
                pass  # Ignore permission errors on Windows
    
    def test_export_processed_data_invalid_format(self, processor, sample_excel_data):
        """Test export with invalid format"""
        success = processor.export_processed_data(sample_excel_data, 'test.txt', 'invalid')
        
        assert not success
    
    def test_get_processing_statistics(self, processor):
        """Test processing statistics retrieval"""
        stats = processor.get_processing_statistics()
        
        assert isinstance(stats, dict)
        assert 'start_time' in stats
        assert 'excel_flights' in stats
        assert 'scraped_flights' in stats


class TestIntegration:
    """Integration tests for data processing pipeline"""
    
    @patch('src.data.processors.ExcelDataProcessor.load_flight_data')
    @patch('src.data.scrapers.FlightScrapingManager.scrape_all_sources')
    def test_full_pipeline_integration(self, mock_scrape, mock_excel):
        """Test complete pipeline integration"""
        # Mock data
        excel_data = pd.DataFrame({
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
            ]
        })
        
        scraped_data = pd.DataFrame({
            'flight_number': ['UK103'],
            'origin_airport': ['BOM'],
            'destination_airport': ['DEL'],
            'scheduled_departure': [datetime(2023, 12, 1, 14, 0)],
            'scheduled_arrival': [datetime(2023, 12, 1, 16, 30)]
        })
        
        mock_excel.return_value = excel_data
        mock_scrape.return_value = scraped_data
        
        # Create processor with web scraping enabled
        config = ProcessingConfig(enable_web_scraping=True)
        processor = DataProcessor(config)
        
        # Process data
        result = processor.process_all_data()
        
        # Verify results
        assert not result.empty
        assert len(result) == 3  # 2 from Excel + 1 from scraping
        assert 'data_quality_score' in result.columns
        assert 'route' in result.columns
        assert 'is_domestic' in result.columns
        
        # Verify processing stats
        stats = processor.get_processing_statistics()
        assert stats['excel_flights'] == 2
        assert stats['scraped_flights'] == 1
        assert stats['combined_flights'] == 3
    
    def test_pipeline_with_duplicates(self):
        """Test pipeline handling of duplicates"""
        config = ProcessingConfig(enable_web_scraping=False)
        processor = DataProcessor(config)
        
        # Create data with duplicates
        excel_data = pd.DataFrame({
            'flight_number': ['AI101', 'AI101', 'SG102'],  # Duplicate AI101
            'origin_airport': ['BOM', 'BOM', 'DEL'],
            'destination_airport': ['DEL', 'DEL', 'BOM'],
            'scheduled_departure': [
                datetime(2023, 12, 1, 12, 0),
                datetime(2023, 12, 1, 12, 0),  # Exact duplicate
                datetime(2023, 12, 1, 13, 0)
            ],
            'scheduled_arrival': [
                datetime(2023, 12, 1, 14, 30),
                datetime(2023, 12, 1, 14, 30),
                datetime(2023, 12, 1, 15, 30)
            ]
        })
        
        with patch.object(processor.excel_processor, 'load_flight_data', return_value=excel_data):
            result = processor.process_all_data()
            
            assert len(result) == 2  # Duplicate should be removed
            assert processor.processing_stats['duplicates_removed'] == 1
    
    def test_pipeline_error_recovery(self):
        """Test pipeline error recovery"""
        config = ProcessingConfig(enable_web_scraping=True)
        processor = DataProcessor(config)
        
        # Mock Excel success but scraping failure
        excel_data = pd.DataFrame({
            'flight_number': ['AI101'],
            'origin_airport': ['BOM'],
            'destination_airport': ['DEL'],
            'scheduled_departure': [datetime(2023, 12, 1, 12, 0)],
            'scheduled_arrival': [datetime(2023, 12, 1, 14, 30)]
        })
        
        with patch.object(processor.excel_processor, 'load_flight_data', return_value=excel_data), \
             patch.object(processor.scraping_manager, 'scrape_all_sources', side_effect=Exception("Scraping failed")):
            
            result = processor.process_all_data()
            
            # Should still process Excel data despite scraping failure
            assert not result.empty
            assert len(result) == 1
            assert processor.processing_stats['excel_flights'] == 1
            assert processor.processing_stats['scraped_flights'] == 0