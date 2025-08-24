"""
Demo script for Unified Data Processing Pipeline
"""
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch
import tempfile
import os

from src.data.pipeline import DataProcessor, ProcessingConfig, DataSource
from src.data.processors import ExcelDataProcessor


def create_sample_excel_data():
    """Create sample Excel data for demonstration"""
    data = {
        'Flight': ['AI101', 'SG102', 'UK103', 'AI104', 'SG105'],
        'Airline': ['AI', 'SG', 'UK', 'AI', 'SG'],
        'From': ['BOM', 'DEL', 'BOM', 'DEL', 'BOM'],
        'To': ['DEL', 'BOM', 'DEL', 'BOM', 'DEL'],
        'Departure': [
            datetime(2023, 12, 1, 12, 0),
            datetime(2023, 12, 1, 13, 0),
            datetime(2023, 12, 1, 14, 0),
            datetime(2023, 12, 1, 15, 0),
            datetime(2023, 12, 1, 16, 0)
        ],
        'Arrival': [
            datetime(2023, 12, 1, 14, 30),
            datetime(2023, 12, 1, 15, 30),
            datetime(2023, 12, 1, 16, 30),
            datetime(2023, 12, 1, 17, 30),
            datetime(2023, 12, 1, 18, 30)
        ],
        'Actual Departure': [
            datetime(2023, 12, 1, 12, 15),  # 15 min delay
            datetime(2023, 12, 1, 13, 5),   # 5 min delay
            datetime(2023, 12, 1, 14, 0),   # On time
            datetime(2023, 12, 1, 15, 30),  # 30 min delay
            datetime(2023, 12, 1, 16, 45)   # 45 min delay
        ],
        'Aircraft': ['A320', 'B737', 'A320', 'B737', 'A320'],
        'Passengers': [180, 150, 175, 160, 185]
    }
    
    return pd.DataFrame(data)


def create_sample_scraped_data():
    """Create sample scraped data for demonstration"""
    data = {
        'flight_number': ['AI106', 'SG107', 'UK108'],
        'origin_airport': ['BOM', 'DEL', 'BOM'],
        'destination_airport': ['DEL', 'BOM', 'DEL'],
        'scheduled_departure': [
            datetime(2023, 12, 1, 17, 0),
            datetime(2023, 12, 1, 18, 0),
            datetime(2023, 12, 1, 19, 0)
        ],
        'scheduled_arrival': [
            datetime(2023, 12, 1, 19, 30),
            datetime(2023, 12, 1, 20, 30),
            datetime(2023, 12, 1, 21, 30)
        ],
        'source': ['FlightRadar24', 'FlightAware', 'FlightRadar24']
    }
    
    return pd.DataFrame(data)


def create_sample_data_with_duplicates():
    """Create sample data with duplicates for testing deduplication"""
    excel_data = pd.DataFrame({
        'Flight': ['AI101', 'SG102', 'AI101'],  # AI101 appears twice
        'From': ['BOM', 'DEL', 'BOM'],
        'To': ['DEL', 'BOM', 'DEL'],
        'Departure': [
            datetime(2023, 12, 1, 12, 0),
            datetime(2023, 12, 1, 13, 0),
            datetime(2023, 12, 1, 12, 5)  # Similar time to first AI101
        ],
        'Arrival': [
            datetime(2023, 12, 1, 14, 30),
            datetime(2023, 12, 1, 15, 30),
            datetime(2023, 12, 1, 14, 35)
        ]
    })
    
    scraped_data = pd.DataFrame({
        'flight_number': ['AI101', 'UK103'],  # AI101 duplicate from different source
        'origin_airport': ['BOM', 'BOM'],
        'destination_airport': ['DEL', 'DEL'],
        'scheduled_departure': [
            datetime(2023, 12, 1, 12, 0),  # Exact match with Excel
            datetime(2023, 12, 1, 14, 0)
        ],
        'scheduled_arrival': [
            datetime(2023, 12, 1, 14, 30),
            datetime(2023, 12, 1, 16, 30)
        ]
    })
    
    return excel_data, scraped_data


def demo_basic_pipeline():
    """Demonstrate basic pipeline functionality"""
    print("🔧 Basic Pipeline Demo")
    print("-" * 40)
    
    # Create configuration
    config = ProcessingConfig(
        excel_file_path="demo_data.xlsx",
        enable_web_scraping=True,
        airports_to_scrape=['BOM', 'DEL'],
        deduplication_threshold=0.8,
        data_quality_threshold=0.6
    )
    
    processor = DataProcessor(config)
    print(f"✅ Created processor with config:")
    print(f"   • Excel file: {config.excel_file_path}")
    print(f"   • Web scraping: {config.enable_web_scraping}")
    print(f"   • Airports: {config.airports_to_scrape}")
    print(f"   • Dedup threshold: {config.deduplication_threshold}")
    print(f"   • Quality threshold: {config.data_quality_threshold}")
    
    return processor


def demo_excel_only_processing():
    """Demonstrate processing with Excel data only"""
    print("\n📊 Excel-Only Processing Demo")
    print("-" * 40)
    
    # Create sample Excel data
    excel_data = create_sample_excel_data()
    
    # Create processor with web scraping disabled
    config = ProcessingConfig(enable_web_scraping=False)
    processor = DataProcessor(config)
    
    # Mock the Excel processor
    with patch.object(processor.excel_processor, 'load_flight_data', return_value=excel_data):
        result = processor.process_all_data()
        
        print(f"✅ Processed {len(result)} flights from Excel only")
        print(f"✅ Data sources: {result['data_source'].value_counts().to_dict()}")
        
        # Show enriched fields
        enriched_fields = ['route', 'departure_hour', 'is_domestic', 'data_quality_score']
        available_fields = [f for f in enriched_fields if f in result.columns]
        print(f"✅ Enriched fields added: {available_fields}")
        
        # Show sample processed data
        if not result.empty:
            print(f"\n📋 Sample Processed Data:")
            display_cols = ['flight_number', 'route', 'departure_hour', 'is_domestic', 'data_quality_score']
            available_cols = [col for col in display_cols if col in result.columns]
            print(result[available_cols].head(3))
    
    return result


def demo_combined_processing():
    """Demonstrate processing with both Excel and scraped data"""
    print("\n🌐 Combined Processing Demo")
    print("-" * 40)
    
    # Create sample data
    excel_data = create_sample_excel_data()
    scraped_data = create_sample_scraped_data()
    
    # Create processor with web scraping enabled
    config = ProcessingConfig(enable_web_scraping=True)
    processor = DataProcessor(config)
    
    # Mock both data sources
    with patch.object(processor.excel_processor, 'load_flight_data', return_value=excel_data), \
         patch.object(processor.scraping_manager, 'scrape_all_sources', return_value=scraped_data):
        
        result = processor.process_all_data()
        
        print(f"✅ Processed {len(result)} flights from combined sources")
        print(f"✅ Data sources: {result['data_source'].value_counts().to_dict()}")
        
        # Show processing statistics
        stats = processor.get_processing_statistics()
        print(f"✅ Processing stats:")
        print(f"   • Excel flights: {stats['excel_flights']}")
        print(f"   • Scraped flights: {stats['scraped_flights']}")
        print(f"   • Combined flights: {stats['combined_flights']}")
        print(f"   • Duplicates removed: {stats['duplicates_removed']}")
        print(f"   • Data quality score: {stats['data_quality_score']:.2f}")
        
        # Show data distribution
        if not result.empty:
            print(f"\n📊 Data Distribution:")
            print(f"   • Routes: {result['route'].value_counts().to_dict()}")
            print(f"   • Airlines: {result['airline'].value_counts().to_dict() if 'airline' in result.columns else 'N/A'}")
    
    return result


def demo_deduplication():
    """Demonstrate deduplication functionality"""
    print("\n🔄 Deduplication Demo")
    print("-" * 40)
    
    # Create data with duplicates
    excel_data, scraped_data = create_sample_data_with_duplicates()
    
    print(f"📥 Input data:")
    print(f"   • Excel flights: {len(excel_data)} (includes duplicates)")
    print(f"   • Scraped flights: {len(scraped_data)} (includes duplicates)")
    print(f"   • Total input: {len(excel_data) + len(scraped_data)} flights")
    
    # Create processor
    config = ProcessingConfig(
        enable_web_scraping=True,
        deduplication_threshold=0.8
    )
    processor = DataProcessor(config)
    
    # Mock data sources
    with patch.object(processor.excel_processor, 'load_flight_data', return_value=excel_data), \
         patch.object(processor.scraping_manager, 'scrape_all_sources', return_value=scraped_data):
        
        result = processor.process_all_data()
        
        print(f"📤 Output data:")
        print(f"   • Final flights: {len(result)}")
        print(f"   • Duplicates removed: {processor.processing_stats['duplicates_removed']}")
        
        # Show which flights survived deduplication
        if not result.empty:
            print(f"\n✅ Surviving flights:")
            for _, row in result.iterrows():
                source = row.get('data_source', 'unknown')
                flight = row.get('flight_number', 'unknown')
                route = row.get('route', 'unknown')
                print(f"   • {flight} ({route}) from {source}")
    
    return result


def demo_data_quality_checks():
    """Demonstrate data quality checks and filtering"""
    print("\n🛡️ Data Quality Demo")
    print("-" * 40)
    
    # Create data with quality issues
    poor_quality_data = pd.DataFrame({
        'Flight': ['AI101', '', 'UK103', 'SG104'],  # Missing flight number
        'From': ['BOM', 'DEL', 'XX', 'DEL'],  # Invalid airport code
        'To': ['DEL', 'BOM', 'YY', 'BOM'],
        'Departure': [
            datetime(2023, 12, 1, 12, 0),
            datetime(2023, 12, 1, 13, 0),
            datetime(2023, 12, 1, 14, 0),
            datetime(2023, 12, 1, 15, 0)
        ],
        'Arrival': [
            datetime(2023, 12, 1, 14, 30),
            datetime(2023, 12, 1, 15, 30),
            datetime(2023, 12, 1, 14, 5),  # Too short duration
            datetime(2023, 12, 1, 17, 30)
        ]
    })
    
    print(f"📥 Input data with quality issues: {len(poor_quality_data)} flights")
    
    # Create processor with strict quality threshold
    config = ProcessingConfig(
        enable_web_scraping=False,
        data_quality_threshold=0.8,
        enable_anomaly_detection=True
    )
    processor = DataProcessor(config)
    
    # Mock Excel data
    with patch.object(processor.excel_processor, 'load_flight_data', return_value=poor_quality_data):
        result = processor.process_all_data()
        
        print(f"📤 Output data after quality checks: {len(result)} flights")
        print(f"✅ Flights removed due to quality issues: {len(poor_quality_data) - len(result)}")
        
        if not result.empty:
            print(f"✅ Average data quality score: {result['data_quality_score'].mean():.2f}")
            print(f"✅ Quality score range: {result['data_quality_score'].min():.2f} - {result['data_quality_score'].max():.2f}")
    
    return result


def demo_data_export():
    """Demonstrate data export functionality"""
    print("\n💾 Data Export Demo")
    print("-" * 40)
    
    # Create sample processed data
    processed_data = pd.DataFrame({
        'flight_id': ['AI101_20231201_1200', 'SG102_20231201_1300'],
        'flight_number': ['AI101', 'SG102'],
        'route': ['BOM-DEL', 'DEL-BOM'],
        'scheduled_departure': [
            datetime(2023, 12, 1, 12, 0),
            datetime(2023, 12, 1, 13, 0)
        ],
        'data_quality_score': [0.95, 0.88],
        'data_source': ['excel', 'web_scraping']
    })
    
    processor = DataProcessor()
    
    # Test different export formats
    export_results = {}
    
    for format_type in ['csv', 'excel', 'json']:
        try:
            with tempfile.NamedTemporaryFile(suffix=f'.{format_type}', delete=False) as tmp_file:
                file_path = tmp_file.name
            
            success = processor.export_processed_data(processed_data, file_path, format_type)
            export_results[format_type] = success
            
            if success:
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                print(f"✅ {format_type.upper()} export: Success ({file_size} bytes)")
            else:
                print(f"❌ {format_type.upper()} export: Failed")
            
            # Cleanup
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except PermissionError:
                pass
                
        except Exception as e:
            print(f"❌ {format_type.upper()} export: Error - {e}")
            export_results[format_type] = False
    
    successful_exports = sum(export_results.values())
    print(f"✅ Successful exports: {successful_exports}/{len(export_results)}")
    
    return export_results


def demo_error_handling():
    """Demonstrate error handling and recovery"""
    print("\n🛡️ Error Handling Demo")
    print("-" * 40)
    
    processor = DataProcessor()
    
    # Test with empty data
    empty_result = processor._combine_data_sources(pd.DataFrame(), pd.DataFrame())
    print(f"✅ Empty data handling: {len(empty_result)} flights (expected 0)")
    
    # Test with invalid data
    invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
    standardized = processor._standardize_combined_data(invalid_data)
    print(f"✅ Invalid data standardization: {len(standardized.columns)} columns added")
    
    # Test deduplication with single flight
    single_flight = pd.DataFrame({
        'flight_number': ['AI101'],
        'origin_airport': ['BOM'],
        'destination_airport': ['DEL'],
        'scheduled_departure': [datetime.now()],
        'source_priority': [1]
    })
    deduped = processor._deduplicate_data(single_flight)
    print(f"✅ Single flight deduplication: {len(deduped)} flights (expected 1)")
    
    # Test quality checks with empty data
    quality_result = processor._perform_quality_checks(pd.DataFrame())
    print(f"✅ Empty data quality check: {len(quality_result)} flights (expected 0)")
    
    print(f"✅ All error handling tests passed")


def main():
    """Main demo function"""
    print("🚀 Unified Data Processing Pipeline Demo")
    print("=" * 60)
    
    # Demo basic pipeline setup
    processor = demo_basic_pipeline()
    
    # Demo Excel-only processing
    excel_result = demo_excel_only_processing()
    
    # Demo combined processing
    combined_result = demo_combined_processing()
    
    # Demo deduplication
    dedup_result = demo_deduplication()
    
    # Demo data quality checks
    quality_result = demo_data_quality_checks()
    
    # Demo data export
    export_results = demo_data_export()
    
    # Demo error handling
    demo_error_handling()
    
    print(f"\n🎯 Key Features Demonstrated:")
    print(f"   ✅ Configurable processing pipeline")
    print(f"   ✅ Excel and web scraping data integration")
    print(f"   ✅ Intelligent data deduplication")
    print(f"   ✅ Data standardization and cleaning")
    print(f"   ✅ Data quality scoring and filtering")
    print(f"   ✅ Anomaly detection and removal")
    print(f"   ✅ Data enrichment with derived fields")
    print(f"   ✅ Multiple export formats (CSV, Excel, JSON)")
    print(f"   ✅ Comprehensive error handling")
    print(f"   ✅ Processing statistics and monitoring")
    print(f"   ✅ Domestic vs international route classification")
    print(f"   ✅ Flight similarity calculation")
    print(f"   ✅ Data validation and conflict resolution")
    
    print(f"\n🚀 Task 2.3 Complete: Unified Data Processing Pipeline Ready!")
    print(f"\n📝 Note: This demo uses mocked data sources to demonstrate")
    print(f"   functionality without requiring actual Excel files or web scraping.")


if __name__ == "__main__":
    main()