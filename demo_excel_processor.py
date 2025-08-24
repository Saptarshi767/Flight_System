"""
Demo script for Excel Data Processor
"""
import pandas as pd
from datetime import datetime
from src.data.processors import ExcelDataProcessor, DataValidator


def create_sample_excel_file():
    """Create a sample Excel file for demonstration"""
    sample_data = {
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
    
    df = pd.DataFrame(sample_data)
    df.to_excel('Flight_Data_Demo.xlsx', index=False)
    print("✅ Created sample Excel file: Flight_Data_Demo.xlsx")
    return 'Flight_Data_Demo.xlsx'


def main():
    """Main demo function"""
    print("🛫 Flight Scheduling Analysis - Excel Data Processor Demo")
    print("=" * 60)
    
    # Create sample Excel file
    excel_file = create_sample_excel_file()
    
    # Initialize processor
    processor = ExcelDataProcessor(excel_file)
    print(f"\n📊 Processing Excel file: {excel_file}")
    
    # Load and process data
    df = processor.load_flight_data()
    print(f"✅ Loaded and cleaned {len(df)} flight records")
    
    # Display processed data info
    print(f"\n📋 Processed Data Columns:")
    for col in df.columns:
        print(f"   • {col}")
    
    # Show sample data
    print(f"\n🔍 Sample Processed Data:")
    print(df[['flight_id', 'flight_number', 'route', 'delay_minutes', 'delay_category']].head())
    
    # Validate data
    validator = DataValidator()
    validation_results = validator.validate_flight_data(df)
    
    print(f"\n✅ Data Validation Results:")
    print(f"   • Valid: {validation_results['is_valid']}")
    print(f"   • Errors: {len(validation_results['errors'])}")
    print(f"   • Warnings: {len(validation_results['warnings'])}")
    
    if validation_results['warnings']:
        print("   • Warning details:")
        for warning in validation_results['warnings']:
            print(f"     - {warning}")
    
    # Get data summary
    summary = processor.get_data_summary(df)
    print(f"\n📈 Data Summary:")
    print(f"   • Total flights: {summary['total_flights']}")
    print(f"   • Unique airlines: {summary['unique_airlines']}")
    print(f"   • Unique routes: {summary['unique_routes']}")
    print(f"   • Average delay: {summary['delay_stats']['average_delay']:.1f} minutes")
    print(f"   • Delayed flights: {summary['delay_stats']['delayed_flights']}")
    
    # Convert to CSV
    csv_success = processor.convert_to_csv('flight_data_demo.csv')
    if csv_success:
        print(f"\n💾 Successfully converted to CSV: flight_data_demo.csv")
    
    print(f"\n🎯 Key Features Demonstrated:")
    print(f"   ✅ Excel file loading and processing")
    print(f"   ✅ Data cleaning and standardization")
    print(f"   ✅ Column name mapping and normalization")
    print(f"   ✅ Datetime parsing and delay calculation")
    print(f"   ✅ Airport code validation and cleaning")
    print(f"   ✅ Flight ID generation")
    print(f"   ✅ Delay categorization")
    print(f"   ✅ Data validation and quality checks")
    print(f"   ✅ Excel to CSV conversion")
    print(f"   ✅ Comprehensive data summary statistics")
    
    print(f"\n🚀 Task 2.1 Complete: Excel Data Processor Ready!")


if __name__ == "__main__":
    main()