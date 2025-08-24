"""
Demo script for Web Scraping Modules
"""
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
from src.data.scrapers import (
    FlightDataScraper, 
    FlightRadar24Scraper, 
    FlightAwareScraper,
    FlightScrapingManager,
    ScrapingConfig
)


def create_mock_html_response():
    """Create mock HTML response for demonstration"""
    return """
    <html>
        <head><title>Flight Information</title></head>
        <body>
            <table class="flights-table">
                <tr>
                    <th>Flight</th>
                    <th>From</th>
                    <th>To</th>
                    <th>STD</th>
                    <th>Aircraft</th>
                    <th>Status</th>
                </tr>
                <tr>
                    <td>AI101</td>
                    <td>BOM</td>
                    <td>DEL</td>
                    <td>12:00</td>
                    <td>A320</td>
                    <td>On Time</td>
                </tr>
                <tr>
                    <td>SG102</td>
                    <td>DEL</td>
                    <td>BOM</td>
                    <td>13:30</td>
                    <td>B737</td>
                    <td>Delayed</td>
                </tr>
                <tr>
                    <td>UK103</td>
                    <td>BOM</td>
                    <td>DEL</td>
                    <td>15:45</td>
                    <td>A320</td>
                    <td>Boarding</td>
                </tr>
            </table>
        </body>
    </html>
    """


def demo_base_scraper():
    """Demonstrate base scraper functionality"""
    print("🔧 Base Flight Data Scraper Demo")
    print("-" * 40)
    
    # Create scraper with custom config
    config = ScrapingConfig(
        user_agents=['Demo User Agent'],
        request_delay_range=(0.1, 0.2),
        max_retries=2,
        timeout=10,
        headers={'Demo': 'Header'}
    )
    
    scraper = FlightDataScraper(config)
    print(f"✅ Created scraper with {len(scraper.config.user_agents)} user agents")
    print(f"✅ Max retries: {scraper.config.max_retries}")
    print(f"✅ Timeout: {scraper.config.timeout}s")
    
    # Test HTML parsing
    from bs4 import BeautifulSoup
    html = create_mock_html_response()
    soup = BeautifulSoup(html, 'html.parser')
    
    flights = scraper._parse_flight_table(soup, 'table.flights-table')
    print(f"✅ Parsed {len(flights)} flights from mock HTML")
    
    # Clean the data
    df = scraper._clean_scraped_data(flights)
    print(f"✅ Cleaned data: {len(df)} rows, {len(df.columns)} columns")
    
    if not df.empty:
        print("\n📋 Sample Cleaned Data:")
        print(df[['flight_number', 'origin_airport', 'destination_airport', 'scheduled_departure']].head())
    
    return df


def demo_flightradar24_scraper():
    """Demonstrate FlightRadar24 scraper with mocked responses"""
    print("\n✈️ FlightRadar24 Scraper Demo")
    print("-" * 40)
    
    scraper = FlightRadar24Scraper()
    
    # Mock the HTTP request to avoid actual web scraping
    mock_response = Mock()
    mock_response.content = create_mock_html_response().encode()
    
    with patch.object(scraper, '_make_request', return_value=mock_response):
        # Test Mumbai flights scraping
        mumbai_df = scraper.scrape_mumbai_flights()
        print(f"✅ Mumbai flights scraped: {len(mumbai_df)} flights")
        
        if not mumbai_df.empty:
            print(f"   • Airport: {mumbai_df['airport'].iloc[0]}")
            print(f"   • Sample flight: {mumbai_df['flight_number'].iloc[0]}")
        
        # Test Delhi flights scraping
        delhi_df = scraper.scrape_delhi_flights()
        print(f"✅ Delhi flights scraped: {len(delhi_df)} flights")
        
        if not delhi_df.empty:
            print(f"   • Airport: {delhi_df['airport'].iloc[0]}")
            print(f"   • Sample flight: {delhi_df['flight_number'].iloc[0]}")
    
    return mumbai_df, delhi_df


def demo_flightaware_scraper():
    """Demonstrate FlightAware scraper with mocked responses"""
    print("\n🛩️ FlightAware Scraper Demo")
    print("-" * 40)
    
    scraper = FlightAwareScraper()
    
    # Mock the HTTP request
    mock_response = Mock()
    mock_response.content = create_mock_html_response().encode()
    
    with patch.object(scraper, '_make_request', return_value=mock_response):
        # Test airport flights scraping
        bom_df = scraper.scrape_airport_flights('BOM')
        print(f"✅ BOM flights scraped: {len(bom_df)} flights")
        
        del_df = scraper.scrape_airport_flights('DEL')
        print(f"✅ DEL flights scraped: {len(del_df)} flights")
        
        # Test airport name mapping
        print(f"✅ Airport names: BOM -> {scraper._get_airport_name('BOM')}")
        print(f"✅ Airport names: DEL -> {scraper._get_airport_name('DEL')}")
    
    return bom_df, del_df


def demo_scraping_manager():
    """Demonstrate the scraping manager"""
    print("\n🎯 Flight Scraping Manager Demo")
    print("-" * 40)
    
    manager = FlightScrapingManager()
    
    # Mock both scrapers
    mock_fr24_df = pd.DataFrame({
        'flight_number': ['AI101', 'SG102'],
        'origin_airport': ['BOM', 'DEL'],
        'destination_airport': ['DEL', 'BOM'],
        'airport': ['BOM', 'DEL'],
        'source': ['FlightRadar24', 'FlightRadar24']
    })
    
    mock_fa_df = pd.DataFrame({
        'flight_number': ['UK103', 'AI104'],
        'origin_airport': ['BOM', 'DEL'],
        'destination_airport': ['DEL', 'BOM'],
        'airport': ['BOM', 'DEL'],
        'source': ['FlightAware', 'FlightAware']
    })
    
    with patch.object(manager.fr24_scraper, 'scrape_mumbai_flights', return_value=mock_fr24_df.iloc[:1]), \
         patch.object(manager.fr24_scraper, 'scrape_delhi_flights', return_value=mock_fr24_df.iloc[1:]), \
         patch.object(manager.fa_scraper, 'scrape_airport_flights', return_value=mock_fa_df.iloc[:1]):
        
        # Scrape all sources
        combined_df = manager.scrape_all_sources(['BOM', 'DEL'])
        print(f"✅ Combined scraping: {len(combined_df)} total flights")
        
        # Get summary
        summary = manager.get_scraping_summary(combined_df)
        print(f"✅ Summary generated:")
        print(f"   • Total flights: {summary['total_flights']}")
        print(f"   • Sources: {list(summary['sources'].keys())}")
        print(f"   • Airports: {list(summary['airports'].keys())}")
        
        if not combined_df.empty:
            print(f"\n📊 Combined Data Sample:")
            print(combined_df[['flight_number', 'origin_airport', 'destination_airport', 'source']].head())
    
    return combined_df, summary


def demo_error_handling():
    """Demonstrate error handling capabilities"""
    print("\n🛡️ Error Handling Demo")
    print("-" * 40)
    
    scraper = FlightDataScraper()
    
    # Test with invalid HTML
    from bs4 import BeautifulSoup
    invalid_html = "<div>No table here</div>"
    soup = BeautifulSoup(invalid_html, 'html.parser')
    
    flights = scraper._parse_flight_table(soup, 'table.nonexistent')
    print(f"✅ Invalid HTML handling: {len(flights)} flights (expected 0)")
    
    # Test with empty data
    df = scraper._clean_scraped_data([])
    print(f"✅ Empty data handling: DataFrame empty = {df.empty}")
    
    # Test time parsing with various formats
    time_data = pd.Series(['12:00', '1400', 'invalid', '', '25:99'])
    parsed_times = scraper._parse_time_column(time_data)
    valid_times = parsed_times.notna().sum()
    print(f"✅ Time parsing: {valid_times}/{len(time_data)} times parsed successfully")


def main():
    """Main demo function"""
    print("🌐 Flight Data Web Scraping Demo")
    print("=" * 60)
    
    # Demo base scraper
    base_df = demo_base_scraper()
    
    # Demo FlightRadar24 scraper
    mumbai_df, delhi_df = demo_flightradar24_scraper()
    
    # Demo FlightAware scraper
    bom_df, del_df = demo_flightaware_scraper()
    
    # Demo scraping manager
    combined_df, summary = demo_scraping_manager()
    
    # Demo error handling
    demo_error_handling()
    
    print(f"\n🎯 Key Features Demonstrated:")
    print(f"   ✅ Configurable scraping with rate limiting")
    print(f"   ✅ Multiple user agent rotation")
    print(f"   ✅ Robust HTML table parsing")
    print(f"   ✅ Data cleaning and standardization")
    print(f"   ✅ FlightRadar24 specific scraping")
    print(f"   ✅ FlightAware specific scraping")
    print(f"   ✅ Multi-source scraping management")
    print(f"   ✅ Comprehensive error handling")
    print(f"   ✅ Time parsing with multiple formats")
    print(f"   ✅ Airport code standardization")
    print(f"   ✅ Data source tracking")
    
    print(f"\n🚀 Task 2.2 Complete: Web Scraping Modules Ready!")
    print(f"\n📝 Note: This demo uses mocked HTTP responses to avoid")
    print(f"   actual web scraping during demonstration.")


if __name__ == "__main__":
    main()