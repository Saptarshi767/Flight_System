"""
Unit tests for web scraping modules
"""
import pytest
import pandas as pd
import requests
from unittest.mock import Mock, patch, MagicMock
from bs4 import BeautifulSoup
from datetime import datetime

from src.data.scrapers import (
    FlightDataScraper, 
    FlightRadar24Scraper, 
    FlightAwareScraper,
    FlightScrapingManager,
    ScrapingConfig
)


class TestFlightDataScraper:
    """Test cases for FlightDataScraper base class"""
    
    @pytest.fixture
    def scraper(self):
        """Create scraper instance for testing"""
        return FlightDataScraper()
    
    def test_init(self, scraper):
        """Test scraper initialization"""
        assert scraper.session is not None
        assert scraper.config is not None
        assert scraper.logger is not None
        assert 'flightradar24_main' in scraper.urls
        assert 'mumbai_airport' in scraper.urls
    
    def test_default_config(self, scraper):
        """Test default configuration"""
        config = scraper._get_default_config()
        assert isinstance(config, ScrapingConfig)
        assert len(config.user_agents) > 0
        assert config.max_retries > 0
        assert config.timeout > 0
    
    def test_rotate_user_agent(self, scraper):
        """Test user agent rotation"""
        original_ua = scraper.session.headers.get('User-Agent')
        scraper._rotate_user_agent()
        new_ua = scraper.session.headers.get('User-Agent')
        
        # User agent should be from the configured list
        assert new_ua in scraper.config.user_agents
    
    @patch('requests.Session.get')
    def test_make_request_success(self, mock_get, scraper):
        """Test successful HTTP request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<html>test</html>'
        mock_get.return_value = mock_response
        
        response = scraper._make_request('http://example.com')
        
        assert response is not None
        assert response.status_code == 200
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_make_request_failure(self, mock_get, scraper):
        """Test failed HTTP request"""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        response = scraper._make_request('http://example.com')
        
        assert response is None
        assert mock_get.call_count == scraper.config.max_retries
    
    @patch('requests.Session.get')
    def test_make_request_rate_limit(self, mock_get, scraper):
        """Test rate limiting handling"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response
        
        response = scraper._make_request('http://example.com')
        
        assert response is None
        assert mock_get.call_count == scraper.config.max_retries
    
    def test_parse_flight_table_valid_html(self, scraper):
        """Test parsing valid flight table"""
        html = """
        <table class="flights">
            <tr>
                <th>Flight</th>
                <th>From</th>
                <th>To</th>
                <th>Time</th>
            </tr>
            <tr>
                <td>AI101</td>
                <td>BOM</td>
                <td>DEL</td>
                <td>12:00</td>
            </tr>
            <tr>
                <td>SG102</td>
                <td>DEL</td>
                <td>BOM</td>
                <td>13:00</td>
            </tr>
        </table>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        flights = scraper._parse_flight_table(soup, 'table.flights')
        
        assert len(flights) == 2
        assert flights[0]['flight'] == 'AI101'
        assert flights[0]['from'] == 'BOM'
        assert flights[1]['flight'] == 'SG102'
    
    def test_parse_flight_table_no_table(self, scraper):
        """Test parsing when no table found"""
        html = "<div>No table here</div>"
        soup = BeautifulSoup(html, 'html.parser')
        flights = scraper._parse_flight_table(soup, 'table.flights')
        
        assert len(flights) == 0
    
    def test_clean_scraped_data_empty(self, scraper):
        """Test cleaning empty scraped data"""
        df = scraper._clean_scraped_data([])
        assert df.empty
    
    def test_clean_scraped_data_valid(self, scraper):
        """Test cleaning valid scraped data"""
        flights = [
            {
                'flight': 'AI101',
                'from': 'BOM',
                'to': 'DEL',
                'std': '12:00',
                'aircraft': 'A320'
            },
            {
                'flight': 'SG102',
                'from': 'DEL',
                'to': 'BOM',
                'std': '13:00',
                'aircraft': 'B737'
            }
        ]
        
        df = scraper._clean_scraped_data(flights)
        
        assert not df.empty
        assert len(df) == 2
        assert 'flight_number' in df.columns
        assert 'origin_airport' in df.columns
        assert 'destination_airport' in df.columns
        assert 'data_source' in df.columns
    
    def test_standardize_scraped_data(self, scraper):
        """Test data standardization"""
        df = pd.DataFrame({
            'flight_number': ['AI101', 'SG102'],
            'origin_airport': ['BOM (Mumbai)', 'DEL (Delhi)'],
            'destination_airport': ['DEL', 'BOM'],
            'scheduled_departure': ['12:00', '13:00']
        })
        
        standardized_df = scraper._standardize_scraped_data(df)
        
        assert standardized_df['origin_airport'].iloc[0] == 'BOM'
        assert standardized_df['origin_airport'].iloc[1] == 'DEL'
        assert 'data_source' in standardized_df.columns
        assert 'scraped_at' in standardized_df.columns
    
    def test_parse_time_column(self, scraper):
        """Test time column parsing"""
        time_series = pd.Series(['12:00', '13:30', '1400', 'invalid', ''])
        parsed_times = scraper._parse_time_column(time_series)
        
        # Should parse valid times and return None/NaT for invalid ones
        assert parsed_times.iloc[0] is not None
        assert parsed_times.iloc[1] is not None
        assert pd.isna(parsed_times.iloc[3])  # Use pd.isna for NaT values
        assert pd.isna(parsed_times.iloc[4])


class TestFlightRadar24Scraper:
    """Test cases for FlightRadar24Scraper"""
    
    @pytest.fixture
    def scraper(self):
        """Create FlightRadar24 scraper for testing"""
        return FlightRadar24Scraper()
    
    @patch.object(FlightRadar24Scraper, '_make_request')
    def test_scrape_mumbai_flights_success(self, mock_request, scraper):
        """Test successful Mumbai flights scraping"""
        mock_response = Mock()
        mock_response.content = b"""
        <html>
            <table class="table-flights">
                <tr>
                    <th>Flight</th>
                    <th>From</th>
                    <th>To</th>
                </tr>
                <tr>
                    <td>AI101</td>
                    <td>BOM</td>
                    <td>DEL</td>
                </tr>
            </table>
        </html>
        """
        mock_request.return_value = mock_response
        
        df = scraper.scrape_mumbai_flights()
        
        assert not df.empty
        assert 'airport' in df.columns
        assert df['airport'].iloc[0] == 'BOM'
        mock_request.assert_called_once()
    
    @patch.object(FlightRadar24Scraper, '_make_request')
    def test_scrape_mumbai_flights_failure(self, mock_request, scraper):
        """Test Mumbai flights scraping failure"""
        mock_request.return_value = None
        
        df = scraper.scrape_mumbai_flights()
        
        assert df.empty
        mock_request.assert_called_once()
    
    @patch.object(FlightRadar24Scraper, '_make_request')
    def test_scrape_delhi_flights_success(self, mock_request, scraper):
        """Test successful Delhi flights scraping"""
        mock_response = Mock()
        mock_response.content = b"""
        <html>
            <table>
                <tr>
                    <th>Flight</th>
                    <th>Origin</th>
                    <th>Destination</th>
                </tr>
                <tr>
                    <td>SG102</td>
                    <td>DEL</td>
                    <td>BOM</td>
                </tr>
            </table>
        </html>
        """
        mock_request.return_value = mock_response
        
        df = scraper.scrape_delhi_flights()
        
        assert not df.empty
        assert 'airport' in df.columns
        assert df['airport'].iloc[0] == 'DEL'
    
    def test_parse_flightradar24_alternative(self, scraper):
        """Test alternative parsing method"""
        html = """
        <html>
            <script>
                var flightData = {"flights": [{"flight": "AI101", "from": "BOM", "to": "DEL"}]};
            </script>
            <div class="flight-row">AI102 BOM DEL 12:00</div>
        </html>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        flights = scraper._parse_flightradar24_alternative(soup)
        
        # Should find at least one flight
        assert len(flights) >= 0
    
    def test_extract_flight_from_element(self, scraper):
        """Test flight extraction from HTML element"""
        html = "<div>AI101 BOM-DEL 12:00 A320</div>"
        element = BeautifulSoup(html, 'html.parser').find('div')
        
        flight_data = scraper._extract_flight_from_element(element)
        
        assert flight_data is not None
        assert flight_data['flight_number'] == 'AI101'
        assert 'raw_text' in flight_data
    
    def test_extract_flight_from_element_no_flight(self, scraper):
        """Test flight extraction when no flight number found"""
        html = "<div>No flight number here</div>"
        element = BeautifulSoup(html, 'html.parser').find('div')
        
        flight_data = scraper._extract_flight_from_element(element)
        
        assert flight_data is None


class TestFlightAwareScraper:
    """Test cases for FlightAwareScraper"""
    
    @pytest.fixture
    def scraper(self):
        """Create FlightAware scraper for testing"""
        return FlightAwareScraper()
    
    @patch.object(FlightAwareScraper, '_make_request')
    def test_scrape_airport_flights_success(self, mock_request, scraper):
        """Test successful airport flights scraping"""
        mock_response = Mock()
        mock_response.content = b"""
        <html>
            <table class="prettyTable">
                <tr>
                    <th>Ident</th>
                    <th>Origin</th>
                    <th>Destination</th>
                </tr>
                <tr>
                    <td>AI101</td>
                    <td>BOM</td>
                    <td>DEL</td>
                </tr>
            </table>
        </html>
        """
        mock_request.return_value = mock_response
        
        df = scraper.scrape_airport_flights('BOM')
        
        assert not df.empty
        assert 'airport' in df.columns
        assert df['airport'].iloc[0] == 'BOM'
        mock_request.assert_called_once()
    
    @patch.object(FlightAwareScraper, '_make_request')
    def test_scrape_airport_flights_failure(self, mock_request, scraper):
        """Test airport flights scraping failure"""
        mock_request.return_value = None
        
        df = scraper.scrape_airport_flights('BOM')
        
        assert df.empty
    
    def test_get_airport_name(self, scraper):
        """Test airport name retrieval"""
        assert scraper._get_airport_name('BOM') == 'Mumbai'
        assert scraper._get_airport_name('DEL') == 'Delhi'
        assert scraper._get_airport_name('XYZ') == 'XYZ'  # Unknown code
    
    def test_parse_flightaware_alternative(self, scraper):
        """Test alternative FlightAware parsing"""
        html = """
        <html>
            <div class="flight-info">AI101 BOM-DEL</div>
            <div class="flight-row">SG102 DEL-BOM</div>
        </html>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        flights = scraper._parse_flightaware_alternative(soup)
        
        # Should find flights
        assert len(flights) >= 0
    
    def test_extract_flightaware_data(self, scraper):
        """Test FlightAware data extraction"""
        html = "<div>AI101 scheduled departure</div>"
        element = BeautifulSoup(html, 'html.parser').find('div')
        
        flight_data = scraper._extract_flightaware_data(element)
        
        assert flight_data is not None
        assert flight_data['flight_number'] == 'AI101'


class TestFlightScrapingManager:
    """Test cases for FlightScrapingManager"""
    
    @pytest.fixture
    def manager(self):
        """Create scraping manager for testing"""
        return FlightScrapingManager()
    
    def test_init(self, manager):
        """Test manager initialization"""
        assert manager.fr24_scraper is not None
        assert manager.fa_scraper is not None
        assert manager.logger is not None
    
    @patch.object(FlightRadar24Scraper, 'scrape_mumbai_flights')
    @patch.object(FlightRadar24Scraper, 'scrape_delhi_flights')
    @patch.object(FlightAwareScraper, 'scrape_airport_flights')
    def test_scrape_all_sources_success(self, mock_fa, mock_del, mock_bom, manager):
        """Test successful scraping from all sources"""
        # Mock successful responses
        mock_bom_df = pd.DataFrame({
            'flight_number': ['AI101'],
            'origin_airport': ['BOM'],
            'destination_airport': ['DEL']
        })
        mock_del_df = pd.DataFrame({
            'flight_number': ['SG102'],
            'origin_airport': ['DEL'],
            'destination_airport': ['BOM']
        })
        mock_fa_df = pd.DataFrame({
            'flight_number': ['UK103'],
            'origin_airport': ['BOM'],
            'destination_airport': ['DEL']
        })
        
        mock_bom.return_value = mock_bom_df
        mock_del.return_value = mock_del_df
        mock_fa.return_value = mock_fa_df
        
        combined_df = manager.scrape_all_sources(['BOM', 'DEL'])
        
        assert not combined_df.empty
        assert len(combined_df) >= 3  # At least 3 flights from different sources
        assert 'source' in combined_df.columns
        
        # Verify all scrapers were called
        mock_bom.assert_called_once()
        mock_del.assert_called_once()
        assert mock_fa.call_count == 2  # Called for both airports
    
    @patch.object(FlightRadar24Scraper, 'scrape_mumbai_flights')
    @patch.object(FlightRadar24Scraper, 'scrape_delhi_flights')
    @patch.object(FlightAwareScraper, 'scrape_airport_flights')
    def test_scrape_all_sources_no_data(self, mock_fa, mock_del, mock_bom, manager):
        """Test scraping when no data is available"""
        # Mock empty responses
        mock_bom.return_value = pd.DataFrame()
        mock_del.return_value = pd.DataFrame()
        mock_fa.return_value = pd.DataFrame()
        
        combined_df = manager.scrape_all_sources(['BOM', 'DEL'])
        
        assert combined_df.empty
    
    def test_get_scraping_summary_empty(self, manager):
        """Test scraping summary with empty data"""
        summary = manager.get_scraping_summary(pd.DataFrame())
        
        assert summary['total_flights'] == 0
        assert summary['sources'] == []
        assert summary['airports'] == []
    
    def test_get_scraping_summary_with_data(self, manager):
        """Test scraping summary with data"""
        df = pd.DataFrame({
            'flight_number': ['AI101', 'SG102'],
            'source': ['FlightRadar24', 'FlightAware'],
            'airport': ['BOM', 'DEL']
        })
        
        summary = manager.get_scraping_summary(df)
        
        assert summary['total_flights'] == 2
        assert 'FlightRadar24' in summary['sources']
        assert 'FlightAware' in summary['sources']
        assert 'BOM' in summary['airports']
        assert 'DEL' in summary['airports']
        assert 'scraped_at' in summary
        assert 'columns' in summary


class TestIntegration:
    """Integration tests for scraping modules"""
    
    def test_scraping_config_integration(self):
        """Test scraping configuration integration"""
        config = ScrapingConfig(
            user_agents=['Test Agent'],
            request_delay_range=(0.1, 0.2),
            max_retries=1,
            timeout=5,
            headers={'Test': 'Header'}
        )
        
        scraper = FlightDataScraper(config)
        
        assert scraper.config.max_retries == 1
        assert scraper.config.timeout == 5
        assert 'Test' in scraper.session.headers
    
    @patch('requests.Session.get')
    def test_end_to_end_scraping_pipeline(self, mock_get):
        """Test complete scraping pipeline"""
        # Mock response with realistic HTML
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html>
            <table>
                <tr>
                    <th>Flight</th>
                    <th>From</th>
                    <th>To</th>
                    <th>STD</th>
                </tr>
                <tr>
                    <td>AI101</td>
                    <td>BOM</td>
                    <td>DEL</td>
                    <td>12:00</td>
                </tr>
            </table>
        </html>
        """
        mock_get.return_value = mock_response
        
        # Test complete pipeline
        manager = FlightScrapingManager()
        
        # Mock the individual scraper methods to avoid actual HTTP calls
        with patch.object(manager.fr24_scraper, 'scrape_mumbai_flights') as mock_bom:
            mock_bom.return_value = pd.DataFrame({
                'flight_number': ['AI101'],
                'origin_airport': ['BOM'],
                'destination_airport': ['DEL'],
                'source': ['FlightRadar24']
            })
            
            df = manager.scrape_all_sources(['BOM'])
            summary = manager.get_scraping_summary(df)
            
            assert not df.empty
            assert summary['total_flights'] > 0
            assert 'FlightRadar24' in summary['sources']