"""
Unit tests for data source connectors.
Tests Grid Status API connector, weather data connector, and economic data connector.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import requests
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.connectors import (
    GridStatusConnector,
    WeatherDataConnector,
    EconomicDataConnector,
    DataConnectorManager
)


class TestGridStatusConnector(unittest.TestCase):
    """Test cases for Grid Status API connector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.connector = GridStatusConnector()
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 2)
    
    def test_initialization(self):
        """Test connector initialization."""
        self.assertIsNotNone(self.connector.session)
        self.assertEqual(self.connector.base_url, "https://api.gridstatus.io")
    
    def test_initialization_with_api_key(self):
        """Test connector initialization with API key."""
        connector = GridStatusConnector(api_key="test_key")
        self.assertEqual(connector.api_key, "test_key")
    
    @patch('requests.Session.get')
    def test_test_connection_success(self, mock_get):
        """Test successful connection test."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = self.connector.test_connection()
        self.assertTrue(result)
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_test_connection_failure(self, mock_get):
        """Test failed connection test."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
        
        result = self.connector.test_connection()
        self.assertFalse(result)
    
    @patch('requests.Session.get')
    def test_fetch_ercot_load_data_success(self, mock_get):
        """Test successful ERCOT load data fetch."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'timestamp': '2024-01-01T00:00:00Z',
                    'load_mw': 45000.0
                },
                {
                    'timestamp': '2024-01-01T01:00:00Z',
                    'load_mw': 46000.0
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = self.connector.fetch_ercot_load_data(self.start_date, self.end_date)
        
        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('timestamp', result.columns)
        self.assertIn('load_mw', result.columns)
        self.assertIn('region', result.columns)
        self.assertIn('data_source', result.columns)
        self.assertEqual(result['region'].iloc[0], 'ERCOT')
        self.assertEqual(result['data_source'].iloc[0], 'GridStatus')
    
    @patch('requests.Session.get')
    def test_fetch_ercot_load_data_api_error(self, mock_get):
        """Test ERCOT load data fetch with API error."""
        mock_get.side_effect = requests.exceptions.RequestException("API Error")
        
        result = self.connector.fetch_ercot_load_data(self.start_date, self.end_date)
        
        # Should return empty DataFrame on error
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
    
    @patch('requests.Session.get')
    def test_fetch_ercot_generation_data_success(self, mock_get):
        """Test successful ERCOT generation data fetch."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'timestamp': '2024-01-01T00:00:00Z',
                    'natural_gas_generation_mw': 25000.0,
                    'wind_generation_mw': 15000.0,
                    'solar_generation_mw': 5000.0
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = self.connector.fetch_ercot_generation_data(self.start_date, self.end_date)
        
        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertIn('timestamp', result.columns)
        self.assertIn('region', result.columns)
        self.assertEqual(result['region'].iloc[0], 'ERCOT')
    
    @patch('requests.Session.get')
    def test_fetch_ercot_price_data_success(self, mock_get):
        """Test successful ERCOT price data fetch."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'timestamp': '2024-01-01T00:00:00Z',
                    'price_per_mwh': 35.50
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = self.connector.fetch_ercot_price_data(self.start_date, self.end_date)
        
        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertIn('price_per_mwh', result.columns)
        self.assertEqual(result['region'].iloc[0], 'ERCOT')
    
    @patch('requests.Session.get')
    def test_fetch_ercot_official_forecasts_success(self, mock_get):
        """Test successful ERCOT official forecasts fetch."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'timestamp': '2024-01-01T00:00:00Z',
                    'forecast_timestamp': '2023-12-31T12:00:00Z',
                    'forecast_load_mw': 47000.0
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = self.connector.fetch_ercot_official_forecasts('load')
        
        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertIn('forecast_timestamp', result.columns)
        self.assertEqual(result['data_source'].iloc[0], 'ERCOT_Official')
        self.assertEqual(result['forecast_type'].iloc[0], 'load')
    
    def test_fetch_data_load(self):
        """Test generic fetch_data method for load data."""
        with patch.object(self.connector, 'fetch_ercot_load_data') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({'test': [1, 2, 3]})
            
            result = self.connector.fetch_data(self.start_date, self.end_date, 'load')
            
            mock_fetch.assert_called_once_with(self.start_date, self.end_date)
            self.assertIsInstance(result, pd.DataFrame)
    
    def test_fetch_data_invalid_type(self):
        """Test generic fetch_data method with invalid data type."""
        result = self.connector.fetch_data(self.start_date, self.end_date, 'invalid')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)


class TestWeatherDataConnector(unittest.TestCase):
    """Test cases for Weather Data connector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.connector = WeatherDataConnector(api_key="test_key")
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 2)
    
    def test_initialization_openweathermap(self):
        """Test connector initialization with OpenWeatherMap."""
        connector = WeatherDataConnector(provider="openweathermap")
        self.assertEqual(connector.provider, "openweathermap")
        self.assertEqual(connector.base_url, "https://api.openweathermap.org/data/2.5")
    
    def test_initialization_noaa(self):
        """Test connector initialization with NOAA."""
        connector = WeatherDataConnector(provider="noaa")
        self.assertEqual(connector.provider, "noaa")
        self.assertEqual(connector.base_url, "https://api.weather.gov")
    
    def test_initialization_invalid_provider(self):
        """Test connector initialization with invalid provider."""
        with self.assertRaises(ValueError):
            WeatherDataConnector(provider="invalid")
    
    @patch('requests.Session.get')
    def test_test_connection_openweathermap_success(self, mock_get):
        """Test successful connection test for OpenWeatherMap."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = self.connector.test_connection()
        self.assertTrue(result)
    
    @patch('requests.Session.get')
    def test_test_connection_noaa_success(self, mock_get):
        """Test successful connection test for NOAA."""
        connector = WeatherDataConnector(provider="noaa")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = connector.test_connection()
        self.assertTrue(result)
    
    def test_fetch_data_openweathermap(self):
        """Test fetch_data method for OpenWeatherMap."""
        result = self.connector.fetch_data(self.start_date, self.end_date)
        
        # Should return DataFrame with proper structure
        self.assertIsInstance(result, pd.DataFrame)
        expected_columns = ['timestamp', 'temperature', 'humidity', 'wind_speed', 'location', 'data_source']
        for col in expected_columns:
            self.assertIn(col, result.columns)
    
    def test_fetch_data_noaa(self):
        """Test fetch_data method for NOAA."""
        connector = WeatherDataConnector(provider="noaa")
        result = connector.fetch_data(self.start_date, self.end_date)
        
        # Should return DataFrame with proper structure
        self.assertIsInstance(result, pd.DataFrame)


class TestEconomicDataConnector(unittest.TestCase):
    """Test cases for Economic Data connector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.connector = EconomicDataConnector(api_key="test_key")
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 2)
    
    def test_initialization(self):
        """Test connector initialization."""
        self.assertEqual(self.connector.api_key, "test_key")
        self.assertEqual(self.connector.base_url, "https://api.stlouisfed.org/fred")
    
    @patch('requests.Session.get')
    def test_test_connection_success(self, mock_get):
        """Test successful connection test."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = self.connector.test_connection()
        self.assertTrue(result)
    
    def test_fetch_data_default_indicators(self):
        """Test fetch_data method with default indicators."""
        result = self.connector.fetch_data(self.start_date, self.end_date)
        
        # Should return DataFrame with proper structure
        self.assertIsInstance(result, pd.DataFrame)
        expected_columns = ['timestamp', 'indicator', 'value', 'data_source']
        for col in expected_columns:
            self.assertIn(col, result.columns)
    
    def test_fetch_data_custom_indicators(self):
        """Test fetch_data method with custom indicators."""
        custom_indicators = ['GDP', 'UNRATE']
        result = self.connector.fetch_data(self.start_date, self.end_date, custom_indicators)
        
        self.assertIsInstance(result, pd.DataFrame)


class TestDataConnectorManager(unittest.TestCase):
    """Test cases for Data Connector Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = DataConnectorManager()
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 2)
    
    def test_initialization(self):
        """Test manager initialization."""
        self.assertIn('gridstatus', self.manager.connectors)
        self.assertIn('weather', self.manager.connectors)
        self.assertIn('economic', self.manager.connectors)
    
    def test_get_connector(self):
        """Test get_connector method."""
        gridstatus = self.manager.get_connector('gridstatus')
        self.assertIsInstance(gridstatus, GridStatusConnector)
        
        invalid = self.manager.get_connector('invalid')
        self.assertIsNone(invalid)
    
    @patch('data.connectors.GridStatusConnector.test_connection')
    @patch('data.connectors.WeatherDataConnector.test_connection')
    @patch('data.connectors.EconomicDataConnector.test_connection')
    def test_test_all_connections(self, mock_economic, mock_weather, mock_gridstatus):
        """Test test_all_connections method."""
        mock_gridstatus.return_value = True
        mock_weather.return_value = True
        mock_economic.return_value = False
        
        results = self.manager.test_all_connections()
        
        self.assertTrue(results['gridstatus'])
        self.assertTrue(results['weather'])
        self.assertFalse(results['economic'])
    
    @patch('data.connectors.GridStatusConnector.fetch_data')
    def test_fetch_ercot_data(self, mock_fetch):
        """Test fetch_ercot_data method."""
        mock_fetch.return_value = pd.DataFrame({'test': [1, 2, 3]})
        
        results = self.manager.fetch_ercot_data(self.start_date, self.end_date, ['load'])
        
        self.assertIn('load', results)
        self.assertIsInstance(results['load'], pd.DataFrame)
    
    def test_fetch_ercot_data_no_connector(self):
        """Test fetch_ercot_data method when connector is not available."""
        # Remove gridstatus connector
        del self.manager.connectors['gridstatus']
        
        results = self.manager.fetch_ercot_data(self.start_date, self.end_date)
        
        self.assertEqual(len(results), 0)
    
    @patch('data.connectors.DataConnectorManager.fetch_ercot_data')
    @patch('data.connectors.WeatherDataConnector.fetch_data')
    @patch('data.connectors.EconomicDataConnector.fetch_data')
    def test_fetch_all_data(self, mock_economic, mock_weather, mock_ercot):
        """Test fetch_all_data method."""
        mock_ercot.return_value = {'load': pd.DataFrame({'load': [1, 2, 3]})}
        mock_weather.return_value = pd.DataFrame({'temp': [20, 21, 22]})
        mock_economic.return_value = pd.DataFrame({'gdp': [100, 101, 102]})
        
        results = self.manager.fetch_all_data(self.start_date, self.end_date)
        
        self.assertIn('load', results)
        self.assertIn('weather', results)
        self.assertIn('economic', results)


class TestDataConnectorIntegration(unittest.TestCase):
    """Integration tests for data connectors."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 2)
    
    def test_gridstatus_connector_integration(self):
        """Test Grid Status connector integration."""
        connector = GridStatusConnector()
        
        # Test that connector can be created without errors
        self.assertIsNotNone(connector)
        self.assertIsNotNone(connector.session)
        
        # Test that methods return proper DataFrame structure
        result = connector.fetch_data(self.start_date, self.end_date, 'invalid_type')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
    
    def test_data_connector_manager_integration(self):
        """Test Data Connector Manager integration."""
        manager = DataConnectorManager()
        
        # Test that manager initializes all connectors
        self.assertEqual(len(manager.connectors), 3)
        
        # Test that all connectors are of correct type
        self.assertIsInstance(manager.get_connector('gridstatus'), GridStatusConnector)
        self.assertIsInstance(manager.get_connector('weather'), WeatherDataConnector)
        self.assertIsInstance(manager.get_connector('economic'), EconomicDataConnector)


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)