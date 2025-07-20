"""
Unit tests for weather data integration and correlation features.
Tests weather API integration and energy-weather feature creation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.weather_integration import WeatherDataIntegrator


class TestWeatherDataIntegrator(unittest.TestCase):
    """Test cases for WeatherDataIntegrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integrator = WeatherDataIntegrator()
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 2)
        
        # Sample weather data
        self.sample_weather_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='h'),
            'city': ['houston'] * 24,
            'latitude': [29.7604] * 24,
            'longitude': [-95.3698] * 24,
            'weight': [0.3] * 24,
            'temperature_f': np.random.uniform(40, 80, 24),
            'temperature_c': np.random.uniform(4, 27, 24),
            'humidity': np.random.uniform(30, 90, 24),
            'wind_speed_mph': np.random.uniform(5, 25, 24),
            'pressure_pa': np.random.uniform(101000, 102000, 24),
            'data_source': ['NWS'] * 24
        })
        
        # Sample energy data
        self.sample_energy_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='h'),
            'ercot_load_mw': np.random.uniform(30000, 70000, 24),
            'ercot_price_per_mwh': np.random.uniform(20, 100, 24)
        })
    
    def test_initialization(self):
        """Test WeatherDataIntegrator initialization."""
        self.assertIsNotNone(self.integrator.session)
        self.assertEqual(len(self.integrator.ercot_locations), 5)
        self.assertIn('houston', self.integrator.ercot_locations)
        self.assertIn('dallas', self.integrator.ercot_locations)
    
    def test_ercot_locations_weights(self):
        """Test that ERCOT location weights sum to 1.0."""
        total_weight = sum(loc['weight'] for loc in self.integrator.ercot_locations.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_celsius_to_fahrenheit(self):
        """Test temperature conversion."""
        # Test known conversions
        self.assertAlmostEqual(self.integrator._celsius_to_fahrenheit(0), 32, places=1)
        self.assertAlmostEqual(self.integrator._celsius_to_fahrenheit(100), 212, places=1)
        self.assertAlmostEqual(self.integrator._celsius_to_fahrenheit(20), 68, places=1)
        
        # Test None handling
        self.assertIsNone(self.integrator._celsius_to_fahrenheit(None))
    
    def test_mps_to_mph(self):
        """Test wind speed conversion."""
        # Test known conversions
        self.assertAlmostEqual(self.integrator._mps_to_mph(10), 22.37, places=1)
        self.assertAlmostEqual(self.integrator._mps_to_mph(0), 0, places=1)
        
        # Test None handling
        self.assertIsNone(self.integrator._mps_to_mph(None))
    
    def test_extract_value(self):
        """Test NWS weather property value extraction."""
        # Test valid property
        weather_property = {'value': 25.5, 'unitCode': 'unit:degC'}
        self.assertEqual(self.integrator._extract_value(weather_property), 25.5)
        
        # Test None property
        self.assertIsNone(self.integrator._extract_value(None))
        
        # Test invalid property
        self.assertIsNone(self.integrator._extract_value({'invalid': 'data'}))
    
    @patch('requests.Session.get')
    def test_get_nws_station_success(self, mock_get):
        """Test successful NWS station retrieval."""
        # Mock points API response
        points_response = Mock()
        points_response.status_code = 200
        points_response.json.return_value = {
            'properties': {
                'observationStations': 'https://api.weather.gov/gridpoints/HGX/64,68/stations'
            }
        }
        
        # Mock stations API response
        stations_response = Mock()
        stations_response.status_code = 200
        stations_response.json.return_value = {
            'features': [
                {'id': 'KIAH'}
            ]
        }
        
        mock_get.side_effect = [points_response, stations_response]
        
        station_id = self.integrator._get_nws_station(29.7604, -95.3698)
        self.assertEqual(station_id, 'KIAH')
        self.assertEqual(mock_get.call_count, 2)
    
    @patch('requests.Session.get')
    def test_get_nws_station_failure(self, mock_get):
        """Test NWS station retrieval failure."""
        mock_get.return_value.status_code = 404
        
        station_id = self.integrator._get_nws_station(29.7604, -95.3698)
        self.assertIsNone(station_id)
    
    @patch('requests.Session.get')
    def test_fetch_nws_observations_success(self, mock_get):
        """Test successful NWS observations fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'features': [
                {
                    'properties': {
                        'timestamp': '2024-01-01T12:00:00Z',
                        'temperature': {'value': 20.0},
                        'relativeHumidity': {'value': 65.0},
                        'windSpeed': {'value': 5.0}
                    }
                }
            ]
        }
        mock_get.return_value = mock_response
        
        observations = self.integrator._fetch_nws_observations('KIAH', self.start_date, self.end_date)
        
        self.assertEqual(len(observations), 1)
        self.assertIn('properties', observations[0])
    
    @patch('requests.Session.get')
    def test_fetch_openweather_current_success(self, mock_get):
        """Test successful OpenWeatherMap current weather fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'main': {
                'temp': 293.15,  # 20°C in Kelvin
                'humidity': 65,
                'pressure': 1013
            },
            'wind': {
                'speed': 5.0,
                'deg': 180
            },
            'clouds': {
                'all': 25
            },
            'weather': [
                {'description': 'partly cloudy'}
            ]
        }
        mock_get.return_value = mock_response
        
        # Test with API key
        integrator_with_key = WeatherDataIntegrator(openweather_api_key='test_key')
        weather_data = integrator_with_key._fetch_openweather_current(29.7604, -95.3698)
        
        self.assertIsNotNone(weather_data)
        self.assertIn('main', weather_data)
        self.assertEqual(weather_data['main']['temp'], 293.15)
    
    def test_create_ercot_weighted_weather(self):
        """Test ERCOT weighted weather calculation."""
        # Create multi-city weather data
        multi_city_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1, 12)] * 3,
            'city': ['houston', 'dallas', 'austin'],
            'weight': [0.3, 0.25, 0.2],
            'temperature_f': [75, 70, 72],
            'temperature_c': [24, 21, 22],
            'humidity': [60, 55, 58],
            'wind_speed_mph': [10, 12, 8],
            'pressure_pa': [101300, 101200, 101250]
        })
        
        weighted_weather = self.integrator.create_ercot_weighted_weather(multi_city_data)
        
        self.assertFalse(weighted_weather.empty)
        self.assertEqual(len(weighted_weather), 1)
        self.assertIn('ercot_temperature_f', weighted_weather.columns)
        
        # Check weighted average calculation
        expected_temp = (75 * 0.3 + 70 * 0.25 + 72 * 0.2) / (0.3 + 0.25 + 0.2)
        actual_temp = weighted_weather['ercot_temperature_f'].iloc[0]
        self.assertAlmostEqual(actual_temp, expected_temp, places=1)
    
    def test_create_ercot_weighted_weather_empty_data(self):
        """Test ERCOT weighted weather with empty data."""
        empty_data = pd.DataFrame()
        result = self.integrator.create_ercot_weighted_weather(empty_data)
        self.assertTrue(result.empty)
    
    def test_create_weather_energy_features(self):
        """Test weather-energy feature creation."""
        # Create ERCOT weighted weather data
        weather_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='h'),
            'ercot_temperature_f': np.random.uniform(40, 90, 24),
            'ercot_humidity': np.random.uniform(30, 90, 24),
            'ercot_wind_speed_mph': np.random.uniform(5, 25, 24)
        })
        
        combined_features = self.integrator.create_weather_energy_features(
            weather_data, self.sample_energy_data
        )
        
        # Check that weather-energy features were created
        expected_features = [
            'heating_degree_days', 'cooling_degree_days',
            'temp_very_cold', 'temp_cold', 'temp_mild', 'temp_warm', 'temp_hot', 'temp_very_hot',
            'wind_power_potential', 'wind_calm', 'wind_light', 'wind_moderate', 'wind_strong',
            'heat_index', 'heat_stress'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, combined_features.columns)
        
        # Check that original energy data is preserved
        self.assertIn('ercot_load_mw', combined_features.columns)
        self.assertIn('ercot_price_per_mwh', combined_features.columns)
    
    def test_create_temperature_features(self):
        """Test temperature-based feature creation."""
        data = pd.DataFrame({
            'ercot_temperature_f': [30, 45, 65, 80, 90, 100]
        })
        
        result = self.integrator._create_temperature_features(data)
        
        # Check heating degree days
        self.assertEqual(result['heating_degree_days'].iloc[0], 35)  # 65 - 30
        self.assertEqual(result['heating_degree_days'].iloc[2], 0)   # 65 - 65
        
        # Check cooling degree days
        self.assertEqual(result['cooling_degree_days'].iloc[2], 0)   # 65 - 65
        self.assertEqual(result['cooling_degree_days'].iloc[4], 25)  # 90 - 65
        
        # Check temperature categories
        self.assertEqual(result['temp_very_cold'].iloc[0], 1)  # 30°F
        self.assertEqual(result['temp_mild'].iloc[2], 1)       # 65°F
        self.assertEqual(result['temp_very_hot'].iloc[5], 1)   # 100°F
    
    def test_create_wind_features(self):
        """Test wind-based feature creation."""
        data = pd.DataFrame({
            'ercot_wind_speed_mph': [5, 10, 20, 35, 60]
        })
        
        result = self.integrator._create_wind_features(data)
        
        # Check wind power potential
        self.assertEqual(result['wind_power_potential'].iloc[0], 0)     # Below cut-in
        self.assertGreater(result['wind_power_potential'].iloc[1], 0)   # Above cut-in
        self.assertEqual(result['wind_power_potential'].iloc[3], 1)     # Rated power
        self.assertEqual(result['wind_power_potential'].iloc[4], 0)     # Above cut-out
        
        # Check wind categories
        self.assertEqual(result['wind_calm'].iloc[0], 1)        # 5 mph
        self.assertEqual(result['wind_light'].iloc[1], 1)       # 10 mph
        self.assertEqual(result['wind_moderate'].iloc[2], 1)    # 20 mph
        self.assertEqual(result['wind_strong'].iloc[3], 1)      # 35 mph
    
    def test_create_comfort_indices(self):
        """Test comfort index creation."""
        data = pd.DataFrame({
            'ercot_temperature_f': [85, 95, 75],
            'ercot_humidity': [70, 80, 50]
        })
        
        result = self.integrator._create_comfort_indices(data)
        
        # Check that heat index is calculated
        self.assertIn('heat_index', result.columns)
        self.assertIn('heat_stress', result.columns)
        
        # Heat index should be higher than temperature for high humidity
        self.assertGreater(result['heat_index'].iloc[0], result['ercot_temperature_f'].iloc[0])
        
        # Check heat stress indicators
        self.assertIn(result['heat_stress'].iloc[1], [0, 1])  # Should be binary
    
    def test_create_seasonal_weather_features(self):
        """Test seasonal weather feature creation."""
        data = pd.DataFrame({
            'timestamp': [
                datetime(2024, 1, 15, 12),  # Winter
                datetime(2024, 7, 15, 12),  # Summer
                datetime(2024, 4, 15, 12)   # Spring
            ],
            'ercot_temperature_f': [45, 95, 75]
        })
        
        result = self.integrator._create_seasonal_weather_features(data)
        
        # Check seasonal features
        self.assertIn('peak_heating_season', result.columns)
        self.assertIn('peak_cooling_season', result.columns)
        self.assertIn('temp_vs_seasonal_normal', result.columns)
        
        # Check seasonal indicators
        self.assertEqual(result['peak_heating_season'].iloc[0], 1)  # January
        self.assertEqual(result['peak_cooling_season'].iloc[1], 1)  # July
        self.assertEqual(result['peak_heating_season'].iloc[2], 0)  # April
    
    def test_get_weather_summary(self):
        """Test weather data summary generation."""
        summary = self.integrator.get_weather_summary(self.sample_weather_data)
        
        self.assertIn('total_records', summary)
        self.assertIn('date_range', summary)
        self.assertIn('temperature_stats', summary)
        self.assertIn('weather_features', summary)
        
        self.assertEqual(summary['total_records'], 24)
        self.assertIn('mean', summary['temperature_stats'])
        self.assertIn('min', summary['temperature_stats'])
        self.assertIn('max', summary['temperature_stats'])
    
    def test_get_weather_summary_empty_data(self):
        """Test weather summary with empty data."""
        empty_data = pd.DataFrame()
        summary = self.integrator.get_weather_summary(empty_data)
        
        self.assertIn('error', summary)
        self.assertEqual(summary['error'], 'No weather data available')
    
    def test_create_weather_energy_features_empty_data(self):
        """Test weather-energy features with empty data."""
        empty_weather = pd.DataFrame()
        empty_energy = pd.DataFrame()
        
        # Test with empty weather data
        result1 = self.integrator.create_weather_energy_features(empty_weather, self.sample_energy_data)
        self.assertEqual(len(result1), len(self.sample_energy_data))
        
        # Test with empty energy data
        result2 = self.integrator.create_weather_energy_features(self.sample_weather_data, empty_energy)
        self.assertTrue(result2.empty)
    
    @patch('features.weather_integration.WeatherDataIntegrator._get_nws_station')
    @patch('features.weather_integration.WeatherDataIntegrator._fetch_nws_observations')
    def test_fetch_nws_weather_data_integration(self, mock_observations, mock_station):
        """Test NWS weather data fetch integration."""
        # Mock station lookup
        mock_station.return_value = 'KIAH'
        
        # Mock observations
        mock_observations.return_value = [
            {
                'properties': {
                    'timestamp': '2024-01-01T12:00:00Z',
                    'temperature': {'value': 20.0},
                    'relativeHumidity': {'value': 65.0},
                    'windSpeed': {'value': 5.0},
                    'windDirection': {'value': 180.0},
                    'barometricPressure': {'value': 101300.0},
                    'visibility': {'value': 10000.0}
                }
            }
        ]
        
        result = self.integrator.fetch_nws_weather_data(self.start_date, self.end_date)
        
        # Should have data for all ERCOT cities
        self.assertFalse(result.empty)
        self.assertIn('temperature_f', result.columns)
        self.assertIn('city', result.columns)
        self.assertIn('data_source', result.columns)
        
        # Check that all cities are represented
        cities_in_result = result['city'].unique()
        self.assertGreater(len(cities_in_result), 0)


class TestWeatherIntegrationIntegration(unittest.TestCase):
    """Integration tests for weather integration system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integrator = WeatherDataIntegrator()
        
        # Create realistic ERCOT energy data
        dates = pd.date_range('2024-01-01', periods=168, freq='h')  # One week
        self.ercot_energy_data = pd.DataFrame({
            'timestamp': dates,
            'ercot_load_mw': 45000 + 15000 * np.sin(2 * np.pi * np.arange(168) / 24) + np.random.normal(0, 2000, 168),
            'ercot_price_per_mwh': 50 + 20 * np.sin(2 * np.pi * np.arange(168) / 24) + np.random.normal(0, 5, 168),
            'wind_generation_mw': np.random.uniform(5000, 25000, 168)
        })
        
        # Create realistic weather data
        self.ercot_weather_data = pd.DataFrame({
            'timestamp': dates,
            'ercot_temperature_f': 70 + 20 * np.sin(2 * np.pi * np.arange(168) / (24 * 7)) + np.random.normal(0, 5, 168),
            'ercot_humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(168) / 24) + np.random.normal(0, 10, 168),
            'ercot_wind_speed_mph': 15 + 10 * np.sin(2 * np.pi * np.arange(168) / 12) + np.random.normal(0, 3, 168)
        })
    
    def test_complete_weather_energy_workflow(self):
        """Test complete weather-energy integration workflow."""
        # 1. Create weather-energy features
        combined_data = self.integrator.create_weather_energy_features(
            self.ercot_weather_data, self.ercot_energy_data
        )
        
        # Should have more columns than original energy data
        self.assertGreater(len(combined_data.columns), len(self.ercot_energy_data.columns))
        
        # Should preserve all original data
        self.assertEqual(len(combined_data), len(self.ercot_energy_data))
        
        # Check key weather-energy features
        weather_energy_features = [
            'heating_degree_days', 'cooling_degree_days',
            'wind_power_potential', 'heat_index',
            'peak_cooling_season', 'peak_heating_season'
        ]
        
        for feature in weather_energy_features:
            self.assertIn(feature, combined_data.columns)
        
        # 2. Generate weather summary
        summary = self.integrator.get_weather_summary(self.ercot_weather_data)
        self.assertIn('total_records', summary)
        self.assertEqual(summary['total_records'], 168)
        
        # 3. Check that features make sense for energy forecasting
        # Temperature-load correlation should exist
        temp_load_corr = combined_data['ercot_temperature_f'].corr(combined_data['ercot_load_mw'])
        self.assertIsNotNone(temp_load_corr)
        
        # Wind power potential should correlate with wind speed
        wind_corr = combined_data['ercot_wind_speed_mph'].corr(combined_data['wind_power_potential'])
        self.assertGreater(wind_corr, 0.5)  # Should be positively correlated
    
    def test_seasonal_weather_patterns(self):
        """Test seasonal weather pattern recognition."""
        # Create year-long data with seasonal patterns
        dates = pd.date_range('2024-01-01', periods=8760, freq='h')  # Full year
        
        # Simulate seasonal temperature pattern
        day_of_year = dates.dayofyear
        seasonal_temp = 70 + 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in summer
        
        weather_data = pd.DataFrame({
            'timestamp': dates,
            'ercot_temperature_f': seasonal_temp + np.random.normal(0, 5, len(dates)),
            'ercot_humidity': np.random.uniform(40, 80, len(dates)),
            'ercot_wind_speed_mph': np.random.uniform(5, 25, len(dates))
        })
        
        # Create seasonal features
        seasonal_features = self.integrator._create_seasonal_weather_features(weather_data)
        
        # Check seasonal indicators
        summer_data = seasonal_features[seasonal_features['month'].isin([6, 7, 8, 9])]
        winter_data = seasonal_features[seasonal_features['month'].isin([12, 1, 2])]
        
        # Summer should have more cooling season indicators
        summer_cooling = summer_data['peak_cooling_season'].mean()
        winter_cooling = winter_data['peak_cooling_season'].mean()
        self.assertGreater(summer_cooling, winter_cooling)
        
        # Winter should have more heating season indicators
        summer_heating = summer_data['peak_heating_season'].mean()
        winter_heating = winter_data['peak_heating_season'].mean()
        self.assertGreater(winter_heating, summer_heating)
    
    def test_extreme_weather_handling(self):
        """Test handling of extreme weather conditions."""
        # Create data with extreme weather
        extreme_weather = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
            'ercot_temperature_f': [10, 110, 32, 95, 65, -5, 120, 75, 85, 50],  # Extreme temps
            'ercot_humidity': [10, 95, 50, 90, 60, 20, 85, 45, 70, 55],
            'ercot_wind_speed_mph': [0, 60, 15, 45, 8, 2, 70, 12, 25, 18]  # Extreme winds
        })
        
        energy_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
            'ercot_load_mw': np.random.uniform(40000, 80000, 10)
        })
        
        # Should handle extreme values without errors
        result = self.integrator.create_weather_energy_features(extreme_weather, energy_data)
        
        # Check extreme temperature categories
        self.assertGreater(result['temp_very_cold'].sum(), 0)  # Should detect very cold
        self.assertGreater(result['temp_very_hot'].sum(), 0)   # Should detect very hot
        
        # Check extreme wind conditions
        self.assertGreater(result['wind_calm'].sum(), 0)       # Should detect calm winds
        self.assertGreater(result['wind_very_strong'].sum(), 0) # Should detect strong winds
        
        # Check extreme heat stress
        self.assertGreater(result['extreme_heat_stress'].sum(), 0)  # Should detect heat stress


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)