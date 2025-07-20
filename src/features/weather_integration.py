"""
Weather data integration and correlation features for energy forecasting.
Integrates with free weather APIs and creates energy-specific weather features.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherDataIntegrator:
    """
    Integrates weather data from free APIs and creates energy-specific weather features.
    Focuses on ERCOT region (Texas) weather patterns.
    """
    
    def __init__(self, openweather_api_key: Optional[str] = None):
        """
        Initialize weather data integrator.
        
        Args:
            openweather_api_key: Optional API key for OpenWeatherMap (free tier available)
        """
        self.openweather_api_key = openweather_api_key or os.getenv('OPENWEATHER_API_KEY')
        self.session = self._create_session()
        
        # ERCOT region coordinates (Texas major cities)
        self.ercot_locations = {
            'houston': {'lat': 29.7604, 'lon': -95.3698, 'weight': 0.3},
            'dallas': {'lat': 32.7767, 'lon': -96.7970, 'weight': 0.25},
            'austin': {'lat': 30.2672, 'lon': -97.7431, 'weight': 0.2},
            'san_antonio': {'lat': 29.4241, 'lon': -98.4936, 'weight': 0.15},
            'fort_worth': {'lat': 32.7555, 'lon': -97.3308, 'weight': 0.1}
        }
        
        logger.info("Initialized WeatherDataIntegrator for ERCOT region")
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': 'EnergyTech-ML-Platform/1.0 (Educational/Research)'
        })
        
        return session
    
    def fetch_nws_weather_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch weather data from NOAA/NWS API (free, no API key required).
        
        Args:
            start_date: Start date for weather data
            end_date: End date for weather data
            
        Returns:
            DataFrame with weather data for ERCOT region
        """
        try:
            weather_data = []
            
            for city, coords in self.ercot_locations.items():
                logger.info(f"Fetching NWS weather data for {city}")
                
                # Get weather station for this location
                station_data = self._get_nws_station(coords['lat'], coords['lon'])
                if not station_data:
                    continue
                
                # Fetch observations
                observations = self._fetch_nws_observations(station_data, start_date, end_date)
                
                for obs in observations:
                    weather_data.append({
                        'timestamp': pd.to_datetime(obs.get('timestamp')),
                        'city': city,
                        'latitude': coords['lat'],
                        'longitude': coords['lon'],
                        'weight': coords['weight'],
                        'temperature_c': self._extract_value(obs.get('temperature')),
                        'temperature_f': self._celsius_to_fahrenheit(self._extract_value(obs.get('temperature'))),
                        'humidity': self._extract_value(obs.get('relativeHumidity')),
                        'wind_speed_mps': self._extract_value(obs.get('windSpeed')),
                        'wind_speed_mph': self._mps_to_mph(self._extract_value(obs.get('windSpeed'))),
                        'wind_direction': self._extract_value(obs.get('windDirection')),
                        'pressure_pa': self._extract_value(obs.get('barometricPressure')),
                        'visibility_m': self._extract_value(obs.get('visibility')),
                        'data_source': 'NWS'
                    })
                
                # Be respectful to the API
                time.sleep(1)
            
            if weather_data:
                df = pd.DataFrame(weather_data)
                df = df.sort_values('timestamp').reset_index(drop=True)
                logger.info(f"Successfully fetched {len(df)} weather observations from NWS")
                return df
            else:
                logger.warning("No weather data retrieved from NWS")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching NWS weather data: {e}")
            return pd.DataFrame()
    
    def _get_nws_station(self, lat: float, lon: float) -> Optional[str]:
        """Get nearest NWS weather station for coordinates."""
        try:
            url = f"https://api.weather.gov/points/{lat},{lon}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                stations_url = data['properties']['observationStations']
                
                # Get stations
                stations_response = self.session.get(stations_url, timeout=10)
                if stations_response.status_code == 200:
                    stations_data = stations_response.json()
                    if stations_data['features']:
                        return stations_data['features'][0]['id']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting NWS station for {lat}, {lon}: {e}")
            return None
    
    def _fetch_nws_observations(self, station_id: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch observations from NWS station."""
        try:
            # NWS API has limitations on date range, so we'll get recent data
            url = f"https://api.weather.gov/stations/{station_id}/observations"
            params = {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('features', [])
            else:
                logger.warning(f"NWS API returned status {response.status_code} for station {station_id}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching NWS observations for {station_id}: {e}")
            return []
    
    def fetch_openweather_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch weather data from OpenWeatherMap API (requires free API key).
        
        Args:
            start_date: Start date for weather data
            end_date: End date for weather data
            
        Returns:
            DataFrame with weather data for ERCOT region
        """
        if not self.openweather_api_key:
            logger.warning("OpenWeatherMap API key not provided, skipping OpenWeather data")
            return pd.DataFrame()
        
        try:
            weather_data = []
            
            for city, coords in self.ercot_locations.items():
                logger.info(f"Fetching OpenWeather data for {city}")
                
                # For historical data, we'd use the historical API (paid)
                # For current/forecast, we use the free current weather API
                current_weather = self._fetch_openweather_current(coords['lat'], coords['lon'])
                
                if current_weather:
                    weather_data.append({
                        'timestamp': datetime.now(),
                        'city': city,
                        'latitude': coords['lat'],
                        'longitude': coords['lon'],
                        'weight': coords['weight'],
                        'temperature_c': current_weather.get('main', {}).get('temp', 0) - 273.15,  # Kelvin to Celsius
                        'temperature_f': self._celsius_to_fahrenheit(current_weather.get('main', {}).get('temp', 0) - 273.15),
                        'humidity': current_weather.get('main', {}).get('humidity'),
                        'wind_speed_mps': current_weather.get('wind', {}).get('speed'),
                        'wind_speed_mph': self._mps_to_mph(current_weather.get('wind', {}).get('speed', 0)),
                        'wind_direction': current_weather.get('wind', {}).get('deg'),
                        'pressure_pa': current_weather.get('main', {}).get('pressure', 0) * 100,  # hPa to Pa
                        'visibility_m': current_weather.get('visibility'),
                        'cloud_cover': current_weather.get('clouds', {}).get('all'),
                        'weather_description': current_weather.get('weather', [{}])[0].get('description'),
                        'data_source': 'OpenWeatherMap'
                    })
                
                # Be respectful to the API
                time.sleep(0.5)
            
            if weather_data:
                df = pd.DataFrame(weather_data)
                logger.info(f"Successfully fetched {len(df)} weather observations from OpenWeatherMap")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching OpenWeatherMap data: {e}")
            return pd.DataFrame()
    
    def _fetch_openweather_current(self, lat: float, lon: float) -> Optional[Dict]:
        """Fetch current weather from OpenWeatherMap."""
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.openweather_api_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"OpenWeatherMap API returned status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching OpenWeather current data: {e}")
            return None
    
    def _extract_value(self, weather_property: Optional[Dict]) -> Optional[float]:
        """Extract numeric value from NWS weather property."""
        if weather_property and isinstance(weather_property, dict):
            return weather_property.get('value')
        return None
    
    def _celsius_to_fahrenheit(self, celsius: Optional[float]) -> Optional[float]:
        """Convert Celsius to Fahrenheit."""
        if celsius is not None:
            return celsius * 9/5 + 32
        return None
    
    def _mps_to_mph(self, mps: Optional[float]) -> Optional[float]:
        """Convert meters per second to miles per hour."""
        if mps is not None:
            return mps * 2.237
        return None
    
    def create_ercot_weighted_weather(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create ERCOT region-weighted weather averages.
        
        Args:
            weather_data: DataFrame with weather data for multiple cities
            
        Returns:
            DataFrame with weighted weather averages for ERCOT region
        """
        if weather_data.empty:
            return pd.DataFrame()
        
        try:
            # Group by timestamp and calculate weighted averages
            weather_features = []
            
            for timestamp, group in weather_data.groupby('timestamp'):
                if len(group) == 0:
                    continue
                
                # Calculate weighted averages
                total_weight = group['weight'].sum()
                
                weighted_weather = {
                    'timestamp': timestamp,
                    'ercot_temperature_f': (group['temperature_f'] * group['weight']).sum() / total_weight,
                    'ercot_temperature_c': (group['temperature_c'] * group['weight']).sum() / total_weight,
                    'ercot_humidity': (group['humidity'] * group['weight']).sum() / total_weight,
                    'ercot_wind_speed_mph': (group['wind_speed_mph'] * group['weight']).sum() / total_weight,
                    'ercot_pressure_pa': (group['pressure_pa'] * group['weight']).sum() / total_weight,
                    'data_source': 'ERCOT_Weighted'
                }
                
                # Add cloud cover if available
                if 'cloud_cover' in group.columns:
                    weighted_weather['ercot_cloud_cover'] = (group['cloud_cover'] * group['weight']).sum() / total_weight
                
                weather_features.append(weighted_weather)
            
            if weather_features:
                df = pd.DataFrame(weather_features)
                df = df.sort_values('timestamp').reset_index(drop=True)
                logger.info(f"Created {len(df)} ERCOT weighted weather records")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error creating ERCOT weighted weather: {e}")
            return pd.DataFrame()
    
    def create_weather_energy_features(self, weather_data: pd.DataFrame, energy_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-energy correlation features for forecasting.
        
        Args:
            weather_data: DataFrame with weather data
            energy_data: DataFrame with energy market data
            
        Returns:
            DataFrame with combined weather-energy features
        """
        if weather_data.empty or energy_data.empty:
            logger.warning("Empty weather or energy data provided")
            return energy_data
        
        try:
            # Merge weather and energy data on timestamp
            merged_data = pd.merge(energy_data, weather_data, on='timestamp', how='left')
            
            # Create energy-specific weather features
            merged_data = self._create_temperature_features(merged_data)
            merged_data = self._create_wind_features(merged_data)
            merged_data = self._create_comfort_indices(merged_data)
            merged_data = self._create_seasonal_weather_features(merged_data)
            
            logger.info(f"Created weather-energy features for {len(merged_data)} records")
            return merged_data
            
        except Exception as e:
            logger.error(f"Error creating weather-energy features: {e}")
            return energy_data
    
    def _create_temperature_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create temperature-based energy features."""
        if 'ercot_temperature_f' not in data.columns:
            return data
        
        # Heating and Cooling Degree Days (base 65°F)
        data['heating_degree_days'] = np.maximum(0, 65 - data['ercot_temperature_f'])
        data['cooling_degree_days'] = np.maximum(0, data['ercot_temperature_f'] - 65)
        
        # Temperature categories for energy demand
        data['temp_very_cold'] = (data['ercot_temperature_f'] < 32).astype(int)
        data['temp_cold'] = ((data['ercot_temperature_f'] >= 32) & (data['ercot_temperature_f'] < 50)).astype(int)
        data['temp_mild'] = ((data['ercot_temperature_f'] >= 50) & (data['ercot_temperature_f'] < 75)).astype(int)
        data['temp_warm'] = ((data['ercot_temperature_f'] >= 75) & (data['ercot_temperature_f'] < 85)).astype(int)
        data['temp_hot'] = ((data['ercot_temperature_f'] >= 85) & (data['ercot_temperature_f'] < 95)).astype(int)
        data['temp_very_hot'] = (data['ercot_temperature_f'] >= 95).astype(int)
        
        # Temperature deviation from seasonal normal (simplified)
        data['temp_deviation_from_normal'] = data['ercot_temperature_f'] - 70  # Rough Texas average
        
        return data
    
    def _create_wind_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create wind-based energy features."""
        if 'ercot_wind_speed_mph' not in data.columns:
            return data
        
        # Wind power generation potential (simplified wind power curve)
        # Typical wind turbine: cut-in ~7mph, rated ~31mph, cut-out ~56mph
        data['wind_power_potential'] = np.where(
            data['ercot_wind_speed_mph'] < 7, 0,
            np.where(
                data['ercot_wind_speed_mph'] < 31,
                (data['ercot_wind_speed_mph'] - 7) / (31 - 7),  # Linear ramp up
                np.where(
                    data['ercot_wind_speed_mph'] < 56, 1,  # Rated power
                    0  # Cut-out
                )
            )
        )
        
        # Wind categories
        data['wind_calm'] = (data['ercot_wind_speed_mph'] < 7).astype(int)
        data['wind_light'] = ((data['ercot_wind_speed_mph'] >= 7) & (data['ercot_wind_speed_mph'] < 15)).astype(int)
        data['wind_moderate'] = ((data['ercot_wind_speed_mph'] >= 15) & (data['ercot_wind_speed_mph'] < 25)).astype(int)
        data['wind_strong'] = ((data['ercot_wind_speed_mph'] >= 25) & (data['ercot_wind_speed_mph'] < 40)).astype(int)
        data['wind_very_strong'] = (data['ercot_wind_speed_mph'] >= 40).astype(int)
        
        return data
    
    def _create_comfort_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comfort indices that affect energy demand."""
        if 'ercot_temperature_f' not in data.columns or 'ercot_humidity' not in data.columns:
            return data
        
        # Heat Index (apparent temperature)
        temp_f = data['ercot_temperature_f']
        humidity = data['ercot_humidity']
        
        # Simplified heat index calculation
        data['heat_index'] = np.where(
            temp_f >= 80,
            -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity 
            - 0.22475541 * temp_f * humidity - 6.83783e-3 * temp_f**2
            - 5.481717e-2 * humidity**2 + 1.22874e-3 * temp_f**2 * humidity
            + 8.5282e-4 * temp_f * humidity**2 - 1.99e-6 * temp_f**2 * humidity**2,
            temp_f
        )
        
        # Comfort categories
        data['heat_stress'] = (data['heat_index'] > 90).astype(int)
        data['extreme_heat_stress'] = (data['heat_index'] > 105).astype(int)
        
        return data
    
    def _create_seasonal_weather_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal weather-based features."""
        if 'timestamp' not in data.columns:
            return data
        
        # Extract time components
        data['month'] = data['timestamp'].dt.month
        data['hour'] = data['timestamp'].dt.hour
        
        # Seasonal temperature expectations (Texas-specific)
        seasonal_temp_normal = {
            1: 50, 2: 55, 3: 65, 4: 75, 5: 82, 6: 88,
            7: 92, 8: 92, 9: 87, 10: 78, 11: 65, 12: 52
        }
        
        if 'ercot_temperature_f' in data.columns:
            data['temp_vs_seasonal_normal'] = data.apply(
                lambda row: row['ercot_temperature_f'] - seasonal_temp_normal.get(row['month'], 70),
                axis=1
            )
        
        # Peak cooling/heating periods
        data['peak_cooling_season'] = ((data['month'] >= 6) & (data['month'] <= 9)).astype(int)
        data['peak_heating_season'] = ((data['month'] <= 2) | (data['month'] == 12)).astype(int)
        
        return data
    
    def get_weather_summary(self, weather_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for weather data."""
        if weather_data.empty:
            return {'error': 'No weather data available'}
        
        summary = {
            'total_records': len(weather_data),
            'date_range': {
                'start': weather_data['timestamp'].min().isoformat() if 'timestamp' in weather_data.columns else None,
                'end': weather_data['timestamp'].max().isoformat() if 'timestamp' in weather_data.columns else None
            },
            'temperature_stats': {},
            'weather_features': list(weather_data.columns)
        }
        
        # Temperature statistics - check for both column name formats
        temp_column = None
        if 'ercot_temperature_f' in weather_data.columns:
            temp_column = 'ercot_temperature_f'
        elif 'temperature_f' in weather_data.columns:
            temp_column = 'temperature_f'
        
        if temp_column:
            temp_data = weather_data[temp_column].dropna()
            if len(temp_data) > 0:
                summary['temperature_stats'] = {
                    'mean': float(temp_data.mean()),
                    'min': float(temp_data.min()),
                    'max': float(temp_data.max()),
                    'std': float(temp_data.std())
                }
        
        return summary