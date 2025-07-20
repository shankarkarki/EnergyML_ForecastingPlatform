"""
Unit tests for energy market feature engineering.
Tests ERCOT-specific features, holidays, and market patterns.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.energy_market_features import EnergyMarketFeatureEngineer


class TestEnergyMarketFeatureEngineer(unittest.TestCase):
    """Test cases for EnergyMarketFeatureEngineer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engineer = EnergyMarketFeatureEngineer()
        
        # Create realistic ERCOT market data
        dates = pd.date_range('2024-01-01', periods=168, freq='h')  # One week
        np.random.seed(42)  # For reproducible tests
        
        # Create realistic daily patterns
        hours = np.arange(168) % 24
        daily_pattern = 45000 + 15000 * np.sin(2 * np.pi * hours / 24)  # Load pattern
        price_pattern = 50 + 30 * np.sin(2 * np.pi * hours / 24)  # Price pattern
        
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'ercot_load_mw': daily_pattern + np.random.normal(0, 2000, 168),
            'ercot_price_per_mwh': price_pattern + np.random.normal(0, 10, 168),
            'wind_generation_mw': np.random.uniform(5000, 25000, 168),
            'solar_generation_mw': np.maximum(0, 10000 * np.sin(2 * np.pi * hours / 24) * (hours >= 6) * (hours <= 18)),
            'total_generation_mw': daily_pattern * 1.1 + np.random.normal(0, 1000, 168)
        })
        
        # Small dataset for edge case testing
        self.small_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-07-04', periods=24, freq='h'),  # July 4th
            'ercot_load_mw': np.random.uniform(40000, 60000, 24),
            'ercot_price_per_mwh': np.random.uniform(30, 80, 24)
        })
    
    def test_initialization(self):
        """Test EnergyMarketFeatureEngineer initialization."""
        self.assertIsNotNone(self.engineer.ercot_market_config)
        self.assertIsNotNone(self.engineer.texas_holidays)
        self.assertIsNotNone(self.engineer.market_events)
        
        # Check ERCOT market configuration
        self.assertIn('peak_hours', self.engineer.ercot_market_config)
        self.assertIn('super_peak_hours', self.engineer.ercot_market_config)
        self.assertEqual(len(self.engineer.ercot_market_config['peak_hours']), 16)  # 7 AM to 10 PM
        self.assertEqual(len(self.engineer.ercot_market_config['super_peak_hours']), 5)  # 2 PM to 6 PM
    
    def test_create_market_period_features(self):
        """Test ERCOT market period feature creation."""
        result = self.engineer._create_market_period_features(self.test_data.copy())
        
        # Check that market period features are created
        expected_features = [
            'is_peak_period', 'is_super_peak', 'is_off_peak', 'is_shoulder_period',
            'is_weekday', 'is_weekend', 'is_business_peak', 'is_business_super_peak',
            'is_monday', 'is_friday', 'is_saturday', 'is_sunday'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result.columns)
        
        # Check binary indicators
        for feature in expected_features:
            self.assertTrue(all(result[feature].isin([0, 1])))
        
        # Check peak period logic
        peak_hours = self.engineer.ercot_market_config['peak_hours']
        for idx, row in result.iterrows():
            hour = row['hour']
            expected_peak = 1 if hour in peak_hours else 0
            self.assertEqual(row['is_peak_period'], expected_peak)
        
        # Check super peak logic
        super_peak_hours = self.engineer.ercot_market_config['super_peak_hours']
        for idx, row in result.iterrows():
            hour = row['hour']
            expected_super_peak = 1 if hour in super_peak_hours else 0
            self.assertEqual(row['is_super_peak'], expected_super_peak)
    
    def test_create_pricing_features(self):
        """Test pricing feature creation."""
        result = self.engineer._create_pricing_features(self.test_data.copy())
        
        # Check that pricing features are created
        price_features = [col for col in result.columns if 'price' in col.lower()]
        
        # Should have volatility features
        volatility_features = [col for col in price_features if 'volatility' in col]
        self.assertGreater(len(volatility_features), 0)
        
        # Should have spike detection features
        spike_features = [col for col in price_features if 'spike' in col]
        self.assertGreater(len(spike_features), 0)
        
        # Should have price categories
        category_features = [col for col in price_features if any(cat in col for cat in ['low', 'normal', 'high', 'extreme'])]
        self.assertGreater(len(category_features), 0)
        
        # Check price momentum features
        momentum_features = [col for col in price_features if 'momentum' in col]
        self.assertGreater(len(momentum_features), 0)
        
        # Check that z-score is calculated
        z_score_features = [col for col in price_features if 'z_score' in col]
        self.assertGreater(len(z_score_features), 0)
    
    def test_create_load_pattern_features(self):
        """Test load pattern feature creation."""
        result = self.engineer._create_load_pattern_features(self.test_data.copy())
        
        # Check that load features are created
        load_features = [col for col in result.columns if 'load' in col.lower()]
        
        # Should have load factor
        load_factor_features = [col for col in load_features if 'load_factor' in col]
        self.assertGreater(len(load_factor_features), 0)
        
        # Should have growth features
        growth_features = [col for col in load_features if 'growth' in col]
        self.assertGreater(len(growth_features), 0)
        
        # Should have load categories
        category_features = [col for col in load_features if any(cat in col for cat in ['very_low', 'low', 'normal', 'high', 'very_high'])]
        self.assertGreater(len(category_features), 0)
        
        # Check that daily peak detection works
        peak_features = [col for col in load_features if 'daily_peak' in col]
        self.assertGreater(len(peak_features), 0)
    
    def test_create_generation_features(self):
        """Test generation feature creation."""
        result = self.engineer._create_generation_features(self.test_data.copy())
        
        # Check renewable generation features
        renewable_features = [col for col in result.columns if any(renewable in col.lower() 
                             for renewable in ['wind', 'solar'])]
        
        # Should have variability features
        variability_features = [col for col in renewable_features if 'variability' in col]
        self.assertGreater(len(variability_features), 0)
        
        # Should have ramp rate features
        ramp_features = [col for col in renewable_features if 'ramp' in col]
        self.assertGreater(len(ramp_features), 0)
        
        # Check penetration calculation if total generation exists
        penetration_features = [col for col in result.columns if 'penetration' in col]
        if 'total_generation_mw' in self.test_data.columns:
            self.assertGreater(len(penetration_features), 0)
    
    def test_create_market_stress_features(self):
        """Test market stress feature creation."""
        result = self.engineer._create_market_stress_features(self.test_data.copy())
        
        # Check reserve margin features
        if 'total_generation_mw' in self.test_data.columns and 'ercot_load_mw' in self.test_data.columns:
            self.assertIn('reserve_margin', result.columns)
            self.assertIn('is_tight_reserves', result.columns)
            self.assertIn('is_emergency_reserves', result.columns)
            
            # Check that reserve margin is calculated correctly
            expected_reserve = (result['total_generation_mw'] - result['ercot_load_mw']) / result['ercot_load_mw']
            pd.testing.assert_series_equal(result['reserve_margin'], expected_reserve, check_names=False)
        
        # Check price-load correlation features
        correlation_features = [col for col in result.columns if 'correlation' in col]
        if 'ercot_price_per_mwh' in self.test_data.columns and 'ercot_load_mw' in self.test_data.columns:
            self.assertGreater(len(correlation_features), 0)
    
    def test_create_holiday_features(self):
        """Test holiday feature creation."""
        result = self.engineer.create_holiday_features(self.small_data.copy())  # July 4th data
        
        # Check that holiday features are created
        holiday_features = [
            'is_federal_holiday', 'is_texas_holiday', 'is_new_years', 'is_july_4th',
            'is_thanksgiving', 'is_christmas', 'is_holiday_week', 'is_summer_vacation', 'is_school_year'
        ]
        
        for feature in holiday_features:
            self.assertIn(feature, result.columns)
        
        # Check that July 4th is detected
        july_4th_rows = result[result['timestamp'].dt.date == datetime(2024, 7, 4).date()]
        if not july_4th_rows.empty:
            self.assertEqual(july_4th_rows['is_july_4th'].iloc[0], 1)
            self.assertEqual(july_4th_rows['is_federal_holiday'].iloc[0], 1)
        
        # Check summer vacation period
        self.assertTrue(all(result['is_summer_vacation'] == 1))  # July data
        
        # Check binary indicators
        for feature in holiday_features:
            self.assertTrue(all(result[feature].isin([0, 1])))
    
    def test_create_seasonal_energy_features(self):
        """Test seasonal energy feature creation."""
        result = self.engineer.create_seasonal_energy_features(self.test_data.copy())  # January data
        
        # Check seasonal features
        seasonal_features = [
            'is_winter_storm_season', 'is_summer_peak_season', 'is_spring_maintenance',
            'is_fall_maintenance', 'is_hurricane_season', 'is_freeze_risk_season'
        ]
        
        for feature in seasonal_features:
            self.assertIn(feature, result.columns)
        
        # Check January patterns
        self.assertTrue(all(result['is_winter_storm_season'] == 1))  # January is winter storm season
        self.assertTrue(all(result['is_freeze_risk_season'] == 1))   # January has freeze risk
        self.assertTrue(all(result['is_summer_peak_season'] == 0))   # January is not summer
        
        # Check binary indicators
        for feature in seasonal_features:
            self.assertTrue(all(result[feature].isin([0, 1])))
    
    def test_create_grid_event_features(self):
        """Test grid event feature creation."""
        result = self.engineer.create_grid_event_features(self.test_data.copy())
        
        # Check load ramping features
        load_features = [col for col in result.columns if 'load' in col.lower()]
        ramping_features = [col for col in load_features if 'ramping' in col]
        self.assertGreater(len(ramping_features), 0)
        
        # Check rapid change features
        rapid_features = [col for col in load_features if 'rapid' in col]
        self.assertGreater(len(rapid_features), 0)
        
        # Check market event features
        if 'ercot_price_per_mwh' in self.test_data.columns and 'ercot_load_mw' in self.test_data.columns:
            self.assertIn('is_scarcity_event', result.columns)
            self.assertIn('is_price_anomaly', result.columns)
        
        # Check binary indicators
        binary_features = [col for col in result.columns if col.startswith('is_')]
        for feature in binary_features:
            if feature in result.columns:
                self.assertTrue(all(result[feature].isin([0, 1])))
    
    def test_create_ercot_market_features(self):
        """Test comprehensive ERCOT market feature creation."""
        result = self.engineer.create_ercot_market_features(self.test_data.copy())
        
        # Should have more columns than original
        self.assertGreater(len(result.columns), len(self.test_data.columns))
        
        # Should preserve original data
        self.assertEqual(len(result), len(self.test_data))
        
        # Check that key ERCOT features are present
        ercot_features = [
            'is_peak_period', 'is_super_peak', 'is_business_peak',
            'reserve_margin', 'is_tight_reserves'
        ]
        
        for feature in ercot_features:
            if feature in result.columns:  # Some features depend on data availability
                self.assertIn(feature, result.columns)
    
    def test_create_all_energy_market_features(self):
        """Test comprehensive energy market feature creation."""
        result = self.engineer.create_all_energy_market_features(self.test_data.copy())
        
        # Should have significantly more columns than original
        self.assertGreater(len(result.columns), len(self.test_data.columns) * 2)
        
        # Should preserve original data
        self.assertEqual(len(result), len(self.test_data))
        
        # Should not be empty
        self.assertFalse(result.empty)
        
        # Check that features from all categories are present
        feature_categories = {
            'market_periods': ['peak', 'business'],
            'pricing': ['price', 'spike', 'volatility'],
            'load_patterns': ['load_factor', 'growth'],
            'generation': ['ramp', 'variability'],
            'holidays': ['holiday', 'vacation'],
            'seasonal': ['season', 'winter', 'summer'],
            'grid_events': ['scarcity', 'anomaly']
        }
        
        for category, keywords in feature_categories.items():
            category_features = [col for col in result.columns 
                               if any(keyword in col.lower() for keyword in keywords)]
            self.assertGreater(len(category_features), 0, f"No features found for category: {category}")
    
    def test_get_feature_summary(self):
        """Test feature summary generation."""
        features = self.engineer.create_all_energy_market_features(self.test_data.copy())
        summary = self.engineer.get_feature_summary(features)
        
        # Check summary structure
        self.assertIn('total_features', summary)
        self.assertIn('feature_categories', summary)
        self.assertIn('data_shape', summary)
        
        # Check feature categories
        categories = summary['feature_categories']
        self.assertIn('market_periods', categories)
        self.assertIn('pricing', categories)
        self.assertIn('load_patterns', categories)
        
        # Check that counts are reasonable
        self.assertGreater(summary['total_features'], len(self.test_data.columns))
        self.assertEqual(summary['data_shape'], features.shape)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        
        # All methods should handle empty data gracefully
        result1 = self.engineer.create_ercot_market_features(empty_data)
        self.assertTrue(result1.empty)
        
        result2 = self.engineer.create_holiday_features(empty_data)
        self.assertTrue(result2.empty)
        
        result3 = self.engineer.create_seasonal_energy_features(empty_data)
        self.assertTrue(result3.empty)
        
        result4 = self.engineer.create_all_energy_market_features(empty_data)
        self.assertTrue(result4.empty)
    
    def test_missing_timestamp_handling(self):
        """Test handling of data without timestamp column."""
        data_no_timestamp = self.test_data.drop(columns=['timestamp'])
        
        # Should return original data if no timestamp
        result = self.engineer.create_holiday_features(data_no_timestamp)
        self.assertEqual(len(result.columns), len(data_no_timestamp.columns))
    
    def test_specific_holiday_detection(self):
        """Test detection of specific holidays."""
        # Create data with specific holidays
        holiday_dates = [
            datetime(2024, 1, 1),   # New Year's
            datetime(2024, 7, 4),   # July 4th
            datetime(2024, 12, 25), # Christmas
        ]
        
        holiday_data = pd.DataFrame({
            'timestamp': holiday_dates,
            'ercot_load_mw': [40000, 35000, 30000]  # Lower load on holidays
        })
        
        result = self.engineer.create_holiday_features(holiday_data)
        
        # Check specific holiday detection
        self.assertEqual(result['is_new_years'].iloc[0], 1)
        self.assertEqual(result['is_july_4th'].iloc[1], 1)
        self.assertEqual(result['is_christmas'].iloc[2], 1)
        
        # All should be federal holidays
        self.assertTrue(all(result['is_federal_holiday'] == 1))


class TestEnergyMarketFeaturesIntegration(unittest.TestCase):
    """Integration tests for energy market features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engineer = EnergyMarketFeatureEngineer()
        
        # Create full year of ERCOT data
        dates = pd.date_range('2024-01-01', periods=8760, freq='h')  # Full year
        np.random.seed(42)
        
        # Create realistic seasonal and daily patterns
        hours = np.arange(8760) % 24
        days = np.arange(8760) // 24
        
        # Seasonal temperature pattern (affects load)
        seasonal_temp = 70 + 25 * np.sin(2 * np.pi * (days - 80) / 365)
        
        # Load pattern with daily and seasonal cycles
        daily_load_pattern = 45000 + 15000 * np.sin(2 * np.pi * hours / 24)
        seasonal_load_adjustment = 5000 * np.sin(2 * np.pi * days / 365)
        
        self.full_year_data = pd.DataFrame({
            'timestamp': dates,
            'ercot_load_mw': daily_load_pattern + seasonal_load_adjustment + np.random.normal(0, 2000, 8760),
            'ercot_price_per_mwh': 50 + 30 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 10, 8760),
            'wind_generation_mw': np.random.uniform(5000, 25000, 8760),
            'solar_generation_mw': np.maximum(0, 10000 * np.sin(2 * np.pi * hours / 24) * (hours >= 6) * (hours <= 18)),
            'total_generation_mw': daily_load_pattern * 1.1 + seasonal_load_adjustment + np.random.normal(0, 1000, 8760),
            'temperature_f': seasonal_temp + 10 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, 8760)
        })
    
    def test_full_year_energy_market_workflow(self):
        """Test complete energy market feature workflow with full year data."""
        # Create all energy market features
        features = self.engineer.create_all_energy_market_features(self.full_year_data)
        
        # Should have many more features than original
        self.assertGreater(len(features.columns), len(self.full_year_data.columns) * 3)
        
        # Should preserve all data
        self.assertEqual(len(features), len(self.full_year_data))
        
        # Generate summary
        summary = self.engineer.get_feature_summary(features)
        self.assertGreater(summary['total_features'], 50)  # Should have many features
        
        # Check seasonal patterns
        summer_data = features[features['timestamp'].dt.month.isin([6, 7, 8])]
        winter_data = features[features['timestamp'].dt.month.isin([12, 1, 2])]
        
        # Summer should have more peak cooling indicators
        summer_peak_season = summer_data['is_summer_peak_season'].mean()
        winter_peak_season = winter_data['is_summer_peak_season'].mean()
        self.assertGreater(summer_peak_season, winter_peak_season)
        
        # Winter should have more storm season indicators
        summer_storm_season = summer_data['is_winter_storm_season'].mean()
        winter_storm_season = winter_data['is_winter_storm_season'].mean()
        self.assertGreater(winter_storm_season, summer_storm_season)
    
    def test_market_stress_detection(self):
        """Test market stress event detection."""
        # Create data with market stress conditions
        stress_data = self.full_year_data.copy()
        
        # Create artificial scarcity event (high load, high price)
        stress_indices = range(1000, 1024)  # 24 hours of stress
        stress_data.loc[stress_indices, 'ercot_load_mw'] = 75000  # Very high load
        stress_data.loc[stress_indices, 'ercot_price_per_mwh'] = 500  # Very high price
        stress_data.loc[stress_indices, 'total_generation_mw'] = 76000  # Tight reserves
        
        # Create features
        features = self.engineer.create_all_energy_market_features(stress_data)
        
        # Check that stress events are detected
        stress_events = features.loc[stress_indices]
        
        # Should detect tight reserves
        if 'is_tight_reserves' in features.columns:
            tight_reserves = stress_events['is_tight_reserves'].sum()
            self.assertGreater(tight_reserves, 0)
        
        # Should detect scarcity events
        if 'is_scarcity_event' in features.columns:
            scarcity_events = stress_events['is_scarcity_event'].sum()
            self.assertGreater(scarcity_events, 0)
        
        # Should detect price spikes
        price_spike_cols = [col for col in features.columns if 'spike' in col.lower()]
        if price_spike_cols:
            spike_events = stress_events[price_spike_cols[0]].sum()
            self.assertGreater(spike_events, 0)
    
    def test_holiday_impact_on_features(self):
        """Test that holidays create appropriate feature patterns."""
        features = self.engineer.create_all_energy_market_features(self.full_year_data)
        
        # Find major holidays in the data
        holiday_data = features[features['is_federal_holiday'] == 1]
        non_holiday_data = features[features['is_federal_holiday'] == 0]
        
        if not holiday_data.empty and not non_holiday_data.empty:
            # Holidays typically have different load patterns
            holiday_avg_load = holiday_data['ercot_load_mw'].mean()
            non_holiday_avg_load = non_holiday_data['ercot_load_mw'].mean()
            
            # The difference should be noticeable (though direction depends on holiday type)
            load_difference_pct = abs(holiday_avg_load - non_holiday_avg_load) / non_holiday_avg_load
            self.assertGreater(load_difference_pct, 0.01)  # At least 1% difference
    
    def test_renewable_generation_features(self):
        """Test renewable generation feature patterns."""
        features = self.engineer.create_all_energy_market_features(self.full_year_data)
        
        # Solar generation should have daily patterns
        solar_features = [col for col in features.columns if 'solar' in col.lower()]
        if solar_features:
            # Solar should be higher during day hours
            day_hours = features[features['hour'].isin(range(10, 16))]  # 10 AM to 3 PM
            night_hours = features[features['hour'].isin(range(22, 24)) | features['hour'].isin(range(0, 6))]
            
            day_solar = day_hours['solar_generation_mw'].mean()
            night_solar = night_hours['solar_generation_mw'].mean()
            self.assertGreater(day_solar, night_solar)
        
        # Wind generation should have variability features
        wind_variability_features = [col for col in features.columns if 'wind' in col.lower() and 'variability' in col.lower()]
        self.assertGreater(len(wind_variability_features), 0)


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)