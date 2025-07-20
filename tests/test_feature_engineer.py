"""
Unit tests for feature engineering functionality.
Tests time series feature creation, normalization, and selection.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.feature_engineer import TimeSeriesFeatureEngineer


class TestTimeSeriesFeatureEngineer(unittest.TestCase):
    """Test cases for TimeSeriesFeatureEngineer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engineer = TimeSeriesFeatureEngineer()
        
        # Create realistic energy market data
        dates = pd.date_range('2024-01-01', periods=168, freq='h')  # One week
        np.random.seed(42)  # For reproducible tests
        
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'load_mw': np.random.uniform(30000, 70000, 168) + 
                      5000 * np.sin(2 * np.pi * np.arange(168) / 24),  # Daily pattern
            'price_per_mwh': np.random.uniform(20, 100, 168) + 
                           20 * np.sin(2 * np.pi * np.arange(168) / 24),  # Daily pattern
            'generation_mw': np.random.uniform(25000, 65000, 168),
            'temperature': np.random.uniform(40, 90, 168),
            'region': ['ERCOT'] * 168
        })
        
        # Small dataset for testing edge cases
        self.small_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
            'load_mw': [100, 200, 300, 400, 500],
            'price_per_mwh': [20, 30, 40, 50, 60]
        })
    
    def test_find_timestamp_column(self):
        """Test timestamp column detection."""
        # Test with 'timestamp' column
        timestamp_col = self.engineer._find_timestamp_column(self.test_data)
        self.assertEqual(timestamp_col, 'timestamp')
        
        # Test with different timestamp column name
        data_with_time = self.test_data.rename(columns={'timestamp': 'datetime'})
        timestamp_col = self.engineer._find_timestamp_column(data_with_time)
        self.assertEqual(timestamp_col, 'datetime')
        
        # Test with no timestamp column
        data_no_timestamp = self.test_data.drop(columns=['timestamp'])
        timestamp_col = self.engineer._find_timestamp_column(data_no_timestamp)
        self.assertIsNone(timestamp_col)
    
    def test_create_time_features(self):
        """Test time-based feature creation."""
        features = self.engineer._create_time_features(self.test_data, 'timestamp')
        
        # Check that time features are created
        expected_time_features = [
            'hour', 'day_of_week', 'day_of_month', 'day_of_year', 
            'week_of_year', 'month', 'quarter', 'year',
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos',
            'is_weekend', 'is_business_day', 'is_morning', 'is_afternoon',
            'is_evening', 'is_night', 'is_peak_hour', 'is_super_peak'
        ]
        
        for feature in expected_time_features:
            self.assertIn(feature, features.columns)
        
        # Check value ranges
        self.assertTrue(all(features['hour'].between(0, 23)))
        self.assertTrue(all(features['day_of_week'].between(0, 6)))
        self.assertTrue(all(features['month'].between(1, 12)))
        
        # Check cyclical encoding
        self.assertTrue(all(features['hour_sin'].between(-1, 1)))
        self.assertTrue(all(features['hour_cos'].between(-1, 1)))
        
        # Check binary indicators
        self.assertTrue(all(features['is_weekend'].isin([0, 1])))
        self.assertTrue(all(features['is_business_day'].isin([0, 1])))
    
    def test_create_lag_features(self):
        """Test lag feature creation."""
        target_columns = ['load_mw', 'price_per_mwh']
        lag_periods = [1, 2, 3]
        
        features = self.engineer._create_lag_features(
            self.test_data, target_columns, lag_periods
        )
        
        # Check that lag features are created
        for col in target_columns:
            for lag in lag_periods:
                lag_col = f'{col}_lag_{lag}'
                self.assertIn(lag_col, features.columns)
        
        # Check lag values
        self.assertEqual(features['load_mw_lag_1'].iloc[1], features['load_mw'].iloc[0])
        self.assertEqual(features['load_mw_lag_2'].iloc[2], features['load_mw'].iloc[0])
        
        # Check NaN values at the beginning
        self.assertTrue(pd.isna(features['load_mw_lag_1'].iloc[0]))
        self.assertTrue(pd.isna(features['load_mw_lag_2'].iloc[1]))
    
    def test_create_rolling_features(self):
        """Test rolling window feature creation."""
        target_columns = ['load_mw']
        rolling_windows = [3, 6]
        
        features = self.engineer._create_rolling_features(
            self.test_data, target_columns, rolling_windows
        )
        
        # Check that rolling features are created
        for col in target_columns:
            for window in rolling_windows:
                expected_features = [
                    f'{col}_rolling_mean_{window}',
                    f'{col}_rolling_std_{window}',
                    f'{col}_rolling_min_{window}',
                    f'{col}_rolling_max_{window}',
                    f'{col}_rolling_median_{window}',
                    f'{col}_rolling_q25_{window}',
                    f'{col}_rolling_q75_{window}',
                    f'{col}_rolling_diff_{window}'
                ]
                
                for feature in expected_features:
                    self.assertIn(feature, features.columns)
        
        # Check rolling mean calculation
        window = 3
        expected_mean = features['load_mw'].iloc[:3].mean()
        actual_mean = features['load_mw_rolling_mean_3'].iloc[2]
        self.assertAlmostEqual(expected_mean, actual_mean, places=5)
    
    def test_create_statistical_features(self):
        """Test statistical feature creation."""
        target_columns = ['load_mw', 'price_per_mwh']
        
        features = self.engineer._create_statistical_features(
            self.test_data, target_columns
        )
        
        # Check that statistical features are created
        for col in target_columns:
            expected_features = [
                f'{col}_pct_change',
                f'{col}_diff',
                f'{col}_ema_12',
                f'{col}_ema_24',
                f'{col}_volatility_12',
                f'{col}_volatility_24',
                f'{col}_zscore'
            ]
            
            for feature in expected_features:
                self.assertIn(feature, features.columns)
        
        # Check percentage change calculation
        expected_pct_change = (features['load_mw'].iloc[1] - features['load_mw'].iloc[0]) / features['load_mw'].iloc[0]
        actual_pct_change = features['load_mw_pct_change'].iloc[1]
        self.assertAlmostEqual(expected_pct_change, actual_pct_change, places=5)
    
    def test_create_seasonal_features(self):
        """Test seasonal feature creation."""
        features = self.engineer._create_seasonal_features(self.test_data, 'timestamp')
        
        # Check that seasonal features are created
        expected_seasonal_features = [
            'is_spring', 'is_summer', 'is_fall', 'is_winter',
            'is_holiday', 'is_heating_season', 'is_cooling_season'
        ]
        
        for feature in expected_seasonal_features:
            self.assertIn(feature, features.columns)
        
        # Check binary indicators
        for feature in expected_seasonal_features:
            self.assertTrue(all(features[feature].isin([0, 1])))
        
        # Check that exactly one season is active (for January data)
        season_sum = (features['is_spring'] + features['is_summer'] + 
                     features['is_fall'] + features['is_winter'])
        self.assertTrue(all(season_sum == 1))
    
    def test_create_features_full_pipeline(self):
        """Test the complete feature creation pipeline."""
        original_columns = len(self.test_data.columns)
        
        features = self.engineer.create_features(self.test_data)
        
        # Should have more columns than original
        self.assertGreater(len(features.columns), original_columns)
        
        # Should have fewer rows due to NaN removal
        self.assertLessEqual(len(features), len(self.test_data))
        
        # Should not be empty
        self.assertFalse(features.empty)
        
        # Check that original columns are preserved
        for col in self.test_data.columns:
            if col != 'timestamp':  # timestamp might be modified
                self.assertIn(col, features.columns)
    
    def test_create_features_with_config(self):
        """Test feature creation with custom configuration."""
        config = {
            'time_features': True,
            'lag_features': True,
            'rolling_features': False,  # Disable rolling features
            'statistical_features': False,  # Disable statistical features
            'seasonal_features': True,
            'lag_periods': [1, 2],  # Only short lags
            'target_columns': ['load_mw']  # Only one target
        }
        
        features = self.engineer.create_features(self.test_data, config)
        
        # Should have time and seasonal features
        self.assertIn('hour', features.columns)
        self.assertIn('is_winter', features.columns)
        
        # Should have lag features
        self.assertIn('load_mw_lag_1', features.columns)
        self.assertIn('load_mw_lag_2', features.columns)
        
        # Should not have rolling features
        rolling_features = [col for col in features.columns if 'rolling' in col]
        self.assertEqual(len(rolling_features), 0)
        
        # Should not have statistical features
        statistical_features = [col for col in features.columns if any(stat in col for stat in ['pct_change', 'ema', 'volatility'])]
        self.assertEqual(len(statistical_features), 0)
    
    def test_normalize_features(self):
        """Test feature normalization."""
        # Create features first
        features = self.engineer.create_features(self.small_data)
        
        # Test standard normalization
        normalized_features, scaler = self.engineer.normalize_features(features, 'standard')
        
        self.assertIsNotNone(scaler)
        self.assertEqual(len(normalized_features), len(features))
        
        # Just verify that normalization was applied (some columns should be different)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        # Find at least one column that has variation and was normalized
        found_normalized_column = False
        for col in numeric_cols:
            if (not features[col].isna().all() and 
                not normalized_features[col].isna().all() and 
                features[col].std() > 0):  # Has variation in original data
                
                # Check if this column was actually normalized (values changed)
                if not features[col].equals(normalized_features[col]):
                    found_normalized_column = True
                    break
        
        # At least one column should have been normalized
        self.assertTrue(found_normalized_column, "No columns were normalized")
    
    def test_normalize_features_different_scalers(self):
        """Test different normalization methods."""
        features = self.engineer.create_features(self.small_data)
        
        # Test MinMax scaler
        normalized_minmax, scaler_minmax = self.engineer.normalize_features(features, 'minmax')
        self.assertIsNotNone(scaler_minmax)
        
        # Test Robust scaler
        normalized_robust, scaler_robust = self.engineer.normalize_features(features, 'robust')
        self.assertIsNotNone(scaler_robust)
        
        # Test invalid scaler
        normalized_invalid, scaler_invalid = self.engineer.normalize_features(features, 'invalid')
        self.assertIsNone(scaler_invalid)
    
    def test_select_features(self):
        """Test feature selection."""
        # Create features
        features = self.engineer.create_features(self.test_data)
        
        # Create target variable
        target = features['load_mw'].shift(-1).dropna()  # Next hour load as target
        features_for_selection = features.iloc[:-1]  # Remove last row to match target
        
        # Test mutual information selection
        selected_features, selected_names = self.engineer.select_features(
            features_for_selection, target, method='mutual_info', k=10
        )
        
        self.assertLessEqual(len(selected_names), 10)
        self.assertGreater(len(selected_names), 0)
        
        # Test f_regression selection
        selected_features_f, selected_names_f = self.engineer.select_features(
            features_for_selection, target, method='f_regression', k=5
        )
        
        self.assertLessEqual(len(selected_names_f), 5)
        self.assertGreater(len(selected_names_f), 0)
        
        # Check feature importance is populated
        importance = self.engineer.get_feature_importance()
        self.assertGreater(len(importance), 0)
    
    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        feature_pairs = [('load_mw', 'temperature'), ('price_per_mwh', 'load_mw')]
        
        features_with_interactions = self.engineer.create_interaction_features(
            self.test_data, feature_pairs
        )
        
        # Check that interaction features are created
        self.assertIn('load_mw_x_temperature', features_with_interactions.columns)
        self.assertIn('load_mw_div_temperature', features_with_interactions.columns)
        self.assertIn('price_per_mwh_x_load_mw', features_with_interactions.columns)
        self.assertIn('price_per_mwh_div_load_mw', features_with_interactions.columns)
        
        # Check interaction calculation
        expected_interaction = self.test_data['load_mw'] * self.test_data['temperature']
        actual_interaction = features_with_interactions['load_mw_x_temperature']
        pd.testing.assert_series_equal(expected_interaction, actual_interaction, check_names=False)
    
    def test_create_polynomial_features(self):
        """Test polynomial feature creation."""
        columns = ['load_mw', 'price_per_mwh']
        degree = 3
        
        features_with_poly = self.engineer.create_polynomial_features(
            self.test_data, columns, degree
        )
        
        # Check that polynomial features are created
        for col in columns:
            for d in range(2, degree + 1):
                poly_col = f'{col}_poly_{d}'
                self.assertIn(poly_col, features_with_poly.columns)
        
        # Check polynomial calculation
        expected_poly = self.test_data['load_mw'] ** 2
        actual_poly = features_with_poly['load_mw_poly_2']
        pd.testing.assert_series_equal(expected_poly, actual_poly, check_names=False)
    
    def test_get_feature_summary(self):
        """Test feature summary generation."""
        features = self.engineer.create_features(self.test_data)
        summary = self.engineer.get_feature_summary(features)
        
        # Check summary structure
        self.assertIn('total_features', summary)
        self.assertIn('numeric_features', summary)
        self.assertIn('categorical_features', summary)
        self.assertIn('datetime_features', summary)
        self.assertIn('feature_types', summary)
        self.assertIn('missing_values', summary)
        self.assertIn('data_shape', summary)
        
        # Check values
        self.assertEqual(summary['total_features'], len(features.columns))
        self.assertEqual(summary['data_shape'], features.shape)
        self.assertGreater(summary['numeric_features'], 0)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        
        features = self.engineer.create_features(empty_data)
        self.assertTrue(features.empty)
        
        # Test normalization with empty data
        normalized, scaler = self.engineer.normalize_features(empty_data)
        self.assertTrue(normalized.empty)
        self.assertIsNone(scaler)
    
    def test_missing_timestamp_handling(self):
        """Test handling of data without timestamp column."""
        data_no_timestamp = self.test_data.drop(columns=['timestamp'])
        
        # Should return original data if no timestamp found
        features = self.engineer.create_features(data_no_timestamp)
        self.assertEqual(len(features.columns), len(data_no_timestamp.columns))


class TestFeatureEngineerIntegration(unittest.TestCase):
    """Integration tests for feature engineering system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engineer = TimeSeriesFeatureEngineer()
        
        # Create realistic ERCOT-like data
        dates = pd.date_range('2024-01-01', periods=720, freq='h')  # 30 days
        np.random.seed(42)
        
        # Create realistic patterns
        hours = np.arange(720) % 24
        days = np.arange(720) // 24
        
        # Base load with daily and weekly patterns
        base_load = 45000 + 15000 * np.sin(2 * np.pi * hours / 24)  # Daily pattern
        weekly_pattern = 5000 * np.sin(2 * np.pi * days / 7)  # Weekly pattern
        noise = np.random.normal(0, 2000, 720)
        
        self.ercot_data = pd.DataFrame({
            'timestamp': dates,
            'ercot_load_mw': base_load + weekly_pattern + noise,
            'ercot_price_per_mwh': 30 + 20 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, 720),
            'wind_generation_mw': np.random.uniform(5000, 25000, 720),
            'solar_generation_mw': np.maximum(0, 10000 * np.sin(2 * np.pi * hours / 24) * (hours >= 6) * (hours <= 18)),
            'temperature_f': 70 + 20 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 5, 720),
            'region': ['ERCOT'] * 720,
            'data_source': ['GridStatus'] * 720
        })
    
    def test_complete_feature_engineering_workflow(self):
        """Test complete feature engineering workflow with realistic data."""
        # 1. Create features
        features = self.engineer.create_features(self.ercot_data)
        
        # Should have many more features than original
        self.assertGreater(len(features.columns), len(self.ercot_data.columns))
        
        # Should have reasonable number of rows (some lost due to lags/rolling)
        self.assertGreater(len(features), len(self.ercot_data) * 0.7)  # At least 70% retained
        
        # 2. Normalize features
        normalized_features, scaler = self.engineer.normalize_features(features, 'standard')
        self.assertIsNotNone(scaler)
        
        # 3. Select features
        target = features['ercot_load_mw'].shift(-1).dropna()
        features_for_selection = normalized_features.iloc[:-1]
        
        selected_features, selected_names = self.engineer.select_features(
            features_for_selection, target, method='mutual_info', k=20
        )
        
        self.assertEqual(len(selected_names), 20)
        
        # 4. Get feature importance
        importance = self.engineer.get_feature_importance()
        self.assertEqual(len(importance), len(features_for_selection.select_dtypes(include=[np.number]).columns))
        
        # 5. Generate summary
        summary = self.engineer.get_feature_summary(selected_features)
        self.assertGreater(summary['total_features'], 0)
        self.assertGreater(summary['numeric_features'], 0)
    
    def test_energy_specific_feature_patterns(self):
        """Test that energy-specific patterns are captured in features."""
        features = self.engineer.create_features(self.ercot_data)
        
        # Check that peak hour features are reasonable
        peak_hours = features[features['is_peak_hour'] == 1]['hour'].unique()
        self.assertTrue(all(7 <= hour <= 22 for hour in peak_hours))
        
        # Check that weekend patterns are captured
        weekend_data = features[features['is_weekend'] == 1]
        business_data = features[features['is_business_day'] == 1]
        
        self.assertGreater(len(weekend_data), 0)
        self.assertGreater(len(business_data), 0)
        
        # Check seasonal features for January data
        winter_data = features[features['is_winter'] == 1]
        self.assertEqual(len(winter_data), len(features))  # All January data should be winter
        
        # Check that lag features capture temporal dependencies
        lag_correlation = features['ercot_load_mw'].corr(features['ercot_load_mw_lag_1'])
        self.assertGreater(lag_correlation, 0.8)  # Should be highly correlated
    
    def test_feature_engineering_performance(self):
        """Test feature engineering performance with larger dataset."""
        # Create larger dataset
        dates = pd.date_range('2024-01-01', periods=8760, freq='h')  # Full year
        large_data = pd.DataFrame({
            'timestamp': dates,
            'load_mw': np.random.uniform(30000, 70000, 8760),
            'price_per_mwh': np.random.uniform(20, 100, 8760),
            'temperature': np.random.uniform(40, 90, 8760)
        })
        
        # Should complete without errors
        features = self.engineer.create_features(large_data)
        
        # Should have reasonable performance
        self.assertGreater(len(features), 8000)  # Most data retained
        self.assertGreater(len(features.columns), 50)  # Many features created


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)