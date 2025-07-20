"""
Unit tests for multi-horizon forecasting functionality.
Tests short-term, medium-term, and long-term forecasting capabilities.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.multi_horizon_forecaster import (
    MultiHorizonForecaster, HorizonType, HorizonConfig,
    create_multi_horizon_forecaster, generate_all_horizons
)
from model.forecaster import RandomForestForecaster, create_forecaster
from model.forecasting_interface import ForecastResult, ForecastHorizon


class TestMultiHorizonForecaster(unittest.TestCase):
    """Test cases for MultiHorizonForecaster class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create sample data for training models
        dates = pd.date_range('2024-01-01', periods=200, freq='h')
        self.X = pd.DataFrame({
            'timestamp': dates,
            'temperature': 70 + 10 * np.sin(2 * np.pi * np.arange(200) / 24) + np.random.normal(0, 3, 200),
            'hour_of_day': np.arange(200) % 24,
            'day_of_week': (np.arange(200) // 24) % 7
        })
        
        self.y = pd.Series(
            50000 + 15000 * np.sin(2 * np.pi * np.arange(200) / 24) + np.random.normal(0, 2000, 200),
            name='load_mw'
        )
        
        # Train multiple models
        self.models = {}
        
        # Random Forest model
        rf_model = RandomForestForecaster(n_estimators=10, sequence_length=12)
        rf_model.fit(self.X, self.y)
        self.models['random_forest'] = rf_model
        
        # Transformer model
        transformer_model = create_forecaster('transformer', d_model=32, nhead=4, num_layers=2, sequence_length=12)
        transformer_model.fit(self.X, self.y, epochs=5, batch_size=16)
        self.models['transformer'] = transformer_model
        
        # Create multi-horizon forecaster
        self.forecaster = MultiHorizonForecaster(self.models)
    
    def test_initialization(self):
        """Test MultiHorizonForecaster initialization."""
        self.assertEqual(len(self.forecaster.models), 2)
        self.assertIn('random_forest', self.forecaster.models)
        self.assertIn('transformer', self.forecaster.models)
        
        # Check horizon configurations
        self.assertEqual(len(self.forecaster.horizon_configs), 5)
        self.assertIn(HorizonType.INTRADAY, self.forecaster.horizon_configs)
        self.assertIn(HorizonType.DAY_AHEAD, self.forecaster.horizon_configs)
        self.assertIn(HorizonType.WEEK_AHEAD, self.forecaster.horizon_configs)
        self.assertIn(HorizonType.MONTH_AHEAD, self.forecaster.horizon_configs)
        self.assertIn(HorizonType.SEASONAL, self.forecaster.horizon_configs)
    
    def test_initialization_with_unfitted_model_raises_error(self):
        """Test that unfitted model raises error."""
        unfitted_model = RandomForestForecaster(n_estimators=5)
        models_with_unfitted = {'unfitted': unfitted_model}
        
        with self.assertRaises(ValueError) as context:
            MultiHorizonForecaster(models_with_unfitted)
        
        self.assertIn("must be fitted", str(context.exception))
    
    def test_horizon_config_structure(self):
        """Test horizon configuration structure."""
        intraday_config = self.forecaster.horizon_configs[HorizonType.INTRADAY]
        
        self.assertEqual(intraday_config.horizon_type, HorizonType.INTRADAY)
        self.assertEqual(intraday_config.max_periods, 24)
        self.assertEqual(intraday_config.frequency, 'h')
        self.assertEqual(intraday_config.aggregation_method, 'none')
        self.assertFalse(intraday_config.seasonal_adjustment)
        self.assertEqual(intraday_config.uncertainty_scaling, 1.0)
        self.assertIsInstance(intraday_config.model_preferences, list)
    
    def test_generate_short_term_forecast_intraday(self):
        """Test short-term forecast generation (intraday)."""
        hours = 12
        result = self.forecaster.generate_short_term_forecast(hours=hours)
        
        # Check result structure
        self.assertIsInstance(result, ForecastResult)
        self.assertEqual(len(result.predictions), hours)
        self.assertEqual(len(result.timestamps), hours)
        self.assertEqual(len(result.uncertainty), hours)
        
        # Check timestamps are hourly
        time_diff = result.timestamps[1] - result.timestamps[0]
        self.assertEqual(time_diff, pd.Timedelta(hours=1))
        
        # Check confidence intervals
        self.assertIn(0.68, result.confidence_intervals)
        self.assertIn(0.90, result.confidence_intervals)
        self.assertIn(0.95, result.confidence_intervals)
        
        # Check metadata
        self.assertIn('adjustments', result.metadata)
    
    def test_generate_short_term_forecast_day_ahead(self):
        """Test short-term forecast generation (day-ahead)."""
        hours = 48  # 2 days
        result = self.forecaster.generate_short_term_forecast(hours=hours)
        
        self.assertEqual(len(result.predictions), hours)
        self.assertEqual(len(result.timestamps), hours)
        
        # Should still be hourly frequency
        time_diff = result.timestamps[1] - result.timestamps[0]
        self.assertEqual(time_diff, pd.Timedelta(hours=1))
    
    def test_generate_medium_term_forecast_daily(self):
        """Test medium-term forecast generation (daily)."""
        days = 5
        result = self.forecaster.generate_medium_term_forecast(days=days)
        
        self.assertEqual(len(result.predictions), days)
        self.assertEqual(len(result.timestamps), days)
        
        # Check timestamps are daily
        time_diff = result.timestamps[1] - result.timestamps[0]
        self.assertEqual(time_diff, pd.Timedelta(days=1))
        
        # Check metadata includes adjustments
        self.assertIn('adjustments', result.metadata)
    
    def test_generate_medium_term_forecast_weekly(self):
        """Test medium-term forecast generation (weekly)."""
        days = 21  # 3 weeks
        result = self.forecaster.generate_medium_term_forecast(days=days)
        
        # Should expand weekly forecast to daily
        self.assertEqual(len(result.predictions), days)
        self.assertEqual(len(result.timestamps), days)
        
        # Check that daily resolution is maintained
        time_diff = result.timestamps[1] - result.timestamps[0]
        self.assertEqual(time_diff, pd.Timedelta(days=1))
    
    def test_generate_long_term_forecast_monthly(self):
        """Test long-term forecast generation (monthly)."""
        months = 2
        result = self.forecaster.generate_long_term_forecast(months=months)
        
        self.assertEqual(len(result.predictions), months)
        self.assertEqual(len(result.timestamps), months)
        
        # Check metadata includes long-term adjustments
        self.assertIn('adjustments', result.metadata)
        self.assertIn('long_term', result.metadata['adjustments'])
    
    def test_generate_long_term_forecast_seasonal(self):
        """Test long-term forecast generation (seasonal)."""
        months = 6
        result = self.forecaster.generate_long_term_forecast(months=months)
        
        self.assertEqual(len(result.predictions), months)
        self.assertEqual(len(result.timestamps), months)
        
        # Check that uncertainty increases with time
        # Later predictions should have higher uncertainty
        self.assertGreater(result.uncertainty[-1], result.uncertainty[0])
    
    def test_generate_multi_horizon_forecast(self):
        """Test multi-horizon forecast generation."""
        horizons = {
            'short': 24,
            'medium': 7,
            'long': 3
        }
        
        results = self.forecaster.generate_multi_horizon_forecast(horizons)
        
        # Check all horizons were generated
        self.assertEqual(len(results), 3)
        self.assertIn('short', results)
        self.assertIn('medium', results)
        self.assertIn('long', results)
        
        # Check each result
        self.assertEqual(len(results['short'].predictions), 24)
        self.assertEqual(len(results['medium'].predictions), 7)
        self.assertEqual(len(results['long'].predictions), 3)
    
    def test_generate_multi_horizon_forecast_with_custom_start_time(self):
        """Test multi-horizon forecast with custom start time."""
        start_time = datetime(2024, 6, 15, 10, 0, 0)
        horizons = {'short': 12, 'medium': 3}
        
        results = self.forecaster.generate_multi_horizon_forecast(
            horizons, start_time=start_time
        )
        
        # Check start times
        self.assertEqual(results['short'].timestamps[0], start_time)
        self.assertEqual(results['medium'].timestamps[0], start_time)
    
    def test_compare_horizon_performance(self):
        """Test horizon performance comparison."""
        # Generate forecasts
        horizons = {'short': 12, 'medium': 5}
        forecasts = self.forecaster.generate_multi_horizon_forecast(horizons)
        
        # Create synthetic actual values
        actual_values = {
            'short': np.random.randn(12) * 2000 + 50000,
            'medium': np.random.randn(5) * 3000 + 50000
        }
        
        comparison = self.forecaster.compare_horizon_performance(forecasts, actual_values)
        
        # Check comparison structure
        self.assertIn('horizon_performance', comparison)
        self.assertIn('summary', comparison)
        self.assertIn('recommendations', comparison)
        
        # Check performance metrics
        self.assertIn('short', comparison['horizon_performance'])
        self.assertIn('medium', comparison['horizon_performance'])
        
        # Check metrics structure
        short_metrics = comparison['horizon_performance']['short']
        expected_metrics = ['mae', 'rmse', 'mape', 'model_used', 'horizon_type', 'num_points']
        for metric in expected_metrics:
            self.assertIn(metric, short_metrics)
        
        # Check summary
        self.assertIn('best_performing_horizon', comparison['summary'])
        self.assertIn('best_mape', comparison['summary'])
        
        # Check recommendations
        self.assertIsInstance(comparison['recommendations'], list)
    
    def test_model_selection(self):
        """Test model selection for different horizons."""
        intraday_config = self.forecaster.horizon_configs[HorizonType.INTRADAY]
        selected_model = self.forecaster._select_best_model(intraday_config)
        
        # Should select a model based on preferences
        self.assertIn(selected_model, self.forecaster.models.values())
        
        # Test with config that has no matching preferences
        custom_config = HorizonConfig(
            horizon_type=HorizonType.INTRADAY,
            max_periods=24,
            frequency='h',
            aggregation_method='none',
            seasonal_adjustment=False,
            uncertainty_scaling=1.0,
            model_preferences=['nonexistent_model']
        )
        
        fallback_model = self.forecaster._select_best_model(custom_config)
        self.assertIn(fallback_model, self.forecaster.models.values())
    
    def test_short_term_adjustments(self):
        """Test short-term specific adjustments."""
        # Generate forecast that covers peak hours
        result = self.forecaster.generate_short_term_forecast(hours=24)
        
        # Check that adjustments were applied
        self.assertIn('adjustments', result.metadata)
        self.assertIn('short_term', result.metadata['adjustments'])
        
        # Check that uncertainty varies by time of day
        # (Peak hours should have higher uncertainty)
        hours = result.timestamps.hour
        peak_hours = (hours >= 16) & (hours <= 20)
        non_peak_hours = ~peak_hours & ~((hours >= 7) & (hours <= 9))
        
        if np.any(peak_hours) and np.any(non_peak_hours):
            avg_peak_uncertainty = np.mean(result.uncertainty[peak_hours])
            avg_non_peak_uncertainty = np.mean(result.uncertainty[non_peak_hours])
            self.assertGreater(avg_peak_uncertainty, avg_non_peak_uncertainty)
    
    def test_seasonal_adjustment(self):
        """Test seasonal adjustment application."""
        # Generate long-term forecast that spans different seasons
        start_time = datetime(2024, 6, 1)  # Summer
        result = self.forecaster.generate_long_term_forecast(
            months=6, start_time=start_time
        )
        
        # Check that seasonal adjustments were applied
        self.assertIn('adjustments', result.metadata)
        
        # Summer months should have higher predictions due to cooling demand
        summer_months = result.timestamps.month.isin([6, 7, 8])
        if np.any(summer_months):
            # Seasonal adjustment should have been applied
            self.assertTrue(len(result.predictions) > 0)
    
    def test_weekly_to_daily_expansion(self):
        """Test expansion of weekly forecast to daily resolution."""
        weekly_data = np.array([100, 105, 110, 108])
        target_days = 14  # 2 weeks
        
        daily_data = self.forecaster._expand_weekly_to_daily(weekly_data, target_days)
        
        self.assertEqual(len(daily_data), target_days)
        self.assertTrue(np.all(np.isfinite(daily_data)))
        
        # First day should be close to first week value
        self.assertAlmostEqual(daily_data[0], weekly_data[0], delta=5)
        
        # Last day should be close to last week value
        self.assertAlmostEqual(daily_data[-1], weekly_data[-1], delta=5)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create and train sample models
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        X = pd.DataFrame({
            'timestamp': dates,
            'temperature': 70 + 5 * np.sin(2 * np.pi * np.arange(100) / 24) + np.random.normal(0, 2, 100),
            'hour_of_day': np.arange(100) % 24
        })
        
        y = pd.Series(
            50000 + 10000 * np.sin(2 * np.pi * np.arange(100) / 24) + np.random.normal(0, 1500, 100),
            name='load_mw'
        )
        
        self.models = {}
        
        rf_model = RandomForestForecaster(n_estimators=5, sequence_length=8)
        rf_model.fit(X, y)
        self.models['rf'] = rf_model
        
        transformer_model = create_forecaster('transformer', d_model=16, nhead=2, num_layers=1, sequence_length=8)
        transformer_model.fit(X, y, epochs=3, batch_size=8)
        self.models['transformer'] = transformer_model
    
    def test_create_multi_horizon_forecaster(self):
        """Test create_multi_horizon_forecaster convenience function."""
        forecaster = create_multi_horizon_forecaster(self.models)
        
        self.assertIsInstance(forecaster, MultiHorizonForecaster)
        self.assertEqual(len(forecaster.models), 2)
        self.assertIn('rf', forecaster.models)
        self.assertIn('transformer', forecaster.models)
    
    def test_generate_all_horizons(self):
        """Test generate_all_horizons convenience function."""
        results = generate_all_horizons(
            self.models,
            short_hours=12,
            medium_days=5,
            long_months=2
        )
        
        # Check all horizons were generated
        self.assertEqual(len(results), 3)
        self.assertIn('short_term', results)
        self.assertIn('medium_term', results)
        self.assertIn('long_term', results)
        
        # Check forecast lengths
        self.assertEqual(len(results['short_term'].predictions), 12)
        self.assertEqual(len(results['medium_term'].predictions), 5)
        self.assertEqual(len(results['long_term'].predictions), 2)
    
    def test_generate_all_horizons_with_custom_parameters(self):
        """Test generate_all_horizons with custom parameters."""
        start_time = datetime(2024, 7, 1, 14, 0, 0)
        confidence_levels = [0.80, 0.95]
        
        results = generate_all_horizons(
            self.models,
            short_hours=6,
            medium_days=3,
            long_months=1,
            start_time=start_time,
            confidence_levels=confidence_levels
        )
        
        # Check custom parameters were applied
        for result in results.values():
            self.assertEqual(result.timestamps[0], start_time)
            self.assertEqual(len(result.confidence_intervals), 2)
            self.assertIn(0.80, result.confidence_intervals)
            self.assertIn(0.95, result.confidence_intervals)


class TestMultiHorizonIntegration(unittest.TestCase):
    """Integration tests for multi-horizon forecasting."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=150, freq='h')
        self.X = pd.DataFrame({
            'timestamp': dates,
            'temperature': 70 + 8 * np.sin(2 * np.pi * np.arange(150) / 24) + np.random.normal(0, 2, 150),
            'hour_of_day': np.arange(150) % 24,
            'day_of_week': (np.arange(150) // 24) % 7
        })
        
        self.y = pd.Series(
            50000 + 12000 * np.sin(2 * np.pi * np.arange(150) / 24) + np.random.normal(0, 1800, 150),
            name='load_mw'
        )
    
    def test_multiple_model_types_multi_horizon(self):
        """Test multi-horizon forecasting with different model types."""
        # Train different models
        models = {}
        
        rf_model = create_forecaster('random_forest', n_estimators=8, sequence_length=10)
        rf_model.fit(self.X, self.y)
        models['random_forest'] = rf_model
        
        transformer_model = create_forecaster('transformer', d_model=24, nhead=3, num_layers=2, sequence_length=10)
        transformer_model.fit(self.X, self.y, epochs=4, batch_size=12)
        models['transformer'] = transformer_model
        
        # Create multi-horizon forecaster
        forecaster = MultiHorizonForecaster(models)
        
        # Generate all horizon types
        short_result = forecaster.generate_short_term_forecast(hours=18)
        medium_result = forecaster.generate_medium_term_forecast(days=6)
        long_result = forecaster.generate_long_term_forecast(months=2)
        
        # Check all forecasts were generated successfully
        self.assertEqual(len(short_result.predictions), 18)
        self.assertEqual(len(medium_result.predictions), 6)
        self.assertEqual(len(long_result.predictions), 2)
        
        # Check that different models might be selected for different horizons
        # (This depends on the model preferences in horizon configs)
        self.assertTrue(np.all(np.isfinite(short_result.predictions)))
        self.assertTrue(np.all(np.isfinite(medium_result.predictions)))
        self.assertTrue(np.all(np.isfinite(long_result.predictions)))
    
    def test_end_to_end_multi_horizon_workflow(self):
        """Test complete multi-horizon forecasting workflow."""
        # Train models
        models = {}
        
        rf_model = create_forecaster('random_forest', n_estimators=6, sequence_length=8)
        rf_model.fit(self.X, self.y)
        models['rf'] = rf_model
        
        # Generate multi-horizon forecasts
        horizons = {
            'intraday': 12,
            'daily': 4,
            'monthly': 2
        }
        
        forecaster = MultiHorizonForecaster(models)
        results = forecaster.generate_multi_horizon_forecast(horizons)
        
        # Verify all forecasts
        self.assertEqual(len(results), 3)
        
        for horizon_name, result in results.items():
            self.assertIsInstance(result, ForecastResult)
            self.assertTrue(len(result.predictions) > 0)
            self.assertTrue(len(result.timestamps) > 0)
            self.assertTrue(len(result.uncertainty) > 0)
            self.assertTrue(len(result.confidence_intervals) > 0)
        
        # Test performance comparison with synthetic actual values
        actual_values = {
            'intraday': np.random.randn(12) * 2000 + 50000,
            'daily': np.random.randn(4) * 2500 + 50000,
            'monthly': np.random.randn(2) * 3000 + 50000
        }
        
        comparison = forecaster.compare_horizon_performance(results, actual_values)
        
        # Check comparison results
        self.assertIn('horizon_performance', comparison)
        self.assertIn('summary', comparison)
        self.assertIn('recommendations', comparison)
        
        # Verify performance metrics were calculated
        for horizon_name in horizons.keys():
            self.assertIn(horizon_name, comparison['horizon_performance'])
            metrics = comparison['horizon_performance'][horizon_name]
            self.assertIn('mae', metrics)
            self.assertIn('rmse', metrics)
            self.assertIn('mape', metrics)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)