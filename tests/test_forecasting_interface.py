"""
Unit tests for the forecasting interface.
Tests forecast generation, confidence intervals, and uncertainty measures.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.forecasting_interface import (
    ForecastingInterface, ForecastResult, ForecastHorizon, ConfidenceLevel,
    create_forecasting_interface, generate_forecast, compare_model_forecasts
)
from model.forecaster import RandomForestForecaster, create_forecaster


class TestForecastResult(unittest.TestCase):
    """Test cases for ForecastResult class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create sample forecast result
        self.predictions = np.array([100, 105, 110, 108, 112])
        self.timestamps = pd.date_range('2024-01-01', periods=5, freq='H')
        self.uncertainty = np.array([5, 6, 7, 6, 8])
        
        self.confidence_intervals = {
            0.68: (self.predictions - self.uncertainty, self.predictions + self.uncertainty),
            0.95: (self.predictions - 1.96 * self.uncertainty, self.predictions + 1.96 * self.uncertainty)
        }
        
        self.forecast_result = ForecastResult(
            predictions=self.predictions,
            timestamps=self.timestamps,
            confidence_intervals=self.confidence_intervals,
            uncertainty=self.uncertainty,
            model_name="TestModel",
            model_type="test",
            forecast_horizon=ForecastHorizon.SHORT_TERM,
            created_at=datetime.now(),
            metadata={'test': True}
        )
    
    def test_forecast_result_initialization(self):
        """Test ForecastResult initialization."""
        self.assertEqual(len(self.forecast_result.predictions), 5)
        self.assertEqual(len(self.forecast_result.timestamps), 5)
        self.assertEqual(self.forecast_result.model_name, "TestModel")
        self.assertEqual(self.forecast_result.forecast_horizon, ForecastHorizon.SHORT_TERM)
        self.assertIn(0.68, self.forecast_result.confidence_intervals)
        self.assertIn(0.95, self.forecast_result.confidence_intervals)
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        df = self.forecast_result.to_dataframe()
        
        # Check structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        
        # Check columns
        expected_columns = ['timestamp', 'prediction', 'uncertainty', 'lower_68', 'upper_68', 'lower_95', 'upper_95']
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Check values
        np.testing.assert_array_equal(df['prediction'].values, self.predictions)
        np.testing.assert_array_equal(df['uncertainty'].values, self.uncertainty)
    
    def test_get_confidence_interval(self):
        """Test getting confidence intervals."""
        # Test existing confidence level
        lower_95, upper_95 = self.forecast_result.get_confidence_interval(0.95)
        expected_lower = self.predictions - 1.96 * self.uncertainty
        expected_upper = self.predictions + 1.96 * self.uncertainty
        
        np.testing.assert_array_almost_equal(lower_95, expected_lower)
        np.testing.assert_array_almost_equal(upper_95, expected_upper)
        
        # Test default confidence level
        lower_default, upper_default = self.forecast_result.get_confidence_interval()
        np.testing.assert_array_almost_equal(lower_default, lower_95)
        np.testing.assert_array_almost_equal(upper_default, upper_95)


class TestForecastingInterface(unittest.TestCase):
    """Test cases for ForecastingInterface class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create sample data for training
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        self.X = pd.DataFrame({
            'timestamp': dates,
            'temperature': 70 + 10 * np.sin(2 * np.pi * np.arange(100) / 24) + np.random.normal(0, 3, 100),
            'hour_of_day': np.arange(100) % 24,
            'day_of_week': (np.arange(100) // 24) % 7
        })
        
        self.y = pd.Series(
            50000 + 15000 * np.sin(2 * np.pi * np.arange(100) / 24) + np.random.normal(0, 2000, 100),
            name='load_mw'
        )
        
        # Train a sample model
        self.model = RandomForestForecaster(n_estimators=10, sequence_length=12)
        self.model.fit(self.X, self.y)
        
        # Create forecasting interface
        self.interface = ForecastingInterface(self.model)
    
    def test_interface_initialization(self):
        """Test ForecastingInterface initialization."""
        self.assertEqual(self.interface.model_name, "RandomForest")
        self.assertEqual(self.interface.model_type, "ml")
        self.assertIs(self.interface.model, self.model)
    
    def test_interface_with_unfitted_model_raises_error(self):
        """Test that unfitted model raises error."""
        unfitted_model = RandomForestForecaster(n_estimators=5)
        
        with self.assertRaises(ValueError) as context:
            ForecastingInterface(unfitted_model)
        
        self.assertIn("Model must be fitted", str(context.exception))
    
    def test_generate_forecast_basic(self):
        """Test basic forecast generation."""
        horizon = 24
        result = self.interface.generate_forecast(horizon)
        
        # Check result structure
        self.assertIsInstance(result, ForecastResult)
        self.assertEqual(len(result.predictions), horizon)
        self.assertEqual(len(result.timestamps), horizon)
        self.assertEqual(len(result.uncertainty), horizon)
        
        # Check metadata
        self.assertEqual(result.model_name, "RandomForest")
        self.assertEqual(result.model_type, "ml")
        self.assertEqual(result.forecast_horizon, ForecastHorizon.SHORT_TERM)
        
        # Check confidence intervals
        self.assertIn(0.68, result.confidence_intervals)
        self.assertIn(0.90, result.confidence_intervals)
        self.assertIn(0.95, result.confidence_intervals)
    
    def test_generate_forecast_with_custom_parameters(self):
        """Test forecast generation with custom parameters."""
        horizon = 48
        start_time = datetime(2024, 6, 1, 12, 0, 0)
        confidence_levels = [0.80, 0.95]
        
        result = self.interface.generate_forecast(
            horizon=horizon,
            start_time=start_time,
            frequency='H',
            confidence_levels=confidence_levels
        )
        
        # Check parameters
        self.assertEqual(len(result.predictions), horizon)
        self.assertEqual(result.timestamps[0], start_time)
        self.assertEqual(len(result.confidence_intervals), 2)
        self.assertIn(0.80, result.confidence_intervals)
        self.assertIn(0.95, result.confidence_intervals)
    
    def test_generate_short_term_forecast(self):
        """Test short-term forecast generation."""
        hours = 12
        result = self.interface.generate_short_term_forecast(hours=hours)
        
        self.assertEqual(len(result.predictions), hours)
        self.assertEqual(result.forecast_horizon, ForecastHorizon.SHORT_TERM)
        self.assertEqual(result.metadata['frequency'], 'H')
    
    def test_generate_medium_term_forecast(self):
        """Test medium-term forecast generation."""
        days = 5
        result = self.interface.generate_medium_term_forecast(days=days)
        
        self.assertEqual(len(result.predictions), days)
        self.assertEqual(result.forecast_horizon, ForecastHorizon.MEDIUM_TERM)  # 5 days is medium-term for daily freq
        self.assertEqual(result.metadata['frequency'], 'D')
    
    def test_generate_long_term_forecast(self):
        """Test long-term forecast generation."""
        weeks = 8
        result = self.interface.generate_long_term_forecast(weeks=weeks)
        
        self.assertEqual(len(result.predictions), weeks)
        self.assertEqual(result.forecast_horizon, ForecastHorizon.LONG_TERM)
        self.assertEqual(result.metadata['frequency'], 'W')
    
    def test_compare_forecasts_basic(self):
        """Test basic forecast comparison."""
        # Generate multiple forecasts
        forecast1 = self.interface.generate_forecast(24)
        forecast2 = self.interface.generate_forecast(24)
        
        forecasts = [forecast1, forecast2]
        comparison = self.interface.compare_forecasts(forecasts)
        
        # Check comparison structure
        self.assertIn('forecasts', comparison)
        self.assertIn('summary', comparison)
        self.assertIn('metrics', comparison)
        
        # Check forecasts info
        self.assertEqual(len(comparison['forecasts']), 2)
        self.assertEqual(comparison['summary']['num_forecasts'], 2)
    
    def test_compare_forecasts_with_actual_values(self):
        """Test forecast comparison with actual values."""
        horizon = 24
        forecast1 = self.interface.generate_forecast(horizon)
        forecast2 = self.interface.generate_forecast(horizon)
        
        # Create synthetic actual values
        actual_values = np.random.randn(horizon) * 1000 + 50000
        
        forecasts = [forecast1, forecast2]
        comparison = self.interface.compare_forecasts(forecasts, actual_values)
        
        # Check metrics were calculated
        self.assertIn('metrics', comparison)
        self.assertTrue(len(comparison['metrics']) > 0)
        
        # Check metric types
        first_metric = list(comparison['metrics'].values())[0]
        expected_metrics = ['mae', 'mse', 'rmse', 'mape']
        for metric in expected_metrics:
            self.assertIn(metric, first_metric)
    
    def test_horizon_type_determination(self):
        """Test forecast horizon type determination."""
        # Test hourly forecasts
        self.assertEqual(
            self.interface._determine_horizon_type(24, 'H'),
            ForecastHorizon.SHORT_TERM
        )
        self.assertEqual(
            self.interface._determine_horizon_type(72, 'H'),
            ForecastHorizon.MEDIUM_TERM
        )
        self.assertEqual(
            self.interface._determine_horizon_type(200, 'H'),
            ForecastHorizon.LONG_TERM
        )
        
        # Test daily forecasts
        self.assertEqual(
            self.interface._determine_horizon_type(2, 'D'),
            ForecastHorizon.SHORT_TERM
        )
        self.assertEqual(
            self.interface._determine_horizon_type(15, 'D'),
            ForecastHorizon.MEDIUM_TERM
        )
        self.assertEqual(
            self.interface._determine_horizon_type(40, 'D'),
            ForecastHorizon.LONG_TERM
        )
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        predictions = np.array([100, 105, 110])
        uncertainty = np.array([5, 6, 7])
        
        # Test 95% confidence interval
        lower, upper = self.interface._calculate_confidence_interval(predictions, uncertainty, 0.95)
        
        expected_margin = 1.96 * uncertainty
        expected_lower = predictions - expected_margin
        expected_upper = predictions + expected_margin
        
        np.testing.assert_array_almost_equal(lower, expected_lower, decimal=2)
        np.testing.assert_array_almost_equal(upper, expected_upper, decimal=2)
    
    def test_dummy_input_creation(self):
        """Test dummy input data creation."""
        horizon = 24
        timestamps = pd.date_range('2024-01-01', periods=horizon, freq='H')
        
        dummy_X = self.interface._create_dummy_input(horizon, timestamps)
        
        # Check structure
        self.assertIsInstance(dummy_X, pd.DataFrame)
        self.assertEqual(len(dummy_X), horizon)
        
        # Check basic columns
        expected_columns = ['timestamp', 'hour_of_day', 'day_of_week', 'month', 'is_weekend']
        for col in expected_columns:
            self.assertIn(col, dummy_X.columns)
        
        # Check data types and ranges
        self.assertTrue(dummy_X['hour_of_day'].between(0, 23).all())
        self.assertTrue(dummy_X['day_of_week'].between(0, 6).all())
        self.assertTrue(dummy_X['month'].between(1, 12).all())


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create and train a sample model
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        X = pd.DataFrame({
            'timestamp': dates,
            'temperature': 70 + 5 * np.sin(2 * np.pi * np.arange(50) / 24) + np.random.normal(0, 2, 50),
            'hour_of_day': np.arange(50) % 24
        })
        
        y = pd.Series(
            50000 + 10000 * np.sin(2 * np.pi * np.arange(50) / 24) + np.random.normal(0, 1500, 50),
            name='load_mw'
        )
        
        self.model = RandomForestForecaster(n_estimators=5, sequence_length=8)
        self.model.fit(X, y)
    
    def test_create_forecasting_interface(self):
        """Test create_forecasting_interface convenience function."""
        interface = create_forecasting_interface(self.model)
        
        self.assertIsInstance(interface, ForecastingInterface)
        self.assertEqual(interface.model_name, "RandomForest")
        self.assertIs(interface.model, self.model)
    
    def test_generate_forecast_convenience(self):
        """Test generate_forecast convenience function."""
        horizon = 12
        result = generate_forecast(self.model, horizon)
        
        self.assertIsInstance(result, ForecastResult)
        self.assertEqual(len(result.predictions), horizon)
        self.assertEqual(result.model_name, "RandomForest")
    
    def test_generate_forecast_with_confidence_levels(self):
        """Test generate_forecast with custom confidence levels."""
        horizon = 12
        confidence_levels = [0.80, 0.99]
        
        result = generate_forecast(self.model, horizon, confidence_levels=confidence_levels)
        
        self.assertEqual(len(result.confidence_intervals), 2)
        self.assertIn(0.80, result.confidence_intervals)
        self.assertIn(0.99, result.confidence_intervals)
    
    def test_compare_model_forecasts(self):
        """Test compare_model_forecasts convenience function."""
        # Create another model for comparison
        model2 = RandomForestForecaster(n_estimators=3, sequence_length=6)
        model2.fit(pd.DataFrame({'dummy': np.random.randn(20)}), pd.Series(np.random.randn(20)))
        
        models = [self.model, model2]
        horizon = 10
        
        comparison = compare_model_forecasts(models, horizon)
        
        self.assertIn('forecasts', comparison)
        self.assertIn('summary', comparison)
        self.assertEqual(comparison['summary']['num_forecasts'], 2)
    
    def test_compare_model_forecasts_with_actual_values(self):
        """Test model comparison with actual values."""
        model2 = RandomForestForecaster(n_estimators=3, sequence_length=6)
        model2.fit(pd.DataFrame({'dummy': np.random.randn(20)}), pd.Series(np.random.randn(20)))
        
        models = [self.model, model2]
        horizon = 10
        actual_values = np.random.randn(horizon) * 1000 + 50000
        
        comparison = compare_model_forecasts(models, horizon, actual_values=actual_values)
        
        self.assertIn('metrics', comparison)
        self.assertTrue(len(comparison['metrics']) > 0)


class TestForecastingIntegration(unittest.TestCase):
    """Integration tests for forecasting interface with different model types."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        self.X = pd.DataFrame({
            'timestamp': dates,
            'temperature': 70 + 5 * np.sin(2 * np.pi * np.arange(50) / 24) + np.random.normal(0, 2, 50),
            'hour_of_day': np.arange(50) % 24
        })
        
        self.y = pd.Series(
            50000 + 10000 * np.sin(2 * np.pi * np.arange(50) / 24) + np.random.normal(0, 1500, 50),
            name='load_mw'
        )
    
    def test_random_forest_forecasting(self):
        """Test forecasting interface with Random Forest model."""
        model = create_forecaster('random_forest', n_estimators=5, sequence_length=8)
        model.fit(self.X, self.y)
        
        interface = ForecastingInterface(model)
        result = interface.generate_forecast(24)
        
        self.assertEqual(len(result.predictions), 24)
        self.assertEqual(result.model_type, "ml")
        self.assertTrue(np.all(np.isfinite(result.predictions)))
        self.assertTrue(np.all(result.uncertainty > 0))
    
    def test_transformer_forecasting(self):
        """Test forecasting interface with Transformer model."""
        model = create_forecaster('transformer', d_model=16, nhead=2, num_layers=1, sequence_length=8)
        model.fit(self.X, self.y, epochs=2, batch_size=8)  # Quick training
        
        interface = ForecastingInterface(model)
        result = interface.generate_forecast(12)
        
        self.assertEqual(len(result.predictions), 12)
        self.assertEqual(result.model_type, "deep_learning")
        self.assertTrue(np.all(np.isfinite(result.predictions)))
    
    def test_multiple_model_comparison(self):
        """Test comparison of multiple different model types."""
        # Train different models
        rf_model = create_forecaster('random_forest', n_estimators=5, sequence_length=8)
        rf_model.fit(self.X, self.y)
        
        transformer_model = create_forecaster('transformer', d_model=16, nhead=2, num_layers=1, sequence_length=8)
        transformer_model.fit(self.X, self.y, epochs=2, batch_size=8)
        
        models = [rf_model, transformer_model]
        horizon = 12
        
        comparison = compare_model_forecasts(models, horizon)
        
        # Check that both models generated forecasts
        self.assertEqual(comparison['summary']['num_forecasts'], 2)
        self.assertIn('ml', comparison['summary']['model_types'])
        self.assertIn('deep_learning', comparison['summary']['model_types'])
        
        # Check individual forecast info
        forecast_info = comparison['forecasts']
        model_names = [f['model_name'] for f in forecast_info]
        self.assertIn('RandomForest', model_names)
        self.assertIn('Transformer', model_names)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)