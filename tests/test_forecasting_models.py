"""
Unit tests for forecasting models (traditional and foundation models).
Tests statistical, ML, deep learning, and foundation model implementations.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.forecaster import (
    ARIMAForecaster, ExponentialSmoothingForecaster,
    RandomForestForecaster, XGBoostForecaster, TransformerForecaster,
    create_forecaster, TRADITIONAL_MODELS
)

from model.foundation_models import (
    TimesFMForecaster, ChronosForecaster, LagLlamaForecaster,
    MoiraiForecaster, FoundationModelEnsemble,
    create_foundation_model, FOUNDATION_MODELS
)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


class TestTraditionalForecasters(unittest.TestCase):
    """Test cases for traditional forecasting models."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create realistic time series data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='h')
        
        # Create synthetic energy load data with daily patterns
        hours = np.arange(200) % 24
        daily_pattern = 50000 + 15000 * np.sin(2 * np.pi * hours / 24)
        noise = np.random.normal(0, 2000, 200)
        trend = np.linspace(0, 5000, 200)
        
        self.X = pd.DataFrame({
            'timestamp': dates,
            'temperature': 70 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 3, 200),
            'hour_of_day': hours,
            'day_of_week': (np.arange(200) // 24) % 7
        })
        
        self.y = pd.Series(daily_pattern + trend + noise, name='load_mw')
        
        # Split into train/test
        split_idx = 150
        self.X_train = self.X.iloc[:split_idx]
        self.y_train = self.y.iloc[:split_idx]
        self.X_test = self.X.iloc[split_idx:]
        self.y_test = self.y.iloc[split_idx:]
    
    def test_random_forest_forecaster(self):
        """Test Random Forest forecaster."""
        model = RandomForestForecaster(
            n_estimators=10,  # Small for testing
            sequence_length=12
        )
        
        # Test initialization
        self.assertEqual(model.model_name, "RandomForest")
        self.assertEqual(model.model_type, "ml")
        self.assertEqual(model.n_estimators, 10)
        
        # Test fitting
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.model)
        
        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Test prediction with uncertainty
        pred, unc = model.predict_with_uncertainty(self.X_test)
        self.assertEqual(len(pred), len(self.X_test))
        self.assertEqual(len(unc), len(self.X_test))
        
        # Test feature importance
        importance = model.get_feature_importance()
        self.assertIsInstance(importance, dict)
        
        # Test forecasting
        forecast_df = model.forecast(horizon=24)
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), 24)
    
    def test_arima_forecaster(self):
        """Test ARIMA forecaster."""
        model = ARIMAForecaster(
            order=(1, 1, 1),
            seasonal_order=None
        )
        
        # Test initialization
        self.assertEqual(model.model_name, "ARIMA")
        self.assertEqual(model.model_type, "statistical")
        self.assertEqual(model.order, (1, 1, 1))
        
        # Test fitting
        try:
            model.fit(self.X_train, self.y_train)
            self.assertTrue(model.is_fitted)
            
            # Test prediction
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
            # Test prediction with uncertainty
            pred, unc = model.predict_with_uncertainty(self.X_test)
            self.assertEqual(len(pred), len(self.X_test))
            self.assertEqual(len(unc), len(self.X_test))
            
            # Test forecasting
            forecast_df = model.forecast(horizon=24)
            self.assertIsInstance(forecast_df, pd.DataFrame)
            self.assertEqual(len(forecast_df), 24)
            
        except ImportError:
            # Skip if statsmodels not available
            self.skipTest("statsmodels not available")
    
    def test_exponential_smoothing_forecaster(self):
        """Test Exponential Smoothing forecaster."""
        model = ExponentialSmoothingForecaster(
            trend='add',
            seasonal='add',
            seasonal_periods=24
        )
        
        # Test initialization
        self.assertEqual(model.model_name, "ExponentialSmoothing")
        self.assertEqual(model.model_type, "statistical")
        self.assertEqual(model.trend, 'add')
        
        # Test fitting
        try:
            model.fit(self.X_train, self.y_train)
            self.assertTrue(model.is_fitted)
            
            # Test prediction
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
            # Test prediction with uncertainty
            pred, unc = model.predict_with_uncertainty(self.X_test)
            self.assertEqual(len(pred), len(self.X_test))
            self.assertEqual(len(unc), len(self.X_test))
            
            # Test forecasting
            forecast_df = model.forecast(horizon=24)
            self.assertIsInstance(forecast_df, pd.DataFrame)
            self.assertEqual(len(forecast_df), 24)
            
        except ImportError:
            # Skip if statsmodels not available
            self.skipTest("statsmodels not available")
    
    def test_xgboost_forecaster(self):
        """Test XGBoost forecaster."""
        model = XGBoostForecaster(
            n_estimators=10,  # Small for testing
            max_depth=3,
            sequence_length=12
        )
        
        # Test initialization
        self.assertEqual(model.model_name, "XGBoost")
        self.assertEqual(model.model_type, "ml")
        self.assertEqual(model.n_estimators, 10)
        
        # Test fitting
        try:
            model.fit(self.X_train, self.y_train)
            self.assertTrue(model.is_fitted)
            self.assertIsNotNone(model.model)
            
            # Test prediction
            predictions = model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.X_test))
            
            # Test prediction with uncertainty
            pred, unc = model.predict_with_uncertainty(self.X_test)
            self.assertEqual(len(pred), len(self.X_test))
            self.assertEqual(len(unc), len(self.X_test))
            
            # Test feature importance
            importance = model.get_feature_importance()
            self.assertIsInstance(importance, dict)
            
            # Test forecasting
            forecast_df = model.forecast(horizon=24)
            self.assertIsInstance(forecast_df, pd.DataFrame)
            self.assertEqual(len(forecast_df), 24)
            
        except ImportError:
            # Skip if XGBoost not available
            self.skipTest("XGBoost not available")
    
    # LSTM test removed - LSTM implementation moved to backlog
    
    def test_transformer_forecaster(self):
        """Test Transformer forecaster."""
        model = TransformerForecaster(
            d_model=32,  # Small for testing
            nhead=4,
            num_layers=2,
            sequence_length=12
        )
        
        # Test initialization
        self.assertEqual(model.model_name, "Transformer")
        self.assertEqual(model.model_type, "deep_learning")
        self.assertEqual(model.d_model, 32)
        self.assertEqual(model.nhead, 4)
        
        # Test fitting (with few epochs for speed)
        model.fit(self.X_train, self.y_train, epochs=5, batch_size=16)
        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.model)
        
        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Test prediction with uncertainty
        pred, unc = model.predict_with_uncertainty(self.X_test)
        self.assertEqual(len(pred), len(self.X_test))
        self.assertEqual(len(unc), len(self.X_test))
        
        # Test forecasting
        forecast_df = model.forecast(horizon=24)
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), 24)
    
    def test_create_forecaster_factory(self):
        """Test forecaster factory function."""
        # Test valid model types
        for model_type in ['random_forest', 'transformer']:
            try:
                model = create_forecaster(model_type)
                self.assertIsNotNone(model)
                self.assertIn(model_type.replace('_', ''), model.model_name.lower().replace('_', ''))
            except ImportError:
                # Skip if dependencies not available
                continue
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            create_forecaster('invalid_model')
    
    def test_traditional_models_registry(self):
        """Test traditional models registry."""
        self.assertIsInstance(TRADITIONAL_MODELS, dict)
        
        expected_models = ['arima', 'exponential_smoothing', 'random_forest', 'xgboost', 'transformer']
        for model_name in expected_models:
            self.assertIn(model_name, TRADITIONAL_MODELS)
            
            model_info = TRADITIONAL_MODELS[model_name]
            self.assertIn('class', model_info)
            self.assertIn('type', model_info)
            self.assertIn('description', model_info)
            self.assertIn('strengths', model_info)


class TestFoundationModels(unittest.TestCase):
    """Test cases for foundation models."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        
        # Create synthetic time series data
        hours = np.arange(100) % 24
        daily_pattern = 50000 + 15000 * np.sin(2 * np.pi * hours / 24)
        noise = np.random.normal(0, 2000, 100)
        
        self.X = pd.DataFrame({
            'timestamp': dates,
            'load_mw': daily_pattern + noise,
            'temperature': 70 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 3, 100)
        })
        
        self.y = pd.Series(daily_pattern + noise, name='target')
        
        # Split data
        split_idx = 80
        self.X_train = self.X.iloc[:split_idx]
        self.y_train = self.y.iloc[:split_idx]
        self.X_test = self.X.iloc[split_idx:]
        self.y_test = self.y.iloc[split_idx:]
    
    def test_timesfm_forecaster(self):
        """Test TimesFM forecaster."""
        model = TimesFMForecaster(
            model_size="timesfm-1.0-200m",
            context_length=32,
            prediction_length=24
        )
        
        # Test initialization
        self.assertEqual(model.model_name, "TimesFM-timesfm-1.0-200m")
        self.assertEqual(model.model_type, "foundation")
        self.assertEqual(model.context_length, 32)
        
        # Test model loading (will use mock)
        model.load_pretrained_model()
        self.assertIsNotNone(model.model)
        
        # Test fitting
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_fitted)
        
        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Test prediction with uncertainty
        pred, unc = model.predict_with_uncertainty(self.X_test)
        self.assertEqual(len(pred), len(self.X_test))
        self.assertEqual(len(unc), len(self.X_test))
        
        # Test data preprocessing
        processed = model.preprocess_data(self.X_test)
        self.assertIsInstance(processed, dict)
        self.assertIn('inputs', processed)
    
    def test_chronos_forecaster(self):
        """Test Chronos forecaster."""
        model = ChronosForecaster(
            model_size="chronos-t5-small",
            context_length=64,
            prediction_length=24
        )
        
        # Test initialization
        self.assertEqual(model.model_name, "Chronos-chronos-t5-small")
        self.assertEqual(model.model_type, "foundation")
        
        # Test model loading (will use mock)
        model.load_pretrained_model()
        self.assertIsNotNone(model.model)
        
        # Test fitting
        model.fit(self.X_train, self.y_train)
        self.assertTrue(model.is_fitted)
        
        # Test prediction
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Test prediction with uncertainty
        pred, unc = model.predict_with_uncertainty(self.X_test)
        self.assertEqual(len(pred), len(self.X_test))
        self.assertEqual(len(unc), len(self.X_test))
    
    def test_foundation_model_ensemble(self):
        """Test foundation model ensemble."""
        ensemble = FoundationModelEnsemble(
            models=['timesfm', 'chronos'],
            weights=[0.6, 0.4],
            context_length=64,
            prediction_length=24
        )
        
        # Test initialization
        self.assertEqual(ensemble.model_name, "FoundationEnsemble")
        self.assertEqual(ensemble.model_type, "foundation")
        self.assertEqual(ensemble.model_names, ['timesfm', 'chronos'])
        self.assertEqual(ensemble.weights, [0.6, 0.4])
        
        # Test model loading
        ensemble.load_pretrained_model()
        self.assertGreater(len(ensemble.foundation_models), 0)
        
        # Test fitting
        ensemble.fit(self.X_train, self.y_train)
        self.assertTrue(ensemble.is_fitted)
        
        # Test prediction
        predictions = ensemble.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Test prediction with uncertainty
        pred, unc = ensemble.predict_with_uncertainty(self.X_test)
        self.assertEqual(len(pred), len(self.X_test))
        self.assertEqual(len(unc), len(self.X_test))
    
    def test_create_foundation_model_factory(self):
        """Test foundation model factory function."""
        # Test valid model types
        for model_type in ['timesfm', 'chronos', 'lag-llama', 'moirai', 'ensemble']:
            model = create_foundation_model(model_type)
            self.assertIsNotNone(model)
            self.assertIn(model_type.replace('-', ''), model.model_name.lower().replace('-', ''))
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            create_foundation_model('invalid_foundation_model')
    
    def test_foundation_models_registry(self):
        """Test foundation models registry."""
        self.assertIsInstance(FOUNDATION_MODELS, dict)
        
        expected_models = ['timesfm', 'chronos', 'lag-llama', 'moirai', 'ensemble']
        for model_name in expected_models:
            self.assertIn(model_name, FOUNDATION_MODELS)
            
            model_info = FOUNDATION_MODELS[model_name]
            self.assertIn('class', model_info)
            self.assertIn('description', model_info)
            self.assertIn('sizes', model_info)
            self.assertIn('strengths', model_info)


class TestModelIntegration(unittest.TestCase):
    """Integration tests for forecasting models."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Create time series data
        dates = pd.date_range('2024-01-01', periods=200, freq='h')
        hours = np.arange(200) % 24
        
        # Complex pattern with trend, seasonality, and noise
        trend = np.linspace(45000, 55000, 200)
        daily_seasonal = 10000 * np.sin(2 * np.pi * hours / 24)
        noise = np.random.normal(0, 1500, 200)
        
        load_data = trend + daily_seasonal + noise
        
        self.data = pd.DataFrame({
            'timestamp': dates,
            'load_mw': load_data,
            'temperature': 70 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 2, 200),
            'hour_of_day': hours,
            'day_of_week': (np.arange(200) // 24) % 7
        })
        
        # Split into train/test
        self.train_data = self.data.iloc[:150]
        self.test_data = self.data.iloc[150:]
    
    def test_model_comparison_workflow(self):
        """Test comparing multiple models on the same data."""
        models_to_test = [
            ('RandomForest', RandomForestForecaster(n_estimators=10, sequence_length=12)),
            ('TimesFM', TimesFMForecaster(context_length=24, prediction_length=12)),
        ]
        
        results = {}
        
        for model_name, model in models_to_test:
            try:
                # Prepare data
                X_train = self.train_data.drop(columns=['load_mw'])
                y_train = self.train_data['load_mw']
                X_test = self.test_data.drop(columns=['load_mw'])
                y_test = self.test_data['load_mw']
                
                # Train model
                if hasattr(model, 'load_pretrained_model'):
                    model.load_pretrained_model()
                
                model.fit(X_train, y_train)
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Calculate simple metrics
                mae = np.mean(np.abs(predictions - y_test.values))
                rmse = np.sqrt(np.mean((predictions - y_test.values) ** 2))
                
                results[model_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'predictions': predictions
                }
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        # Check that we got results for at least one model
        successful_models = [name for name, result in results.items() if 'error' not in result]
        self.assertGreater(len(successful_models), 0, "At least one model should work")
        
        # Check that predictions have reasonable values
        for model_name in successful_models:
            predictions = results[model_name]['predictions']
            self.assertEqual(len(predictions), len(self.test_data))
            self.assertTrue(np.all(np.isfinite(predictions)), f"{model_name} produced non-finite predictions")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)