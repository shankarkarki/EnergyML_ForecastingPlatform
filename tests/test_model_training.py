"""
Unit tests for the model training framework.
Tests base models, validation, and training orchestration.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, patch
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.base_model import BaseForecaster, TimeSeriesForecaster, EnsembleForecaster
from model.validation import ModelEvaluator, TimeSeriesValidator, HyperparameterTuner
from model.trainer import ModelTrainer


class MockForecaster(BaseForecaster):
    """Mock forecaster for testing."""
    
    def __init__(self, model_name: str = "mock_model", **kwargs):
        super().__init__(model_name, "mock", **kwargs)
        self.mock_model = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.validate_input(X, y)
        self.feature_names = list(X.columns)
        self.target_name = y.name or 'target'
        self.is_fitted = True
        
        # Mock training metrics
        self.training_metrics = {'mae': 10.0, 'rmse': 15.0, 'mape': 5.0}
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self.validate_input(X)
        
        # Return simple predictions (mean of numeric features only)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return np.random.randn(len(X))  # Fallback if no numeric columns
        return X[numeric_cols].mean(axis=1).values
    
    def predict_with_uncertainty(self, X: pd.DataFrame, **kwargs):
        predictions = self.predict(X, **kwargs)
        uncertainties = np.ones_like(predictions) * 2.0  # Mock uncertainty
        return predictions, uncertainties
    
    def _get_model_state(self):
        return {'mock_state': 'saved'}
    
    def _set_model_state(self, state):
        self.mock_model = state


class TestBaseForecaster(unittest.TestCase):
    """Test cases for BaseForecaster."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockForecaster("test_model")
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'timestamp': dates
        })
        self.y = pd.Series(np.random.randn(100), name='target')
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_name, "test_model")
        self.assertEqual(self.model.model_type, "mock")
        self.assertFalse(self.model.is_fitted)
        self.assertEqual(len(self.model.feature_names), 0)
        self.assertIsInstance(self.model.model_id, str)
        self.assertIn("mock_test_model", self.model.model_id)
    
    def test_fit_predict_workflow(self):
        """Test basic fit and predict workflow."""
        # Test unfitted model
        with self.assertRaises(ValueError):
            self.model.predict(self.X)
        
        # Fit model
        self.model.fit(self.X, self.y)
        self.assertTrue(self.model.is_fitted)
        self.assertEqual(self.model.feature_names, list(self.X.columns))
        self.assertEqual(self.model.target_name, 'target')
        
        # Make predictions
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertIsInstance(predictions, np.ndarray)
        
        # Test predictions with uncertainty
        pred, unc = self.model.predict_with_uncertainty(self.X)
        self.assertEqual(len(pred), len(self.X))
        self.assertEqual(len(unc), len(self.X))
    
    def test_input_validation(self):
        """Test input validation."""
        # Test invalid X type
        with self.assertRaises(ValueError):
            self.model.validate_input("not_a_dataframe")
        
        # Test empty X
        with self.assertRaises(ValueError):
            self.model.validate_input(pd.DataFrame())
        
        # Test mismatched X and y lengths
        with self.assertRaises(ValueError):
            self.model.validate_input(self.X, pd.Series([1, 2, 3]))
        
        # Test y with null values
        y_with_nulls = self.y.copy()
        y_with_nulls.iloc[0] = np.nan
        with self.assertRaises(ValueError):
            self.model.validate_input(self.X, y_with_nulls)
    
    def test_model_info(self):
        """Test model information retrieval."""
        info = self.model.get_model_info()
        
        self.assertIn('model_name', info)
        self.assertIn('model_type', info)
        self.assertIn('is_fitted', info)
        self.assertIn('feature_names', info)
        self.assertIn('training_metrics', info)
        
        self.assertEqual(info['model_name'], "test_model")
        self.assertEqual(info['model_type'], "mock")
        self.assertFalse(info['is_fitted'])
    
    def test_save_load_model(self):
        """Test model saving and loading."""
        # Fit model first
        self.model.fit(self.X, self.y)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'test_model.pkl')
            
            # Save model
            self.model.save_model(filepath)
            self.assertTrue(os.path.exists(filepath))
            
            # Load model
            loaded_model = MockForecaster.load_model(filepath)
            
            # Check loaded model
            self.assertTrue(loaded_model.is_fitted)
            self.assertEqual(loaded_model.model_name, self.model.model_name)
            self.assertEqual(loaded_model.feature_names, self.model.feature_names)
            self.assertEqual(loaded_model.target_name, self.model.target_name)
            
            # Test predictions work
            predictions = loaded_model.predict(self.X)
            self.assertEqual(len(predictions), len(self.X))


class TestTimeSeriesForecaster(unittest.TestCase):
    """Test cases for TimeSeriesForecaster."""
    
    def setUp(self):
        """Set up test fixtures."""
        class MockTimeSeriesForecaster(TimeSeriesForecaster):
            def fit(self, X, y, **kwargs):
                self.is_fitted = True
                return self
            
            def predict(self, X, **kwargs):
                return np.random.randn(len(X))
            
            def predict_with_uncertainty(self, X, **kwargs):
                pred = self.predict(X)
                unc = np.ones_like(pred)
                return pred, unc
            
            def forecast(self, horizon=None, **kwargs):
                horizon = horizon or self.forecast_horizon
                dates = pd.date_range('2024-01-01', periods=horizon, freq='h')
                return pd.DataFrame({
                    'timestamp': dates,
                    'forecast': np.random.randn(horizon)
                })
            
            def _get_model_state(self):
                return {}
            
            def _set_model_state(self, state):
                pass
        
        self.model = MockTimeSeriesForecaster("ts_model", "time_series")
        
        # Create time series data
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        self.data = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.randn(100),
            'feature1': np.random.randn(100)
        })
    
    def test_initialization(self):
        """Test time series forecaster initialization."""
        self.assertEqual(self.model.time_column, 'timestamp')
        self.assertEqual(self.model.forecast_horizon, 24)
        self.assertEqual(self.model.frequency, 'H')
    
    def test_prepare_time_series_data(self):
        """Test time series data preparation."""
        X, y = self.model.prepare_time_series_data(self.data, 'value')
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(self.data))
        self.assertEqual(len(y), len(self.data))
        self.assertNotIn('value', X.columns)
    
    def test_create_forecast_dates(self):
        """Test forecast date creation."""
        last_date = datetime(2024, 1, 1, 12, 0, 0)
        forecast_dates = self.model.create_forecast_dates(last_date, 5)
        
        self.assertEqual(len(forecast_dates), 5)
        self.assertEqual(forecast_dates[0], datetime(2024, 1, 1, 13, 0, 0))
    
    def test_forecast(self):
        """Test forecasting functionality."""
        forecast = self.model.forecast(horizon=10)
        
        self.assertIsInstance(forecast, pd.DataFrame)
        self.assertEqual(len(forecast), 10)
        self.assertIn('timestamp', forecast.columns)
        self.assertIn('forecast', forecast.columns)


class TestEnsembleForecaster(unittest.TestCase):
    """Test cases for EnsembleForecaster."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create base models
        self.base_models = [
            MockForecaster("model1"),
            MockForecaster("model2"),
            MockForecaster("model3")
        ]
        
        self.ensemble = EnsembleForecaster("test_ensemble", self.base_models)
        
        # Create sample data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        })
        self.y = pd.Series(np.random.randn(50), name='target')
    
    def test_initialization(self):
        """Test ensemble initialization."""
        self.assertEqual(len(self.ensemble.base_models), 3)
        self.assertEqual(self.ensemble.combination_method, 'average')
        self.assertIsNone(self.ensemble.model_weights)
    
    def test_initialization_with_weights(self):
        """Test ensemble initialization with weights."""
        weights = [0.5, 0.3, 0.2]
        ensemble = EnsembleForecaster("weighted_ensemble", self.base_models, weights=weights)
        
        self.assertEqual(ensemble.model_weights, weights)
    
    def test_fit_predict_workflow(self):
        """Test ensemble fit and predict workflow."""
        # Fit ensemble
        self.ensemble.fit(self.X, self.y)
        self.assertTrue(self.ensemble.is_fitted)
        
        # Check that all base models are fitted
        for model in self.ensemble.base_models:
            self.assertTrue(model.is_fitted)
        
        # Make predictions
        predictions = self.ensemble.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        
        # Test predictions with uncertainty
        pred, unc = self.ensemble.predict_with_uncertainty(self.X)
        self.assertEqual(len(pred), len(self.X))
        self.assertEqual(len(unc), len(self.X))


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()
        
        # Create sample predictions
        np.random.seed(42)
        self.y_true = np.random.randn(100) * 10 + 50  # Mean around 50
        self.y_pred = self.y_true + np.random.randn(100) * 2  # Add some noise
    
    def test_evaluate_model(self):
        """Test model evaluation with multiple metrics."""
        metrics = ['mae', 'rmse', 'mape', 'r2']
        results = self.evaluator.evaluate_model(self.y_true, self.y_pred, metrics)
        
        self.assertEqual(len(results), len(metrics))
        for metric in metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], (int, float))
            self.assertFalse(np.isnan(results[metric]))
    
    def test_individual_metrics(self):
        """Test individual metric calculations."""
        # Test MAE
        mae = self.evaluator._mean_absolute_error(self.y_true, self.y_pred)
        self.assertGreater(mae, 0)
        
        # Test RMSE
        rmse = self.evaluator._root_mean_squared_error(self.y_true, self.y_pred)
        self.assertGreater(rmse, 0)
        self.assertGreaterEqual(rmse, mae)  # RMSE should be >= MAE
        
        # Test MAPE
        mape = self.evaluator._mean_absolute_percentage_error(self.y_true, self.y_pred)
        self.assertGreater(mape, 0)
        
        # Test R²
        r2 = self.evaluator._r_squared(self.y_true, self.y_pred)
        self.assertLessEqual(r2, 1.0)
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation."""
        # Create data with clear directional pattern
        y_true = np.array([1, 2, 3, 2, 1])
        y_pred = np.array([1.1, 2.1, 2.9, 1.9, 1.1])
        
        accuracy = self.evaluator._directional_accuracy(y_true, y_pred)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 100)
    
    def test_cross_validate_model(self):
        """Test cross-validation functionality."""
        # Create mock model and data
        model = MockForecaster("cv_test")
        
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'timestamp': dates
        })
        y = pd.Series(np.random.randn(100), name='target')
        
        # Perform cross-validation
        cv_results = self.evaluator.cross_validate_model(
            model, X, y, cv_strategy='time_series', n_splits=3
        )
        
        self.assertIn('n_folds', cv_results)
        self.assertIn('fold_results', cv_results)
        self.assertIn('mae_mean', cv_results)
        self.assertIn('mae_std', cv_results)


class TestTimeSeriesValidator(unittest.TestCase):
    """Test cases for TimeSeriesValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = TimeSeriesValidator()
        
        # Create time series data
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        self.data = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.randn(100)
        })
    
    def test_time_series_split(self):
        """Test time series splitting."""
        splits = list(self.validator.time_series_split(self.data, n_splits=3))
        
        self.assertEqual(len(splits), 3)
        
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)
            
            # Check temporal order
            max_train_idx = train_idx.max()
            min_test_idx = test_idx.min()
            self.assertLess(max_train_idx, min_test_idx)
    
    def test_walk_forward_validation(self):
        """Test walk-forward validation."""
        splits = list(self.validator.walk_forward_validation(
            self.data, initial_train_size=50, step_size=5, forecast_horizon=5
        ))
        
        self.assertGreater(len(splits), 0)
        
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertEqual(len(test_idx), 5)  # Should match forecast_horizon
    
    def test_seasonal_validation(self):
        """Test seasonal validation."""
        # Create longer data for seasonal validation
        dates = pd.date_range('2024-01-01', periods=200, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.randn(200)
        })
        
        splits = list(self.validator.seasonal_validation(
            data, season_length=24, n_seasons=3
        ))
        
        self.assertGreater(len(splits), 0)
        
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertEqual(len(test_idx), 24)  # Should match season_length


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trainer = ModelTrainer()
        self.trainer.register_model('mock', MockForecaster)
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        self.X_train = pd.DataFrame({
            'feature1': np.random.randn(80),
            'feature2': np.random.randn(80),
            'timestamp': dates[:80]
        })
        self.y_train = pd.Series(np.random.randn(80), name='target')
        
        self.X_val = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20),
            'timestamp': dates[80:]
        })
        self.y_val = pd.Series(np.random.randn(20), name='target')
    
    def test_model_registration(self):
        """Test model registration."""
        self.assertIn('mock', self.trainer.model_registry)
        self.assertEqual(self.trainer.model_registry['mock'], MockForecaster)
    
    def test_train_single_model(self):
        """Test single model training."""
        model = MockForecaster("single_test")
        
        results = self.trainer.train_single_model(
            model, self.X_train, self.y_train, self.X_val, self.y_val
        )
        
        self.assertTrue(results['success'])
        self.assertIn('train_metrics', results)
        self.assertIn('validation_metrics', results)
        self.assertIn('training_duration_seconds', results)
        
        # Check model is stored
        self.assertIn(model.model_id, self.trainer.trained_models)
    
    def test_train_multiple_models(self):
        """Test multiple model training."""
        model_configs = [
            {'type': 'mock', 'name': 'model1'},
            {'type': 'mock', 'name': 'model2'}
        ]
        
        results = self.trainer.train_multiple_models(
            model_configs, self.X_train, self.y_train, self.X_val, self.y_val
        )
        
        self.assertEqual(len(results), 2)
        
        for result in results.values():
            self.assertTrue(result['success'])
    
    def test_cross_validate_model(self):
        """Test model cross-validation."""
        model = MockForecaster("cv_test")
        
        # Combine train and validation data for CV
        X_full = pd.concat([self.X_train, self.X_val])
        y_full = pd.concat([self.y_train, self.y_val])
        
        cv_results = self.trainer.cross_validate_model(
            model, X_full, y_full, n_splits=3
        )
        
        self.assertIn('n_folds', cv_results)
        self.assertIn('mae_mean', cv_results)
    
    def test_create_ensemble(self):
        """Test ensemble creation."""
        # Train some models first
        model1 = MockForecaster("ensemble_model1")
        model2 = MockForecaster("ensemble_model2")
        
        self.trainer.train_single_model(model1, self.X_train, self.y_train)
        self.trainer.train_single_model(model2, self.X_train, self.y_train)
        
        # Create ensemble
        ensemble = self.trainer.create_ensemble(
            [model1.model_id, model2.model_id], 
            "test_ensemble"
        )
        
        self.assertIsInstance(ensemble, EnsembleForecaster)
        self.assertEqual(len(ensemble.base_models), 2)
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        # Train a few models
        model1 = MockForecaster("comparison_model1")
        model2 = MockForecaster("comparison_model2")
        
        self.trainer.train_single_model(model1, self.X_train, self.y_train, self.X_val, self.y_val)
        self.trainer.train_single_model(model2, self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Get comparison
        comparison = self.trainer.get_model_comparison()
        
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 2)
        self.assertIn('model_name', comparison.columns)
        self.assertIn('model_type', comparison.columns)
    
    def test_training_summary(self):
        """Test training summary."""
        # Train a model
        model = MockForecaster("summary_test")
        self.trainer.train_single_model(model, self.X_train, self.y_train)
        
        summary = self.trainer.get_training_summary()
        
        self.assertIn('total_trainings', summary)
        self.assertIn('successful_trainings', summary)
        self.assertIn('trained_models', summary)
        self.assertIn('success_rate', summary)
        
        self.assertEqual(summary['total_trainings'], 1)
        self.assertEqual(summary['successful_trainings'], 1)
        self.assertEqual(summary['success_rate'], 1.0)


class TestHyperparameterTuner(unittest.TestCase):
    """Test cases for HyperparameterTuner."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple parameterized mock model
        class ParameterizedMockForecaster(MockForecaster):
            def __init__(self, model_name="param_mock", param1=1.0, param2=10, **kwargs):
                super().__init__(model_name, **kwargs)
                self.param1 = param1
                self.param2 = param2
                self.parameters.update({'param1': param1, 'param2': param2})
        
        self.model_class = ParameterizedMockForecaster
        self.param_grid = {
            'param1': [0.5, 1.0, 1.5],
            'param2': [5, 10, 15]
        }
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        self.X = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'timestamp': dates
        })
        self.y = pd.Series(np.random.randn(50), name='target')
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning process."""
        tuner = HyperparameterTuner(
            model_class=self.model_class,
            param_grid=self.param_grid,
            cv_strategy='time_series',
            scoring_metric='mae',
            n_splits=2
        )
        
        tuner.fit(self.X, self.y)
        
        self.assertIsNotNone(tuner.best_params_)
        self.assertIsNotNone(tuner.best_score_)
        self.assertGreater(len(tuner.cv_results_), 0)
        
        # Check that best parameters are from the grid
        self.assertIn(tuner.best_params_['param1'], self.param_grid['param1'])
        self.assertIn(tuner.best_params_['param2'], self.param_grid['param2'])
        
        # Get best model
        best_model = tuner.get_best_model()
        self.assertIsInstance(best_model, self.model_class)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)