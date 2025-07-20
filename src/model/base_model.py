"""
Base model interface and abstract classes for the energy forecasting platform.
Provides common interface for all forecasting models (statistical, ML, deep learning).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pickle
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    Defines the common interface that all models must implement.
    """
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        """
        Initialize the base forecaster.
        
        Args:
            model_name: Human-readable name for the model
            model_type: Type category (e.g., 'statistical', 'ml', 'deep_learning')
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model_id = f"{model_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.parameters = kwargs
        self.is_fitted = False
        self.training_history = []
        self.feature_names = []
        self.target_name = ""
        self.model_metadata = {
            'created_at': datetime.now(),
            'model_name': model_name,
            'model_type': model_type,
            'model_id': self.model_id,
            'version': '1.0.0'
        }
        
        # Model performance tracking
        self.training_metrics = {}
        self.validation_metrics = {}
        self.test_metrics = {}
        
        logger.info(f"Initialized {self.model_type} model: {self.model_name}")
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseForecaster':
        """
        Train the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, X: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (predictions, uncertainty_estimates)
        """
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if available.
        
        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        return None
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get model parameters and hyperparameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.parameters.copy()
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get training history and metrics.
        
        Returns:
            List of training history records
        """
        return self.training_history.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model metadata and performance
        """
        info = self.model_metadata.copy()
        info.update({
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names.copy(),
            'target_name': self.target_name,
            'training_metrics': self.training_metrics.copy(),
            'validation_metrics': self.validation_metrics.copy(),
            'test_metrics': self.test_metrics.copy(),
            'parameters': self.get_model_parameters()
        })
        return info
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model data
        model_data = {
            'model_metadata': self.model_metadata,
            'parameters': self.parameters,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'test_metrics': self.test_metrics,
            'training_history': self.training_history,
            'model_state': self._get_model_state()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'BaseForecaster':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls.__new__(cls)
        instance.model_metadata = model_data['model_metadata']
        instance.parameters = model_data['parameters']
        instance.feature_names = model_data['feature_names']
        instance.target_name = model_data['target_name']
        instance.training_metrics = model_data['training_metrics']
        instance.validation_metrics = model_data['validation_metrics']
        instance.test_metrics = model_data['test_metrics']
        instance.training_history = model_data['training_history']
        instance.is_fitted = True
        
        # Restore basic attributes from metadata
        instance.model_name = instance.model_metadata['model_name']
        instance.model_type = instance.model_metadata['model_type']
        instance.model_id = instance.model_metadata['model_id']
        
        # Restore model-specific state
        instance._set_model_state(model_data['model_state'])
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """
        Get model-specific state for serialization.
        
        Returns:
            Dictionary containing model state
        """
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """
        Restore model-specific state from serialization.
        
        Args:
            state: Dictionary containing model state
        """
        pass
    
    def validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Validate input data format and consistency.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            
        Raises:
            ValueError: If input data is invalid
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("X cannot be empty")
        
        if y is not None:
            if not isinstance(y, pd.Series):
                raise ValueError("y must be a pandas Series")
            
            if len(X) != len(y):
                raise ValueError("X and y must have the same number of samples")
            
            if y.isnull().any():
                raise ValueError("y cannot contain null values")
        
        # Check for required features if model is fitted
        if self.is_fitted and self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.model_type.title()}Model({self.model_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}(name='{self.model_name}', type='{self.model_type}', fitted={self.is_fitted})"


class TimeSeriesForecaster(BaseForecaster):
    """
    Base class for time series forecasting models.
    Extends BaseForecaster with time series specific functionality.
    """
    
    def __init__(self, model_name: str, model_type: str, **kwargs):
        """
        Initialize the time series forecaster.
        
        Args:
            model_name: Human-readable name for the model
            model_type: Type category
            **kwargs: Additional parameters
        """
        super().__init__(model_name, model_type, **kwargs)
        self.time_column = kwargs.get('time_column', 'timestamp')
        self.forecast_horizon = kwargs.get('forecast_horizon', 24)  # Default 24 hours
        self.frequency = kwargs.get('frequency', 'H')  # Hourly by default
        
    def prepare_time_series_data(self, data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare time series data for training.
        
        Args:
            data: Input data with time column
            target_col: Name of target column
            
        Returns:
            Tuple of (features, target)
        """
        if self.time_column not in data.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data")
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Sort by time
        data_sorted = data.sort_values(self.time_column).copy()
        
        # Extract features and target
        feature_cols = [col for col in data_sorted.columns if col != target_col]
        X = data_sorted[feature_cols]
        y = data_sorted[target_col]
        
        return X, y
    
    def create_forecast_dates(self, last_date: datetime, horizon: Optional[int] = None) -> pd.DatetimeIndex:
        """
        Create future dates for forecasting.
        
        Args:
            last_date: Last date in training data
            horizon: Number of periods to forecast (uses default if None)
            
        Returns:
            DatetimeIndex with future dates
        """
        if horizon is None:
            horizon = self.forecast_horizon
        
        return pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=horizon,
            freq=self.frequency.lower()  # Convert 'H' to 'h'
        )
    
    @abstractmethod
    def forecast(self, horizon: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """
        Generate time series forecast.
        
        Args:
            horizon: Number of periods to forecast
            **kwargs: Additional forecasting parameters
            
        Returns:
            DataFrame with forecast results
        """
        pass


class EnsembleForecaster(BaseForecaster):
    """
    Base class for ensemble forecasting models.
    Combines multiple models for improved performance.
    """
    
    def __init__(self, model_name: str, base_models: List[BaseForecaster], **kwargs):
        """
        Initialize the ensemble forecaster.
        
        Args:
            model_name: Name for the ensemble
            base_models: List of base models to ensemble
            **kwargs: Additional parameters
        """
        super().__init__(model_name, 'ensemble', **kwargs)
        self.base_models = base_models
        self.model_weights = kwargs.get('weights', None)
        self.combination_method = kwargs.get('combination_method', 'average')
        
        if self.model_weights and len(self.model_weights) != len(self.base_models):
            raise ValueError("Number of weights must match number of base models")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'EnsembleForecaster':
        """
        Train all base models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        self.validate_input(X, y)
        
        logger.info(f"Training ensemble with {len(self.base_models)} base models")
        
        # Train each base model
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {model.model_name}")
            model.fit(X, y, **kwargs)
        
        self.is_fitted = True
        self.feature_names = list(X.columns)
        self.target_name = y.name or 'target'
        
        logger.info("Ensemble training completed")
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Array of ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        self.validate_input(X)
        
        # Get predictions from all base models
        predictions = []
        for model in self.base_models:
            pred = model.predict(X, **kwargs)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Combine predictions
        if self.combination_method == 'average':
            if self.model_weights:
                ensemble_pred = np.average(predictions, axis=0, weights=self.model_weights)
            else:
                ensemble_pred = np.mean(predictions, axis=0)
        elif self.combination_method == 'median':
            ensemble_pred = np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions with uncertainty.
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (predictions, uncertainty_estimates)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get predictions and uncertainties from all base models
        predictions = []
        uncertainties = []
        
        for model in self.base_models:
            pred, unc = model.predict_with_uncertainty(X, **kwargs)
            predictions.append(pred)
            uncertainties.append(unc)
        
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        
        # Ensemble prediction
        if self.model_weights:
            ensemble_pred = np.average(predictions, axis=0, weights=self.model_weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)
        
        # Ensemble uncertainty (combination of model disagreement and individual uncertainties)
        model_disagreement = np.std(predictions, axis=0)
        avg_uncertainty = np.mean(uncertainties, axis=0)
        ensemble_uncertainty = np.sqrt(model_disagreement**2 + avg_uncertainty**2)
        
        return ensemble_pred, ensemble_uncertainty
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get ensemble-specific state for serialization."""
        return {
            'base_models': [model._get_model_state() for model in self.base_models],
            'model_weights': self.model_weights,
            'combination_method': self.combination_method
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Restore ensemble-specific state from serialization."""
        # Note: This is a simplified implementation
        # In practice, you'd need to properly restore base models
        self.model_weights = state['model_weights']
        self.combination_method = state['combination_method']