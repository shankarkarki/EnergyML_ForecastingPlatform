"""
Cross-validation utilities for time series forecasting models.
Provides time-aware validation strategies and performance evaluation.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesValidator:
    """
    Time series cross-validation with various splitting strategies.
    Ensures temporal order is preserved during validation.
    """
    
    def __init__(self, time_column: str = 'timestamp'):
        """
        Initialize the time series validator.
        
        Args:
            time_column: Name of the timestamp column
        """
        self.time_column = time_column
        
    def time_series_split(self, 
                         data: pd.DataFrame, 
                         n_splits: int = 5, 
                         test_size: Optional[int] = None,
                         gap: int = 0) -> Iterator[Tuple[pd.Index, pd.Index]]:
        """
        Time series cross-validation split.
        
        Args:
            data: Input data with time column
            n_splits: Number of splits
            test_size: Size of test set (if None, uses equal splits)
            gap: Gap between train and test sets (in periods)
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        if self.time_column not in data.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data")
        
        # Sort by time
        data_sorted = data.sort_values(self.time_column)
        indices = data_sorted.index
        
        if test_size is None:
            # Use sklearn's TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
            for train_idx, test_idx in tscv.split(indices):
                yield indices[train_idx], indices[test_idx]
        else:
            # Custom split with fixed test size
            total_size = len(indices)
            
            for i in range(n_splits):
                # Calculate split points
                test_end = total_size - i * (test_size + gap)
                test_start = test_end - test_size
                train_end = test_start - gap
                
                if train_end <= 0:
                    break
                
                train_indices = indices[:train_end]
                test_indices = indices[test_start:test_end]
                
                yield train_indices, test_indices
    
    def walk_forward_validation(self, 
                               data: pd.DataFrame,
                               initial_train_size: int,
                               step_size: int = 1,
                               forecast_horizon: int = 1) -> Iterator[Tuple[pd.Index, pd.Index]]:
        """
        Walk-forward validation for time series.
        
        Args:
            data: Input data with time column
            initial_train_size: Initial training set size
            step_size: Step size for moving window
            forecast_horizon: Number of periods to forecast
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        if self.time_column not in data.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data")
        
        # Sort by time
        data_sorted = data.sort_values(self.time_column)
        indices = data_sorted.index
        total_size = len(indices)
        
        train_start = 0
        train_end = initial_train_size
        
        while train_end + forecast_horizon <= total_size:
            test_start = train_end
            test_end = min(train_end + forecast_horizon, total_size)
            
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
            
            # Move window
            train_end += step_size
            if step_size == forecast_horizon:
                train_start += step_size  # Sliding window
            # Otherwise expanding window (train_start stays 0)
    
    def seasonal_validation(self,
                           data: pd.DataFrame,
                           season_length: int = 24,  # 24 hours for daily seasonality
                           n_seasons: int = 7) -> Iterator[Tuple[pd.Index, pd.Index]]:
        """
        Seasonal cross-validation for time series.
        
        Args:
            data: Input data with time column
            season_length: Length of one season (e.g., 24 for daily)
            n_seasons: Number of seasons to use for validation
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        if self.time_column not in data.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data")
        
        # Sort by time
        data_sorted = data.sort_values(self.time_column)
        indices = data_sorted.index
        total_size = len(indices)
        
        # Calculate number of complete seasons
        total_seasons = total_size // season_length
        
        if total_seasons < n_seasons + 1:
            raise ValueError(f"Not enough data for {n_seasons} seasons of length {season_length}")
        
        for i in range(total_seasons - n_seasons):
            # Use i-th season as test, previous seasons as train
            test_start = i * season_length
            test_end = (i + 1) * season_length
            
            # Train on all data before test season
            train_indices = indices[:test_start]
            test_indices = indices[test_start:test_end]
            
            if len(train_indices) > 0:
                yield train_indices, test_indices


class ModelEvaluator:
    """
    Comprehensive model evaluation with time series specific metrics.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.metrics_registry = {
            'mae': self._mean_absolute_error,
            'mse': self._mean_squared_error,
            'rmse': self._root_mean_squared_error,
            'mape': self._mean_absolute_percentage_error,
            'smape': self._symmetric_mean_absolute_percentage_error,
            'r2': self._r_squared,
            'directional_accuracy': self._directional_accuracy,
            'forecast_bias': self._forecast_bias
        }
    
    def evaluate_model(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metrics to compute (if None, uses all)
            
        Returns:
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = list(self.metrics_registry.keys())
        
        results = {}
        
        for metric in metrics:
            if metric in self.metrics_registry:
                try:
                    value = self.metrics_registry[metric](y_true, y_pred)
                    results[metric] = value
                except Exception as e:
                    logger.warning(f"Failed to compute {metric}: {e}")
                    results[metric] = np.nan
            else:
                logger.warning(f"Unknown metric: {metric}")
        
        return results
    
    def cross_validate_model(self,
                            model,
                            X: pd.DataFrame,
                            y: pd.Series,
                            cv_strategy: str = 'time_series',
                            n_splits: int = 5,
                            metrics: Optional[List[str]] = None,
                            **cv_kwargs) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model instance with fit/predict methods
            X: Feature matrix
            y: Target variable
            cv_strategy: Cross-validation strategy
            n_splits: Number of splits
            metrics: List of metrics to compute
            **cv_kwargs: Additional CV parameters
            
        Returns:
            Dictionary with CV results
        """
        validator = TimeSeriesValidator(time_column=cv_kwargs.get('time_column', 'timestamp'))
        
        # Combine X and y for splitting
        data = X.copy()
        data['_target'] = y
        
        # Choose CV strategy
        if cv_strategy == 'time_series':
            cv_splits = validator.time_series_split(data, n_splits=n_splits, **cv_kwargs)
        elif cv_strategy == 'walk_forward':
            cv_splits = validator.walk_forward_validation(data, **cv_kwargs)
        elif cv_strategy == 'seasonal':
            cv_splits = validator.seasonal_validation(data, **cv_kwargs)
        else:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}")
        
        # Perform cross-validation
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            logger.info(f"Processing fold {fold + 1}")
            
            # Split data
            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]
            
            # Train model
            model_copy = self._copy_model(model)
            model_copy.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model_copy.predict(X_test)
            
            # Evaluate
            fold_metrics = self.evaluate_model(y_test.values, y_pred, metrics)
            fold_metrics['fold'] = fold
            fold_metrics['train_size'] = len(X_train)
            fold_metrics['test_size'] = len(X_test)
            
            fold_results.append(fold_metrics)
        
        # Aggregate results
        cv_results = self._aggregate_cv_results(fold_results)
        
        return cv_results
    
    def _copy_model(self, model):
        """Create a copy of the model for CV."""
        # This is a simplified implementation
        # In practice, you might need model-specific copying logic
        try:
            from copy import deepcopy
            return deepcopy(model)
        except:
            # Fallback: create new instance with same parameters
            return model.__class__(**model.get_model_parameters())
    
    def _aggregate_cv_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation results across folds."""
        if not fold_results:
            return {}
        
        # Get metric names (excluding non-numeric fields)
        metric_names = [k for k in fold_results[0].keys() 
                       if k not in ['fold', 'train_size', 'test_size'] and 
                       isinstance(fold_results[0][k], (int, float))]
        
        aggregated = {
            'n_folds': len(fold_results),
            'fold_results': fold_results
        }
        
        # Calculate statistics for each metric
        for metric in metric_names:
            values = [fold[metric] for fold in fold_results if not np.isnan(fold[metric])]
            
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
                aggregated[f'{metric}_median'] = np.median(values)
        
        return aggregated
    
    # Metric implementations
    def _mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    def _mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return mean_squared_error(y_true, y_pred)
    
    def _root_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def _mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _symmetric_mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        
        if not np.any(mask):
            return 0.0
        
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    def _r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared."""
        return r2_score(y_true, y_pred)
    
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (percentage of correct direction predictions)."""
        if len(y_true) < 2:
            return np.nan
        
        # Calculate direction changes
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        # Calculate accuracy
        correct_directions = np.sum(true_direction == pred_direction)
        total_directions = len(true_direction)
        
        return (correct_directions / total_directions) * 100 if total_directions > 0 else np.nan
    
    def _forecast_bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate forecast bias (mean of residuals)."""
        return np.mean(y_pred - y_true)


class HyperparameterTuner:
    """
    Hyperparameter tuning for forecasting models using time series cross-validation.
    """
    
    def __init__(self, 
                 model_class,
                 param_grid: Dict[str, List[Any]],
                 cv_strategy: str = 'time_series',
                 scoring_metric: str = 'rmse',
                 n_splits: int = 3):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            model_class: Model class to tune
            param_grid: Dictionary of parameters to search
            cv_strategy: Cross-validation strategy
            scoring_metric: Metric to optimize
            n_splits: Number of CV splits
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv_strategy = cv_strategy
        self.scoring_metric = scoring_metric
        self.n_splits = n_splits
        self.evaluator = ModelEvaluator()
        
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **cv_kwargs) -> 'HyperparameterTuner':
        """
        Perform hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target variable
            **cv_kwargs: Additional CV parameters
            
        Returns:
            Self for method chaining
        """
        param_combinations = list(self._generate_param_combinations())
        logger.info(f"Starting hyperparameter tuning with {len(param_combinations)} combinations")
        
        best_score = np.inf if self.scoring_metric in ['mae', 'mse', 'rmse', 'mape'] else -np.inf
        best_params = None
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing parameter combination {i+1}: {params}")
            
            # Create model with current parameters
            model = self.model_class(**params)
            
            # Perform cross-validation
            cv_results = self.evaluator.cross_validate_model(
                model, X, y, 
                cv_strategy=self.cv_strategy,
                n_splits=self.n_splits,
                metrics=[self.scoring_metric],
                **cv_kwargs
            )
            
            # Get mean score
            score_key = f'{self.scoring_metric}_mean'
            if score_key in cv_results:
                score = cv_results[score_key]
                
                # Check if this is the best score
                is_better = (score < best_score if self.scoring_metric in ['mae', 'mse', 'rmse', 'mape'] 
                           else score > best_score)
                
                if is_better:
                    best_score = score
                    best_params = params
                
                # Store results
                result = {
                    'params': params,
                    'score': score,
                    'cv_results': cv_results
                }
                self.cv_results_.append(result)
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {self.scoring_metric}: {best_score}")
        
        return self
    
    def _generate_param_combinations(self) -> Iterator[Dict[str, Any]]:
        """Generate all parameter combinations from the grid."""
        import itertools
        
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
    
    def get_best_model(self) -> Any:
        """
        Get the best model with optimal parameters.
        
        Returns:
            Model instance with best parameters
        """
        if self.best_params_ is None:
            raise ValueError("Must call fit() before getting best model")
        
        return self.model_class(**self.best_params_)