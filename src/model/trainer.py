"""
Model training orchestrator for the energy forecasting platform.
Coordinates model training, validation, and evaluation workflows.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json

from .base_model import BaseForecaster, TimeSeriesForecaster, EnsembleForecaster
from .validation import ModelEvaluator, TimeSeriesValidator, HyperparameterTuner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrates the complete model training workflow.
    Handles data preparation, training, validation, and model management.
    """
    
    def __init__(self, 
                 model_registry: Optional[Dict[str, Any]] = None,
                 default_cv_strategy: str = 'time_series',
                 default_metrics: Optional[List[str]] = None):
        """
        Initialize the model trainer.
        
        Args:
            model_registry: Registry of available model classes
            default_cv_strategy: Default cross-validation strategy
            default_metrics: Default evaluation metrics
        """
        self.model_registry = model_registry or {}
        self.default_cv_strategy = default_cv_strategy
        self.default_metrics = default_metrics or ['mae', 'rmse', 'mape', 'r2']
        
        self.evaluator = ModelEvaluator()
        self.validator = TimeSeriesValidator()
        
        # Training history
        self.training_history = []
        self.trained_models = {}
        
        logger.info("ModelTrainer initialized")
    
    def register_model(self, name: str, model_class: type) -> None:
        """
        Register a model class in the registry.
        
        Args:
            name: Name to register the model under
            model_class: Model class to register
        """
        self.model_registry[name] = model_class
        logger.info(f"Registered model: {name}")
    
    def train_single_model(self,
                          model: BaseForecaster,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None,
                          **training_kwargs) -> Dict[str, Any]:
        """
        Train a single model with validation.
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **training_kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training model: {model.model_name}")
        
        # Record training start
        training_start = datetime.now()
        
        try:
            # Train the model
            model.fit(X_train, y_train, **training_kwargs)
            
            # Evaluate on training data
            y_train_pred = model.predict(X_train)
            train_metrics = self.evaluator.evaluate_model(
                y_train.values, y_train_pred, self.default_metrics
            )
            
            # Evaluate on validation data if provided
            val_metrics = {}
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
                val_metrics = self.evaluator.evaluate_model(
                    y_val.values, y_val_pred, self.default_metrics
                )
            
            # Record training completion
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            
            # Compile results
            results = {
                'model_id': model.model_id,
                'model_name': model.model_name,
                'model_type': model.model_type,
                'training_start': training_start,
                'training_end': training_end,
                'training_duration_seconds': training_duration,
                'train_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'training_data_shape': X_train.shape,
                'validation_data_shape': X_val.shape if X_val is not None else None,
                'feature_names': list(X_train.columns),
                'target_name': y_train.name or 'target',
                'success': True
            }
            
            # Update model metrics
            model.training_metrics = train_metrics
            model.validation_metrics = val_metrics
            
            # Store trained model
            self.trained_models[model.model_id] = model
            self.training_history.append(results)
            
            logger.info(f"Model training completed successfully in {training_duration:.2f} seconds")
            logger.info(f"Training RMSE: {train_metrics.get('rmse', 'N/A'):.4f}")
            if val_metrics:
                logger.info(f"Validation RMSE: {val_metrics.get('rmse', 'N/A'):.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            
            results = {
                'model_id': model.model_id,
                'model_name': model.model_name,
                'model_type': model.model_type,
                'training_start': training_start,
                'training_end': datetime.now(),
                'error': str(e),
                'success': False
            }
            
            self.training_history.append(results)
            return results
    
    def train_multiple_models(self,
                             model_configs: List[Dict[str, Any]],
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_val: Optional[pd.DataFrame] = None,
                             y_val: Optional[pd.Series] = None,
                             **training_kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models with the same data.
        
        Args:
            model_configs: List of model configurations
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **training_kwargs: Additional training parameters
            
        Returns:
            Dictionary mapping model IDs to training results
        """
        logger.info(f"Training {len(model_configs)} models")
        
        results = {}
        
        for i, config in enumerate(model_configs):
            logger.info(f"Training model {i+1}/{len(model_configs)}")
            
            try:
                # Create model instance
                model = self._create_model_from_config(config)
                
                # Train model
                result = self.train_single_model(
                    model, X_train, y_train, X_val, y_val, **training_kwargs
                )
                
                results[model.model_id] = result
                
            except Exception as e:
                logger.error(f"Failed to train model {i+1}: {e}")
                results[f"failed_model_{i}"] = {
                    'error': str(e),
                    'config': config,
                    'success': False
                }
        
        return results
    
    def cross_validate_model(self,
                            model: BaseForecaster,
                            X: pd.DataFrame,
                            y: pd.Series,
                            cv_strategy: Optional[str] = None,
                            n_splits: int = 5,
                            metrics: Optional[List[str]] = None,
                            **cv_kwargs) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model instance to validate
            X: Feature matrix
            y: Target variable
            cv_strategy: Cross-validation strategy
            n_splits: Number of splits
            metrics: List of metrics to compute
            **cv_kwargs: Additional CV parameters
            
        Returns:
            Cross-validation results
        """
        cv_strategy = cv_strategy or self.default_cv_strategy
        metrics = metrics or self.default_metrics
        
        logger.info(f"Cross-validating model: {model.model_name}")
        logger.info(f"CV strategy: {cv_strategy}, splits: {n_splits}")
        
        results = self.evaluator.cross_validate_model(
            model, X, y, cv_strategy, n_splits, metrics, **cv_kwargs
        )
        
        # Log results
        for metric in metrics:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in results and std_key in results:
                logger.info(f"CV {metric.upper()}: {results[mean_key]:.4f} (+/- {results[std_key]:.4f})")
        
        return results
    
    def tune_hyperparameters(self,
                            model_class: type,
                            param_grid: Dict[str, List[Any]],
                            X: pd.DataFrame,
                            y: pd.Series,
                            cv_strategy: Optional[str] = None,
                            scoring_metric: str = 'rmse',
                            n_splits: int = 3,
                            **cv_kwargs) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a model.
        
        Args:
            model_class: Model class to tune
            param_grid: Parameter grid to search
            X: Feature matrix
            y: Target variable
            cv_strategy: Cross-validation strategy
            scoring_metric: Metric to optimize
            n_splits: Number of CV splits
            **cv_kwargs: Additional CV parameters
            
        Returns:
            Tuning results with best parameters
        """
        cv_strategy = cv_strategy or self.default_cv_strategy
        
        logger.info(f"Tuning hyperparameters for {model_class.__name__}")
        logger.info(f"Parameter grid: {param_grid}")
        
        tuner = HyperparameterTuner(
            model_class=model_class,
            param_grid=param_grid,
            cv_strategy=cv_strategy,
            scoring_metric=scoring_metric,
            n_splits=n_splits
        )
        
        tuner.fit(X, y, **cv_kwargs)
        
        results = {
            'best_params': tuner.best_params_,
            'best_score': tuner.best_score_,
            'cv_results': tuner.cv_results_,
            'best_model': tuner.get_best_model()
        }
        
        logger.info(f"Best parameters: {tuner.best_params_}")
        logger.info(f"Best {scoring_metric}: {tuner.best_score_:.4f}")
        
        return results
    
    def create_ensemble(self,
                       model_ids: List[str],
                       ensemble_name: str,
                       combination_method: str = 'average',
                       weights: Optional[List[float]] = None) -> EnsembleForecaster:
        """
        Create an ensemble from trained models.
        
        Args:
            model_ids: List of model IDs to ensemble
            ensemble_name: Name for the ensemble
            combination_method: How to combine predictions
            weights: Optional weights for models
            
        Returns:
            Ensemble forecaster instance
        """
        # Get base models
        base_models = []
        for model_id in model_ids:
            if model_id in self.trained_models:
                base_models.append(self.trained_models[model_id])
            else:
                raise ValueError(f"Model {model_id} not found in trained models")
        
        # Create ensemble
        ensemble = EnsembleForecaster(
            model_name=ensemble_name,
            base_models=base_models,
            combination_method=combination_method,
            weights=weights
        )
        
        logger.info(f"Created ensemble '{ensemble_name}' with {len(base_models)} models")
        
        return ensemble
    
    def save_model(self, model_id: str, filepath: Union[str, Path]) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_id: ID of the model to save
            filepath: Path where to save the model
        """
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found in trained models")
        
        model = self.trained_models[model_id]
        model.save_model(filepath)
        
        logger.info(f"Saved model {model_id} to {filepath}")
    
    def load_model(self, filepath: Union[str, Path], model_class: type) -> BaseForecaster:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            model_class: Class of the model to load
            
        Returns:
            Loaded model instance
        """
        model = model_class.load_model(filepath)
        self.trained_models[model.model_id] = model
        
        logger.info(f"Loaded model {model.model_id} from {filepath}")
        return model
    
    def get_model_comparison(self, metric: str = 'rmse') -> pd.DataFrame:
        """
        Compare performance of all trained models.
        
        Args:
            metric: Metric to compare models by
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_id, model in self.trained_models.items():
            model_info = model.get_model_info()
            
            row = {
                'model_id': model_id,
                'model_name': model_info['model_name'],
                'model_type': model_info['model_type'],
                'is_fitted': model_info['is_fitted']
            }
            
            # Add training metrics
            for m, value in model_info['training_metrics'].items():
                row[f'train_{m}'] = value
            
            # Add validation metrics
            for m, value in model_info['validation_metrics'].items():
                row[f'val_{m}'] = value
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by validation metric if available, otherwise training metric
        sort_column = f'val_{metric}' if f'val_{metric}' in df.columns else f'train_{metric}'
        if sort_column in df.columns:
            ascending = metric in ['mae', 'mse', 'rmse', 'mape']  # Lower is better for these metrics
            df = df.sort_values(sort_column, ascending=ascending)
        
        return df
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> BaseForecaster:
        """
        Create a model instance from configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Model instance
        """
        model_type = config.get('type')
        model_name = config.get('name', 'unnamed_model')
        
        if model_type not in self.model_registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self.model_registry[model_type]
        model_params = config.get('parameters', {})
        
        return model_class(model_name=model_name, **model_params)
    
    def export_training_history(self, filepath: Union[str, Path]) -> None:
        """
        Export training history to JSON file.
        
        Args:
            filepath: Path to save the history
        """
        # Convert datetime objects to strings for JSON serialization
        history_serializable = []
        for record in self.training_history:
            record_copy = record.copy()
            for key, value in record_copy.items():
                if isinstance(value, datetime):
                    record_copy[key] = value.isoformat()
            history_serializable.append(record_copy)
        
        with open(filepath, 'w') as f:
            json.dump(history_serializable, f, indent=2, default=str)
        
        logger.info(f"Training history exported to {filepath}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training activities.
        
        Returns:
            Dictionary with training summary
        """
        successful_trainings = [h for h in self.training_history if h.get('success', False)]
        failed_trainings = [h for h in self.training_history if not h.get('success', False)]
        
        summary = {
            'total_trainings': len(self.training_history),
            'successful_trainings': len(successful_trainings),
            'failed_trainings': len(failed_trainings),
            'trained_models': len(self.trained_models),
            'model_types': list(set(model.model_type for model in self.trained_models.values())),
            'success_rate': len(successful_trainings) / len(self.training_history) if self.training_history else 0
        }
        
        if successful_trainings:
            durations = [h['training_duration_seconds'] for h in successful_trainings]
            summary['avg_training_duration'] = np.mean(durations)
            summary['total_training_time'] = np.sum(durations)
        
        return summary