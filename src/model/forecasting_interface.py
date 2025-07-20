"""
Unified forecasting interface for the energy forecasting platform.
Provides standardized forecast generation with confidence intervals and uncertainty measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass

from .base_model import BaseForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastHorizon(Enum):
    """Enumeration of forecast horizons."""
    SHORT_TERM = "short_term"      # Hours to days
    MEDIUM_TERM = "medium_term"    # Days to weeks  
    LONG_TERM = "long_term"        # Weeks to months


class ConfidenceLevel(Enum):
    """Standard confidence levels for intervals."""
    LOW = 0.68      # 68% (1 sigma)
    MEDIUM = 0.90   # 90%
    HIGH = 0.95     # 95%
    VERY_HIGH = 0.99  # 99%


@dataclass
class ForecastResult:
    """
    Standardized forecast result container.
    Contains predictions, uncertainty measures, and metadata.
    """
    predictions: np.ndarray
    timestamps: pd.DatetimeIndex
    confidence_intervals: Dict[float, Tuple[np.ndarray, np.ndarray]]
    uncertainty: np.ndarray
    model_name: str
    model_type: str
    forecast_horizon: ForecastHorizon
    created_at: datetime
    metadata: Dict[str, Any]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast result to DataFrame."""
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'prediction': self.predictions,
            'uncertainty': self.uncertainty
        })
        
        # Add confidence intervals
        for confidence_level, (lower, upper) in self.confidence_intervals.items():
            df[f'lower_{int(confidence_level*100)}'] = lower
            df[f'upper_{int(confidence_level*100)}'] = upper
        
        return df
    
    def get_confidence_interval(self, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Get confidence interval for specified level."""
        if confidence_level in self.confidence_intervals:
            return self.confidence_intervals[confidence_level]
        
        # Calculate if not available
        z_score = self._get_z_score(confidence_level)
        lower = self.predictions - z_score * self.uncertainty
        upper = self.predictions + z_score * self.uncertainty
        return lower, upper
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for confidence level."""
        from scipy import stats
        return stats.norm.ppf((1 + confidence_level) / 2)


class ForecastingInterface:
    """
    Unified interface for generating forecasts with confidence intervals.
    Provides standardized methods for all forecasting models.
    """
    
    def __init__(self, model: BaseForecaster):
        """
        Initialize forecasting interface.
        
        Args:
            model: Trained forecasting model
        """
        self.model = model
        self.model_name = model.model_name
        self.model_type = model.model_type
        
        if not model.is_fitted:
            raise ValueError("Model must be fitted before creating forecasting interface")
    
    def generate_forecast(self,
                         horizon: int,
                         start_time: Optional[datetime] = None,
                         frequency: str = 'H',
                         confidence_levels: Optional[List[float]] = None,
                         include_uncertainty: bool = True,
                         **kwargs) -> ForecastResult:
        """
        Generate comprehensive forecast with confidence intervals.
        
        Args:
            horizon: Number of periods to forecast
            start_time: Start time for forecast (current time if None)
            frequency: Forecast frequency ('H', 'D', etc.)
            confidence_levels: List of confidence levels (default: [0.68, 0.90, 0.95])
            include_uncertainty: Whether to include uncertainty estimates
            **kwargs: Additional model-specific parameters
            
        Returns:
            ForecastResult object with predictions and confidence intervals
        """
        if confidence_levels is None:
            confidence_levels = [0.68, 0.90, 0.95]
        
        # Generate timestamps
        if start_time is None:
            start_time = datetime.now()
        
        timestamps = pd.date_range(
            start=start_time,
            periods=horizon,
            freq=frequency
        )
        
        # Create dummy input data for forecasting
        dummy_X = self._create_dummy_input(horizon, timestamps)
        
        # Generate predictions with uncertainty
        if include_uncertainty and hasattr(self.model, 'predict_with_uncertainty'):
            predictions, uncertainty = self.model.predict_with_uncertainty(dummy_X, **kwargs)
        else:
            predictions = self.model.predict(dummy_X, **kwargs)
            uncertainty = np.ones_like(predictions) * 0.1  # Default uncertainty
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for level in confidence_levels:
            lower, upper = self._calculate_confidence_interval(predictions, uncertainty, level)
            confidence_intervals[level] = (lower, upper)
        
        # Determine forecast horizon type
        forecast_horizon = self._determine_horizon_type(horizon, frequency)
        
        # Create forecast result
        result = ForecastResult(
            predictions=predictions,
            timestamps=timestamps,
            confidence_intervals=confidence_intervals,
            uncertainty=uncertainty,
            model_name=self.model_name,
            model_type=self.model_type,
            forecast_horizon=forecast_horizon,
            created_at=datetime.now(),
            metadata={
                'horizon': horizon,
                'frequency': frequency,
                'confidence_levels': confidence_levels,
                'model_parameters': self.model.get_model_parameters()
            }
        )
        
        logger.info(f"Generated {forecast_horizon.value} forecast with {len(predictions)} points")
        return result
    
    def generate_short_term_forecast(self,
                                   hours: int = 24,
                                   confidence_levels: Optional[List[float]] = None,
                                   **kwargs) -> ForecastResult:
        """
        Generate short-term forecast (hours to days).
        
        Args:
            hours: Number of hours to forecast
            confidence_levels: Confidence levels for intervals
            **kwargs: Additional parameters
            
        Returns:
            ForecastResult for short-term horizon
        """
        return self.generate_forecast(
            horizon=hours,
            frequency='H',
            confidence_levels=confidence_levels,
            **kwargs
        )
    
    def generate_medium_term_forecast(self,
                                    days: int = 7,
                                    confidence_levels: Optional[List[float]] = None,
                                    **kwargs) -> ForecastResult:
        """
        Generate medium-term forecast (days to weeks).
        
        Args:
            days: Number of days to forecast
            confidence_levels: Confidence levels for intervals
            **kwargs: Additional parameters
            
        Returns:
            ForecastResult for medium-term horizon
        """
        return self.generate_forecast(
            horizon=days,
            frequency='D',
            confidence_levels=confidence_levels,
            **kwargs
        )
    
    def generate_long_term_forecast(self,
                                  weeks: int = 4,
                                  confidence_levels: Optional[List[float]] = None,
                                  **kwargs) -> ForecastResult:
        """
        Generate long-term forecast (weeks to months).
        
        Args:
            weeks: Number of weeks to forecast
            confidence_levels: Confidence levels for intervals
            **kwargs: Additional parameters
            
        Returns:
            ForecastResult for long-term horizon
        """
        return self.generate_forecast(
            horizon=weeks,
            frequency='W',
            confidence_levels=confidence_levels,
            **kwargs
        )
    
    def compare_forecasts(self,
                         forecasts: List[ForecastResult],
                         actual_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compare multiple forecasts and calculate performance metrics.
        
        Args:
            forecasts: List of forecast results to compare
            actual_values: Actual values for comparison (if available)
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'forecasts': [],
            'summary': {},
            'metrics': {}
        }
        
        for i, forecast in enumerate(forecasts):
            forecast_info = {
                'index': i,
                'model_name': forecast.model_name,
                'model_type': forecast.model_type,
                'horizon_type': forecast.forecast_horizon.value,
                'num_points': len(forecast.predictions),
                'mean_prediction': np.mean(forecast.predictions),
                'mean_uncertainty': np.mean(forecast.uncertainty)
            }
            comparison['forecasts'].append(forecast_info)
        
        # Calculate metrics if actual values provided
        if actual_values is not None:
            comparison['metrics'] = self._calculate_comparison_metrics(forecasts, actual_values)
        
        # Summary statistics
        comparison['summary'] = {
            'num_forecasts': len(forecasts),
            'model_types': list(set(f.model_type for f in forecasts)),
            'horizon_types': list(set(f.forecast_horizon.value for f in forecasts))
        }
        
        return comparison
    
    def _create_dummy_input(self, horizon: int, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """Create dummy input data for forecasting."""
        # Create basic time-based features
        dummy_data = {
            'timestamp': timestamps,
            'hour_of_day': timestamps.hour,
            'day_of_week': timestamps.dayofweek,
            'month': timestamps.month,
            'is_weekend': timestamps.dayofweek >= 5
        }
        
        # Add some synthetic features if model expects them
        if hasattr(self.model, 'feature_names') and self.model.feature_names:
            for feature in self.model.feature_names:
                if feature not in dummy_data:
                    # Generate synthetic feature values
                    if 'temperature' in feature.lower():
                        dummy_data[feature] = 70 + 10 * np.sin(2 * np.pi * np.arange(horizon) / 24)
                    elif 'load' in feature.lower() or 'demand' in feature.lower():
                        dummy_data[feature] = 50000 + 5000 * np.sin(2 * np.pi * np.arange(horizon) / 24)
                    else:
                        dummy_data[feature] = np.random.randn(horizon)
        
        return pd.DataFrame(dummy_data)
    
    def _calculate_confidence_interval(self,
                                     predictions: np.ndarray,
                                     uncertainty: np.ndarray,
                                     confidence_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence interval for given level."""
        try:
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
        except ImportError:
            # Fallback z-scores for common confidence levels
            z_scores = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z_score = z_scores.get(confidence_level, 1.96)
        
        margin = z_score * uncertainty
        lower = predictions - margin
        upper = predictions + margin
        
        return lower, upper
    
    def _determine_horizon_type(self, horizon: int, frequency: str) -> ForecastHorizon:
        """Determine forecast horizon type based on horizon and frequency."""
        if frequency == 'H':  # Hourly
            if horizon <= 48:  # Up to 2 days (short-term: hours to days)
                return ForecastHorizon.SHORT_TERM
            elif horizon <= 168:  # Up to 1 week (medium-term: days to weeks)
                return ForecastHorizon.MEDIUM_TERM
            else:  # More than 1 week (long-term: weeks to months)
                return ForecastHorizon.LONG_TERM
        elif frequency == 'D':  # Daily
            if horizon <= 3:  # Up to 3 days (short-term: hours to days)
                return ForecastHorizon.SHORT_TERM
            elif horizon <= 30:  # Up to 1 month (medium-term: days to weeks)
                return ForecastHorizon.MEDIUM_TERM
            else:  # More than 1 month (long-term: months)
                return ForecastHorizon.LONG_TERM
        elif frequency == 'W':  # Weekly
            if horizon <= 4:  # Up to 1 month (medium-term: days to weeks)
                return ForecastHorizon.MEDIUM_TERM
            else:  # More than 1 month (long-term: weeks to months)
                return ForecastHorizon.LONG_TERM
        else:
            return ForecastHorizon.SHORT_TERM  # Default
    
    def _calculate_comparison_metrics(self,
                                    forecasts: List[ForecastResult],
                                    actual_values: np.ndarray) -> Dict[str, Any]:
        """Calculate performance metrics for forecast comparison."""
        metrics = {}
        
        for i, forecast in enumerate(forecasts):
            model_key = f"{forecast.model_name}_{i}"
            
            # Ensure same length for comparison
            min_length = min(len(forecast.predictions), len(actual_values))
            pred = forecast.predictions[:min_length]
            actual = actual_values[:min_length]
            
            # Calculate metrics
            mae = np.mean(np.abs(pred - actual))
            mse = np.mean((pred - actual) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual - pred) / actual)) * 100
            
            metrics[model_key] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'mean_prediction': np.mean(pred),
                'std_prediction': np.std(pred)
            }
        
        return metrics


# Convenience functions for easy forecasting
def create_forecasting_interface(model: BaseForecaster) -> ForecastingInterface:
    """
    Create a forecasting interface for a trained model.
    
    Args:
        model: Trained forecasting model
        
    Returns:
        ForecastingInterface instance
    """
    return ForecastingInterface(model)


def generate_forecast(model: BaseForecaster,
                     horizon: int,
                     confidence_levels: Optional[List[float]] = None,
                     **kwargs) -> ForecastResult:
    """
    Convenience function to generate forecast from a model.
    
    Args:
        model: Trained forecasting model
        horizon: Number of periods to forecast
        confidence_levels: Confidence levels for intervals
        **kwargs: Additional parameters
        
    Returns:
        ForecastResult with predictions and confidence intervals
    """
    interface = ForecastingInterface(model)
    return interface.generate_forecast(horizon, confidence_levels=confidence_levels, **kwargs)


def compare_model_forecasts(models: List[BaseForecaster],
                          horizon: int,
                          actual_values: Optional[np.ndarray] = None,
                          **kwargs) -> Dict[str, Any]:
    """
    Compare forecasts from multiple models.
    
    Args:
        models: List of trained forecasting models
        horizon: Number of periods to forecast
        actual_values: Actual values for comparison (if available)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with comparison results
    """
    forecasts = []
    
    for model in models:
        try:
            forecast = generate_forecast(model, horizon, **kwargs)
            forecasts.append(forecast)
        except Exception as e:
            logger.warning(f"Failed to generate forecast for {model.model_name}: {e}")
    
    if forecasts:
        interface = ForecastingInterface(models[0])  # Use first model for comparison
        return interface.compare_forecasts(forecasts, actual_values)
    else:
        return {'error': 'No forecasts could be generated'}