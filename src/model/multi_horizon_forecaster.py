"""
Multi-horizon forecasting module for the energy forecasting platform.
Implements specialized forecasting for short-term, medium-term, and long-term horizons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass

from .base_model import BaseForecaster
from .forecasting_interface import ForecastResult, ForecastHorizon, ForecastingInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HorizonType(Enum):
    """Energy market specific horizon types."""
    INTRADAY = "intraday"          # 1-24 hours
    DAY_AHEAD = "day_ahead"        # 1-7 days  
    WEEK_AHEAD = "week_ahead"      # 1-4 weeks
    MONTH_AHEAD = "month_ahead"    # 1-3 months
    SEASONAL = "seasonal"          # 3-12 months


@dataclass
class HorizonConfig:
    """Configuration for specific forecast horizon."""
    horizon_type: HorizonType
    max_periods: int
    frequency: str
    aggregation_method: str
    seasonal_adjustment: bool
    uncertainty_scaling: float
    model_preferences: List[str]  # Preferred model types for this horizon


class MultiHorizonForecaster:
    """
    Specialized forecaster for different time horizons in energy markets.
    Adapts forecasting approach based on horizon requirements.
    """
    
    def __init__(self, models: Dict[str, BaseForecaster]):
        """
        Initialize multi-horizon forecaster.
        
        Args:
            models: Dictionary of trained models {model_name: model_instance}
        """
        self.models = models
        self.horizon_configs = self._create_horizon_configs()
        
        # Validate models are fitted
        for name, model in models.items():
            if not model.is_fitted:
                raise ValueError(f"Model {name} must be fitted")
        
        logger.info(f"Initialized multi-horizon forecaster with {len(models)} models")
    
    def _create_horizon_configs(self) -> Dict[HorizonType, HorizonConfig]:
        """Create configuration for each horizon type."""
        return {
            HorizonType.INTRADAY: HorizonConfig(
                horizon_type=HorizonType.INTRADAY,
                max_periods=24,
                frequency='h',
                aggregation_method='none',
                seasonal_adjustment=False,
                uncertainty_scaling=1.0,
                model_preferences=['transformer', 'lstm', 'random_forest']
            ),
            HorizonType.DAY_AHEAD: HorizonConfig(
                horizon_type=HorizonType.DAY_AHEAD,
                max_periods=7,
                frequency='D',
                aggregation_method='daily_avg',
                seasonal_adjustment=True,
                uncertainty_scaling=1.2,
                model_preferences=['random_forest', 'xgboost', 'arima']
            ),
            HorizonType.WEEK_AHEAD: HorizonConfig(
                horizon_type=HorizonType.WEEK_AHEAD,
                max_periods=4,
                frequency='W',
                aggregation_method='weekly_avg',
                seasonal_adjustment=True,
                uncertainty_scaling=1.5,
                model_preferences=['arima', 'exponential_smoothing', 'random_forest']
            ),
            HorizonType.MONTH_AHEAD: HorizonConfig(
                horizon_type=HorizonType.MONTH_AHEAD,
                max_periods=3,
                frequency='M',
                aggregation_method='monthly_avg',
                seasonal_adjustment=True,
                uncertainty_scaling=2.0,
                model_preferences=['arima', 'exponential_smoothing']
            ),
            HorizonType.SEASONAL: HorizonConfig(
                horizon_type=HorizonType.SEASONAL,
                max_periods=12,
                frequency='M',
                aggregation_method='monthly_avg',
                seasonal_adjustment=True,
                uncertainty_scaling=3.0,
                model_preferences=['arima', 'exponential_smoothing']
            )
        }
    
    def generate_short_term_forecast(self,
                                   hours: int = 24,
                                   start_time: Optional[datetime] = None,
                                   confidence_levels: Optional[List[float]] = None,
                                   **kwargs) -> ForecastResult:
        """
        Generate short-term forecast (hours to days).
        Optimized for intraday and day-ahead energy trading.
        
        Args:
            hours: Number of hours to forecast (1-168)
            start_time: Start time for forecast
            confidence_levels: Confidence levels for intervals
            **kwargs: Additional parameters
            
        Returns:
            ForecastResult for short-term horizon
        """
        if hours <= 24:
            horizon_type = HorizonType.INTRADAY
        else:
            horizon_type = HorizonType.DAY_AHEAD
        
        config = self.horizon_configs[horizon_type]
        
        # Select best model for this horizon
        model = self._select_best_model(config)
        
        # Generate base forecast
        interface = ForecastingInterface(model)
        
        if start_time is None:
            start_time = datetime.now()
        
        # For short-term, use hourly frequency
        result = interface.generate_forecast(
            horizon=hours,
            start_time=start_time,
            frequency='h',
            confidence_levels=confidence_levels or [0.68, 0.90, 0.95],
            **kwargs
        )
        
        # Apply short-term specific adjustments
        result = self._apply_short_term_adjustments(result, config)
        
        logger.info(f"Generated short-term forecast for {hours} hours using {model.model_name}")
        return result
    
    def generate_medium_term_forecast(self,
                                    days: int = 7,
                                    start_time: Optional[datetime] = None,
                                    confidence_levels: Optional[List[float]] = None,
                                    **kwargs) -> ForecastResult:
        """
        Generate medium-term forecast (days to weeks).
        Optimized for weekly energy planning and operations.
        
        Args:
            days: Number of days to forecast (1-30)
            start_time: Start time for forecast
            confidence_levels: Confidence levels for intervals
            **kwargs: Additional parameters
            
        Returns:
            ForecastResult for medium-term horizon
        """
        if days <= 7:
            horizon_type = HorizonType.DAY_AHEAD
            frequency = 'D'
            horizon = days
        else:
            horizon_type = HorizonType.WEEK_AHEAD
            frequency = 'W'
            horizon = max(1, days // 7)  # Convert to weeks
        
        config = self.horizon_configs[horizon_type]
        model = self._select_best_model(config)
        
        # Generate base forecast
        interface = ForecastingInterface(model)
        
        if start_time is None:
            start_time = datetime.now()
        
        result = interface.generate_forecast(
            horizon=horizon,
            start_time=start_time,
            frequency=frequency,
            confidence_levels=confidence_levels or [0.68, 0.90, 0.95],
            **kwargs
        )
        
        # Apply medium-term specific adjustments
        result = self._apply_medium_term_adjustments(result, config, days)
        
        logger.info(f"Generated medium-term forecast for {days} days using {model.model_name}")
        return result
    
    def generate_long_term_forecast(self,
                                  months: int = 3,
                                  start_time: Optional[datetime] = None,
                                  confidence_levels: Optional[List[float]] = None,
                                  **kwargs) -> ForecastResult:
        """
        Generate long-term forecast (months).
        Optimized for strategic energy planning and capacity decisions.
        
        Args:
            months: Number of months to forecast (1-12)
            start_time: Start time for forecast
            confidence_levels: Confidence levels for intervals
            **kwargs: Additional parameters
            
        Returns:
            ForecastResult for long-term horizon
        """
        if months <= 3:
            horizon_type = HorizonType.MONTH_AHEAD
        else:
            horizon_type = HorizonType.SEASONAL
        
        config = self.horizon_configs[horizon_type]
        model = self._select_best_model(config)
        
        # Generate base forecast
        interface = ForecastingInterface(model)
        
        if start_time is None:
            start_time = datetime.now()
        
        result = interface.generate_forecast(
            horizon=months,
            start_time=start_time,
            frequency='M',
            confidence_levels=confidence_levels or [0.68, 0.90, 0.95],
            **kwargs
        )
        
        # Apply long-term specific adjustments
        result = self._apply_long_term_adjustments(result, config)
        
        logger.info(f"Generated long-term forecast for {months} months using {model.model_name}")
        return result
    
    def generate_multi_horizon_forecast(self,
                                      horizons: Dict[str, int],
                                      start_time: Optional[datetime] = None,
                                      confidence_levels: Optional[List[float]] = None,
                                      **kwargs) -> Dict[str, ForecastResult]:
        """
        Generate forecasts for multiple horizons simultaneously.
        
        Args:
            horizons: Dictionary of {horizon_name: periods} e.g., {'short': 24, 'medium': 7, 'long': 3}
            start_time: Start time for forecasts
            confidence_levels: Confidence levels for intervals
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of {horizon_name: ForecastResult}
        """
        results = {}
        
        for horizon_name, periods in horizons.items():
            try:
                if horizon_name.lower() in ['short', 'short_term', 'intraday', 'hours']:
                    result = self.generate_short_term_forecast(
                        hours=periods, start_time=start_time, 
                        confidence_levels=confidence_levels, **kwargs
                    )
                elif horizon_name.lower() in ['medium', 'medium_term', 'daily', 'days']:
                    result = self.generate_medium_term_forecast(
                        days=periods, start_time=start_time,
                        confidence_levels=confidence_levels, **kwargs
                    )
                elif horizon_name.lower() in ['long', 'long_term', 'monthly', 'months']:
                    result = self.generate_long_term_forecast(
                        months=periods, start_time=start_time,
                        confidence_levels=confidence_levels, **kwargs
                    )
                else:
                    logger.warning(f"Unknown horizon type: {horizon_name}")
                    continue
                
                results[horizon_name] = result
                
            except Exception as e:
                logger.error(f"Failed to generate {horizon_name} forecast: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        logger.info(f"Generated multi-horizon forecasts for {len(results)} horizons")
        return results
    
    def compare_horizon_performance(self,
                                  forecasts: Dict[str, ForecastResult],
                                  actual_values: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compare performance across different forecast horizons.
        
        Args:
            forecasts: Dictionary of {horizon_name: ForecastResult}
            actual_values: Dictionary of {horizon_name: actual_values}
            
        Returns:
            Dictionary with performance comparison results
        """
        comparison = {
            'horizon_performance': {},
            'summary': {},
            'recommendations': []
        }
        
        for horizon_name, forecast in forecasts.items():
            if horizon_name in actual_values:
                actual = actual_values[horizon_name]
                
                # Calculate metrics
                min_length = min(len(forecast.predictions), len(actual))
                pred = forecast.predictions[:min_length]
                actual_trimmed = actual[:min_length]
                
                mae = np.mean(np.abs(pred - actual_trimmed))
                mse = np.mean((pred - actual_trimmed) ** 2)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_trimmed - pred) / actual_trimmed)) * 100
                
                comparison['horizon_performance'][horizon_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'model_used': forecast.model_name,
                    'horizon_type': forecast.forecast_horizon.value,
                    'num_points': min_length
                }
        
        # Generate summary and recommendations
        if comparison['horizon_performance']:
            best_horizon = min(comparison['horizon_performance'].items(), 
                             key=lambda x: x[1]['mape'])
            comparison['summary']['best_performing_horizon'] = best_horizon[0]
            comparison['summary']['best_mape'] = best_horizon[1]['mape']
            
            # Generate recommendations
            comparison['recommendations'] = self._generate_horizon_recommendations(
                comparison['horizon_performance']
            )
        
        return comparison
    
    def _select_best_model(self, config: HorizonConfig) -> BaseForecaster:
        """Select the best model for a specific horizon configuration."""
        # Try preferred models in order
        for preferred_model in config.model_preferences:
            for model_name, model in self.models.items():
                if preferred_model.lower() in model_name.lower():
                    return model
        
        # Fallback to first available model
        return list(self.models.values())[0]
    
    def _apply_short_term_adjustments(self, 
                                    result: ForecastResult, 
                                    config: HorizonConfig) -> ForecastResult:
        """Apply short-term specific adjustments to forecast."""
        # For short-term forecasts, apply minimal adjustments
        # Focus on high-frequency patterns and recent trends
        
        # Adjust uncertainty based on time of day (higher uncertainty during peak hours)
        if len(result.timestamps) > 0:
            hours = result.timestamps.hour
            peak_hours = (hours >= 16) & (hours <= 20)  # Evening peak
            morning_peak = (hours >= 7) & (hours <= 9)   # Morning peak
            
            uncertainty_multiplier = np.ones_like(result.uncertainty)
            uncertainty_multiplier[peak_hours] *= 1.3
            uncertainty_multiplier[morning_peak] *= 1.2
            
            result.uncertainty = result.uncertainty * uncertainty_multiplier
            
            # Recalculate confidence intervals with adjusted uncertainty
            for confidence_level in result.confidence_intervals.keys():
                lower, upper = self._calculate_confidence_interval(
                    result.predictions, result.uncertainty, confidence_level
                )
                result.confidence_intervals[confidence_level] = (lower, upper)
        
        result.metadata['adjustments'] = 'short_term_peak_hour_uncertainty'
        return result
    
    def _apply_medium_term_adjustments(self, 
                                     result: ForecastResult, 
                                     config: HorizonConfig,
                                     original_days: int) -> ForecastResult:
        """Apply medium-term specific adjustments to forecast."""
        # For medium-term forecasts, consider weekly patterns and seasonal effects
        
        # If we forecasted in weeks but need daily resolution, interpolate
        if config.frequency == 'W' and original_days > 7:
            # Expand weekly forecast to daily
            daily_predictions = self._expand_weekly_to_daily(result.predictions, original_days)
            daily_uncertainty = self._expand_weekly_to_daily(result.uncertainty, original_days)
            
            # Create new timestamps
            daily_timestamps = pd.date_range(
                start=result.timestamps[0],
                periods=original_days,
                freq='D'
            )
            
            result.predictions = daily_predictions
            result.uncertainty = daily_uncertainty
            result.timestamps = daily_timestamps
            
            # Recalculate confidence intervals
            for confidence_level in result.confidence_intervals.keys():
                lower, upper = self._calculate_confidence_interval(
                    result.predictions, result.uncertainty, confidence_level
                )
                result.confidence_intervals[confidence_level] = (lower, upper)
        
        # Apply seasonal adjustment if configured
        if config.seasonal_adjustment:
            result = self._apply_seasonal_adjustment(result, 'medium_term')
        
        # Scale uncertainty
        result.uncertainty *= config.uncertainty_scaling
        
        result.metadata['adjustments'] = 'medium_term_seasonal_weekly_patterns'
        return result
    
    def _apply_long_term_adjustments(self, 
                                   result: ForecastResult, 
                                   config: HorizonConfig) -> ForecastResult:
        """Apply long-term specific adjustments to forecast."""
        # For long-term forecasts, focus on seasonal patterns and trends
        
        # Apply strong seasonal adjustment
        if config.seasonal_adjustment:
            result = self._apply_seasonal_adjustment(result, 'long_term')
        
        # Increase uncertainty for longer horizons
        result.uncertainty *= config.uncertainty_scaling
        
        # Add trend uncertainty (uncertainty increases with time)
        time_factor = np.linspace(1.0, 1.5, len(result.predictions))
        result.uncertainty *= time_factor
        
        # Recalculate confidence intervals with adjusted uncertainty
        for confidence_level in result.confidence_intervals.keys():
            lower, upper = self._calculate_confidence_interval(
                result.predictions, result.uncertainty, confidence_level
            )
            result.confidence_intervals[confidence_level] = (lower, upper)
        
        result.metadata['adjustments'] = 'long_term_seasonal_trend_uncertainty'
        return result
    
    def _expand_weekly_to_daily(self, weekly_data: np.ndarray, target_days: int) -> np.ndarray:
        """Expand weekly forecast to daily resolution."""
        # Simple linear interpolation for now
        # In practice, you might use more sophisticated methods
        
        weeks = len(weekly_data)
        days_per_week = target_days / weeks
        
        daily_data = []
        for i in range(weeks):
            if i < weeks - 1:
                # Interpolate between current and next week
                start_val = weekly_data[i]
                end_val = weekly_data[i + 1]
                week_days = int(days_per_week)
                
                for day in range(week_days):
                    alpha = day / week_days
                    daily_val = start_val * (1 - alpha) + end_val * alpha
                    daily_data.append(daily_val)
            else:
                # Last week - just repeat the value
                remaining_days = target_days - len(daily_data)
                daily_data.extend([weekly_data[i]] * remaining_days)
        
        return np.array(daily_data[:target_days])
    
    def _apply_seasonal_adjustment(self, 
                                 result: ForecastResult, 
                                 adjustment_type: str) -> ForecastResult:
        """Apply seasonal adjustments to forecast."""
        # Simple seasonal adjustment based on month
        if len(result.timestamps) > 0:
            months = result.timestamps.month
            
            # Energy demand seasonal factors (higher in summer/winter)
            seasonal_factors = np.ones_like(result.predictions, dtype=float)
            
            # Summer months (higher cooling demand)
            summer_months = (months >= 6) & (months <= 8)
            seasonal_factors[summer_months] *= 1.1
            
            # Winter months (higher heating demand)
            winter_months = (months >= 12) | (months <= 2)
            seasonal_factors[winter_months] *= 1.15
            
            # Apply seasonal adjustment
            result.predictions *= seasonal_factors
            
            # Adjust confidence intervals
            for confidence_level in result.confidence_intervals.keys():
                lower, upper = result.confidence_intervals[confidence_level]
                result.confidence_intervals[confidence_level] = (
                    lower * seasonal_factors,
                    upper * seasonal_factors
                )
        
        return result
    
    def _calculate_confidence_interval(self,
                                     predictions: np.ndarray,
                                     uncertainty: np.ndarray,
                                     confidence_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence interval for given level."""
        try:
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
        except ImportError:
            # Fallback z-scores
            z_scores = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z_score = z_scores.get(confidence_level, 1.96)
        
        margin = z_score * uncertainty
        return predictions - margin, predictions + margin
    
    def _generate_horizon_recommendations(self, 
                                        performance_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on horizon performance."""
        recommendations = []
        
        # Find best and worst performing horizons
        sorted_horizons = sorted(performance_data.items(), key=lambda x: x[1]['mape'])
        
        if len(sorted_horizons) >= 2:
            best_horizon = sorted_horizons[0]
            worst_horizon = sorted_horizons[-1]
            
            recommendations.append(
                f"Best performing horizon: {best_horizon[0]} (MAPE: {best_horizon[1]['mape']:.2f}%)"
            )
            
            if worst_horizon[1]['mape'] > best_horizon[1]['mape'] * 2:
                recommendations.append(
                    f"Consider improving {worst_horizon[0]} forecasting model or parameters"
                )
        
        # Check for high uncertainty horizons
        for horizon_name, metrics in performance_data.items():
            if metrics['mape'] > 15:  # High error threshold
                recommendations.append(
                    f"High uncertainty in {horizon_name} forecasts - consider ensemble methods"
                )
        
        return recommendations


# Convenience functions
def create_multi_horizon_forecaster(models: Dict[str, BaseForecaster]) -> MultiHorizonForecaster:
    """
    Create a multi-horizon forecaster from a dictionary of models.
    
    Args:
        models: Dictionary of {model_name: trained_model}
        
    Returns:
        MultiHorizonForecaster instance
    """
    return MultiHorizonForecaster(models)


def generate_all_horizons(models: Dict[str, BaseForecaster],
                         short_hours: int = 24,
                         medium_days: int = 7,
                         long_months: int = 3,
                         **kwargs) -> Dict[str, ForecastResult]:
    """
    Generate forecasts for all standard horizons.
    
    Args:
        models: Dictionary of trained models
        short_hours: Hours for short-term forecast
        medium_days: Days for medium-term forecast
        long_months: Months for long-term forecast
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with forecasts for all horizons
    """
    forecaster = MultiHorizonForecaster(models)
    
    return forecaster.generate_multi_horizon_forecast({
        'short_term': short_hours,
        'medium_term': medium_days,
        'long_term': long_months
    }, **kwargs)