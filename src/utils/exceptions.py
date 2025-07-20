"""
Custom exceptions for power market forecasting system.
"""


class PowerMarketForecastingError(Exception):
    """Base exception for power market forecasting system."""
    pass


class DataCollectionError(PowerMarketForecastingError):
    """Exception raised during data collection."""
    pass


class DataValidationError(PowerMarketForecastingError):
    """Exception raised during data validation."""
    pass


class ModelTrainingError(PowerMarketForecastingError):
    """Exception raised during model training."""
    pass


class ForecastingError(PowerMarketForecastingError):
    """Exception raised during forecast generation."""
    pass


class APIError(PowerMarketForecastingError):
    """Exception raised in API operations."""
    pass