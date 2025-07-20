"""
Data models for power market forecasting system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional


@dataclass
class MarketData:
    """Model for power market data."""
    id: str
    source: str
    region: str
    timestamp: datetime
    load: float
    price: float
    generation_mix: Dict[str, float]  # e.g., {"coal": 0.3, "natural_gas": 0.4, "renewable": 0.3}
    weather_data: Dict[str, float]  # e.g., {"temperature": 75.0, "humidity": 0.6}
    is_holiday: bool
    special_events: List[str]


@dataclass
class Forecast:
    """Model for forecast data."""
    id: str
    region: str
    created_at: datetime
    forecast_type: str  # "short_term", "medium_term", "long_term"
    start_time: datetime
    end_time: datetime
    time_points: List[datetime]
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]  # List of (lower_bound, upper_bound)
    model_id: str
    features_used: List[str]
    performance_metrics: Dict[str, float]  # e.g., {"MAPE": 3.2, "RMSE": 45.6}


@dataclass
class ModelMetadata:
    """Model for ML model metadata."""
    id: str
    name: str
    type: str  # e.g., "ARIMA", "LSTM", "RandomForest"
    parameters: Dict[str, Any]
    features: List[str]
    training_date: datetime
    training_dataset: str
    validation_metrics: Dict[str, float]
    regions: List[str]  # Regions this model is applicable to