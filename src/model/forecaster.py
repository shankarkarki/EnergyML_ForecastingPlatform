"""
Forecasting module for power market predictions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd


class Forecaster(ABC):
    """Abstract base class for forecasting."""
    
    @abstractmethod
    def generate_forecast(self, model: Any, features: pd.DataFrame, horizon: int, 
                         confidence_level: float) -> Dict[str, Any]:
        """Generate forecast using trained model."""
        pass
    
    @abstractmethod
    def generate_regional_forecast(self, region: str, model: Any, features: pd.DataFrame, 
                                  horizon: int) -> Dict[str, Any]:
        """Generate region-specific forecast."""
        pass
    
    @abstractmethod
    def adjust_for_special_events(self, forecast: Dict[str, Any], 
                                 events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adjust forecast for special events."""
        pass