"""
Visualization module for power market forecasting.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class Visualizer(ABC):
    """Abstract base class for visualization."""
    
    @abstractmethod
    def create_forecast_chart(self, forecast: Dict[str, Any], actual: Optional[pd.Series] = None) -> Any:
        """Create forecast visualization chart."""
        pass
    
    @abstractmethod
    def create_performance_chart(self, metrics: Dict[str, float]) -> Any:
        """Create model performance visualization."""
        pass
    
    @abstractmethod
    def export_visualization(self, visualization: Any, format_type: str) -> bytes:
        """Export visualization in specified format."""
        pass