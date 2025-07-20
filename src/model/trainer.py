"""
Model training module for power market forecasting.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd


class ModelTrainer(ABC):
    """Abstract base class for model training."""
    
    @abstractmethod
    def train_model(self, features: pd.DataFrame, target: pd.Series, model_type: str, params: Dict[str, Any]) -> Any:
        """Train a model using features and target data."""
        pass
    
    @abstractmethod
    def cross_validate(self, features: pd.DataFrame, target: pd.Series, model_type: str, 
                      params: Dict[str, Any], cv_strategy: str) -> Dict[str, Any]:
        """Perform cross-validation on model."""
        pass
    
    @abstractmethod
    def evaluate_model(self, model: Any, features: pd.DataFrame, target: pd.Series, 
                      metrics: List[str]) -> Dict[str, float]:
        """Evaluate model performance using specified metrics."""
        pass
    
    @abstractmethod
    def save_model(self, model: Any, model_name: str) -> str:
        """Save trained model to storage."""
        pass
    
    @abstractmethod
    def load_model(self, model_name: str) -> Any:
        """Load trained model from storage."""
        pass