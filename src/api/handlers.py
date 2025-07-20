"""
API handlers for power market forecasting system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any


class APIHandler(ABC):
    """Abstract base class for API handling."""
    
    @abstractmethod
    def validate_request(self, request: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate incoming API request."""
        pass
    
    @abstractmethod
    def process_forecast_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process forecast generation request."""
        pass
    
    @abstractmethod
    def format_response(self, data: Any, format_type: str) -> Any:
        """Format response data in specified format."""
        pass
    
    @abstractmethod
    def handle_error(self, error_code: int, error_message: str) -> Dict[str, Any]:
        """Handle and format error responses."""
        pass