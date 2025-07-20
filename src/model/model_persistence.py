"""
Model persistence and versioning system for the energy forecasting platform.
Handles saving, loading, and versioning of trained models.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import joblib
import pandas as pd

from .base_model import BaseForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelVersionManager:
    """
    Manages model versions and persistence operations.
    Provides versioning, metadata tracking, and model lifecycle management.
    """
    
    def __init__(self, base_path: Union[str, Path] = "models"):
        """
        Initialize the model version manager.
        
        Args:
            base_path: Base directory for storing models
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.base_path / "saved_models"
        self.metadata_dir = self.base_path / "metadata"
        self.versions_dir = self.base_path / "versions"
        
        for directory in [self.models_dir, self.metadata_dir, self.versions_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model version manager initialized at {self.base_path}")
    
    def save_model(self, 
                   model: BaseForecaster, 
                   model_name: str,
                   version: Optional[str] = None,
                   description: str = "",
                   tags: Optional[List[str]] = None) -> str:
        """
        Save a model with versioning support.
        
        Args:
            model: Trained model to save
            model_name: Name for the model
            version: Version string (auto-generated if None)
            description: Description of this model version
            tags: Optional tags for categorization
            
        Returns:
            Version string of the saved model
        """
        if not model.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version(model_name)
        
        # Create model directory
        model_dir = self.models_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file
        model_file = model_dir / "model.joblib"
        model.save_model(model_file)
        
        # Create version metadata
        version_metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'description': description,
            'tags': tags or [],
            'model_type': model.model_type,
            'model_class': model.__class__.__name__,
            'model_info': model.get_model_info(),
            'file_path': str(model_file.relative_to(self.base_path)),
            'file_size': model_file.stat().st_size
        }
        
        # Save version metadata
        metadata_file = self.metadata_dir / f"{model_name}_{version}.json"
        with open(metadata_file, 'w') as f:
            json.dump(version_metadata, f, indent=2, default=str)
        
        # Update model registry
        self._update_model_registry(model_name, version, version_metadata)
        
        logger.info(f"Model saved: {model_name} v{version}")
        return version
    
    def load_model(self, 
                   model_name: str, 
                   version: Optional[str] = None,
                   model_class: Optional[type] = None) -> BaseForecaster:
        """
        Load a model by name and version.
        
        Args:
            model_name: Name of the model to load
            version: Version to load (latest if None)
            model_class: Model class for loading (auto-detected if None)
            
        Returns:
            Loaded model instance
        """
        # Get version to load
        if version is None:
            version = self.get_latest_version(model_name)
            if version is None:
                raise ValueError(f"No versions found for model: {model_name}")
        
        # Load version metadata
        metadata = self.get_version_metadata(model_name, version)
        if metadata is None:
            raise ValueError(f"Version not found: {model_name} v{version}")
        
        # Load model file
        model_file = self.base_path / metadata['file_path']
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Determine model class if not provided
        if model_class is None:
            model_class = self._get_model_class(metadata['model_class'])
        
        # Load model
        model = model_class.load_model(model_file)
        
        logger.info(f"Model loaded: {model_name} v{version}")
        return model
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model information dictionaries
        """
        registry_file = self.base_path / "model_registry.json"
        if not registry_file.exists():
            return []
        
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        models = []
        for model_name, model_info in registry.items():
            models.append({
                'name': model_name,
                'latest_version': model_info.get('latest_version'),
                'total_versions': len(model_info.get('versions', [])),
                'created_at': model_info.get('created_at'),
                'last_updated': model_info.get('last_updated'),
                'model_type': model_info.get('model_type'),
                'tags': model_info.get('tags', [])
            })
        
        return models
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version information dictionaries
        """
        versions = []
        metadata_pattern = f"{model_name}_*.json"
        
        for metadata_file in self.metadata_dir.glob(metadata_pattern):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                versions.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x['created_at'], reverse=True)
        return versions
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest version string or None if no versions exist
        """
        versions = self.list_versions(model_name)
        return versions[0]['version'] if versions else None
    
    def get_version_metadata(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific model version.
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            Version metadata dictionary or None if not found
        """
        metadata_file = self.metadata_dir / f"{model_name}_{version}.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None
    
    def delete_version(self, model_name: str, version: str) -> bool:
        """
        Delete a specific model version.
        
        Args:
            model_name: Name of the model
            version: Version to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            # Delete model files
            model_dir = self.models_dir / model_name / version
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Delete metadata
            metadata_file = self.metadata_dir / f"{model_name}_{version}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Update registry
            self._remove_from_registry(model_name, version)
            
            logger.info(f"Deleted model version: {model_name} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete all versions of a model.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            # Delete all model files
            model_dir = self.models_dir / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Delete all metadata files
            metadata_pattern = f"{model_name}_*.json"
            for metadata_file in self.metadata_dir.glob(metadata_pattern):
                metadata_file.unlink()
            
            # Remove from registry
            self._remove_model_from_registry(model_name)
            
            logger.info(f"Deleted model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False
    
    def compare_versions(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two versions of a model.
        
        Args:
            model_name: Name of the model
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Comparison results dictionary
        """
        metadata1 = self.get_version_metadata(model_name, version1)
        metadata2 = self.get_version_metadata(model_name, version2)
        
        if not metadata1 or not metadata2:
            raise ValueError("One or both versions not found")
        
        comparison = {
            'model_name': model_name,
            'version1': version1,
            'version2': version2,
            'created_at_diff': metadata2['created_at'] > metadata1['created_at'],
            'size_diff': metadata2['file_size'] - metadata1['file_size'],
            'description_diff': {
                'v1': metadata1['description'],
                'v2': metadata2['description']
            },
            'tags_diff': {
                'v1': metadata1['tags'],
                'v2': metadata2['tags']
            }
        }
        
        # Compare model performance if available
        if 'model_info' in metadata1 and 'model_info' in metadata2:
            info1 = metadata1['model_info']
            info2 = metadata2['model_info']
            
            comparison['performance_diff'] = {
                'training_metrics': {
                    'v1': info1.get('training_metrics', {}),
                    'v2': info2.get('training_metrics', {})
                },
                'validation_metrics': {
                    'v1': info1.get('validation_metrics', {}),
                    'v2': info2.get('validation_metrics', {})
                }
            }
        
        return comparison
    
    def _generate_version(self, model_name: str) -> str:
        """Generate a new version string for a model."""
        existing_versions = self.list_versions(model_name)
        
        if not existing_versions:
            return "1.0.0"
        
        # Simple versioning: increment patch version
        latest_version = existing_versions[0]['version']
        try:
            major, minor, patch = map(int, latest_version.split('.'))
            return f"{major}.{minor}.{patch + 1}"
        except ValueError:
            # Fallback to timestamp-based versioning
            return datetime.now().strftime("%Y%m%d.%H%M%S")
    
    def _update_model_registry(self, model_name: str, version: str, metadata: Dict[str, Any]) -> None:
        """Update the model registry with new version information."""
        registry_file = self.base_path / "model_registry.json"
        
        # Load existing registry
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        else:
            registry = {}
        
        # Update registry
        if model_name not in registry:
            registry[model_name] = {
                'created_at': metadata['created_at'],
                'model_type': metadata['model_type'],
                'versions': [],
                'tags': []
            }
        
        registry[model_name]['last_updated'] = metadata['created_at']
        registry[model_name]['latest_version'] = version
        
        if version not in registry[model_name]['versions']:
            registry[model_name]['versions'].append(version)
        
        # Merge tags
        existing_tags = set(registry[model_name]['tags'])
        new_tags = set(metadata['tags'])
        registry[model_name]['tags'] = list(existing_tags.union(new_tags))
        
        # Save registry
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
    
    def _remove_from_registry(self, model_name: str, version: str) -> None:
        """Remove a version from the model registry."""
        registry_file = self.base_path / "model_registry.json"
        
        if not registry_file.exists():
            return
        
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        if model_name in registry and version in registry[model_name]['versions']:
            registry[model_name]['versions'].remove(version)
            
            # Update latest version
            if registry[model_name]['versions']:
                # Get the most recent version
                versions = self.list_versions(model_name)
                if versions:
                    registry[model_name]['latest_version'] = versions[0]['version']
            else:
                # No versions left, remove model from registry
                del registry[model_name]
        
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
    
    def _remove_model_from_registry(self, model_name: str) -> None:
        """Remove a model completely from the registry."""
        registry_file = self.base_path / "model_registry.json"
        
        if not registry_file.exists():
            return
        
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        if model_name in registry:
            del registry[model_name]
        
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
    
    def _get_model_class(self, class_name: str) -> type:
        """Get model class by name."""
        # Import model classes
        from .forecaster import (
            ARIMAForecaster, ExponentialSmoothingForecaster,
            RandomForestForecaster, XGBoostForecaster, TransformerForecaster
        )
        from .foundation_models import (
            TimesFMForecaster, ChronosForecaster, LagLlamaForecaster,
            MoiraiForecaster, FoundationModelEnsemble
        )
        
        # Map class names to classes
        class_map = {
            'ARIMAForecaster': ARIMAForecaster,
            'ExponentialSmoothingForecaster': ExponentialSmoothingForecaster,
            'RandomForestForecaster': RandomForestForecaster,
            'XGBoostForecaster': XGBoostForecaster,
            'TransformerForecaster': TransformerForecaster,
            'TimesFMForecaster': TimesFMForecaster,
            'ChronosForecaster': ChronosForecaster,
            'LagLlamaForecaster': LagLlamaForecaster,
            'MoiraiForecaster': MoiraiForecaster,
            'FoundationModelEnsemble': FoundationModelEnsemble
        }
        
        if class_name not in class_map:
            raise ValueError(f"Unknown model class: {class_name}")
        
        return class_map[class_name]


# Convenience functions for easy model persistence
def save_model(model: BaseForecaster, 
               model_name: str,
               version: Optional[str] = None,
               description: str = "",
               tags: Optional[List[str]] = None,
               base_path: Union[str, Path] = "models") -> str:
    """
    Convenience function to save a model with versioning.
    
    Args:
        model: Trained model to save
        model_name: Name for the model
        version: Version string (auto-generated if None)
        description: Description of this model version
        tags: Optional tags for categorization
        base_path: Base directory for storing models
        
    Returns:
        Version string of the saved model
    """
    manager = ModelVersionManager(base_path)
    return manager.save_model(model, model_name, version, description, tags)


def load_model(model_name: str, 
               version: Optional[str] = None,
               base_path: Union[str, Path] = "models") -> BaseForecaster:
    """
    Convenience function to load a model by name and version.
    
    Args:
        model_name: Name of the model to load
        version: Version to load (latest if None)
        base_path: Base directory where models are stored
        
    Returns:
        Loaded model instance
    """
    manager = ModelVersionManager(base_path)
    return manager.load_model(model_name, version)


def list_models(base_path: Union[str, Path] = "models") -> List[Dict[str, Any]]:
    """
    Convenience function to list all available models.
    
    Args:
        base_path: Base directory where models are stored
        
    Returns:
        List of model information dictionaries
    """
    manager = ModelVersionManager(base_path)
    return manager.list_models()