"""
Unit tests for model persistence and versioning functionality.
Tests model saving, loading, versioning, and metadata management.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.model_persistence import ModelVersionManager, save_model, load_model, list_models
from model.forecaster import RandomForestForecaster, create_forecaster
from model.base_model import BaseForecaster


class TestModelVersionManager(unittest.TestCase):
    """Test cases for ModelVersionManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.manager = ModelVersionManager(self.test_dir)
        
        # Create sample data for training models
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        
        self.X = pd.DataFrame({
            'timestamp': dates,
            'temperature': 70 + 10 * np.sin(2 * np.pi * np.arange(100) / 24) + np.random.normal(0, 3, 100),
            'hour_of_day': np.arange(100) % 24,
            'day_of_week': (np.arange(100) // 24) % 7
        })
        
        self.y = pd.Series(
            50000 + 15000 * np.sin(2 * np.pi * np.arange(100) / 24) + np.random.normal(0, 2000, 100),
            name='load_mw'
        )
        
        # Train a sample model
        self.sample_model = RandomForestForecaster(n_estimators=10, sequence_length=12)
        self.sample_model.fit(self.X, self.y)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test ModelVersionManager initialization."""
        # Test directory structure creation
        self.assertTrue(self.manager.base_path.exists())
        self.assertTrue(self.manager.models_dir.exists())
        self.assertTrue(self.manager.metadata_dir.exists())
        self.assertTrue(self.manager.versions_dir.exists())
        
        # Test with custom path
        custom_path = Path(self.test_dir) / "custom_models"
        custom_manager = ModelVersionManager(custom_path)
        self.assertTrue(custom_path.exists())
    
    def test_save_model_basic(self):
        """Test basic model saving functionality."""
        model_name = "test_rf_model"
        
        # Save model
        version = self.manager.save_model(
            self.sample_model,
            model_name,
            description="Test Random Forest model"
        )
        
        # Check version was returned
        self.assertIsInstance(version, str)
        self.assertTrue(version.startswith("1.0."))
        
        # Check model file was created
        model_dir = self.manager.models_dir / model_name / version
        model_file = model_dir / "model.joblib"
        self.assertTrue(model_file.exists())
        
        # Check metadata was created
        metadata_file = self.manager.metadata_dir / f"{model_name}_{version}.json"
        self.assertTrue(metadata_file.exists())
        
        # Verify metadata content
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata['model_name'], model_name)
        self.assertEqual(metadata['version'], version)
        self.assertEqual(metadata['description'], "Test Random Forest model")
        self.assertEqual(metadata['model_type'], "ml")
        self.assertEqual(metadata['model_class'], "RandomForestForecaster")
    
    def test_save_model_with_custom_version(self):
        """Test saving model with custom version."""
        model_name = "test_custom_version"
        custom_version = "2.1.0"
        
        version = self.manager.save_model(
            self.sample_model,
            model_name,
            version=custom_version,
            description="Custom version test",
            tags=["test", "custom"]
        )
        
        self.assertEqual(version, custom_version)
        
        # Check metadata includes tags
        metadata = self.manager.get_version_metadata(model_name, version)
        self.assertEqual(metadata['tags'], ["test", "custom"])
    
    def test_save_unfitted_model_raises_error(self):
        """Test that saving unfitted model raises error."""
        unfitted_model = RandomForestForecaster(n_estimators=10)
        
        with self.assertRaises(ValueError) as context:
            self.manager.save_model(unfitted_model, "unfitted_model")
        
        self.assertIn("Cannot save unfitted model", str(context.exception))
    
    def test_load_model_basic(self):
        """Test basic model loading functionality."""
        model_name = "test_load_model"
        
        # Save model first
        version = self.manager.save_model(self.sample_model, model_name)
        
        # Load model
        loaded_model = self.manager.load_model(model_name, version)
        
        # Verify loaded model
        self.assertIsInstance(loaded_model, RandomForestForecaster)
        self.assertTrue(loaded_model.is_fitted)
        self.assertEqual(loaded_model.model_name, "RandomForest")
        self.assertEqual(loaded_model.model_type, "ml")
        
        # Test prediction works
        predictions = loaded_model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_load_latest_version(self):
        """Test loading latest version when version not specified."""
        model_name = "test_latest_version"
        
        # Save multiple versions
        version1 = self.manager.save_model(self.sample_model, model_name, description="Version 1")
        version2 = self.manager.save_model(self.sample_model, model_name, description="Version 2")
        
        # Load without specifying version (should get latest)
        loaded_model = self.manager.load_model(model_name)
        
        # Should load the latest version
        self.assertIsInstance(loaded_model, RandomForestForecaster)
        self.assertTrue(loaded_model.is_fitted)
    
    def test_load_nonexistent_model_raises_error(self):
        """Test that loading nonexistent model raises error."""
        with self.assertRaises(ValueError) as context:
            self.manager.load_model("nonexistent_model")
        
        self.assertIn("No versions found", str(context.exception))
        
        # Test loading nonexistent version
        self.manager.save_model(self.sample_model, "existing_model")
        
        with self.assertRaises(ValueError) as context:
            self.manager.load_model("existing_model", "nonexistent_version")
        
        self.assertIn("Version not found", str(context.exception))
    
    def test_list_models(self):
        """Test listing all models."""
        # Initially should be empty
        models = self.manager.list_models()
        self.assertEqual(len(models), 0)
        
        # Save some models
        self.manager.save_model(self.sample_model, "model1", tags=["tag1"])
        self.manager.save_model(self.sample_model, "model2", tags=["tag2"])
        
        # List models
        models = self.manager.list_models()
        self.assertEqual(len(models), 2)
        
        model_names = [m['name'] for m in models]
        self.assertIn("model1", model_names)
        self.assertIn("model2", model_names)
        
        # Check model info structure
        model1_info = next(m for m in models if m['name'] == "model1")
        self.assertIn('latest_version', model1_info)
        self.assertIn('total_versions', model1_info)
        self.assertIn('created_at', model1_info)
        self.assertIn('model_type', model1_info)
        self.assertEqual(model1_info['tags'], ["tag1"])
    
    def test_list_versions(self):
        """Test listing versions of a specific model."""
        model_name = "test_versions"
        
        # Save multiple versions
        version1 = self.manager.save_model(self.sample_model, model_name, description="First version")
        version2 = self.manager.save_model(self.sample_model, model_name, description="Second version")
        
        # List versions
        versions = self.manager.list_versions(model_name)
        self.assertEqual(len(versions), 2)
        
        # Should be sorted by creation date (newest first)
        self.assertEqual(versions[0]['version'], version2)
        self.assertEqual(versions[1]['version'], version1)
        
        # Check version info structure
        self.assertEqual(versions[0]['description'], "Second version")
        self.assertEqual(versions[1]['description'], "First version")
    
    def test_get_latest_version(self):
        """Test getting latest version of a model."""
        model_name = "test_latest"
        
        # No versions initially
        latest = self.manager.get_latest_version(model_name)
        self.assertIsNone(latest)
        
        # Save versions
        version1 = self.manager.save_model(self.sample_model, model_name)
        latest = self.manager.get_latest_version(model_name)
        self.assertEqual(latest, version1)
        
        version2 = self.manager.save_model(self.sample_model, model_name)
        latest = self.manager.get_latest_version(model_name)
        self.assertEqual(latest, version2)
    
    def test_get_version_metadata(self):
        """Test getting metadata for specific version."""
        model_name = "test_metadata"
        description = "Test metadata retrieval"
        tags = ["test", "metadata"]
        
        version = self.manager.save_model(
            self.sample_model,
            model_name,
            description=description,
            tags=tags
        )
        
        # Get metadata
        metadata = self.manager.get_version_metadata(model_name, version)
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['model_name'], model_name)
        self.assertEqual(metadata['version'], version)
        self.assertEqual(metadata['description'], description)
        self.assertEqual(metadata['tags'], tags)
        self.assertEqual(metadata['model_type'], "ml")
        self.assertEqual(metadata['model_class'], "RandomForestForecaster")
        
        # Test nonexistent metadata
        nonexistent = self.manager.get_version_metadata("nonexistent", "1.0.0")
        self.assertIsNone(nonexistent)
    
    def test_delete_version(self):
        """Test deleting a specific model version."""
        model_name = "test_delete_version"
        
        # Save multiple versions
        version1 = self.manager.save_model(self.sample_model, model_name, description="Version 1")
        version2 = self.manager.save_model(self.sample_model, model_name, description="Version 2")
        
        # Verify both versions exist
        versions = self.manager.list_versions(model_name)
        self.assertEqual(len(versions), 2)
        
        # Delete one version
        success = self.manager.delete_version(model_name, version1)
        self.assertTrue(success)
        
        # Verify version was deleted
        versions = self.manager.list_versions(model_name)
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]['version'], version2)
        
        # Verify metadata was deleted
        metadata = self.manager.get_version_metadata(model_name, version1)
        self.assertIsNone(metadata)
    
    def test_delete_model(self):
        """Test deleting all versions of a model."""
        model_name = "test_delete_model"
        
        # Save multiple versions
        self.manager.save_model(self.sample_model, model_name, description="Version 1")
        self.manager.save_model(self.sample_model, model_name, description="Version 2")
        
        # Verify model exists
        models = self.manager.list_models()
        model_names = [m['name'] for m in models]
        self.assertIn(model_name, model_names)
        
        # Delete model
        success = self.manager.delete_model(model_name)
        self.assertTrue(success)
        
        # Verify model was deleted
        models = self.manager.list_models()
        model_names = [m['name'] for m in models]
        self.assertNotIn(model_name, model_names)
        
        # Verify no versions remain
        versions = self.manager.list_versions(model_name)
        self.assertEqual(len(versions), 0)
    
    def test_compare_versions(self):
        """Test comparing two versions of a model."""
        model_name = "test_compare"
        
        # Save two versions with different descriptions
        version1 = self.manager.save_model(
            self.sample_model, 
            model_name, 
            description="First version",
            tags=["v1"]
        )
        
        version2 = self.manager.save_model(
            self.sample_model, 
            model_name, 
            description="Second version",
            tags=["v2"]
        )
        
        # Compare versions
        comparison = self.manager.compare_versions(model_name, version1, version2)
        
        self.assertEqual(comparison['model_name'], model_name)
        self.assertEqual(comparison['version1'], version1)
        self.assertEqual(comparison['version2'], version2)
        self.assertTrue(comparison['created_at_diff'])  # v2 should be newer
        self.assertEqual(comparison['description_diff']['v1'], "First version")
        self.assertEqual(comparison['description_diff']['v2'], "Second version")
        self.assertEqual(comparison['tags_diff']['v1'], ["v1"])
        self.assertEqual(comparison['tags_diff']['v2'], ["v2"])
    
    def test_version_generation(self):
        """Test automatic version generation."""
        model_name = "test_version_gen"
        
        # First version should be 1.0.0
        version1 = self.manager.save_model(self.sample_model, model_name)
        self.assertEqual(version1, "1.0.0")
        
        # Second version should increment patch
        version2 = self.manager.save_model(self.sample_model, model_name)
        self.assertEqual(version2, "1.0.1")
        
        # Third version should increment patch again
        version3 = self.manager.save_model(self.sample_model, model_name)
        self.assertEqual(version3, "1.0.2")


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample data and model
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        
        self.X = pd.DataFrame({
            'timestamp': dates,
            'temperature': 70 + 5 * np.sin(2 * np.pi * np.arange(50) / 24) + np.random.normal(0, 2, 50),
            'hour_of_day': np.arange(50) % 24
        })
        
        self.y = pd.Series(
            50000 + 10000 * np.sin(2 * np.pi * np.arange(50) / 24) + np.random.normal(0, 1500, 50),
            name='load_mw'
        )
        
        self.sample_model = RandomForestForecaster(n_estimators=5, sequence_length=8)
        self.sample_model.fit(self.X, self.y)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_save_model_convenience(self):
        """Test convenience save_model function."""
        model_name = "convenience_test"
        
        version = save_model(
            self.sample_model,
            model_name,
            description="Convenience function test",
            tags=["convenience", "test"],
            base_path=self.test_dir
        )
        
        self.assertIsInstance(version, str)
        
        # Verify model was saved
        manager = ModelVersionManager(self.test_dir)
        metadata = manager.get_version_metadata(model_name, version)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['description'], "Convenience function test")
        self.assertEqual(metadata['tags'], ["convenience", "test"])
    
    def test_load_model_convenience(self):
        """Test convenience load_model function."""
        model_name = "convenience_load_test"
        
        # Save model first
        version = save_model(self.sample_model, model_name, base_path=self.test_dir)
        
        # Load using convenience function
        loaded_model = load_model(model_name, version, base_path=self.test_dir)
        
        self.assertIsInstance(loaded_model, RandomForestForecaster)
        self.assertTrue(loaded_model.is_fitted)
        
        # Test prediction works
        predictions = loaded_model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
    
    def test_load_latest_convenience(self):
        """Test loading latest version using convenience function."""
        model_name = "convenience_latest_test"
        
        # Save multiple versions
        save_model(self.sample_model, model_name, base_path=self.test_dir)
        save_model(self.sample_model, model_name, base_path=self.test_dir)
        
        # Load latest (no version specified)
        loaded_model = load_model(model_name, base_path=self.test_dir)
        
        self.assertIsInstance(loaded_model, RandomForestForecaster)
        self.assertTrue(loaded_model.is_fitted)
    
    def test_list_models_convenience(self):
        """Test convenience list_models function."""
        # Save some models
        save_model(self.sample_model, "model1", base_path=self.test_dir)
        save_model(self.sample_model, "model2", base_path=self.test_dir)
        
        # List models
        models = list_models(base_path=self.test_dir)
        
        self.assertEqual(len(models), 2)
        model_names = [m['name'] for m in models]
        self.assertIn("model1", model_names)
        self.assertIn("model2", model_names)


class TestModelPersistenceIntegration(unittest.TestCase):
    """Integration tests for model persistence with different model types."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = ModelVersionManager(self.test_dir)
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        
        self.X = pd.DataFrame({
            'timestamp': dates,
            'temperature': 70 + 5 * np.sin(2 * np.pi * np.arange(50) / 24) + np.random.normal(0, 2, 50),
            'hour_of_day': np.arange(50) % 24
        })
        
        self.y = pd.Series(
            50000 + 10000 * np.sin(2 * np.pi * np.arange(50) / 24) + np.random.normal(0, 1500, 50),
            name='load_mw'
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_random_forest_persistence(self):
        """Test persistence of Random Forest model."""
        model = create_forecaster('random_forest', n_estimators=5, sequence_length=8)
        model.fit(self.X, self.y)
        
        # Save and load
        version = self.manager.save_model(model, "rf_test")
        loaded_model = self.manager.load_model("rf_test", version)
        
        # Test predictions are consistent
        original_pred = model.predict(self.X)
        loaded_pred = loaded_model.predict(self.X)
        
        # Should be very close (allowing for small numerical differences)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)
    
    def test_transformer_persistence(self):
        """Test persistence of Transformer model."""
        model = create_forecaster('transformer', d_model=16, nhead=2, num_layers=1, sequence_length=8)
        model.fit(self.X, self.y, epochs=2, batch_size=8)  # Quick training for test
        
        # Save and load
        version = self.manager.save_model(model, "transformer_test")
        loaded_model = self.manager.load_model("transformer_test", version)
        
        # Test that model structure is preserved
        self.assertEqual(loaded_model.d_model, 16)
        self.assertEqual(loaded_model.nhead, 2)
        self.assertEqual(loaded_model.num_layers, 1)
        self.assertTrue(loaded_model.is_fitted)
    
    def test_multiple_model_types_workflow(self):
        """Test workflow with multiple different model types."""
        # Train different models
        rf_model = create_forecaster('random_forest', n_estimators=5, sequence_length=8)
        rf_model.fit(self.X, self.y)
        
        transformer_model = create_forecaster('transformer', d_model=16, nhead=2, num_layers=1, sequence_length=8)
        transformer_model.fit(self.X, self.y, epochs=2, batch_size=8)
        
        # Save models
        rf_version = self.manager.save_model(rf_model, "energy_rf", tags=["ml", "ensemble"])
        transformer_version = self.manager.save_model(transformer_model, "energy_transformer", tags=["deep_learning", "attention"])
        
        # List all models
        models = self.manager.list_models()
        self.assertEqual(len(models), 2)
        
        model_names = [m['name'] for m in models]
        self.assertIn("energy_rf", model_names)
        self.assertIn("energy_transformer", model_names)
        
        # Load and test each model
        loaded_rf = self.manager.load_model("energy_rf")
        loaded_transformer = self.manager.load_model("energy_transformer")
        
        self.assertIsInstance(loaded_rf, RandomForestForecaster)
        self.assertIsInstance(loaded_transformer, type(transformer_model))
        
        # Test predictions work
        rf_pred = loaded_rf.predict(self.X)
        transformer_pred = loaded_transformer.predict(self.X)
        
        self.assertEqual(len(rf_pred), len(self.X))
        self.assertEqual(len(transformer_pred), len(self.X))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)