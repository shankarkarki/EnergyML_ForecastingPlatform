"""
Unit tests for data storage and retrieval functionality.
Tests file system storage, SQLite storage, and the storage manager.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.storage import (
    FileSystemStorage,
    SQLiteStorage,
    DataStorageManager
)


class TestFileSystemStorage(unittest.TestCase):
    """Test cases for FileSystemStorage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = FileSystemStorage(base_path=self.temp_dir)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
            'load_mw': np.random.uniform(30000, 70000, 100),
            'price_per_mwh': np.random.uniform(20, 100, 100),
            'region': ['ERCOT'] * 100
        })
        
        self.small_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
            'value': range(10)
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_store_and_retrieve_data(self):
        """Test basic store and retrieve operations."""
        dataset_name = 'test_dataset'
        
        # Store data
        result = self.storage.store_data(self.test_data, dataset_name)
        self.assertTrue(result)
        
        # Retrieve data
        retrieved_data = self.storage.retrieve_data(dataset_name)
        
        # Verify data integrity
        self.assertEqual(len(retrieved_data), len(self.test_data))
        self.assertEqual(list(retrieved_data.columns), list(self.test_data.columns))
    
    def test_store_with_metadata(self):
        """Test storing data with metadata."""
        dataset_name = 'test_with_metadata'
        metadata = {
            'source': 'test',
            'description': 'Test dataset with metadata',
            'version': '1.0'
        }
        
        # Store data with metadata
        result = self.storage.store_data(self.test_data, dataset_name, metadata)
        self.assertTrue(result)
        
        # Retrieve metadata
        info = self.storage.get_dataset_info(dataset_name)
        self.assertEqual(info['source'], 'test')
        self.assertEqual(info['description'], 'Test dataset with metadata')
        self.assertEqual(info['version'], '1.0')
    
    def test_partitioned_storage(self):
        """Test partitioned storage for large datasets."""
        dataset_name = 'test_partitioned'
        
        # Store large dataset
        result = self.storage.store_data(self.test_data, dataset_name)
        self.assertTrue(result)
        
        # Check metadata (partitioning may not be available without Parquet)
        info = self.storage.get_dataset_info(dataset_name)
        self.assertIn('storage_format', info)
        
        # If using pickle format, partitioning won't be available
        if info.get('storage_format') == 'pickle':
            self.assertFalse(info.get('partitioned', False))
        else:
            # If Parquet is available, should be partitioned for large datasets
            self.assertTrue(info.get('partitioned', False))
            self.assertIn('partition_column', info)
        
        # Retrieve data should work regardless
        retrieved_data = self.storage.retrieve_data(dataset_name)
        self.assertEqual(len(retrieved_data), len(self.test_data))
    
    def test_date_range_filtering(self):
        """Test date range filtering during retrieval."""
        dataset_name = 'test_date_filter'
        
        # Store data
        self.storage.store_data(self.test_data, dataset_name)
        
        # Retrieve with date filtering
        start_date = datetime(2024, 1, 1, 12)  # Start from 12:00
        end_date = datetime(2024, 1, 2, 12)    # End at next day 12:00
        
        filtered_data = self.storage.retrieve_data(dataset_name, start_date, end_date)
        
        # Should have fewer rows than original
        self.assertLess(len(filtered_data), len(self.test_data))
        self.assertGreater(len(filtered_data), 0)
    
    def test_list_datasets(self):
        """Test listing available datasets."""
        # Initially should be empty
        datasets = self.storage.list_datasets()
        self.assertEqual(len(datasets), 0)
        
        # Store some datasets
        self.storage.store_data(self.test_data, 'dataset1')
        self.storage.store_data(self.small_data, 'dataset2')
        
        # Should list both datasets
        datasets = self.storage.list_datasets()
        self.assertEqual(len(datasets), 2)
        self.assertIn('dataset1', datasets)
        self.assertIn('dataset2', datasets)
    
    def test_delete_dataset(self):
        """Test dataset deletion."""
        dataset_name = 'test_delete'
        
        # Store data
        self.storage.store_data(self.test_data, dataset_name)
        self.assertIn(dataset_name, self.storage.list_datasets())
        
        # Delete dataset
        result = self.storage.delete_dataset(dataset_name)
        self.assertTrue(result)
        
        # Should no longer be listed
        self.assertNotIn(dataset_name, self.storage.list_datasets())
        
        # Should return empty DataFrame when retrieved
        retrieved_data = self.storage.retrieve_data(dataset_name)
        self.assertTrue(retrieved_data.empty)
    
    def test_empty_data_handling(self):
        """Test handling of empty DataFrames."""
        empty_data = pd.DataFrame()
        result = self.storage.store_data(empty_data, 'empty_dataset')
        
        # Should handle gracefully
        self.assertEqual(result, "")
    
    def test_additional_filters(self):
        """Test additional filtering during retrieval."""
        dataset_name = 'test_filters'
        
        # Store data
        self.storage.store_data(self.test_data, dataset_name)
        
        # Retrieve with filters
        filters = {'region': 'ERCOT'}
        filtered_data = self.storage.retrieve_data(dataset_name, filters=filters)
        
        # All rows should have region = 'ERCOT'
        self.assertTrue(all(filtered_data['region'] == 'ERCOT'))


class TestSQLiteStorage(unittest.TestCase):
    """Test cases for SQLiteStorage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(self.temp_dir, 'test.db')
        self.storage = SQLiteStorage(db_path=db_path)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='h'),
            'load_mw': np.random.uniform(30000, 70000, 50),
            'price_per_mwh': np.random.uniform(20, 100, 50),
            'region': ['ERCOT'] * 50
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_store_and_retrieve_data(self):
        """Test basic store and retrieve operations."""
        dataset_name = 'test_dataset'
        
        # Store data
        result = self.storage.store_data(self.test_data, dataset_name)
        self.assertTrue(result)
        
        # Retrieve data
        retrieved_data = self.storage.retrieve_data(dataset_name)
        
        # Verify data integrity
        self.assertEqual(len(retrieved_data), len(self.test_data))
        self.assertEqual(set(retrieved_data.columns), set(self.test_data.columns))
    
    def test_table_name_sanitization(self):
        """Test table name sanitization."""
        # Test with problematic characters
        dataset_name = 'test-dataset.with@special#chars!'
        
        # Should store successfully
        result = self.storage.store_data(self.test_data, dataset_name)
        self.assertTrue(result)
        
        # Should be able to retrieve
        retrieved_data = self.storage.retrieve_data(dataset_name)
        self.assertEqual(len(retrieved_data), len(self.test_data))
    
    def test_date_filtering(self):
        """Test date range filtering."""
        dataset_name = 'test_date_filter'
        
        # Store data
        self.storage.store_data(self.test_data, dataset_name)
        
        # Retrieve with date filtering
        start_date = datetime(2024, 1, 1, 12)
        end_date = datetime(2024, 1, 2, 12)
        
        filtered_data = self.storage.retrieve_data(dataset_name, start_date, end_date)
        
        # Should have fewer rows
        self.assertLess(len(filtered_data), len(self.test_data))
        self.assertGreater(len(filtered_data), 0)
    
    def test_sql_filters(self):
        """Test SQL-based filtering."""
        dataset_name = 'test_sql_filters'
        
        # Store data
        self.storage.store_data(self.test_data, dataset_name)
        
        # Retrieve with filters
        filters = {'region': 'ERCOT'}
        filtered_data = self.storage.retrieve_data(dataset_name, filters=filters)
        
        # All rows should match filter
        self.assertTrue(all(filtered_data['region'] == 'ERCOT'))
    
    def test_list_and_delete_datasets(self):
        """Test listing and deleting datasets."""
        # Store datasets
        self.storage.store_data(self.test_data, 'dataset1')
        self.storage.store_data(self.test_data, 'dataset2')
        
        # List datasets
        datasets = self.storage.list_datasets()
        self.assertEqual(len(datasets), 2)
        
        # Delete one dataset
        result = self.storage.delete_dataset('dataset1')
        self.assertTrue(result)
        
        # Should have one less dataset
        datasets = self.storage.list_datasets()
        self.assertEqual(len(datasets), 1)
        self.assertNotIn('dataset1', datasets)
    
    def test_metadata_storage(self):
        """Test metadata storage and retrieval."""
        dataset_name = 'test_metadata'
        metadata = {'source': 'test', 'version': '1.0'}
        
        # Store with metadata
        self.storage.store_data(self.test_data, dataset_name, metadata)
        
        # Retrieve metadata
        info = self.storage.get_dataset_info(dataset_name)
        self.assertEqual(info['source'], 'test')
        self.assertEqual(info['version'], '1.0')


class TestDataStorageManager(unittest.TestCase):
    """Test cases for DataStorageManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=30, freq='h'),
            'load_mw': np.random.uniform(30000, 70000, 30),
            'price_per_mwh': np.random.uniform(20, 100, 30)
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_filesystem_backend(self):
        """Test manager with filesystem backend."""
        manager = DataStorageManager(
            backend='filesystem',
            base_path=self.temp_dir
        )
        
        # Test basic operations
        self._test_basic_operations(manager)
    
    def test_sqlite_backend(self):
        """Test manager with SQLite backend."""
        db_path = os.path.join(self.temp_dir, 'test.db')
        manager = DataStorageManager(
            backend='sqlite',
            db_path=db_path
        )
        
        # Test basic operations
        self._test_basic_operations(manager)
    
    def _test_basic_operations(self, manager):
        """Test basic operations for any storage manager."""
        dataset_name = 'test_dataset'
        
        # Store dataset
        result = manager.store_dataset(self.test_data, dataset_name)
        self.assertTrue(result)
        
        # Retrieve dataset
        retrieved_data = manager.retrieve_dataset(dataset_name)
        self.assertEqual(len(retrieved_data), len(self.test_data))
        
        # List datasets
        datasets = manager.list_datasets()
        self.assertIn(dataset_name, datasets)
        
        # Get dataset info
        info = manager.get_dataset_info(dataset_name)
        self.assertIn('row_count', info)
        self.assertEqual(info['row_count'], len(self.test_data))
        
        # Delete dataset
        result = manager.delete_dataset(dataset_name)
        self.assertTrue(result)
        
        # Should no longer be listed
        datasets = manager.list_datasets()
        self.assertNotIn(dataset_name, datasets)
    
    def test_overwrite_protection(self):
        """Test overwrite protection."""
        manager = DataStorageManager(
            backend='filesystem',
            base_path=self.temp_dir
        )
        
        dataset_name = 'test_overwrite'
        
        # Store dataset
        result = manager.store_dataset(self.test_data, dataset_name)
        self.assertTrue(result)
        
        # Try to store again with overwrite=False
        result = manager.store_dataset(self.test_data, dataset_name, overwrite=False)
        self.assertFalse(result)
        
        # Should work with overwrite=True (default)
        result = manager.store_dataset(self.test_data, dataset_name, overwrite=True)
        self.assertTrue(result)
    
    def test_storage_summary(self):
        """Test storage summary generation."""
        manager = DataStorageManager(
            backend='filesystem',
            base_path=self.temp_dir
        )
        
        # Store multiple datasets
        manager.store_dataset(self.test_data, 'dataset1')
        manager.store_dataset(self.test_data, 'dataset2')
        
        # Get summary
        summary = manager.get_storage_summary()
        
        self.assertEqual(summary['total_datasets'], 2)
        self.assertIn('dataset1', summary['datasets'])
        self.assertIn('dataset2', summary['datasets'])
        self.assertEqual(summary['total_rows'], len(self.test_data) * 2)
    
    def test_backup_and_restore(self):
        """Test backup and restore functionality."""
        manager = DataStorageManager(
            backend='filesystem',
            base_path=self.temp_dir
        )
        
        dataset_name = 'test_backup'
        backup_path = os.path.join(self.temp_dir, 'backup.pkl.gz')
        
        # Store dataset
        manager.store_dataset(self.test_data, dataset_name)
        
        # Create backup
        result = manager.backup_dataset(dataset_name, backup_path)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(backup_path))
        
        # Delete original dataset
        manager.delete_dataset(dataset_name)
        self.assertNotIn(dataset_name, manager.list_datasets())
        
        # Restore from backup
        result = manager.restore_dataset(backup_path, dataset_name)
        self.assertTrue(result)
        
        # Should be available again
        self.assertIn(dataset_name, manager.list_datasets())
        
        # Data should be intact
        restored_data = manager.retrieve_dataset(dataset_name)
        self.assertEqual(len(restored_data), len(self.test_data))
    
    def test_date_range_retrieval(self):
        """Test date range retrieval."""
        manager = DataStorageManager(
            backend='filesystem',
            base_path=self.temp_dir
        )
        
        dataset_name = 'test_date_range'
        
        # Store dataset
        manager.store_dataset(self.test_data, dataset_name)
        
        # Retrieve with date range
        start_date = datetime(2024, 1, 1, 12)
        end_date = datetime(2024, 1, 2, 12)
        
        filtered_data = manager.retrieve_dataset(
            dataset_name, 
            start_date=start_date, 
            end_date=end_date
        )
        
        # Should have fewer rows
        self.assertLess(len(filtered_data), len(self.test_data))
        self.assertGreater(len(filtered_data), 0)
    
    def test_invalid_backend(self):
        """Test handling of invalid backend."""
        with self.assertRaises(ValueError):
            DataStorageManager(backend='invalid_backend')


class TestStorageIntegration(unittest.TestCase):
    """Integration tests for storage system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create realistic energy data
        dates = pd.date_range('2024-01-01', periods=168, freq='h')  # One week
        self.energy_data = pd.DataFrame({
            'timestamp': dates,
            'ercot_load_mw': np.random.uniform(30000, 70000, 168),
            'ercot_price_per_mwh': np.random.uniform(20, 200, 168),
            'wind_generation_mw': np.random.uniform(5000, 25000, 168),
            'solar_generation_mw': np.random.uniform(0, 15000, 168),
            'temperature_f': np.random.uniform(40, 95, 168),
            'region': ['ERCOT'] * 168,
            'data_source': ['GridStatus'] * 168
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_energy_data_workflow(self):
        """Test complete workflow with energy data."""
        manager = DataStorageManager(
            backend='filesystem',
            base_path=self.temp_dir
        )
        
        # Store energy data with metadata
        metadata = {
            'source': 'GridStatus',
            'region': 'ERCOT',
            'data_type': 'market_data',
            'collection_date': datetime.now().isoformat()
        }
        
        result = manager.store_dataset(self.energy_data, 'ercot_market_data', metadata)
        self.assertTrue(result)
        
        # Test various retrieval scenarios
        
        # 1. Retrieve all data
        all_data = manager.retrieve_dataset('ercot_market_data')
        self.assertEqual(len(all_data), len(self.energy_data))
        
        # 2. Retrieve with date range
        start_date = datetime(2024, 1, 2)
        end_date = datetime(2024, 1, 4)
        date_filtered = manager.retrieve_dataset(
            'ercot_market_data',
            start_date=start_date,
            end_date=end_date
        )
        self.assertLess(len(date_filtered), len(self.energy_data))
        self.assertGreater(len(date_filtered), 0)
        
        # 3. Retrieve with filters
        region_filtered = manager.retrieve_dataset(
            'ercot_market_data',
            filters={'region': 'ERCOT'}
        )
        self.assertEqual(len(region_filtered), len(self.energy_data))
        
        # 4. Test metadata retrieval
        info = manager.get_dataset_info('ercot_market_data')
        self.assertEqual(info['source'], 'GridStatus')
        self.assertEqual(info['region'], 'ERCOT')
        
        # 5. Test storage summary
        summary = manager.get_storage_summary()
        self.assertEqual(summary['total_datasets'], 1)
        self.assertEqual(summary['total_rows'], len(self.energy_data))
    
    def test_multiple_datasets_management(self):
        """Test managing multiple datasets."""
        manager = DataStorageManager(
            backend='sqlite',
            db_path=os.path.join(self.temp_dir, 'energy.db')
        )
        
        # Create different types of datasets
        load_data = self.energy_data[['timestamp', 'ercot_load_mw', 'region']].copy()
        price_data = self.energy_data[['timestamp', 'ercot_price_per_mwh', 'region']].copy()
        weather_data = self.energy_data[['timestamp', 'temperature_f']].copy()
        
        # Store datasets
        manager.store_dataset(load_data, 'ercot_load', {'type': 'load'})
        manager.store_dataset(price_data, 'ercot_price', {'type': 'price'})
        manager.store_dataset(weather_data, 'weather', {'type': 'weather'})
        
        # List all datasets
        datasets = manager.list_datasets()
        self.assertEqual(len(datasets), 3)
        self.assertIn('ercot_load', datasets)
        self.assertIn('ercot_price', datasets)
        self.assertIn('weather', datasets)
        
        # Test selective retrieval
        load_retrieved = manager.retrieve_dataset('ercot_load')
        self.assertEqual(len(load_retrieved.columns), 3)
        self.assertIn('ercot_load_mw', load_retrieved.columns)
        
        # Test summary
        summary = manager.get_storage_summary()
        self.assertEqual(summary['total_datasets'], 3)


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)