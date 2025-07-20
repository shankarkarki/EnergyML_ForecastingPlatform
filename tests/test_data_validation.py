"""
Unit tests for data validation and cleaning functionality.
Tests all validators, cleaners, and the data quality manager.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.validation import (
    DataQualityIssue,
    CompletenessValidator,
    ConsistencyValidator,
    OutlierValidator,
    EnergyDataValidator,
    DataCleaner,
    DataQualityManager
)


class TestDataQualityIssue(unittest.TestCase):
    """Test cases for DataQualityIssue class."""
    
    def test_initialization(self):
        """Test DataQualityIssue initialization."""
        issue = DataQualityIssue(
            issue_type='missing_values',
            severity='high',
            description='Test issue',
            affected_rows=[1, 2, 3],
            affected_columns=['col1', 'col2']
        )
        
        self.assertEqual(issue.issue_type, 'missing_values')
        self.assertEqual(issue.severity, 'high')
        self.assertEqual(issue.description, 'Test issue')
        self.assertEqual(issue.affected_rows, [1, 2, 3])
        self.assertEqual(issue.affected_columns, ['col1', 'col2'])
        self.assertIsInstance(issue.timestamp, datetime)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        issue = DataQualityIssue('test_type', 'medium', 'Test description')
        issue_dict = issue.to_dict()
        
        self.assertIsInstance(issue_dict, dict)
        self.assertEqual(issue_dict['issue_type'], 'test_type')
        self.assertEqual(issue_dict['severity'], 'medium')
        self.assertEqual(issue_dict['description'], 'Test description')
        self.assertIn('timestamp', issue_dict)


class TestCompletenessValidator(unittest.TestCase):
    """Test cases for CompletenessValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = CompletenessValidator(
            required_columns=['timestamp', 'load_mw'],
            max_missing_percentage=0.1
        )
    
    def test_missing_required_columns(self):
        """Test detection of missing required columns."""
        data = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'price': [10, 20, 30]
        })
        
        issues = self.validator.validate(data)
        
        # Should find missing 'load_mw' column
        missing_col_issues = [i for i in issues if i.issue_type == 'missing_columns']
        self.assertEqual(len(missing_col_issues), 1)
        self.assertIn('load_mw', missing_col_issues[0].affected_columns)
    
    def test_excessive_missing_values(self):
        """Test detection of excessive missing values."""
        data = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5],
            'load_mw': [100, np.nan, np.nan, 400, 500]  # 40% missing
        })
        
        issues = self.validator.validate(data)
        
        # Should find excessive missing values
        missing_val_issues = [i for i in issues if i.issue_type == 'missing_values']
        self.assertEqual(len(missing_val_issues), 1)
        self.assertEqual(missing_val_issues[0].severity, 'high')
    
    def test_empty_rows(self):
        """Test detection of completely empty rows."""
        data = pd.DataFrame({
            'timestamp': [1, np.nan, 3],
            'load_mw': [100, np.nan, 300]
        })
        
        issues = self.validator.validate(data)
        
        # Should find empty row
        empty_row_issues = [i for i in issues if i.issue_type == 'empty_rows']
        self.assertEqual(len(empty_row_issues), 1)
        self.assertIn(1, empty_row_issues[0].affected_rows)
    
    def test_valid_data(self):
        """Test validation of clean data."""
        data = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5],
            'load_mw': [100, 200, 300, 400, 500]
        })
        
        issues = self.validator.validate(data)
        
        # Should find no issues
        self.assertEqual(len(issues), 0)


class TestConsistencyValidator(unittest.TestCase):
    """Test cases for ConsistencyValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConsistencyValidator(timestamp_column='timestamp')
    
    def test_duplicate_rows(self):
        """Test detection of duplicate rows."""
        data = pd.DataFrame({
            'timestamp': [1, 2, 2, 3],
            'load_mw': [100, 200, 200, 300]
        })
        
        issues = self.validator.validate(data)
        
        # Should find duplicate rows
        duplicate_issues = [i for i in issues if i.issue_type == 'duplicate_rows']
        self.assertEqual(len(duplicate_issues), 1)
    
    def test_negative_values(self):
        """Test detection of negative values in positive columns."""
        data = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'load_mw': [100, -50, 300],  # Negative load
            'price_per_mwh': [20, 30, -10]  # Negative price (can be valid)
        })
        
        issues = self.validator.validate(data)
        
        # Should find negative load (but price can be negative)
        negative_issues = [i for i in issues if i.issue_type == 'negative_values']
        load_issues = [i for i in negative_issues if 'load_mw' in i.affected_columns]
        self.assertEqual(len(load_issues), 1)
    
    def test_duplicate_timestamps(self):
        """Test detection of duplicate timestamps."""
        data = pd.DataFrame({
            'timestamp': ['2024-01-01', '2024-01-02', '2024-01-02'],
            'load_mw': [100, 200, 300]
        })
        
        issues = self.validator.validate(data)
        
        # Should find duplicate timestamps
        timestamp_issues = [i for i in issues if i.issue_type == 'duplicate_timestamps']
        self.assertEqual(len(timestamp_issues), 1)
    
    def test_future_timestamps(self):
        """Test detection of unreasonable future timestamps."""
        future_date = datetime.now() + timedelta(days=400)
        data = pd.DataFrame({
            'timestamp': [datetime.now(), future_date],
            'load_mw': [100, 200]
        })
        
        issues = self.validator.validate(data)
        
        # Should find future timestamps
        future_issues = [i for i in issues if i.issue_type == 'future_timestamps']
        self.assertEqual(len(future_issues), 1)


class TestOutlierValidator(unittest.TestCase):
    """Test cases for OutlierValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = OutlierValidator(z_score_threshold=2.0, iqr_multiplier=1.5)
    
    def test_z_score_outliers(self):
        """Test Z-score based outlier detection."""
        # Create data with clear outliers
        normal_data = np.random.normal(100, 10, 50)
        outlier_data = np.append(normal_data, [200, 300])  # Clear outliers
        
        data = pd.DataFrame({
            'load_mw': outlier_data
        })
        
        issues = self.validator.validate(data)
        
        # Should find Z-score outliers
        z_outlier_issues = [i for i in issues if i.issue_type == 'z_score_outliers']
        self.assertGreater(len(z_outlier_issues), 0)
    
    def test_iqr_outliers(self):
        """Test IQR based outlier detection."""
        # Create data with clear outliers - need more extreme values
        data = pd.DataFrame({
            'load_mw': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 1000]  # 1000 is clear outlier
        })
        
        issues = self.validator.validate(data)
        
        # Should find IQR outliers
        iqr_outlier_issues = [i for i in issues if i.issue_type == 'iqr_outliers']
        self.assertGreater(len(iqr_outlier_issues), 0)
    
    def test_insufficient_data(self):
        """Test handling of insufficient data for outlier detection."""
        data = pd.DataFrame({
            'load_mw': [10, 20, 30]  # Too few points
        })
        
        issues = self.validator.validate(data)
        
        # Should not attempt outlier detection
        outlier_issues = [i for i in issues if 'outlier' in i.issue_type]
        self.assertEqual(len(outlier_issues), 0)


class TestEnergyDataValidator(unittest.TestCase):
    """Test cases for EnergyDataValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = EnergyDataValidator()
    
    def test_value_range_validation(self):
        """Test validation of energy data value ranges."""
        data = pd.DataFrame({
            'load_mw': [50000, 150000],  # Second value out of range
            'price_per_mwh': [50, 15000],  # Second value out of range
            'temperature': [75, 200]  # Second value out of range
        })
        
        issues = self.validator.validate(data)
        
        # Should find out-of-range values
        range_issues = [i for i in issues if i.issue_type == 'value_out_of_range']
        self.assertGreater(len(range_issues), 0)
    
    def test_energy_balance_check(self):
        """Test energy balance validation."""
        data = pd.DataFrame({
            'load_mw': [1000, 2000],
            'generation_mw': [500, 1800]  # Second row has significant imbalance
        })
        
        issues = self.validator.validate(data)
        
        # Should find energy imbalance
        balance_issues = [i for i in issues if i.issue_type == 'energy_imbalance']
        self.assertGreater(len(balance_issues), 0)
    
    def test_valid_energy_data(self):
        """Test validation of valid energy data."""
        data = pd.DataFrame({
            'load_mw': [45000, 50000],
            'price_per_mwh': [35, 45],
            'temperature': [75, 80]
        })
        
        issues = self.validator.validate(data)
        
        # Should find no range issues
        range_issues = [i for i in issues if i.issue_type == 'value_out_of_range']
        self.assertEqual(len(range_issues), 0)


class TestDataCleaner(unittest.TestCase):
    """Test cases for DataCleaner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cleaner = DataCleaner()
    
    def test_handle_missing_values_conservative(self):
        """Test conservative missing value handling."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'load_mw': [100, np.nan, 300, np.nan, 500]
        })
        
        # Create mock issue
        issue = DataQualityIssue(
            issue_type='missing_values',
            severity='medium',
            description='Missing values in load_mw',
            affected_columns=['load_mw']
        )
        
        cleaned_data, report = self.cleaner.clean_data(data, [issue], 'conservative')
        
        # Should have imputed missing values
        self.assertEqual(cleaned_data['load_mw'].isnull().sum(), 0)
        self.assertGreater(report['values_imputed'], 0)
    
    def test_handle_duplicates(self):
        """Test duplicate row handling."""
        data = pd.DataFrame({
            'timestamp': [1, 2, 2, 3],
            'load_mw': [100, 200, 200, 300]
        })
        
        # Create mock issue
        issue = DataQualityIssue(
            issue_type='duplicate_rows',
            severity='medium',
            description='Duplicate rows found',
            affected_rows=[2]
        )
        
        cleaned_data, report = self.cleaner.clean_data(data, [issue], 'conservative')
        
        # Should have removed duplicates
        self.assertEqual(len(cleaned_data), 3)
        self.assertGreater(report['rows_removed'], 0)
    
    def test_handle_outliers_conservative(self):
        """Test conservative outlier handling."""
        data = pd.DataFrame({
            'load_mw': [100, 200, 300, 10000]  # Last value is outlier
        })
        
        # Create mock issue
        issue = DataQualityIssue(
            issue_type='z_score_outliers',
            severity='medium',
            description='Z-score outliers found',
            affected_columns=['load_mw'],
            affected_rows=[3]
        )
        
        cleaned_data, report = self.cleaner.clean_data(data, [issue], 'conservative')
        
        # Should have processed the issue
        self.assertIn('issues_addressed', report)
        self.assertGreater(len(report['issues_addressed']), 0)
        
        # The outlier should be capped - check that max value is reasonable
        # The 99th percentile of [100, 200, 300, 10000] should be much less than 10000
        self.assertLess(cleaned_data['load_mw'].max(), 10000)
    
    def test_handle_outliers_aggressive(self):
        """Test aggressive outlier handling."""
        data = pd.DataFrame({
            'load_mw': [100, 200, 300, 10000]  # Last value is outlier
        })
        
        # Create mock issue
        issue = DataQualityIssue(
            issue_type='z_score_outliers',
            severity='medium',
            description='Z-score outliers found',
            affected_rows=[3]
        )
        
        cleaned_data, report = self.cleaner.clean_data(data, [issue], 'aggressive')
        
        # Should have removed outlier rows - check that we have fewer rows
        self.assertLess(len(cleaned_data), len(data))


class TestDataQualityManager(unittest.TestCase):
    """Test cases for DataQualityManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = DataQualityManager()
    
    def test_validate_data(self):
        """Test data validation."""
        data = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'load_mw': [100, np.nan, 300]  # Has missing value
        })
        
        _, report = self.manager.validate_data(data)
        
        # Should generate validation report
        self.assertIn('total_issues', report)
        self.assertIn('issues_by_severity', report)
        self.assertIn('issues_by_type', report)
        self.assertIn('detailed_issues', report)
        self.assertGreater(report['total_issues'], 0)
    
    def test_validate_and_clean_data(self):
        """Test combined validation and cleaning."""
        data = pd.DataFrame({
            'timestamp': [1, 2, 2, 3],  # Has duplicate
            'load_mw': [100, np.nan, 200, 300]  # Has missing value
        })
        
        cleaned_data, report = self.manager.validate_and_clean_data(data, 'conservative')
        
        # Should have cleaned data and comprehensive report
        self.assertIn('original_validation', report)
        self.assertIn('cleaning_report', report)
        self.assertIn('post_cleaning_validation', report)
        self.assertIn('improvement_summary', report)
        
        # Should show improvement
        improvement = report['improvement_summary']
        self.assertGreaterEqual(improvement['issues_before'], improvement['issues_after'])
    
    def test_generate_quality_report(self):
        """Test quality report generation."""
        data = pd.DataFrame({
            'timestamp': [1, 2, 3],
            'load_mw': [100, np.nan, 300]
        })
        
        report_text = self.manager.generate_quality_report(data)
        
        # Should generate readable report
        self.assertIsInstance(report_text, str)
        self.assertIn('DATA QUALITY REPORT', report_text)
        self.assertIn('Dataset Shape', report_text)
    
    def test_custom_validators(self):
        """Test manager with custom validators."""
        custom_validators = [CompletenessValidator()]
        manager = DataQualityManager(validators=custom_validators)
        
        self.assertEqual(len(manager.validators), 1)
        self.assertIsInstance(manager.validators[0], CompletenessValidator)


class TestDataValidationIntegration(unittest.TestCase):
    """Integration tests for data validation system."""
    
    def test_complete_validation_workflow(self):
        """Test complete validation and cleaning workflow."""
        # Create realistic energy data with various issues
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'load_mw': np.random.normal(45000, 5000, 100),
            'price_per_mwh': np.random.normal(50, 20, 100),
            'temperature': np.random.normal(75, 15, 100)
        })
        
        # Introduce various issues
        data.loc[10:15, 'load_mw'] = np.nan  # Missing values
        data.loc[20, :] = data.loc[19, :]  # Duplicate row
        data.loc[30, 'load_mw'] = 200000  # Outlier
        data.loc[40, 'price_per_mwh'] = -2000  # Extreme negative price
        
        # Run complete workflow
        manager = DataQualityManager()
        cleaned_data, report = manager.validate_and_clean_data(data, 'conservative')
        
        # Verify results
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertIsInstance(report, dict)
        
        # Should have found and addressed issues
        self.assertGreater(report['original_validation']['total_issues'], 0)
        self.assertLessEqual(
            report['post_cleaning_validation']['total_issues'],
            report['original_validation']['total_issues']
        )
        
        # Generate readable report
        quality_report = manager.generate_quality_report(data)
        self.assertIn('DATA QUALITY REPORT', quality_report)
    
    def test_energy_specific_validation(self):
        """Test energy-specific validation scenarios."""
        # Create ERCOT-like data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='h'),
            'ercot_load_mw': np.random.uniform(30000, 70000, 24),
            'ercot_price_per_mwh': np.random.uniform(20, 100, 24),
            'wind_generation_mw': np.random.uniform(5000, 25000, 24),
            'solar_generation_mw': np.random.uniform(0, 15000, 24),
            'temperature_f': np.random.uniform(60, 95, 24)
        })
        
        # Add some energy-specific issues
        data.loc[5, 'ercot_load_mw'] = 150000  # Unrealistic load
        data.loc[10, 'ercot_price_per_mwh'] = 15000  # Price spike
        data.loc[15, 'temperature_f'] = 200  # Unrealistic temperature
        
        # Validate with energy validator
        validator = EnergyDataValidator()
        issues = validator.validate(data)
        
        # Should find energy-specific issues
        range_issues = [i for i in issues if i.issue_type == 'value_out_of_range']
        self.assertGreater(len(range_issues), 0)


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)