"""
Feature engineering module for power market forecasting.
Implements comprehensive feature creation for energy time series data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from abc import ABC, abstractmethod
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Some features will be limited.")
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer(ABC):
    """Abstract base class for feature engineering."""
    
    @abstractmethod
    def create_features(self, data: pd.DataFrame, feature_config: Dict[str, Any]) -> pd.DataFrame:
        """Create features from raw data."""
        pass
    
    @abstractmethod
    def normalize_features(self, features: pd.DataFrame, scaler_type: str) -> Tuple[pd.DataFrame, Any]:
        """Normalize features using specified scaler."""
        pass


class TimeSeriesFeatureEngineer(FeatureEngineer):
    """
    Feature engineer specialized for time series data.
    Creates temporal, statistical, and domain-specific features.
    """
    
    def __init__(self):
        """Initialize the time series feature engineer."""
        self.scalers = {}
        self.feature_importance = {}
        self.feature_history = []
        
    def create_features(self, data: pd.DataFrame, feature_config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Create comprehensive features from time series data.
        
        Args:
            data: Input DataFrame with time series data
            feature_config: Configuration for feature creation
            
        Returns:
            DataFrame with engineered features
        """
        if data.empty:
            logger.warning("Empty data provided for feature engineering")
            return pd.DataFrame()
        
        # Default configuration - auto-detect numeric columns if not specified
        if feature_config is None:
            # Auto-detect numeric columns for target_columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove timestamp-like columns
            numeric_cols = [col for col in numeric_cols if not any(time_word in col.lower() 
                           for time_word in ['timestamp', 'time', 'date', 'year', 'month', 'day', 'hour'])]
            
            feature_config = {
                'time_features': True,
                'lag_features': True,
                'rolling_features': True,
                'statistical_features': True,
                'seasonal_features': True,
                'lag_periods': [1, 2, 3, 6, 12, 24],  # Shorter lags for smaller datasets
                'rolling_windows': [3, 6, 12, 24],     # Shorter windows for smaller datasets
                'target_columns': numeric_cols[:5]  # Limit to first 5 numeric columns
            }
        
        try:
            features_df = data.copy()
            
            # Identify timestamp column
            timestamp_col = self._find_timestamp_column(features_df)
            if not timestamp_col:
                logger.error("No timestamp column found in data")
                return features_df
            
            # Ensure timestamp is datetime
            features_df[timestamp_col] = pd.to_datetime(features_df[timestamp_col])
            features_df = features_df.sort_values(timestamp_col).reset_index(drop=True)
            
            logger.info("Creating time series features...")
            
            # 1. Time-based features
            if feature_config.get('time_features', True):
                features_df = self._create_time_features(features_df, timestamp_col)
            
            # 2. Lag features
            if feature_config.get('lag_features', True):
                features_df = self._create_lag_features(
                    features_df, 
                    feature_config.get('target_columns', []),
                    feature_config.get('lag_periods', [1, 2, 3, 6, 12, 24])
                )
            
            # 3. Rolling window features
            if feature_config.get('rolling_features', True):
                features_df = self._create_rolling_features(
                    features_df,
                    feature_config.get('target_columns', []),
                    feature_config.get('rolling_windows', [3, 6, 12, 24])
                )
            
            # 4. Statistical features
            if feature_config.get('statistical_features', True):
                features_df = self._create_statistical_features(
                    features_df,
                    feature_config.get('target_columns', [])
                )
            
            # 5. Seasonal features
            if feature_config.get('seasonal_features', True):
                features_df = self._create_seasonal_features(features_df, timestamp_col)
            
            # Remove rows with excessive NaN values (keep rows with some valid data)
            initial_rows = len(features_df)
            
            # Only drop rows where more than 50% of values are NaN
            nan_threshold = len(features_df.columns) * 0.5
            features_df = features_df.dropna(thresh=int(nan_threshold))
            
            final_rows = len(features_df)
            
            if initial_rows != final_rows:
                logger.info(f"Removed {initial_rows - final_rows} rows with excessive NaN values after feature engineering")
            
            logger.info(f"Created {len(features_df.columns) - len(data.columns)} new features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return data
    
    def _find_timestamp_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find timestamp column in the data."""
        timestamp_candidates = [col for col in data.columns 
                              if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower()]
        
        if timestamp_candidates:
            return timestamp_candidates[0]
        
        # Check for datetime columns
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                return col
        
        return None
    
    def _create_time_features(self, data: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Create time-based features."""
        features_df = data.copy()
        
        # Extract time components
        features_df['hour'] = features_df[timestamp_col].dt.hour
        features_df['day_of_week'] = features_df[timestamp_col].dt.dayofweek
        features_df['day_of_month'] = features_df[timestamp_col].dt.day
        features_df['day_of_year'] = features_df[timestamp_col].dt.dayofyear
        features_df['week_of_year'] = features_df[timestamp_col].dt.isocalendar().week
        features_df['month'] = features_df[timestamp_col].dt.month
        features_df['quarter'] = features_df[timestamp_col].dt.quarter
        features_df['year'] = features_df[timestamp_col].dt.year
        
        # Cyclical encoding for periodic features
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        features_df['day_of_week_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['day_of_week_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['day_of_year_sin'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
        features_df['day_of_year_cos'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        
        # Business day indicators
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        features_df['is_business_day'] = (features_df['day_of_week'] < 5).astype(int)
        
        # Time of day categories
        features_df['is_morning'] = ((features_df['hour'] >= 6) & (features_df['hour'] < 12)).astype(int)
        features_df['is_afternoon'] = ((features_df['hour'] >= 12) & (features_df['hour'] < 18)).astype(int)
        features_df['is_evening'] = ((features_df['hour'] >= 18) & (features_df['hour'] < 22)).astype(int)
        features_df['is_night'] = ((features_df['hour'] >= 22) | (features_df['hour'] < 6)).astype(int)
        
        # Peak hours for energy markets
        features_df['is_peak_hour'] = ((features_df['hour'] >= 7) & (features_df['hour'] <= 22) & 
                                      (features_df['day_of_week'] < 5)).astype(int)
        features_df['is_super_peak'] = ((features_df['hour'] >= 14) & (features_df['hour'] <= 18) & 
                                       (features_df['day_of_week'] < 5)).astype(int)
        
        return features_df
    
    def _create_lag_features(self, data: pd.DataFrame, target_columns: List[str], 
                           lag_periods: List[int]) -> pd.DataFrame:
        """Create lag features for target columns."""
        features_df = data.copy()
        
        # Find numeric columns if target_columns not specified
        if not target_columns:
            target_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to existing columns
        target_columns = [col for col in target_columns if col in features_df.columns]
        
        for col in target_columns:
            for lag in lag_periods:
                lag_col_name = f'{col}_lag_{lag}'
                features_df[lag_col_name] = features_df[col].shift(lag)
        
        return features_df
    
    def _create_rolling_features(self, data: pd.DataFrame, target_columns: List[str], 
                               rolling_windows: List[int]) -> pd.DataFrame:
        """Create rolling window features."""
        features_df = data.copy()
        
        # Find numeric columns if target_columns not specified
        if not target_columns:
            target_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to existing columns
        target_columns = [col for col in target_columns if col in features_df.columns]
        
        for col in target_columns:
            for window in rolling_windows:
                # Rolling statistics
                features_df[f'{col}_rolling_mean_{window}'] = features_df[col].rolling(window=window).mean()
                features_df[f'{col}_rolling_std_{window}'] = features_df[col].rolling(window=window).std()
                features_df[f'{col}_rolling_min_{window}'] = features_df[col].rolling(window=window).min()
                features_df[f'{col}_rolling_max_{window}'] = features_df[col].rolling(window=window).max()
                features_df[f'{col}_rolling_median_{window}'] = features_df[col].rolling(window=window).median()
                
                # Rolling percentiles
                features_df[f'{col}_rolling_q25_{window}'] = features_df[col].rolling(window=window).quantile(0.25)
                features_df[f'{col}_rolling_q75_{window}'] = features_df[col].rolling(window=window).quantile(0.75)
                
                # Rolling differences
                features_df[f'{col}_rolling_diff_{window}'] = features_df[col] - features_df[f'{col}_rolling_mean_{window}']
        
        return features_df
    
    def _create_statistical_features(self, data: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """Create statistical features."""
        features_df = data.copy()
        
        # Find numeric columns if target_columns not specified
        if not target_columns:
            target_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to existing columns
        target_columns = [col for col in target_columns if col in features_df.columns]
        
        for col in target_columns:
            # Rate of change
            features_df[f'{col}_pct_change'] = features_df[col].pct_change()
            features_df[f'{col}_diff'] = features_df[col].diff()
            
            # Exponential moving averages
            features_df[f'{col}_ema_12'] = features_df[col].ewm(span=12).mean()
            features_df[f'{col}_ema_24'] = features_df[col].ewm(span=24).mean()
            
            # Volatility (rolling standard deviation)
            features_df[f'{col}_volatility_12'] = features_df[col].rolling(window=12).std()
            features_df[f'{col}_volatility_24'] = features_df[col].rolling(window=24).std()
            
            # Z-score (standardized values)
            rolling_mean = features_df[col].rolling(window=24).mean()
            rolling_std = features_df[col].rolling(window=24).std()
            features_df[f'{col}_zscore'] = (features_df[col] - rolling_mean) / rolling_std
        
        return features_df
    
    def _create_seasonal_features(self, data: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Create seasonal and calendar features."""
        features_df = data.copy()
        
        # Season indicators
        month = features_df[timestamp_col].dt.month
        features_df['is_spring'] = ((month >= 3) & (month <= 5)).astype(int)
        features_df['is_summer'] = ((month >= 6) & (month <= 8)).astype(int)
        features_df['is_fall'] = ((month >= 9) & (month <= 11)).astype(int)
        features_df['is_winter'] = ((month == 12) | (month <= 2)).astype(int)
        
        # Holiday indicators (simplified - can be expanded)
        features_df['is_holiday'] = 0  # Placeholder - would need holiday calendar
        
        # Special energy market periods
        features_df['is_heating_season'] = ((month >= 11) | (month <= 3)).astype(int)
        features_df['is_cooling_season'] = ((month >= 5) & (month <= 9)).astype(int)
        
        return features_df
    
    def normalize_features(self, features: pd.DataFrame, scaler_type: str = 'standard') -> Tuple[pd.DataFrame, Any]:
        """
        Normalize features using specified scaler.
        
        Args:
            features: DataFrame with features to normalize
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple of (normalized_features, fitted_scaler)
        """
        try:
            # Select numeric columns only
            numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                logger.warning("No numeric columns found for normalization")
                return features, None
            
            # Initialize scaler
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                logger.error(f"Unknown scaler type: {scaler_type}")
                return features, None
            
            # Clean data before normalization (remove inf and extreme values)
            features_clean = features.copy()
            for col in numeric_columns:
                # Replace inf values with NaN
                features_clean[col] = features_clean[col].replace([np.inf, -np.inf], np.nan)
                # Fill NaN with median
                if not features_clean[col].isna().all():
                    features_clean[col] = features_clean[col].fillna(features_clean[col].median())
            
            # Fit and transform
            features_normalized = features_clean.copy()
            features_normalized[numeric_columns] = scaler.fit_transform(features_clean[numeric_columns])
            
            # Store scaler for later use
            self.scalers[scaler_type] = scaler
            
            logger.info(f"Normalized {len(numeric_columns)} features using {scaler_type} scaler")
            return features_normalized, scaler
            
        except Exception as e:
            logger.error(f"Error in feature normalization: {e}")
            return features, None
    
    def select_features(self, features: pd.DataFrame, target: pd.Series, 
                       method: str = 'mutual_info', k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features using specified method.
        
        Args:
            features: DataFrame with features
            target: Target variable
            method: Feature selection method ('mutual_info', 'f_regression')
            k: Number of features to select
            
        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        try:
            # Select numeric columns only
            numeric_features = features.select_dtypes(include=[np.number])
            
            if numeric_features.empty:
                logger.warning("No numeric features found for selection")
                return features, list(features.columns)
            
            # Remove rows with NaN values
            combined_data = pd.concat([numeric_features, target], axis=1).dropna()
            if combined_data.empty:
                logger.warning("No data left after removing NaN values")
                return features, list(features.columns)
            
            X = combined_data.iloc[:, :-1]
            y = combined_data.iloc[:, -1]
            
            # Select feature selection method
            if method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
            elif method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
            else:
                logger.error(f"Unknown feature selection method: {method}")
                return features, list(features.columns)
            
            # Fit selector
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Create feature importance dictionary
            scores = selector.scores_
            feature_scores = dict(zip(X.columns, scores))
            self.feature_importance = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))
            
            # Return selected features from original dataframe
            non_numeric_cols = [col for col in features.columns if col not in numeric_features.columns]
            selected_columns = selected_features + non_numeric_cols
            
            logger.info(f"Selected {len(selected_features)} features using {method}")
            return features[selected_columns], selected_features
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return features, list(features.columns)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from last selection."""
        return self.feature_importance
    
    def create_interaction_features(self, data: pd.DataFrame, 
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified feature pairs.
        
        Args:
            data: DataFrame with features
            feature_pairs: List of tuples with feature pairs to interact
            
        Returns:
            DataFrame with interaction features added
        """
        features_df = data.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in features_df.columns and feat2 in features_df.columns:
                # Multiplicative interaction
                features_df[f'{feat1}_x_{feat2}'] = features_df[feat1] * features_df[feat2]
                
                # Ratio interaction (avoid division by zero)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    features_df[f'{feat1}_div_{feat2}'] = features_df[feat1] / (features_df[feat2] + 1e-8)
        
        return features_df
    
    def create_polynomial_features(self, data: pd.DataFrame, columns: List[str], 
                                 degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.
        
        Args:
            data: DataFrame with features
            columns: List of columns to create polynomial features for
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features added
        """
        features_df = data.copy()
        
        for col in columns:
            if col in features_df.columns and pd.api.types.is_numeric_dtype(features_df[col]):
                for d in range(2, degree + 1):
                    features_df[f'{col}_poly_{d}'] = features_df[col] ** d
        
        return features_df
    
    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of features in the dataset.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Dictionary with feature summary
        """
        summary = {
            'total_features': len(data.columns),
            'numeric_features': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(data.select_dtypes(include=['object', 'category']).columns),
            'datetime_features': len(data.select_dtypes(include=['datetime64']).columns),
            'feature_types': {},
            'missing_values': {},
            'data_shape': data.shape
        }
        
        # Feature types
        for col in data.columns:
            summary['feature_types'][col] = str(data[col].dtype)
            summary['missing_values'][col] = data[col].isnull().sum()
        
        return summary