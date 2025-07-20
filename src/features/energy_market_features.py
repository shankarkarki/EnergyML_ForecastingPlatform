"""
Domain-specific energy market feature engineering for ERCOT and power markets.
Creates features specific to energy trading, grid operations, and market dynamics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import holidays

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyMarketFeatureEngineer:
    """
    Creates domain-specific features for energy markets, with focus on ERCOT.
    Includes market-specific patterns, holidays, grid events, and trading features.
    """
    
    def __init__(self):
        """Initialize the energy market feature engineer."""
        # ERCOT-specific market hours and patterns
        self.ercot_market_config = {
            'peak_hours': list(range(7, 23)),  # 7 AM to 10 PM
            'super_peak_hours': list(range(14, 19)),  # 2 PM to 6 PM
            'off_peak_hours': list(range(23, 24)) + list(range(0, 7)),  # 11 PM to 6 AM
            'shoulder_hours': [6, 23],  # Transition hours
        }
        
        # Texas holidays (affects energy demand)
        self.texas_holidays = holidays.US(state='TX')
        
        # Energy market event patterns
        self.market_events = {
            'winter_storm_months': [12, 1, 2],
            'summer_peak_months': [6, 7, 8, 9],
            'spring_maintenance_months': [3, 4, 5],
            'fall_maintenance_months': [10, 11]
        }
        
        logger.info("Initialized EnergyMarketFeatureEngineer for ERCOT market")
    
    def create_ercot_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create ERCOT-specific market features.
        
        Args:
            data: DataFrame with timestamp and energy market data
            
        Returns:
            DataFrame with ERCOT market features added
        """
        if data.empty or 'timestamp' not in data.columns:
            logger.warning("Data missing timestamp column for ERCOT market features")
            return data
        
        try:
            features_df = data.copy()
            
            # Ensure timestamp is datetime
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            
            # Create ERCOT market period features
            features_df = self._create_market_period_features(features_df)
            
            # Create ERCOT pricing features
            features_df = self._create_pricing_features(features_df)
            
            # Create load pattern features
            features_df = self._create_load_pattern_features(features_df)
            
            # Create generation mix features
            features_df = self._create_generation_features(features_df)
            
            # Create market stress indicators
            features_df = self._create_market_stress_features(features_df)
            
            logger.info(f"Created ERCOT market features for {len(features_df)} records")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating ERCOT market features: {e}")
            return data
    
    def _create_market_period_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ERCOT market period features."""
        # Extract hour for market period classification
        data['hour'] = data['timestamp'].dt.hour
        
        # ERCOT market periods
        data['is_peak_period'] = data['hour'].isin(self.ercot_market_config['peak_hours']).astype(int)
        data['is_super_peak'] = data['hour'].isin(self.ercot_market_config['super_peak_hours']).astype(int)
        data['is_off_peak'] = data['hour'].isin(self.ercot_market_config['off_peak_hours']).astype(int)
        data['is_shoulder_period'] = data['hour'].isin(self.ercot_market_config['shoulder_hours']).astype(int)
        
        # Business day vs weekend patterns
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['is_weekday'] = (data['day_of_week'] < 5).astype(int)
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Combined market periods (business days have different patterns)
        data['is_business_peak'] = (data['is_peak_period'] & data['is_weekday']).astype(int)
        data['is_business_super_peak'] = (data['is_super_peak'] & data['is_weekday']).astype(int)
        
        # Market day categories
        data['is_monday'] = (data['day_of_week'] == 0).astype(int)  # Often higher demand
        data['is_friday'] = (data['day_of_week'] == 4).astype(int)  # Often lower demand
        data['is_saturday'] = (data['day_of_week'] == 5).astype(int)
        data['is_sunday'] = (data['day_of_week'] == 6).astype(int)  # Lowest demand day
        
        return data
    
    def _create_pricing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create energy pricing-related features."""
        # Look for price columns
        price_columns = [col for col in data.columns if 'price' in col.lower()]
        
        for price_col in price_columns:
            if pd.api.types.is_numeric_dtype(data[price_col]):
                # Price volatility indicators
                data[f'{price_col}_rolling_volatility_24h'] = data[price_col].rolling(window=24).std()
                data[f'{price_col}_rolling_volatility_7d'] = data[price_col].rolling(window=168).std()
                
                # Price spike indicators (ERCOT is known for price spikes)
                price_mean = data[price_col].rolling(window=168).mean()  # 7-day average
                price_std = data[price_col].rolling(window=168).std()
                
                data[f'{price_col}_z_score'] = (data[price_col] - price_mean) / price_std
                data[f'{price_col}_is_spike'] = (data[f'{price_col}_z_score'] > 2).astype(int)
                data[f'{price_col}_is_extreme_spike'] = (data[f'{price_col}_z_score'] > 3).astype(int)
                
                # Price categories (ERCOT-specific thresholds)
                data[f'{price_col}_is_negative'] = (data[price_col] < 0).astype(int)
                data[f'{price_col}_is_low'] = ((data[price_col] >= 0) & (data[price_col] < 25)).astype(int)
                data[f'{price_col}_is_normal'] = ((data[price_col] >= 25) & (data[price_col] < 100)).astype(int)
                data[f'{price_col}_is_high'] = ((data[price_col] >= 100) & (data[price_col] < 1000)).astype(int)
                data[f'{price_col}_is_extreme'] = (data[price_col] >= 1000).astype(int)
                
                # Price momentum
                data[f'{price_col}_momentum_1h'] = data[price_col].diff(1)
                data[f'{price_col}_momentum_3h'] = data[price_col].diff(3)
                data[f'{price_col}_momentum_24h'] = data[price_col].diff(24)
        
        return data
    
    def _create_load_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create load pattern features specific to ERCOT."""
        # Look for load columns
        load_columns = [col for col in data.columns if 'load' in col.lower()]
        
        for load_col in load_columns:
            if pd.api.types.is_numeric_dtype(data[load_col]):
                # Load factor (efficiency indicator)
                rolling_max = data[load_col].rolling(window=24).max()
                rolling_avg = data[load_col].rolling(window=24).mean()
                data[f'{load_col}_load_factor'] = rolling_avg / rolling_max
                
                # Peak load indicators
                daily_max = data[load_col].rolling(window=24).max()
                data[f'{load_col}_is_daily_peak'] = (data[load_col] == daily_max).astype(int)
                
                # Load growth patterns
                data[f'{load_col}_growth_1h'] = data[load_col].pct_change(1)
                data[f'{load_col}_growth_24h'] = data[load_col].pct_change(24)
                data[f'{load_col}_growth_7d'] = data[load_col].pct_change(168)
                
                # Load categories (ERCOT typical ranges)
                load_percentiles = data[load_col].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
                data[f'{load_col}_is_very_low'] = (data[load_col] <= load_percentiles[0.1]).astype(int)
                data[f'{load_col}_is_low'] = ((data[load_col] > load_percentiles[0.1]) & 
                                            (data[load_col] <= load_percentiles[0.25])).astype(int)
                data[f'{load_col}_is_normal'] = ((data[load_col] > load_percentiles[0.25]) & 
                                               (data[load_col] <= load_percentiles[0.75])).astype(int)
                data[f'{load_col}_is_high'] = ((data[load_col] > load_percentiles[0.75]) & 
                                             (data[load_col] <= load_percentiles[0.9])).astype(int)
                data[f'{load_col}_is_very_high'] = (data[load_col] > load_percentiles[0.9]).astype(int)
        
        return data
    
    def _create_generation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create generation mix and renewable features."""
        # Look for generation columns
        generation_columns = [col for col in data.columns if 'generation' in col.lower() or 'gen' in col.lower()]
        
        # Renewable generation features
        renewable_cols = [col for col in generation_columns if any(renewable in col.lower() 
                         for renewable in ['wind', 'solar', 'renewable'])]
        
        for gen_col in renewable_cols:
            if pd.api.types.is_numeric_dtype(data[gen_col]):
                # Renewable penetration (if total generation available)
                total_gen_cols = [col for col in data.columns if 'total' in col.lower() and 'gen' in col.lower()]
                if total_gen_cols:
                    total_gen_col = total_gen_cols[0]
                    data[f'{gen_col}_penetration'] = data[gen_col] / (data[total_gen_col] + 1e-6)
                
                # Renewable variability
                data[f'{gen_col}_variability_1h'] = data[gen_col].rolling(window=1).std()
                data[f'{gen_col}_variability_3h'] = data[gen_col].rolling(window=3).std()
                
                # Renewable ramp rates (important for grid stability)
                data[f'{gen_col}_ramp_1h'] = data[gen_col].diff(1)
                data[f'{gen_col}_ramp_3h'] = data[gen_col].diff(3)
                
                # Renewable forecast error (if forecast data available)
                forecast_cols = [col for col in data.columns if 'forecast' in col.lower() and gen_col.split('_')[0] in col.lower()]
                if forecast_cols:
                    forecast_col = forecast_cols[0]
                    data[f'{gen_col}_forecast_error'] = data[gen_col] - data[forecast_col]
                    data[f'{gen_col}_forecast_error_abs'] = abs(data[f'{gen_col}_forecast_error'])
        
        return data
    
    def _create_market_stress_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market stress and reliability indicators."""
        # Reserve margin indicators (if generation and load data available)
        load_cols = [col for col in data.columns if 'load' in col.lower()]
        gen_cols = [col for col in data.columns if 'generation' in col.lower() or 'gen' in col.lower()]
        
        if load_cols and gen_cols:
            load_col = load_cols[0]  # Use first load column
            
            # Calculate total available generation
            total_gen_cols = [col for col in gen_cols if 'total' in col.lower()]
            if total_gen_cols:
                gen_col = total_gen_cols[0]
                
                # Reserve margin
                data['reserve_margin'] = (data[gen_col] - data[load_col]) / data[load_col]
                data['is_tight_reserves'] = (data['reserve_margin'] < 0.1).astype(int)  # Less than 10%
                data['is_emergency_reserves'] = (data['reserve_margin'] < 0.05).astype(int)  # Less than 5%
        
        # Price-load correlation (market efficiency indicator)
        if load_cols:
            price_cols = [col for col in data.columns if 'price' in col.lower()]
            if price_cols:
                load_col = load_cols[0]
                price_col = price_cols[0]
                
                # Rolling correlation between price and load
                data['price_load_correlation_24h'] = data[price_col].rolling(window=24).corr(data[load_col])
                data['price_load_correlation_7d'] = data[price_col].rolling(window=168).corr(data[load_col])
        
        return data
    
    def create_holiday_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create holiday and special event features.
        
        Args:
            data: DataFrame with timestamp column
            
        Returns:
            DataFrame with holiday features added
        """
        if data.empty or 'timestamp' not in data.columns:
            logger.warning("Data missing timestamp column for holiday features")
            return data
        
        try:
            features_df = data.copy()
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            
            # Extract date for holiday lookup
            features_df['date'] = features_df['timestamp'].dt.date
            
            # US Federal holidays
            features_df['is_federal_holiday'] = features_df['date'].apply(
                lambda x: x in holidays.US()
            ).astype(int)
            
            # Texas state holidays
            features_df['is_texas_holiday'] = features_df['date'].apply(
                lambda x: x in self.texas_holidays
            ).astype(int)
            
            # Specific holidays that affect energy demand
            features_df['is_new_years'] = ((features_df['timestamp'].dt.month == 1) & 
                                         (features_df['timestamp'].dt.day == 1)).astype(int)
            features_df['is_july_4th'] = ((features_df['timestamp'].dt.month == 7) & 
                                        (features_df['timestamp'].dt.day == 4)).astype(int)
            features_df['is_thanksgiving'] = features_df['date'].apply(
                lambda x: x in holidays.US() and 'Thanksgiving' in str(holidays.US().get(x, ''))
            ).astype(int)
            features_df['is_christmas'] = ((features_df['timestamp'].dt.month == 12) & 
                                         (features_df['timestamp'].dt.day == 25)).astype(int)
            
            # Holiday periods (affect demand patterns)
            features_df['is_holiday_week'] = 0
            for _, row in features_df.iterrows():
                date = row['date']
                # Check if within 3 days of a major holiday
                for days_offset in range(-3, 4):
                    check_date = date + timedelta(days=days_offset)
                    if check_date in holidays.US():
                        features_df.loc[features_df['date'] == date, 'is_holiday_week'] = 1
                        break
            
            # Summer vacation period (affects commercial load)
            features_df['is_summer_vacation'] = ((features_df['timestamp'].dt.month >= 6) & 
                                               (features_df['timestamp'].dt.month <= 8)).astype(int)
            
            # School year indicators (affects residential patterns)
            features_df['is_school_year'] = ((features_df['timestamp'].dt.month >= 9) | 
                                           (features_df['timestamp'].dt.month <= 5)).astype(int)
            
            logger.info(f"Created holiday features for {len(features_df)} records")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating holiday features: {e}")
            return data
    
    def create_seasonal_energy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonal features specific to energy markets.
        
        Args:
            data: DataFrame with timestamp column
            
        Returns:
            DataFrame with seasonal energy features added
        """
        if data.empty or 'timestamp' not in data.columns:
            logger.warning("Data missing timestamp column for seasonal features")
            return data
        
        try:
            features_df = data.copy()
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            
            # Extract time components
            features_df['month'] = features_df['timestamp'].dt.month
            features_df['week_of_year'] = features_df['timestamp'].dt.isocalendar().week
            
            # ERCOT seasonal patterns
            features_df['is_winter_storm_season'] = features_df['month'].isin(
                self.market_events['winter_storm_months']
            ).astype(int)
            
            features_df['is_summer_peak_season'] = features_df['month'].isin(
                self.market_events['summer_peak_months']
            ).astype(int)
            
            features_df['is_spring_maintenance'] = features_df['month'].isin(
                self.market_events['spring_maintenance_months']
            ).astype(int)
            
            features_df['is_fall_maintenance'] = features_df['month'].isin(
                self.market_events['fall_maintenance_months']
            ).astype(int)
            
            # Texas-specific seasonal patterns
            features_df['is_hurricane_season'] = ((features_df['month'] >= 6) & 
                                                (features_df['month'] <= 11)).astype(int)
            
            features_df['is_freeze_risk_season'] = ((features_df['month'] <= 3) | 
                                                  (features_df['month'] == 12)).astype(int)
            
            # Daylight saving time effects
            features_df['is_dst_transition_week'] = 0
            # Spring forward (second Sunday in March)
            spring_dst = features_df[(features_df['month'] == 3) & 
                                   (features_df['timestamp'].dt.day >= 8) & 
                                   (features_df['timestamp'].dt.day <= 14)]
            features_df.loc[spring_dst.index, 'is_dst_transition_week'] = 1
            
            # Fall back (first Sunday in November)
            fall_dst = features_df[(features_df['month'] == 11) & 
                                 (features_df['timestamp'].dt.day >= 1) & 
                                 (features_df['timestamp'].dt.day <= 7)]
            features_df.loc[fall_dst.index, 'is_dst_transition_week'] = 1
            
            logger.info(f"Created seasonal energy features for {len(features_df)} records")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating seasonal energy features: {e}")
            return data
    
    def create_grid_event_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to grid events and operational conditions.
        
        Args:
            data: DataFrame with energy market data
            
        Returns:
            DataFrame with grid event features added
        """
        if data.empty:
            return data
        
        try:
            features_df = data.copy()
            
            # Look for relevant columns
            load_cols = [col for col in features_df.columns if 'load' in col.lower()]
            price_cols = [col for col in features_df.columns if 'price' in col.lower()]
            
            # Grid stress indicators
            if load_cols:
                load_col = load_cols[0]
                
                # Load following patterns
                features_df[f'{load_col}_is_ramping_up'] = (features_df[load_col].diff(1) > 0).astype(int)
                features_df[f'{load_col}_is_ramping_down'] = (features_df[load_col].diff(1) < 0).astype(int)
                
                # Rapid load changes (grid stress)
                load_change_threshold = features_df[load_col].std() * 0.5
                features_df[f'{load_col}_rapid_increase'] = (
                    features_df[load_col].diff(1) > load_change_threshold
                ).astype(int)
                features_df[f'{load_col}_rapid_decrease'] = (
                    features_df[load_col].diff(1) < -load_change_threshold
                ).astype(int)
            
            # Market event indicators
            if price_cols and load_cols:
                price_col = price_cols[0]
                load_col = load_cols[0]
                
                # Scarcity pricing events (high price, high load)
                price_95th = features_df[price_col].quantile(0.95)
                load_90th = features_df[load_col].quantile(0.90)
                
                features_df['is_scarcity_event'] = (
                    (features_df[price_col] > price_95th) & 
                    (features_df[load_col] > load_90th)
                ).astype(int)
                
                # Price-load decoupling events (unusual market conditions)
                expected_price = features_df[load_col] * (features_df[price_col].mean() / features_df[load_col].mean())
                price_deviation = abs(features_df[price_col] - expected_price) / expected_price
                features_df['is_price_anomaly'] = (price_deviation > 0.5).astype(int)
            
            logger.info(f"Created grid event features for {len(features_df)} records")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating grid event features: {e}")
            return data
    
    def create_all_energy_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all energy market features in one comprehensive function.
        
        Args:
            data: DataFrame with energy market data
            
        Returns:
            DataFrame with all energy market features added
        """
        if data.empty:
            logger.warning("Empty data provided for energy market feature creation")
            return data
        
        try:
            logger.info("Creating comprehensive energy market features...")
            
            # Start with original data
            features_df = data.copy()
            
            # Apply all feature creation methods
            features_df = self.create_ercot_market_features(features_df)
            features_df = self.create_holiday_features(features_df)
            features_df = self.create_seasonal_energy_features(features_df)
            features_df = self.create_grid_event_features(features_df)
            
            original_cols = len(data.columns)
            new_cols = len(features_df.columns)
            
            logger.info(f"Created {new_cols - original_cols} energy market features")
            logger.info(f"Total features: {original_cols} → {new_cols}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating energy market features: {e}")
            return data
    
    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary of energy market features.
        
        Args:
            data: DataFrame with energy market features
            
        Returns:
            Dictionary with feature summary
        """
        if data.empty:
            return {'error': 'No data available'}
        
        summary = {
            'total_features': len(data.columns),
            'energy_market_features': {},
            'data_shape': data.shape,
            'feature_categories': {
                'market_periods': 0,
                'pricing': 0,
                'load_patterns': 0,
                'generation': 0,
                'holidays': 0,
                'seasonal': 0,
                'grid_events': 0
            }
        }
        
        # Categorize features
        for col in data.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['peak', 'off_peak', 'business']):
                summary['feature_categories']['market_periods'] += 1
            elif any(term in col_lower for term in ['price', 'spike', 'volatility']):
                summary['feature_categories']['pricing'] += 1
            elif any(term in col_lower for term in ['load', 'demand']):
                summary['feature_categories']['load_patterns'] += 1
            elif any(term in col_lower for term in ['generation', 'renewable', 'wind', 'solar']):
                summary['feature_categories']['generation'] += 1
            elif any(term in col_lower for term in ['holiday', 'christmas', 'thanksgiving']):
                summary['feature_categories']['holidays'] += 1
            elif any(term in col_lower for term in ['season', 'winter', 'summer', 'hurricane']):
                summary['feature_categories']['seasonal'] += 1
            elif any(term in col_lower for term in ['scarcity', 'anomaly', 'stress', 'emergency']):
                summary['feature_categories']['grid_events'] += 1
        
        return summary