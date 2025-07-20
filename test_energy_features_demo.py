#!/usr/bin/env python3
"""
Demo script to test the energy market features with sample data.
Shows the comprehensive feature engineering capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
from features.energy_market_features import EnergyMarketFeatureEngineer

def create_sample_ercot_data():
    """Create realistic sample ERCOT data for testing."""
    # Create one week of hourly data
    dates = pd.date_range('2024-07-04', periods=168, freq='h')  # July 4th week
    np.random.seed(42)
    
    # Create realistic daily and weekly patterns
    hours = np.arange(168) % 24
    days = np.arange(168) // 24
    
    # Summer load pattern (higher during day, AC usage)
    daily_pattern = 50000 + 20000 * np.sin(2 * np.pi * (hours - 6) / 24)  # Peak around 2 PM
    weekend_adjustment = np.where(days % 7 >= 5, -5000, 0)  # Lower weekend load
    
    # Price pattern (follows load but with spikes)
    price_base = 40 + 30 * np.sin(2 * np.pi * (hours - 6) / 24)
    price_spikes = np.random.choice([0, 0, 0, 0, 200], size=168)  # Occasional spikes
    
    data = pd.DataFrame({
        'timestamp': dates,
        'ercot_load_mw': daily_pattern + weekend_adjustment + np.random.normal(0, 3000, 168),
        'ercot_price_per_mwh': price_base + price_spikes + np.random.normal(0, 15, 168),
        'wind_generation_mw': np.random.uniform(8000, 30000, 168),
        'solar_generation_mw': np.maximum(0, 15000 * np.sin(2 * np.pi * hours / 24) * (hours >= 6) * (hours <= 19)),
        'total_generation_mw': daily_pattern * 1.15 + weekend_adjustment + np.random.normal(0, 2000, 168),
        'temperature_f': 85 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 3, 168)
    })
    
    return data

def main():
    """Run the energy market features demo."""
    print("🔋 Energy Market Features Demo")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample ERCOT data...")
    sample_data = create_sample_ercot_data()
    print(f"   Created {len(sample_data)} hours of data")
    print(f"   Date range: {sample_data['timestamp'].min()} to {sample_data['timestamp'].max()}")
    
    # Initialize feature engineer
    print("\n🛠️  Initializing Energy Market Feature Engineer...")
    engineer = EnergyMarketFeatureEngineer()
    
    # Create features
    print("\n⚡ Creating comprehensive energy market features...")
    features = engineer.create_all_energy_market_features(sample_data)
    
    # Show results
    print(f"\n📈 Feature Engineering Results:")
    print(f"   Original columns: {len(sample_data.columns)}")
    print(f"   Enhanced columns: {len(features.columns)}")
    print(f"   New features added: {len(features.columns) - len(sample_data.columns)}")
    
    # Get feature summary
    summary = engineer.get_feature_summary(features)
    print(f"\n📋 Feature Categories:")
    for category, count in summary['feature_categories'].items():
        if count > 0:
            print(f"   {category.replace('_', ' ').title()}: {count} features")
    
    # Show some sample features
    print(f"\n🔍 Sample Feature Values (first 5 rows):")
    
    # Market period features
    market_features = ['is_peak_period', 'is_super_peak', 'is_business_peak', 'is_weekend']
    if all(col in features.columns for col in market_features):
        print("\n   Market Period Features:")
        print(features[['timestamp'] + market_features].head().to_string(index=False))
    
    # Holiday features (July 4th should be detected)
    holiday_features = ['is_federal_holiday', 'is_july_4th', 'is_summer_vacation']
    if all(col in features.columns for col in holiday_features):
        print("\n   Holiday Features (July 4th detection):")
        july_4th_data = features[features['timestamp'].dt.date == datetime(2024, 7, 4).date()]
        if not july_4th_data.empty:
            print(july_4th_data[['timestamp'] + holiday_features].head().to_string(index=False))
    
    # Pricing features
    price_features = [col for col in features.columns if 'price' in col.lower() and 'spike' in col][:3]
    if price_features:
        print(f"\n   Price Spike Detection:")
        spike_data = features[features[price_features[0]] == 1] if price_features else pd.DataFrame()
        if not spike_data.empty:
            print(f"   Found {len(spike_data)} price spike events")
            print(spike_data[['timestamp', 'ercot_price_per_mwh'] + price_features[:2]].head().to_string(index=False))
        else:
            print("   No price spikes detected in sample data")
    
    # Load pattern features
    load_features = [col for col in features.columns if 'load' in col.lower() and ('growth' in col or 'factor' in col)][:2]
    if load_features:
        print(f"\n   Load Pattern Features:")
        print(features[['timestamp', 'ercot_load_mw'] + load_features].head().to_string(index=False))
    
    # Seasonal features
    seasonal_features = ['is_summer_peak_season', 'is_hurricane_season', 'is_freeze_risk_season']
    if all(col in features.columns for col in seasonal_features):
        print(f"\n   Seasonal Features (July patterns):")
        print(features[['timestamp'] + seasonal_features].head().to_string(index=False))
    
    # Grid stress features
    stress_features = [col for col in features.columns if any(term in col.lower() for term in ['reserve', 'scarcity', 'stress'])][:3]
    if stress_features:
        print(f"\n   Grid Stress Indicators:")
        print(features[['timestamp'] + stress_features].head().to_string(index=False))
    
    print(f"\n✅ Energy Market Features Demo Complete!")
    print(f"   The feature engineering successfully created {len(features.columns) - len(sample_data.columns)} domain-specific features")
    print(f"   These features capture ERCOT market patterns, holidays, seasonal effects, and grid events")
    print(f"   Ready for use in ML forecasting models! 🚀")

if __name__ == "__main__":
    main()