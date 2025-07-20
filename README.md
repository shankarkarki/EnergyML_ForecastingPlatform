# EnergyML_ForecastingPlatform

> Universal energy forecasting platform that works with any electricity market worldwide

## 🎯 Project Overview

EnergyML_Forecasting is a market-agnostic energy forecasting platform designed to predict electricity ( load in first phase and prices in second phase ) across different regional markets. Built with production-grade architecture, the platform can adapt to any electricity market structure (US ISOs, European markets, Asian markets) with minimal configuration changes.

**Current Focus:** ERCOT (Texas) market as the first implementation
**Future Scope:** CAISO, PJM, NYISO, European markets

## 🚀 Key Features

- **Real-time data ingestion** via GridStatus API
- **Multi-model forecasting** (Classical time series + Machine Learning)
- **Market-agnostic architecture** (easily extensible to new markets)
- **REST API** for programmatic access
- **Live dashboard** for visualization
- **Production-ready deployment**

## 📊 Technical Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ Data Source │ -> │ Data Pipeline│ -> │ ML Pipeline │ -> │ API/Frontend │
│ (GridStatus)│    │ (Normalize)  │    │ (Forecast)  │    │ (Serve)      │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

### Tech Stack
- **Backend:** Python, FastAPI
- **ML/Data:** pandas, scikit-learn, XGBoost
- **Database:** SQLite -> PostgreSQL
- **API:** GridStatus for energy data
- **Deployment:** Railway/Render
- **Frontend:** Streamlit/React


## 🎯 MVP Success Criteria

### Technical Targets
- **24-hour ERCOT price predictions** with <15% MAPE
- **API response time** <200ms
- **Real-time data refresh** every 5 minutes
- **Multiple model comparison** capabilities

### Demo Objectives
- Live website showing current ERCOT forecasts
- Historical accuracy visualization
- Professional API documentation
- Scalable architecture demonstration

## 🏗️ Market-Agnostic Design

The platform is designed to work with any electricity market by:

1. **Universal data schema** - normalized format for any market
2. **Configurable adapters** - easy integration with new data sources
3. **Generic feature engineering** - works across market structures
4. **Flexible model framework** - adapts to different market patterns

**Adding a new market requires:**
- New data adapter configuration
- Market-specific feature mappings  
- Validation rules adjustment
- No core architecture changes

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8+
- GridStatus API key

### Installation
```bash
git clone https://github.com/yourusername/energyMlplatform.git
cd energyMlplatform
pip install -r requirements.txt
```

### Configuration
```bash
# Create .env file
echo "GRIDSTATUS_API_KEY=your_api_key_here" > .env
```

### Quick Start
```python
from src.data import EnergyDataManager
from src.models import UniversalForecaster

# Initialize data manager for ERCOT
data_manager = EnergyDataManager(market="ERCOT")

# Get latest data
current_data = data_manager.get_latest_prices()

# Generate forecasts
forecaster = UniversalForecaster()
predictions = forecaster.predict_24h(current_data)
```

## 📚 Resources

- [GridStatus API Documentation](https://gridstatus.io/docs)
- [ERCOT Market Overview](http://www.ercot.com/)
- [Energy Forecasting Papers](./docs/references.md)

## 🤝 Contributing

This is a learning project showcasing energy market forecasting capabilities. Future enhancements:

- Additional market support (CAISO, PJM, European markets)
- Advanced ML models (LSTM, Transformer architectures) 
- Real-time streaming capabilities
- Advanced risk management features

---

**Built by:** Shankar Karki - Energy Quant  
**Contact:** shankar.karki660@gmail.com  
