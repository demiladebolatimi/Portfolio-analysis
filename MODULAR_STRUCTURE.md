# Modular Portfolio Analysis System

## Overview
The portfolio analysis system has been refactored into modular components for better organization, maintainability, and scalability.

## File Structure

### Core Modules

**1. `config.py`**
- Configuration constants and settings
- Analysis parameters (time periods, thresholds, windows)
- Email settings
- Trading restriction settings
- Technical indicator parameters

**2. `technical_indicators.py`**
- All technical indicator calculation functions
- RSI, MACD, Momentum, Bollinger Bands
- Relative Strength, Volume Ratio, ATR
- OBV, Accumulation/Distribution
- VIX, Market Regime indicators

**3. `portfolio_manager.py`**
- Portfolio data loading and management
- Trading restrictions system
- CSV file handling
- Restriction checking and validation
- Historical data fetching

**4. `strategies.py`**
- Q-Learner strategy implementation
- Random Forest strategy implementation
- ML model training and prediction
- Enhanced feature engineering
- Strategy performance calculation

**5. `email_service.py`**
- Email sending functionality
- Report generation
- Chart attachment handling
- HTML email formatting
- Top buy/sell candidates with reasons

**6. `portfolio_analysis_modular.py`**
- Main analysis script
- Orchestrates all modules
- Results compilation and reporting
- Recommendations generation

### Data Files

**7. `portfolio.csv`**
- Current portfolio holdings
- Symbol, shares, cost basis, market value

**8. `trading_restrictions.csv`**
- Trading action tracking
- 60-day restriction enforcement
- Last action dates and prices

## Benefits of Modular Structure

### 1. **Separation of Concerns**
- Each module has a single, well-defined responsibility
- Easier to understand and maintain
- Changes to one module don't affect others

### 2. **Reusability**
- Technical indicators can be used in other projects
- Strategy implementations can be tested independently
- Email service can be used for different reports

### 3. **Testing**
- Individual modules can be unit tested
- Easier to debug issues
- Better code quality

### 4. **Scalability**
- Easy to add new technical indicators
- Simple to add new strategies
- Straightforward to extend functionality

### 5. **Configuration Management**
- Centralized settings in `config.py`
- Easy to adjust parameters
- No need to search through code for settings

## Migration Guide

### From Monolithic to Modular

**Old approach:**
```bash
python portfolio_analysis.py
```

**New approach:**
```bash
python portfolio_analysis_modular.py
```

### Using Individual Modules

**Calculate technical indicators:**
```python
from technical_indicators import calculate_rsi, calculate_macd
rsi = calculate_rsi(prices, window=14)
macd = calculate_macd(prices)
```

**Run specific strategy:**
```python
from strategies import run_qlearner_strategy
trades = run_qlearner_strategy('AAPL', start_date, end_date)
```

**Check trading restrictions:**
```python
from portfolio_manager import check_trading_restrictions
allowed, message = check_trading_restrictions('AAPL', 'SELL', 150.0, restrictions_df)
```

**Send email report:**
```python
from email_service import send_email_report
send_email_report(results_df, "Portfolio Report")
```

## Configuration

Edit `config.py` to customize:

```python
# Analysis settings
ANALYSIS_MONTHS = 6  # Change analysis period
TRAINING_YEARS = 5   # Change training data period
ACCURACY_THRESHOLD = 0.55  # Change ML accuracy threshold

# Trading thresholds
BUY_SCORE_THRESHOLD = 55
SELL_SCORE_THRESHOLD = 65
```

## Adding New Features

### New Technical Indicator
1. Add function to `technical_indicators.py`
2. Import in `strategies.py` if needed for ML
3. Update feature DataFrame in strategy functions

### New Trading Strategy
1. Create function in `strategies.py`
2. Follow existing pattern (data fetching, feature calculation, trading logic)
3. Add to main analysis function in `portfolio_analysis_modular.py`

### New Email Section
1. Modify `email_service.py`
2. Add new section to HTML body
3. Update data processing if needed

## Testing Individual Modules

```python
# Test technical indicators
python -c "from technical_indicators import calculate_rsi; import pandas as pd; print(calculate_rsi(pd.Series([1,2,3,4,5])))"

# Test portfolio manager
python -c "from portfolio_manager import load_portfolio; df = load_portfolio('portfolio.csv'); print(df.head())"

# Test email service
python -c "from email_service import send_email_report; print('Email service loaded')"
```

## GitHub Actions Integration

Update `.github/workflows/portfolio-analysis.yml`:
```yaml
- name: Run portfolio analysis
  env:
    GMAIL_PASSWORD: ${{ secrets.GMAIL_PASSWORD }}
  run: |
    python portfolio_analysis_modular.py
```

## Backward Compatibility

The original `portfolio_analysis.py` remains unchanged and can still be used. The modular version (`portfolio_analysis_modular.py`) provides the same functionality with improved structure.

## Future Enhancements

Planned modular improvements:
- Separate visualization module
- Database integration module
- API service module
- Backtesting engine module
- Risk management module

This modular structure provides a solid foundation for building a professional-grade trading system.
