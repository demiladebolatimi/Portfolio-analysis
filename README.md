# Portfolio Analysis System

An automated ML-driven portfolio analysis system with 18 enhanced technical features, trading restriction enforcement, and intelligent email notifications.

## Features

### 🤖 Enhanced ML Features (18 indicators)
- **Price-based**: RSI, Momentum, MACD Histogram, ATR
- **Relative Strength**: RS_20D, RS_60D, RS_120D vs SPY
- **Volume**: Volume Ratio, OBV, Accumulation/Distribution
- **Market Regime**: SPY Returns (20d, 60d), VIX
- **Position**: Distance from MA50/MA200, Bollinger Band Position

### 📊 Analysis Configuration
- **Training Data**: 5 years historical data
- **Analysis Window**: 6 months (126 trading days)
- **Train/Test Split**: 80/20
- **ML Accuracy Threshold**: 55% (lowered from 95%)

### ⚖️ Trading Restrictions
- **60-Day Rule Enforcement**: Cannot buy and sell at higher price, or sell and buy at lower price within 60 days
- **Automatic Violation Detection**: Checks recommendations against restrictions
- **CSV-based Tracking**: All trading actions tracked in `trading_restrictions.csv`

### 📧 Intelligent Email System
- **Confidence-based Triggering**: Only sends emails when signals >65 or <35
- **Top 5 Buy/Sell Candidates**: With detailed reasoning
- **Chart Attachments**: Portfolio comparison, heatmaps, tracking charts
- **HTML Reports**: Professional formatting with insights

### 🔄 Automated Execution
- **GitHub Actions**: Runs weekdays at 9:00 AM ET
- **Manual Trigger**: On-demand execution available
- **Artifact Storage**: Charts retained for 30 days

## Installation

### 1. Clone the repository
```bash
cd /Users/king.botti/Documents/Portfolio-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up email configuration
```bash
cp .env.example .env
# Edit .env and add your Google App Password
```

### 4. Configure your portfolio
Edit `portfolio.csv` with your current holdings:
```csv
Symbol,Shares,PortfolioWeight,CostBasis,MarketValue
AAPL,10.0,5.0,3000.0,3200.0
MSFT,5.0,3.0,2000.0,2100.0
```

### 5. Set up trading restrictions
Edit `trading_restrictions.csv` with your recent trades:
```csv
Symbol,LastAction,LastActionDate,LastActionPrice,DaysSinceAction,RestrictionType,RestrictionEndDate,Notes
AAPL,BOUGHT,2024-08-01,175.00,0,CANNOT_SELL_HIGHER,2024-09-30,Bought at $175.00
```

## Usage

### Local Execution
```bash
python portfolio_analysis.py
```

### Modular Version
```bash
python portfolio_analysis_modular.py
```

### GitHub Actions
- Push to GitHub
- Add `GMAIL_PASSWORD` as repository secret
- Workflow runs automatically weekdays at 9:00 AM ET

## Configuration

Edit `config.py` to customize:

```python
# Analysis settings
ANALYSIS_MONTHS = 6
TRAINING_YEARS = 5
ACCURACY_THRESHOLD = 0.55

# Email confidence threshold
EMAIL_CONFIDENCE_THRESHOLD = 65  # Only send if signals >65 or <35

# Trading thresholds
BUY_SCORE_THRESHOLD = 55
SELL_SCORE_THRESHOLD = 55
```

## File Structure

```
Portfolio-analysis/
├── portfolio_analysis.py          # Main analysis script
├── portfolio_analysis_modular.py # Modular version
├── config.py                      # Configuration settings
├── technical_indicators.py        # Technical indicator functions
├── portfolio_manager.py           # Portfolio & restrictions management
├── strategies.py                  # ML strategy implementations
├── email_service.py               # Email sending functionality
├── portfolio.csv                  # Current portfolio holdings
├── trading_restrictions.csv       # Trading restriction tracking
├── requirements.txt               # Python dependencies
├── .github/workflows/             # GitHub Actions workflows
├── .gitignore                     # Git ignore rules
└── .env.example                   # Environment variables template
```

## Trading Restrictions System

### How It Works
1. **Automatic Checking**: System checks if recommendations violate 60-day rules
2. **Violation Warnings**: Displays "⚠️ VIOLATION" in recommendations table
3. **Manual Updates**: Update `trading_restrictions.csv` after executing trades

### Restriction Types
- **CANNOT_SELL_HIGHER**: After buying, can't sell at higher price for 60 days
- **CANNOT_BUY_LOWER**: After selling, can't buy back at lower price for 60 days

### Example Workflow
1. System recommends: SELL AAPL (current price $180)
2. Check restrictions: Last action was BOUGHT at $175, 15 days ago
3. Result: ⚠️ VIOLATION - "Bought at $175.00 15 days ago, cannot sell at higher price ($180.00) within 60 days"

## Email Confidence Triggering

The system only sends emails when high-confidence signals are detected:
- **Buy Score > 65**: Strong buy signal
- **Buy Score < 35**: Strong sell signal (opposite)
- **Sell Score > 65**: Strong sell signal
- **Sell Score < 35**: Strong buy signal (opposite)

If no signals exceed the confidence threshold, the email is skipped with a message:
```
📧 Email skipped - No high-confidence signals (>65 or <35) detected
```

## GitHub Actions Setup

1. Push to GitHub
2. Go to Settings → Secrets and variables → Actions
3. Add secret: `GMAIL_PASSWORD` (your Google App Password)
4. Workflow runs automatically weekdays at 9:00 AM ET

## Documentation

- **MODULAR_STRUCTURE.md**: Detailed modular architecture guide
- **TRADING_RESTRICTIONS_GUIDE.md**: Complete trading restrictions documentation
- **GITHUB_ACTIONS_SETUP.md**: GitHub Actions setup instructions

## Performance Metrics

The system evaluates multiple strategies:
- **Buy & Hold**: Passive portfolio performance
- **Benchmark**: S&P 500 (SPY) performance
- **Manual Strategy**: Rule-based trading (RSI-based)
- **Q-Learner**: ML-based strategy with score thresholds
- **Random Forest**: ML model (only if accuracy >55%)

## Security Notes

- `.env` file contains sensitive credentials (gitignored)
- `trading_restrictions.csv` contains trading data (gitignored)
- Use Google App Passwords, not regular passwords
- Never commit sensitive data to repository

## License

This is a personal trading system. Use at your own risk.

## Support

For issues or questions, check the documentation files or review the code comments.
