"""Configuration constants for portfolio analysis system."""

import os

# Analysis settings
ANALYSIS_MONTHS = 6  # Analysis period in months
TRAINING_YEARS = 5  # Training data period in years
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
MIN_TRAINING_SAMPLES = 100  # Minimum samples for ML training

# Trading settings
MIN_DATA_DAYS = 252  # Minimum days of data required (1 year)
ACCURACY_THRESHOLD = 0.55  # Minimum accuracy for Random Forest participation
EMAIL_CONFIDENCE_THRESHOLD = 65  # Only send email if signals >65 or <35

# Email settings
EMAIL_SENDER = "demibotti2000@gmail.com"
EMAIL_RECEIVER = "demibotti2000@gmail.com"
EMAIL_BCC_ENABLED = os.getenv('EMAIL_BCC_ENABLED', 'false').lower() == 'true'  # Disabled by default
EMAIL_BCC = ["laetitiarakotoarisoa@gmail.com", "nifemibolatimi@gmail.com"]  # Hidden recipients

# Technical indicator windows
RSI_WINDOW = 14
MOMENTUM_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RS_WINDOWS = [20, 60, 120]  # Relative Strength windows
VOLUME_RATIO_WINDOW = 20
ATR_WINDOW = 14
MA_WINDOWS = [50, 200]  # Moving Average windows
BB_WINDOW = 20
BB_STD = 2

# Trading thresholds
BUY_SCORE_THRESHOLD = 55
SELL_SCORE_THRESHOLD = 55
STRONG_BUY_THRESHOLD = 65
STRONG_SELL_THRESHOLD = 65
NEUTRAL_RANGE = (35, 65)

# Trading restriction settings
RESTRICTION_DAYS = 60  # 60-day trading restriction

# Position sizing settings
MAX_POSITION_SIZE = 0.10  # Maximum 10% of portfolio per position
MIN_POSITION_SIZE = 0.01  # Minimum 1% of portfolio per position
DEFAULT_POSITION_SIZE = 0.05  # Default 5% position size
RISK_LEVELS = {
    'conservative': 0.03,  # 3% position size
    'moderate': 0.05,     # 5% position size
    'aggressive': 0.08    # 8% position size
}

# Portfolio weight limits
MAX_SINGLE_STOCK_WEIGHT = 0.15  # Maximum 15% in any single stock
MAX_SECTOR_WEIGHT = 0.25  # Maximum 25% in any sector
MIN_CASH_RESERVE = 0.10  # Keep at least 10% cash

# Gradual entry/exit settings
GRADUAL_ENTRY_ENABLED = True
GRADUAL_EXIT_ENABLED = True
ENTRY_CONFIDENCE_LEVELS = {
    'low': (35, 45),      # Entry 25% of position
    'medium': (45, 55),   # Entry 50% of position
    'high': (55, 65),     # Entry 75% of position
    'very_high': (65, 100) # Entry 100% of position
}
EXIT_CONFIDENCE_LEVELS = {
    'low': (35, 45),      # Exit 25% of position
    'medium': (45, 55),   # Exit 50% of position
    'high': (55, 65),     # Exit 75% of position
    'very_high': (65, 100) # Exit 100% of position
}

# Portfolio management settings
KEEP_SOLD_STOCKS = True  # Keep sold stocks with 0 shares for future analysis
AUTO_UPDATE_PORTFOLIO = False  # Auto-update portfolio.csv (requires manual confirmation)
