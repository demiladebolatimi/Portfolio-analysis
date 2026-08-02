"""Configuration constants for portfolio analysis system."""

# Analysis settings
ANALYSIS_MONTHS = 6  # Analysis period in months
TRAINING_YEARS = 5  # Training data period in years
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
MIN_TRAINING_SAMPLES = 100  # Minimum samples for ML training

# Trading settings
MIN_DATA_DAYS = 252  # Minimum days of data required (1 year)
ACCURACY_THRESHOLD = 0.55  # Minimum accuracy for Random Forest participation

# Email settings
EMAIL_SENDER = "demibotti2000@gmail.com"
EMAIL_RECEIVER = "demibotti2000@gmail.com"

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
