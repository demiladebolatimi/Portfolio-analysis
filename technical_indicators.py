"""Technical indicator calculation functions."""

import numpy as np
import pandas as pd


def calculate_rsi(prices, window=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_momentum(prices, window=14):
    """Calculate Momentum indicator."""
    return (prices / prices.shift(window)) - 1


def calculate_macd(prices):
    """Calculate MACD indicator."""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    return macd


def calculate_macd_histogram(prices):
    """Calculate MACD Histogram."""
    macd = calculate_macd(prices)
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return histogram


def calculate_relative_strength(stock_prices, spy_prices, window):
    """Calculate Relative Strength vs SPY."""
    stock_return = (stock_prices / stock_prices.shift(window)) - 1
    spy_return = (spy_prices / spy_prices.shift(window)) - 1
    rs = stock_return - spy_return
    return rs


def calculate_relative_strength_120d(stock_prices, spy_prices):
    """Calculate 120-day Relative Strength vs SPY."""
    return calculate_relative_strength(stock_prices, spy_prices, window=120)


def calculate_volume_ratio(volume, window=20):
    """Calculate Volume Ratio (current volume / average volume)."""
    avg_volume = volume.rolling(window=window).mean()
    volume_ratio = volume / avg_volume
    return volume_ratio


def calculate_atr(high, low, close, window=14):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


def calculate_distance_from_ma(prices, ma_window):
    """Calculate distance from moving average as percentage."""
    ma = prices.rolling(window=ma_window).mean()
    distance = (prices - ma) / ma * 100
    return distance


def calculate_bollinger_band_position(prices, window=20, num_std=2):
    """Calculate position within Bollinger Bands (0=lower, 0.5=middle, 1=upper)."""
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return bb_position


def calculate_obv(prices, volume):
    """Calculate On Balance Volume."""
    obv = (np.sign(prices.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_accumulation_distribution(high, low, close, volume):
    """Calculate Accumulation/Distribution Line."""
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)
    ad = (clv * volume).fillna(0).cumsum()
    return ad


def calculate_vix():
    """Get current VIX value."""
    try:
        import yfinance as yf
        vix_ticker = yf.Ticker("^VIX")
        vix_hist = vix_ticker.history(period="1mo")
        if not vix_hist.empty:
            return vix_hist['Close'].iloc[-1]
        return None
    except Exception as e:
        print(f"Error getting VIX: {e}")
        return None


def calculate_market_regime(spy_prices, window=20):
    """Calculate market regime indicators."""
    spy_return_20d = (spy_prices / spy_prices.shift(window)) - 1
    spy_return_60d = (spy_prices / spy_prices.shift(60)) - 1
    return spy_return_20d, spy_return_60d
