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


def calculate_macd_histogram_pct(prices):
    """MACD histogram as a percentage of price (comparable across tickers)."""
    return calculate_macd_histogram(prices) / prices * 100


def calculate_adx(high, low, close, window=14):
    """Average Directional Index with +DI/-DI (Wilder smoothing)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    alpha = 1 / window
    atr = true_range.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx, plus_di, minus_di


def calculate_chaikin_money_flow(high, low, close, volume, window=20):
    """Chaikin Money Flow (-1..1): volume-weighted buying vs selling pressure."""
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
    return (clv * volume).rolling(window).sum() / volume.rolling(window).sum()


def calculate_obv_slope(prices, volume, window=20):
    """OBV change over `window` days normalised by volume traded (-1..1)."""
    obv = calculate_obv(prices, volume)
    return obv.diff(window) / volume.rolling(window).sum()


def calculate_momentum_12_1(prices, lookback=252, skip=21):
    """12-month return excluding the most recent month (classic momentum factor)."""
    return prices.shift(skip) / prices.shift(lookback) - 1


def calculate_52w_high_proximity(prices, window=252):
    """Price as a fraction of its trailing 52-week high (1.0 = at the high)."""
    return prices / prices.rolling(window, min_periods=window // 2).max()


def calculate_ma_slope(prices, ma_window=200, slope_window=20):
    """Percentage change of the moving average over `slope_window` days."""
    ma = prices.rolling(ma_window).mean()
    return (ma / ma.shift(slope_window) - 1) * 100


def calculate_realized_volatility(prices, window=20):
    """Annualised realised volatility of daily returns."""
    return prices.pct_change().rolling(window).std() * np.sqrt(252)


def calculate_vix_term_structure(vix, vix3m):
    """VIX / VIX3M ratio; values above 1 (backwardation) signal market stress."""
    return vix / vix3m
