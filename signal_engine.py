"""Composite signal engine (v2).

Combines trend, relative strength, pullback, market-stress and volume
components into a single 0-100 score per day, plus a volatility-scaled
target exposure. All components use only data available at each date.
"""

import numpy as np
import pandas as pd

import config
from technical_indicators import (
    calculate_rsi, calculate_relative_strength, calculate_distance_from_ma,
    calculate_bollinger_band_position, calculate_adx, calculate_chaikin_money_flow,
    calculate_obv_slope, calculate_momentum_12_1, calculate_ma_slope,
    calculate_realized_volatility, calculate_vix_term_structure,
    calculate_macd_histogram_pct,
)


def _clip(series, scale):
    return (series / scale).clip(-1, 1)


def compute_indicator_frame(ohlcv, spy_close, vix_close=None, vix3m_close=None):
    """Compute every indicator used by the engine, aligned to `ohlcv.index`."""
    close, high, low, volume = ohlcv['Close'], ohlcv['High'], ohlcv['Low'], ohlcv['Volume']
    spy = spy_close.reindex(close.index).ffill()
    adx, plus_di, minus_di = calculate_adx(high, low, close, window=config.ADX_WINDOW)

    frame = pd.DataFrame(index=close.index)
    frame['Close'] = close
    frame['RSI'] = calculate_rsi(close, window=config.RSI_WINDOW)
    frame['BB_Position'] = calculate_bollinger_band_position(close, window=config.BB_WINDOW, num_std=config.BB_STD)
    frame['MACD_Hist_Pct'] = calculate_macd_histogram_pct(close)
    frame['Dist_MA200'] = calculate_distance_from_ma(close, 200)
    frame['MA50_Above_MA200'] = (close.rolling(50).mean() > close.rolling(200).mean()).astype(float)
    frame['MA200_Slope'] = calculate_ma_slope(close, 200, 20)
    frame['Momentum_12_1'] = calculate_momentum_12_1(close)
    frame['RS_60D'] = calculate_relative_strength(close, spy, 60)
    frame['RS_120D'] = calculate_relative_strength(close, spy, 120)
    frame['ADX'] = adx
    frame['DI_Spread'] = plus_di - minus_di
    frame['CMF'] = calculate_chaikin_money_flow(high, low, close, volume, window=config.CMF_WINDOW)
    frame['OBV_Slope'] = calculate_obv_slope(close, volume, window=config.CMF_WINDOW)
    frame['Realized_Vol'] = calculate_realized_volatility(close, window=config.VOL_WINDOW)
    frame['SPY_MA200_Slope'] = calculate_ma_slope(spy, 200, 20)

    if vix_close is not None and vix3m_close is not None:
        term = calculate_vix_term_structure(vix_close, vix3m_close)
        frame['VIX_Term'] = term.reindex(close.index).ffill()
    else:
        frame['VIX_Term'] = np.nan
    return frame


def compute_signal_scores(frame):
    """Return a DataFrame with component scores (-1..1), Composite (0-100) and TargetExposure (0..1)."""
    w = config.SIGNAL_WEIGHTS

    trend = (
        np.sign(frame['Dist_MA200'])
        + (frame['MA50_Above_MA200'] * 2 - 1)
        + _clip(frame['MA200_Slope'], 2.0)
        + _clip(frame['Momentum_12_1'], 0.30)
    ) / 4
    # A strong ADX amplifies the trend reading in the direction of the DI spread.
    adx_strength = ((frame['ADX'] - 20) / 20).clip(0, 1)
    trend = (trend + 0.25 * adx_strength * np.sign(frame['DI_Spread'])).clip(-1, 1)

    rel_strength = _clip((frame['RS_60D'] + frame['RS_120D']) / 2, 0.15)

    # Short-term oversold readings are only rewarded inside an established uptrend.
    oversold = (((0.5 - frame['BB_Position']) * 2).clip(-1, 1) + _clip(50 - frame['RSI'], 25)) / 2
    # Contrarian components only ever add conviction; used as sell signals they cut exposure
    # during strong uptrends and historically destroyed returns.
    pullback = oversold.clip(lower=0).where(trend > 0, 0.0)

    # VIX backwardation (VIX > VIX3M) has historically preceded above-average returns;
    # dampen the contrarian read when the broad market's long-term trend is falling.
    stress = _clip(frame['VIX_Term'] - 0.9, 0.15).clip(lower=0).fillna(0)
    stress = stress.where(frame['SPY_MA200_Slope'] >= 0, stress * 0.5)

    volume = (_clip(frame['CMF'], 0.2) + _clip(frame['OBV_Slope'], 0.3)) / 2

    components = pd.DataFrame({
        'Trend': trend,
        'RelativeStrength': rel_strength,
        'Pullback': pullback,
        'MarketStress': stress,
        'Volume': volume,
    })
    raw = sum(components[name] * weight for name, weight in w.items())
    composite = (50 + 50 * raw / sum(w.values())).clip(0, 100)

    vol_scale = (config.TARGET_VOLATILITY / frame['Realized_Vol']).clip(upper=1.0)
    exposure = ((composite - 35) / 30).clip(config.MIN_EXPOSURE, 1) * vol_scale

    out = components.copy()
    out['Composite'] = composite
    out['TargetExposure'] = exposure
    valid = frame[['Dist_MA200', 'Momentum_12_1', 'RS_120D', 'Realized_Vol']].notna().all(axis=1)
    out.loc[~valid, ['Composite', 'TargetExposure']] = np.nan
    return out


def describe_signal(components_row):
    """Short human-readable reasons for the strongest components of one day's signal."""
    labels = {
        'Trend': ('long-term uptrend', 'long-term downtrend'),
        'RelativeStrength': ('outperforming SPY', 'lagging SPY'),
        'Pullback': ('oversold pullback in uptrend', 'short-term overbought'),
        'MarketStress': ('market fear elevated (contrarian buy)', 'complacent market'),
        'Volume': ('accumulation (volume)', 'distribution (volume)'),
    }
    reasons = []
    for name, (pos, neg) in labels.items():
        value = components_row.get(name)
        if value is None or pd.isna(value) or abs(value) < 0.3:
            continue
        reasons.append(pos if value > 0 else neg)
    return reasons
