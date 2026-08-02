"""Trading strategy implementations including Q-Learner and Random Forest."""

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

import config
from technical_indicators import (
    calculate_rsi, calculate_momentum, calculate_macd_histogram,
    calculate_relative_strength, calculate_volume_ratio, calculate_atr,
    calculate_distance_from_ma, calculate_bollinger_band_position,
    calculate_obv, calculate_accumulation_distribution, calculate_vix,
    calculate_market_regime
)


def run_qlearner_strategy(symbol, start_date, end_date, sv=100000):
    """Run Q-Learner Strategy on a symbol using yfinance data with score-based trading."""
    try:
        import yfinance as yf
        
        # Get extended historical data for better training (5 years)
        ticker = yf.Ticker(symbol)
        time.sleep(0.3)  # Delay to avoid rate limiting
        hist = ticker.history(period=f"{config.TRAINING_YEARS}y")
        if hist.empty:
            print(f"  No data found for {symbol} for Q-Learner")
            return None
        
        # Get SPY data for relative strength calculation
        spy_ticker = yf.Ticker("SPY")
        time.sleep(0.3)
        spy_hist = spy_ticker.history(period=f"{config.TRAINING_YEARS}y")
        if spy_hist.empty:
            print(f"  No SPY data found for relative strength")
            return None
        
        # Filter to analysis period for final return calculation
        analysis_start = pd.Timestamp(start_date).tz_localize('America/New_York')
        analysis_end = pd.Timestamp(end_date).tz_localize('America/New_York')
        hist_analysis = hist[(hist.index >= analysis_start) & (hist.index <= analysis_end)]
        
        if hist_analysis.empty:
            print(f"  No analysis period data for {symbol}")
            return None
        
        prices = hist['Close']
        spy_prices = spy_hist['Close']
        volume = hist['Volume']
        high = hist['High']
        low = hist['Low']
        
        if len(prices) < config.MIN_DATA_DAYS:
            print(f"  Insufficient data for Q-Learner {symbol} (got {len(prices)} days)")
            return None
        
        # Train on first 80% of data, test on last 20%
        split_point = int(len(prices) * config.TRAIN_TEST_SPLIT)
        train_prices = prices.iloc[:split_point]
        test_prices = prices.iloc[split_point:]
        
        train_spy = spy_prices.iloc[:split_point]
        train_volume = volume.iloc[:split_point]
        train_high = high.iloc[:split_point]
        train_low = low.iloc[:split_point]
        
        # Calculate enhanced indicators for training data
        train_rsi = calculate_rsi(train_prices, window=config.RSI_WINDOW)
        train_mom = calculate_momentum(train_prices, window=config.MOMENTUM_WINDOW)
        train_macd_hist = calculate_macd_histogram(train_prices)
        train_rs_20d = calculate_relative_strength(train_prices, train_spy, window=20)
        train_rs_60d = calculate_relative_strength(train_prices, train_spy, window=60)
        train_rs_120d = calculate_relative_strength(train_prices, train_spy, window=120)
        train_vol_ratio = calculate_volume_ratio(train_volume, window=config.VOLUME_RATIO_WINDOW)
        train_obv = calculate_obv(train_prices, train_volume)
        train_ad = calculate_accumulation_distribution(train_high, train_low, train_prices, train_volume)
        train_atr = calculate_atr(train_high, train_low, train_prices, window=config.ATR_WINDOW)
        train_dist_ma50 = calculate_distance_from_ma(train_prices, 50)
        train_dist_ma200 = calculate_distance_from_ma(train_prices, 200)
        train_bb_pos = calculate_bollinger_band_position(train_prices, window=config.BB_WINDOW)
        
        # Market regime features
        train_spy_return_20d, train_spy_return_60d = calculate_market_regime(train_spy)
        
        # Get VIX for market sentiment
        current_vix = calculate_vix()
        if current_vix:
            train_vix = pd.Series([current_vix] * len(train_prices), index=train_prices.index)
        else:
            train_vix = pd.Series([20] * len(train_prices), index=train_prices.index)  # Default VIX
        
        # Create target variable: 1 if price goes up next day, 0 if goes down
        train_returns = train_prices.pct_change().shift(-1)
        train_target = (train_returns > 0).astype(int)
        
        # Create feature DataFrame with all enhanced features
        train_features = pd.DataFrame({
            'RSI': train_rsi,
            'Momentum': train_mom,
            'MACD_Histogram': train_macd_hist,
            'RS_20D': train_rs_20d,
            'RS_60D': train_rs_60d,
            'RS_120D': train_rs_120d,
            'Volume_Ratio': train_vol_ratio,
            'OBV': train_obv,
            'Accum_Dist': train_ad,
            'ATR': train_atr,
            'Distance_MA50': train_dist_ma50,
            'Distance_MA200': train_dist_ma200,
            'BB_Position': train_bb_pos,
            'SPY_Return_20D': train_spy_return_20d,
            'SPY_Return_60D': train_spy_return_60d,
            'VIX': train_vix
        })
        
        # Remove NaN values
        valid_indices = ~(train_features.isna().any(axis=1) | train_target.isna())
        train_features = train_features[valid_indices]
        train_target = train_target[valid_indices]
        
        if len(train_features) < config.MIN_TRAINING_SAMPLES:
            print(f"  Insufficient valid training data for {symbol} (got {len(train_features)} samples)")
            return None
        
        # Train Random Forest model for Q-Learner (simplified approach)
        X_train, X_val, y_train, y_val = train_test_split(
            train_features, train_target, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Test on test data with enhanced features
        test_spy = spy_prices.iloc[split_point:]
        test_volume = volume.iloc[split_point:]
        test_high = high.iloc[split_point:]
        test_low = low.iloc[split_point:]
        
        test_rsi = calculate_rsi(test_prices, window=config.RSI_WINDOW)
        test_mom = calculate_momentum(test_prices, window=config.MOMENTUM_WINDOW)
        test_macd_hist = calculate_macd_histogram(test_prices)
        test_rs_20d = calculate_relative_strength(test_prices, test_spy, window=20)
        test_rs_60d = calculate_relative_strength(test_prices, test_spy, window=60)
        test_rs_120d = calculate_relative_strength(test_prices, test_spy, window=120)
        test_vol_ratio = calculate_volume_ratio(test_volume, window=config.VOLUME_RATIO_WINDOW)
        test_obv = calculate_obv(test_prices, test_volume)
        test_ad = calculate_accumulation_distribution(test_high, test_low, test_prices, test_volume)
        test_atr = calculate_atr(test_high, test_low, test_prices, window=config.ATR_WINDOW)
        test_dist_ma50 = calculate_distance_from_ma(test_prices, 50)
        test_dist_ma200 = calculate_distance_from_ma(test_prices, 200)
        test_bb_pos = calculate_bollinger_band_position(test_prices, window=config.BB_WINDOW)
        
        # Market regime features for test data
        test_spy_return_20d, test_spy_return_60d = calculate_market_regime(test_spy)
        
        # Use same VIX for test data
        test_vix = pd.Series([current_vix if current_vix else 20] * len(test_prices), index=test_prices.index)
        
        test_features = pd.DataFrame({
            'RSI': test_rsi,
            'Momentum': test_mom,
            'MACD_Histogram': test_macd_hist,
            'RS_20D': test_rs_20d,
            'RS_60D': test_rs_60d,
            'RS_120D': test_rs_120d,
            'Volume_Ratio': test_vol_ratio,
            'OBV': test_obv,
            'Accum_Dist': test_ad,
            'ATR': test_atr,
            'Distance_MA50': test_dist_ma50,
            'Distance_MA200': test_dist_ma200,
            'BB_Position': test_bb_pos,
            'SPY_Return_20D': test_spy_return_20d,
            'SPY_Return_60D': test_spy_return_60d,
            'VIX': test_vix
        })
        
        # Make predictions (probability of price going up)
        predictions = model.predict_proba(test_features)[:, 1]
        
        # Convert predictions to 0-100 score
        scores = predictions * 100
        
        # Generate trades based on score thresholds
        trades = pd.DataFrame(index=test_prices.index, columns=["Trades"])
        trades["Trades"] = 0.0
        
        curr = 0
        for i in range(len(test_prices)):
            if pd.isna(scores[i]):
                continue
            
            score = scores[i]
            
            # Trading strategy based on score thresholds
            if score > config.STRONG_SELL_THRESHOLD and curr > -1000:  # Score > 65 = SELL signal
                trades.iloc[i] = -1000 - curr
                curr = -1000
            elif score < config.STRONG_BUY_THRESHOLD and curr < 1000:  # Score < 35 = BUY signal
                trades.iloc[i] = 1000 - curr
                curr = 1000
            elif config.NEUTRAL_RANGE[0] <= score <= config.NEUTRAL_RANGE[1] and curr != 0:  # Neutral range = HOLD
                trades.iloc[i] = -curr
                curr = 0
        
        return trades
        
    except Exception as e:
        print(f"Error running Q-Learner for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_randomforest_strategy(symbol, start_date, end_date, sv=100000):
    """Run Random Forest Strategy with enhanced features."""
    try:
        import yfinance as yf
        
        # Get extended historical data for better training (5 years)
        ticker = yf.Ticker(symbol)
        time.sleep(0.3)  # Delay to avoid rate limiting
        hist = ticker.history(period=f"{config.TRAINING_YEARS}y")
        if hist.empty:
            print(f"  No data found for {symbol} for Random Forest")
            return None, None, None
        
        # Get SPY data for relative strength calculation
        spy_ticker = yf.Ticker("SPY")
        time.sleep(0.3)
        spy_hist = spy_ticker.history(period=f"{config.TRAINING_YEARS}y")
        if spy_hist.empty:
            print(f"  No SPY data found for relative strength")
            return None, None, None
        
        # Filter to analysis period for final return calculation
        analysis_start = pd.Timestamp(start_date).tz_localize('America/New_York')
        analysis_end = pd.Timestamp(end_date).tz_localize('America/New_York')
        hist_analysis = hist[(hist.index >= analysis_start) & (hist.index <= analysis_end)]
        
        if hist_analysis.empty:
            print(f"  No analysis period data for {symbol}")
            return None, None, None
        
        prices = hist['Close']
        spy_prices = spy_hist['Close']
        volume = hist['Volume']
        high = hist['High']
        low = hist['Low']
        
        if len(prices) < config.MIN_DATA_DAYS:
            print(f"  Insufficient data for Random Forest {symbol} (got {len(prices)} days)")
            return None, None, None
        
        # Train on first 80% of data, test on last 20%
        split_point = int(len(prices) * config.TRAIN_TEST_SPLIT)
        train_prices = prices.iloc[:split_point]
        test_prices = prices.iloc[split_point:]
        
        train_spy = spy_prices.iloc[:split_point]
        train_volume = volume.iloc[:split_point]
        train_high = high.iloc[:split_point]
        train_low = low.iloc[:split_point]
        
        # Calculate enhanced indicators for training data
        train_rsi = calculate_rsi(train_prices, window=config.RSI_WINDOW)
        train_mom = calculate_momentum(train_prices, window=config.MOMENTUM_WINDOW)
        train_macd_hist = calculate_macd_histogram(train_prices)
        train_rs_20d = calculate_relative_strength(train_prices, train_spy, window=20)
        train_rs_60d = calculate_relative_strength(train_prices, train_spy, window=60)
        train_rs_120d = calculate_relative_strength(train_prices, train_spy, window=120)
        train_vol_ratio = calculate_volume_ratio(train_volume, window=config.VOLUME_RATIO_WINDOW)
        train_obv = calculate_obv(train_prices, train_volume)
        train_ad = calculate_accumulation_distribution(train_high, train_low, train_prices, train_volume)
        train_atr = calculate_atr(train_high, train_low, train_prices, window=config.ATR_WINDOW)
        train_dist_ma50 = calculate_distance_from_ma(train_prices, 50)
        train_dist_ma200 = calculate_distance_from_ma(train_prices, 200)
        train_bb_pos = calculate_bollinger_band_position(train_prices, window=config.BB_WINDOW)
        
        # Market regime features
        train_spy_return_20d, train_spy_return_60d = calculate_market_regime(train_spy)
        
        # Get VIX for market sentiment
        current_vix = calculate_vix()
        if current_vix:
            train_vix = pd.Series([current_vix] * len(train_prices), index=train_prices.index)
        else:
            train_vix = pd.Series([20] * len(train_prices), index=train_prices.index)  # Default VIX
        
        # Create target variable: 1 if price goes up next day, 0 if goes down
        train_returns = train_prices.pct_change().shift(-1)
        train_target = (train_returns > 0).astype(int)
        
        # Create feature DataFrame with all enhanced features
        train_features = pd.DataFrame({
            'RSI': train_rsi,
            'Momentum': train_mom,
            'MACD_Histogram': train_macd_hist,
            'RS_20D': train_rs_20d,
            'RS_60D': train_rs_60d,
            'RS_120D': train_rs_120d,
            'Volume_Ratio': train_vol_ratio,
            'OBV': train_obv,
            'Accum_Dist': train_ad,
            'ATR': train_atr,
            'Distance_MA50': train_dist_ma50,
            'Distance_MA200': train_dist_ma200,
            'BB_Position': train_bb_pos,
            'SPY_Return_20D': train_spy_return_20d,
            'SPY_Return_60D': train_spy_return_60d,
            'VIX': train_vix
        })
        
        # Remove NaN values
        valid_indices = ~(train_features.isna().any(axis=1) | train_target.isna())
        train_features = train_features[valid_indices]
        train_target = train_target[valid_indices]
        
        if len(train_features) < config.MIN_TRAINING_SAMPLES:
            print(f"  Insufficient valid training data for {symbol} (got {len(train_features)} samples)")
            return None, None, None
        
        # Train Random Forest model
        X_train, X_val, y_train, y_val = train_test_split(
            train_features, train_target, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Calculate out-of-sample accuracy
        val_predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, val_predictions)
        
        # Test on test data with enhanced features
        test_spy = spy_prices.iloc[split_point:]
        test_volume = volume.iloc[split_point:]
        test_high = high.iloc[split_point:]
        test_low = low.iloc[split_point:]
        
        test_rsi = calculate_rsi(test_prices, window=config.RSI_WINDOW)
        test_mom = calculate_momentum(test_prices, window=config.MOMENTUM_WINDOW)
        test_macd_hist = calculate_macd_histogram(test_prices)
        test_rs_20d = calculate_relative_strength(test_prices, test_spy, window=20)
        test_rs_60d = calculate_relative_strength(test_prices, test_spy, window=60)
        test_rs_120d = calculate_relative_strength(test_prices, test_spy, window=120)
        test_vol_ratio = calculate_volume_ratio(test_volume, window=config.VOLUME_RATIO_WINDOW)
        test_obv = calculate_obv(test_prices, test_volume)
        test_ad = calculate_accumulation_distribution(test_high, test_low, test_prices, test_volume)
        test_atr = calculate_atr(test_high, test_low, test_prices, window=config.ATR_WINDOW)
        test_dist_ma50 = calculate_distance_from_ma(test_prices, 50)
        test_dist_ma200 = calculate_distance_from_ma(test_prices, 200)
        test_bb_pos = calculate_bollinger_band_position(test_prices, window=config.BB_WINDOW)
        
        # Market regime features for test data
        test_spy_return_20d, test_spy_return_60d = calculate_market_regime(test_spy)
        
        # Use same VIX for test data
        test_vix = pd.Series([current_vix if current_vix else 20] * len(test_prices), index=test_prices.index)
        
        test_features = pd.DataFrame({
            'RSI': test_rsi,
            'Momentum': test_mom,
            'MACD_Histogram': test_macd_hist,
            'RS_20D': test_rs_20d,
            'RS_60D': test_rs_60d,
            'RS_120D': test_rs_120d,
            'Volume_Ratio': test_vol_ratio,
            'OBV': test_obv,
            'Accum_Dist': test_ad,
            'ATR': test_atr,
            'Distance_MA50': test_dist_ma50,
            'Distance_MA200': test_dist_ma200,
            'BB_Position': test_bb_pos,
            'SPY_Return_20D': test_spy_return_20d,
            'SPY_Return_60D': test_spy_return_60d,
            'VIX': test_vix
        })
        
        # Make predictions
        predictions = model.predict_proba(test_features)[:, 1]
        
        # Generate trades based on predictions
        trades = pd.DataFrame(index=test_prices.index, columns=["Trades"])
        trades["Trades"] = 0.0
        
        curr = 0
        for i in range(len(test_prices)):
            if pd.isna(predictions[i]):
                continue
            
            pred = predictions[i]
            
            # Trading strategy based on prediction confidence
            if pred > 0.6 and curr < 1000:  # Strong buy signal
                trades.iloc[i] = 1000 - curr
                curr = 1000
            elif pred < 0.4 and curr > -1000:  # Strong sell signal
                trades.iloc[i] = -1000 - curr
                curr = -1000
            elif 0.4 <= pred <= 0.6 and curr != 0:  # Neutral range
                trades.iloc[i] = -curr
                curr = 0
        
        return trades, accuracy, model
        
    except Exception as e:
        print(f"Error running Random Forest for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
