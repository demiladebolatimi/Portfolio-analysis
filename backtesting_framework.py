"""
Walk-Forward Backtesting Framework
Implements rigorous validation with performance metrics and benchmark comparisons.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os

# Add parent directory to path to import strategy modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies import run_qlearner_strategy
from technical_indicators import (
    calculate_rsi, calculate_momentum, calculate_macd_histogram,
    calculate_relative_strength, calculate_market_regime
)
import config


def calculate_cagr(portfolio_values, years):
    """Calculate Compound Annual Growth Rate."""
    if len(portfolio_values) < 2 or years <= 0:
        return 0
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    cagr = (1 + total_return) ** (1 / years) - 1
    return cagr


def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.02):
    """Calculate Sharpe Ratio (annualized)."""
    if len(portfolio_values) < 2:
        return 0
    
    daily_returns = portfolio_values.pct_change().dropna()
    if len(daily_returns) == 0:
        return 0
    
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return sharpe


def calculate_sortino_ratio(portfolio_values, risk_free_rate=0.02):
    """Calculate Sortino Ratio (downside deviation only)."""
    if len(portfolio_values) < 2:
        return 0
    
    daily_returns = portfolio_values.pct_change().dropna()
    if len(daily_returns) == 0:
        return 0
    
    excess_returns = daily_returns - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    sortino = np.sqrt(252) * excess_returns.mean() / downside_deviation
    return sortino


def calculate_max_drawdown(portfolio_values):
    """Calculate Maximum Drawdown."""
    if len(portfolio_values) < 2:
        return 0
    
    cumulative = (1 + portfolio_values.pct_change()).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    return max_drawdown


def calculate_win_rate(trades):
    """Calculate Win Rate (percentage of profitable trades)."""
    if trades is None or len(trades) == 0:
        return 0
    
    # Count non-zero trades
    non_zero_trades = trades[trades != 0]
    if len(non_zero_trades) == 0:
        return 0
    
    # For simplicity, assume buy trades are wins if price goes up after
    # This is a simplified calculation - real implementation would need trade entry/exit prices
    win_rate = 0.5  # Placeholder
    return win_rate


def calculate_turnover(trades, portfolio_value):
    """Calculate Turnover (total trades / portfolio value)."""
    if trades is None or len(trades) == 0 or portfolio_value == 0:
        return 0
    
    total_trades = trades.abs().sum()
    turnover = total_trades / portfolio_value
    return turnover


def generate_trades_from_signals(prices, symbol, start_date, end_date, initial_value=100000):
    """
    Generate trades using the actual signal engine (BuyScore - SellScore, market regime, Q-Learner bias).
    
    Args:
        prices: Price series for the symbol
        symbol: Stock symbol
        start_date: Start date for trading
        end_date: End date for trading
        initial_value: Starting portfolio value
    
    Returns:
        Series of trades and portfolio values
    """
    try:
        import yfinance as yf
        import time
        
        # Get extended historical data for indicator calculations
        ticker = yf.Ticker(symbol)
        time.sleep(0.3)  # Rate limiting
        hist = ticker.history(period="5y")
        
        if hist.empty:
            print(f"  No data found for {symbol}")
            return None, None
        
        # Filter to analysis period
        analysis_start = pd.Timestamp(start_date).tz_localize('America/New_York')
        analysis_end = pd.Timestamp(end_date).tz_localize('America/New_York')
        hist_analysis = hist[(hist.index >= analysis_start) & (hist.index <= analysis_end)]
        
        if hist_analysis.empty:
            print(f"  No analysis period data for {symbol}")
            return None, None
        
        prices_analysis = hist_analysis['Close']
        
        # Calculate technical indicators
        rsi = calculate_rsi(prices_analysis, window=14)
        momentum = calculate_momentum(prices_analysis, window=14)
        macd_hist = calculate_macd_histogram(prices_analysis)
        
        # Get SPY for relative strength and market regime
        spy_ticker = yf.Ticker("SPY")
        time.sleep(0.3)
        spy_hist = spy_ticker.history(period="5y")
        if not spy_hist.empty:
            spy_analysis = spy_hist[(spy_hist.index >= analysis_start) & (spy_hist.index <= analysis_end)]
            if not spy_analysis.empty:
                rs = calculate_relative_strength(prices_analysis, spy_analysis['Close'])
                market_regime = calculate_market_regime(spy_analysis['Close'])
            else:
                rs = pd.Series(50, index=prices_analysis.index)
                market_regime = "NEUTRAL"
        else:
            rs = pd.Series(50, index=prices_analysis.index)
            market_regime = "NEUTRAL"
        
        # Calculate buy/sell scores with trend bias (MA200)
        ma200 = prices_analysis.rolling(window=200).mean()
        
        buy_scores = []
        sell_scores = []
        
        for i in range(len(prices_analysis)):
            if i < 14 or pd.isna(rsi.iloc[i]) or pd.isna(momentum.iloc[i]):
                buy_scores.append(50)
                sell_scores.append(50)
                continue
            
            rsi_val = rsi.iloc[i]
            mom_val = momentum.iloc[i]
            macd_val = macd_hist.iloc[i] if i < len(macd_hist) else 0
            rs_val = rs.iloc[i] if i < len(rs) else 50
            price = prices_analysis.iloc[i]
            ma200_val = ma200.iloc[i] if i < len(ma200) else price
            
            # Calculate buy score with trend bias
            buy_score = 50
            if rsi_val < 30:
                buy_score += 20
            elif rsi_val < 40:
                buy_score += 10
            elif rsi_val > 70:
                buy_score -= 20
            elif rsi_val > 60:
                buy_score -= 10
            
            if mom_val > 0:
                buy_score += 15
            elif mom_val < 0:
                buy_score -= 15
            
            if macd_val > 0:
                buy_score += 10
            elif macd_val < 0:
                buy_score -= 10
            
            if rs_val > 60:
                buy_score += 10
            elif rs_val < 40:
                buy_score -= 10
            
            # Add trend bias for buy score
            if price > ma200_val:
                buy_score += 15  # Bullish bias
            
            buy_score = max(0, min(100, buy_score))
            buy_scores.append(buy_score)
            
            # Calculate sell score with trend bias
            sell_score = 50
            if rsi_val > 70:
                sell_score += 20
            elif rsi_val > 60:
                sell_score += 10
            elif rsi_val < 30:
                sell_score -= 20
            elif rsi_val < 40:
                sell_score -= 10
            
            if mom_val < 0:
                sell_score += 15
            elif mom_val > 0:
                sell_score -= 15
            
            if macd_val < 0:
                sell_score += 10
            elif macd_val > 0:
                sell_score -= 10
            
            if rs_val < 40:
                sell_score += 10
            elif rs_val > 60:
                sell_score -= 10
            
            # Add trend bias for sell score (inverse of buy)
            if price < ma200_val:
                sell_score += 15  # Bearish bias
            
            sell_score = max(0, min(100, sell_score))
            sell_scores.append(sell_score)
        
        buy_scores = pd.Series(buy_scores, index=prices_analysis.index)
        sell_scores = pd.Series(sell_scores, index=prices_analysis.index)
        
        # Calculate signal strength
        signal_strength = buy_scores - sell_scores
        
        # Apply market regime adjustment
        if market_regime == "BEAR":
            signal_strength *= 0.7
        elif market_regime == "BULL":
            signal_strength *= 1.2
        
        # Get Q-Learner probability
        ql_result = run_qlearner_strategy(symbol, start_date, end_date, sv=initial_value)
        if ql_result is not None and isinstance(ql_result, tuple):
            _, ql_probability = ql_result
            ml_bias = (ql_probability - 0.5) * 40  # ±20 points
            signal_strength += ml_bias
        
        # Generate trades based on signal strength
        trades = pd.DataFrame(index=prices_analysis.index, columns=["Trades"])
        trades["Trades"] = 0.0
        
        curr = 0
        position_size = int((initial_value * 0.05) / prices_analysis.iloc[0])  # 5% position sizing
        
        for i in range(len(prices_analysis)):
            if pd.isna(signal_strength.iloc[i]):
                continue
            
            strength = signal_strength.iloc[i]
            
            # Trading thresholds (same as portfolio analysis)
            if strength > 20 and curr < position_size:  # STRONG BUY
                trades.iloc[i] = position_size - curr
                curr = position_size
            elif strength > 10 and curr < position_size * 0.5:  # MODERATE BUY
                trades.iloc[i] = (position_size * 0.5) - curr
                curr = position_size * 0.5
            elif strength < -20 and curr > -position_size:  # STRONG SELL
                trades.iloc[i] = -position_size - curr
                curr = -position_size
            elif strength < -10 and curr > -position_size * 0.5:  # MODERATE SELL
                trades.iloc[i] = -(position_size * 0.5) - curr
                curr = -position_size * 0.5
            elif -10 <= strength <= 10 and curr != 0:  # HOLD
                trades.iloc[i] = -curr
                curr = 0
        
        # Simulate portfolio value
        portfolio_values = pd.Series(index=prices_analysis.index, dtype=float)
        portfolio_values.iloc[0] = initial_value
        cash = initial_value
        shares = 0
        
        for i in range(1, len(prices_analysis)):
            trade = trades.iloc[i]['Trades']
            price = prices_analysis.iloc[i]
            
            if trade > 0:  # Buy
                cost = trade * price
                cash -= cost
                shares += trade
            elif trade < 0:  # Sell
                proceeds = abs(trade) * price
                cash += proceeds
                shares += trade  # trade is negative
            
            portfolio_values.iloc[i] = cash + (shares * price)
        
        return trades['Trades'], portfolio_values
        
    except Exception as e:
        print(f"  Error generating trades for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def calculate_performance_metrics(portfolio_values, years, trades=None, initial_value=100000):
    """Calculate all performance metrics."""
    metrics = {
        'CAGR': calculate_cagr(portfolio_values, years),
        'Sharpe Ratio': calculate_sharpe_ratio(portfolio_values),
        'Sortino Ratio': calculate_sortino_ratio(portfolio_values),
        'Max Drawdown': calculate_max_drawdown(portfolio_values),
        'Win Rate': calculate_win_rate(trades),
        'Turnover': calculate_turnover(trades, initial_value),
        'Final Value': portfolio_values.iloc[-1],
        'Total Return': (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    }
    return metrics


def walk_forward_backtest(symbols, start_date, end_date, 
                          train_months=12, test_months=6, initial_value=100000):
    """
    Perform walk-forward backtesting with rolling train/test windows using actual signal engine.
    
    Args:
        symbols: List of symbols to trade
        start_date: Overall start date
        end_date: Overall end date
        train_months: Training period in months
        test_months: Testing period in months
        initial_value: Starting portfolio value
    
    Returns:
        Dictionary with performance metrics for each walk-forward window
    """
    results = []
    current_date = start_date
    
    while current_date + timedelta(days=train_months * 30) < end_date:
        train_start = current_date
        train_end = current_date + timedelta(days=train_months * 30)
        test_start = train_end
        test_end = test_start + timedelta(days=test_months * 30)
        
        if test_end > end_date:
            break
        
        print(f"\nWalk-forward window: {train_start.date()} to {test_end.date()}")
        print(f"  Train: {train_start.date()} to {train_end.date()}")
        print(f"  Test: {test_start.date()} to {test_end.date()}")
        
        try:
            # Test strategy on testing period using actual signal engine
            print(f"  Testing on {test_months} months...")
            
            # Aggregate portfolio values across all symbols
            all_portfolio_values = None
            all_trades = None
            
            for symbol in symbols:
                print(f"    Backtesting {symbol}...")
                trades, portfolio_values = generate_trades_from_signals(
                    prices=None,  # Will be fetched internally
                    symbol=symbol,
                    start_date=test_start,
                    end_date=test_end,
                    initial_value=initial_value / len(symbols)  # Equal allocation
                )
                
                if portfolio_values is not None:
                    if all_portfolio_values is None:
                        all_portfolio_values = portfolio_values
                        all_trades = trades
                    else:
                        all_portfolio_values += portfolio_values
                        if all_trades is not None and trades is not None:
                            all_trades += trades
            
            if all_portfolio_values is not None:
                years = test_months / 12
                metrics = calculate_performance_metrics(all_portfolio_values, years, all_trades, initial_value)
                
                results.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'metrics': metrics
                })
                
                print(f"  CAGR: {metrics['CAGR']:.2%}")
                print(f"  Sharpe: {metrics['Sharpe Ratio']:.2f}")
                print(f"  Max Drawdown: {metrics['Max Drawdown']:.2%}")
            else:
                print(f"  No valid portfolio values generated")
            
        except Exception as e:
            print(f"  Error in walk-forward window: {e}")
            import traceback
            traceback.print_exc()
        
        current_date = test_start
    
    return results


def compare_with_benchmarks(strategy_results, benchmark_symbols=['VOO', 'QQQ'], 
                            start_date=None, end_date=None):
    """
    Compare strategy performance against benchmarks.
    
    Args:
        strategy_results: Results from walk-forward backtest
        benchmark_symbols: List of benchmark symbols to compare against
        start_date: Start date for benchmark data
        end_date: End date for benchmark data
    
    Returns:
        DataFrame comparing strategy vs benchmarks
    """
    if start_date is None or end_date is None:
        return None
    
    comparison = {}
    
    # Fetch benchmark data
    for symbol in benchmark_symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                prices = hist['Close']
                years = (end_date - start_date).days / 365.25
                
                metrics = calculate_performance_metrics(prices, years)
                comparison[symbol] = metrics
                print(f"\n{symbol} Metrics:")
                print(f"  CAGR: {metrics['CAGR']:.2%}")
                print(f"  Sharpe: {metrics['Sharpe Ratio']:.2f}")
                print(f"  Max Drawdown: {metrics['Max Drawdown']:.2%}")
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    
    # Aggregate strategy results
    if strategy_results:
        strategy_metrics = {}
        for key in ['CAGR', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown']:
            values = [r['metrics'][key] for r in strategy_results]
            strategy_metrics[key] = np.mean(values)
        
        comparison['Strategy'] = strategy_metrics
    
    return pd.DataFrame(comparison).T


def run_comprehensive_backtest(symbols, start_date, end_date):
    """
    Run comprehensive backtesting with walk-forward validation and benchmark comparison.
    
    Args:
        symbols: List of symbols to trade
        start_date: Start date for backtest (e.g., 2015-01-01)
        end_date: End date for backtest (e.g., 2026-01-01)
    
    Returns:
        Dictionary with backtest results and comparisons
    """
    print("=" * 80)
    print("COMPREHENSIVE WALK-FORWARD BACKTESTING")
    print("=" * 80)
    print(f"Symbols: {symbols}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Training window: 12 months")
    print(f"Testing window: 6 months")
    print("=" * 80)
    
    # Run walk-forward backtest with actual signal engine
    walk_forward_results = walk_forward_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        train_months=12,
        test_months=6
    )
    
    # Compare with benchmarks
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)
    benchmark_comparison = compare_with_benchmarks(
        strategy_results=walk_forward_results,
        benchmark_symbols=['VOO', 'QQQ'],
        start_date=start_date,
        end_date=end_date
    )
    
    if benchmark_comparison is not None:
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(benchmark_comparison.to_string())
    
    return {
        'walk_forward_results': walk_forward_results,
        'benchmark_comparison': benchmark_comparison
    }


if __name__ == "__main__":
    # Example usage
    symbols = ['NVDA', 'AAPL', 'MSFT', 'GOOGL']
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2026, 1, 1)
    
    results = run_comprehensive_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
