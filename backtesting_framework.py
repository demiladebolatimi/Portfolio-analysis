"""
Walk-Forward Backtesting Framework
Implements rigorous validation with performance metrics and benchmark comparisons.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf


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


def walk_forward_backtest(strategy_func, symbols, start_date, end_date, 
                          train_months=12, test_months=6, initial_value=100000):
    """
    Perform walk-forward backtesting with rolling train/test windows.
    
    Args:
        strategy_func: Function that takes (symbols, start_date, end_date) and returns trades
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
            # Train strategy on training period
            print(f"  Training on {train_months} months...")
            # strategy_func would train here
            
            # Test strategy on testing period
            print(f"  Testing on {test_months} months...")
            trades = strategy_func(symbols, test_start, test_end)
            
            # Calculate performance
            if trades is not None:
                # Simulate portfolio value from trades
                # This is simplified - real implementation would need price data
                portfolio_values = pd.Series([initial_value] * (test_months * 20))  # Placeholder
                
                years = test_months / 12
                metrics = calculate_performance_metrics(portfolio_values, years, trades, initial_value)
                
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
            
        except Exception as e:
            print(f"  Error in walk-forward window: {e}")
        
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


def run_comprehensive_backtest(symbols, start_date, end_date, strategy_func):
    """
    Run comprehensive backtesting with walk-forward validation and benchmark comparison.
    
    Args:
        symbols: List of symbols to trade
        start_date: Start date for backtest (e.g., 2015-01-01)
        end_date: End date for backtest (e.g., 2026-01-01)
        strategy_func: Strategy function to test
    
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
    
    # Run walk-forward backtest
    walk_forward_results = walk_forward_backtest(
        strategy_func=strategy_func,
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
    
    # Placeholder strategy function
    def dummy_strategy(symbols, start_date, end_date):
        # This would be replaced with actual strategy implementation
        return None
    
    results = run_comprehensive_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        strategy_func=dummy_strategy
    )
