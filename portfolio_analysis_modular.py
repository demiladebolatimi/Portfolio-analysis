"""Main portfolio analysis script - modular version."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time

import config
from portfolio_manager import (
    load_portfolio, load_trading_restrictions, 
    check_trading_restrictions, update_trading_restriction,
    update_portfolio_after_trade, maintain_sold_stocks
)
from strategies import run_qlearner_strategy, run_randomforest_strategy
from email_service import send_email_report
from technical_indicators import (
    calculate_rsi, calculate_momentum, calculate_macd
)


def calculate_position_size(current_shares, current_value, portfolio_value, buy_score, sell_score, risk_level='moderate'):
    """Calculate recommended position size based on confidence and risk level."""
    # Get base position size from risk level
    base_position_size = config.RISK_LEVELS.get(risk_level, config.DEFAULT_POSITION_SIZE)
    
    # Adjust based on confidence levels
    if buy_score > config.STRONG_BUY_THRESHOLD:
        # Strong buy signal - increase position size
        position_multiplier = 1.5
    elif buy_score > config.BUY_SCORE_THRESHOLD:
        # Moderate buy signal - standard position size
        position_multiplier = 1.0
    elif sell_score > config.STRONG_SELL_THRESHOLD:
        # Strong sell signal - reduce position significantly
        position_multiplier = 0.25
    elif sell_score > config.SELL_SCORE_THRESHOLD:
        # Moderate sell signal - reduce position
        position_multiplier = 0.5
    else:
        # Neutral - maintain current position
        position_multiplier = 1.0
    
    # Calculate target position value
    target_position_value = portfolio_value * base_position_size * position_multiplier
    
    # Apply limits
    target_position_value = min(target_position_value, portfolio_value * config.MAX_POSITION_SIZE)
    target_position_value = max(target_position_value, portfolio_value * config.MIN_POSITION_SIZE)
    
    return target_position_value


def calculate_trade_size(current_shares, current_value, portfolio_value, buy_score, sell_score, recommendation, risk_level='moderate'):
    """Calculate recommended trade size (shares to buy/sell)."""
    target_position_value = calculate_position_size(current_shares, current_value, portfolio_value, buy_score, sell_score, risk_level)
    
    # Calculate current position value
    current_position_value = current_shares * (current_value / current_shares if current_shares > 0 else 0)
    
    # Calculate trade size
    if recommendation in ['BUY', 'STRONG BUY']:
        # Buy signal - calculate shares to add
        trade_value = target_position_value - current_position_value
        if trade_value > 0:
            shares_to_trade = int(trade_value / (current_value / current_shares if current_shares > 0 else 100))
            return max(0, shares_to_trade), 'BUY'
        else:
            return 0, 'HOLD'
    
    elif recommendation in ['SELL', 'STRONG SELL']:
        # Sell signal - calculate shares to sell
        trade_value = current_position_value - target_position_value
        if trade_value > 0:
            shares_to_trade = int(trade_value / (current_value / current_shares if current_shares > 0 else 100))
            return max(0, min(shares_to_trade, current_shares)), 'SELL'
        else:
            return 0, 'HOLD'
    
    else:
        return 0, 'HOLD'


def calculate_gradual_entry_exit(buy_score, sell_score, current_shares, recommendation):
    """Calculate gradual entry/exit based on confidence levels."""
    if not config.GRADUAL_ENTRY_ENABLED and not config.GRADUAL_EXIT_ENABLED:
        return 1.0  # Full position
    
    if recommendation in ['BUY', 'STRONG BUY'] and config.GRADUAL_ENTRY_ENABLED:
        # Gradual entry based on buy score
        for level, (min_score, max_score) in config.ENTRY_CONFIDENCE_LEVELS.items():
            if min_score <= buy_score < max_score:
                if level == 'low':
                    return 0.25  # Entry 25%
                elif level == 'medium':
                    return 0.50  # Entry 50%
                elif level == 'high':
                    return 0.75  # Entry 75%
                else:
                    return 1.0  # Entry 100%
    
    elif recommendation in ['SELL', 'STRONG SELL'] and config.GRADUAL_EXIT_ENABLED:
        # Gradual exit based on sell score
        for level, (min_score, max_score) in config.EXIT_CONFIDENCE_LEVELS.items():
            if min_score <= sell_score < max_score:
                if level == 'low':
                    return 0.25  # Exit 25%
                elif level == 'medium':
                    return 0.50  # Exit 50%
                elif level == 'high':
                    return 0.75  # Exit 75%
                else:
                    return 1.0  # Exit 100%
    
    return 1.0  # Default to full position


def calculate_daily_sell_scores(prices):
    """Calculate daily sell likelihood scores (0-100) based on indicators."""
    if prices is None or len(prices) < 30:
        return None
    
    # Calculate indicators
    rsi_values = calculate_rsi(prices, window=14)
    mom_values = calculate_momentum(prices, window=14)
    macd_values = calculate_macd(prices)
    
    daily_scores = pd.Series(index=prices.index, dtype=float)
    
    for i in range(len(prices)):
        try:
            if i < 14:  # Need enough data for indicators
                daily_scores.iloc[i] = 50  # Neutral
                continue
                
            rsi_val = rsi_values.iloc[i] if i < len(rsi_values) else None
            mom_val = mom_values.iloc[i] if i < len(mom_values) else None
            macd_val = macd_values.iloc[i] if i < len(macd_values) else None
            
            if pd.isna(rsi_val) or pd.isna(mom_val) or pd.isna(macd_val):
                daily_scores.iloc[i] = 50
                continue
            
            # Calculate individual sell signals (0-100 scale)
            # RSI: >70 = high sell score, <30 = low sell score
            rsi_score = min(100, max(0, (rsi_val - 30) / 40 * 100))  # 30->0%, 70->100%
            
            # Momentum: positive = lower sell score, negative = higher sell score
            mom_score = min(100, max(0, (0.05 - mom_val) / 0.10 * 100))  # +5% -> 0%, -5% -> 100%
            
            # MACD: positive = lower sell score, negative = higher sell score
            macd_score = min(100, max(0, (0 - macd_val) / 2 * 100))  # +2 -> 0%, -2 -> 100%
            
            # Combined score (weighted average)
            combined_score = (rsi_score * 0.4 + mom_score * 0.3 + macd_score * 0.3)
            daily_scores.iloc[i] = combined_score
            
        except Exception:
            daily_scores.iloc[i] = 50
    
    return daily_scores


def calculate_daily_buy_scores(prices):
    """Calculate daily buy likelihood scores (0-100) based on indicators."""
    if prices is None or len(prices) < 30:
        return None
    
    # Calculate indicators
    rsi_values = calculate_rsi(prices, window=14)
    mom_values = calculate_momentum(prices, window=14)
    macd_values = calculate_macd(prices)
    
    daily_scores = pd.Series(index=prices.index, dtype=float)
    
    for i in range(len(prices)):
        try:
            if i < 14:  # Need enough data for indicators
                daily_scores.iloc[i] = 50  # Neutral
                continue
                
            rsi_val = rsi_values.iloc[i] if i < len(rsi_values) else None
            mom_val = mom_values.iloc[i] if i < len(mom_values) else None
            macd_val = macd_values.iloc[i] if i < len(macd_values) else None
            
            if pd.isna(rsi_val) or pd.isna(mom_val) or pd.isna(macd_val):
                daily_scores.iloc[i] = 50
                continue
            
            # Calculate individual buy signals (0-100 scale)
            # RSI: <30 = high buy score, >70 = low buy score
            rsi_score = min(100, max(0, (70 - rsi_val) / 40 * 100))  # 70->0%, 30->100%
            
            # Momentum: positive = higher buy score, negative = lower buy score
            mom_score = min(100, max(0, (mom_val + 0.05) / 0.10 * 100))  # -5% -> 0%, +5% -> 100%
            
            # MACD: positive = higher buy score, negative = lower buy score
            macd_score = min(100, max(0, (macd_val + 2) / 4 * 100))  # -2 -> 0%, +2 -> 100%
            
            # Combined score (weighted average)
            combined_score = (rsi_score * 0.4 + mom_score * 0.3 + macd_score * 0.3)
            daily_scores.iloc[i] = combined_score
            
        except Exception:
            daily_scores.iloc[i] = 50
    
    return daily_scores


def calculate_actual_performance(portfolio, trades_dict):
    """Calculate actual portfolio performance."""
    total_cost = portfolio['CostBasis'].sum()
    total_value = portfolio['MarketValue'].sum()
    actual_return = (total_value - total_cost) / total_cost if total_cost > 0 else 0
    return total_cost, total_value, actual_return


def calculate_buy_hold_return(prices):
    """Calculate buy and hold return."""
    if len(prices) < 2:
        return None
    return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]


def calculate_benchmark_return(start_date, end_date):
    """Calculate benchmark (SPY) return."""
    try:
        import yfinance as yf
        spy_ticker = yf.Ticker("SPY")
        days = (end_date - start_date).days
        period = f"{days}d" if days > 0 else "2mo"
        time.sleep(0.5)  # Delay to avoid rate limiting
        spy_hist = spy_ticker.history(period=period)
        if not spy_hist.empty:
            spy_prices = spy_hist['Close']
            return (spy_prices.iloc[-1] - spy_prices.iloc[0]) / spy_prices.iloc[0]
        return None
    except Exception as e:
        print(f"Error calculating benchmark return: {e}")
        return None


def compute_portfolio_value_from_trades(trades, prices):
    """Compute portfolio value from trades."""
    if trades is None or prices is None:
        return None
    
    # Align trades with prices
    aligned_trades = trades.reindex(prices.index).fillna(0)
    
    # Calculate portfolio value
    shares = aligned_trades.cumsum()
    portfolio_value = shares * prices
    
    return portfolio_value


def calculate_daily_sell_scores(prices):
    """Calculate daily sell likelihood scores (0-100) based on indicators."""
    if prices is None or len(prices) < 30:
        return None
    
    # Calculate indicators
    rsi_values = calculate_rsi(prices, window=14)
    mom_values = calculate_momentum(prices, window=14)
    macd_values = calculate_macd(prices)
    
    daily_scores = pd.Series(index=prices.index, dtype=float)
    
    for i in range(len(prices)):
        try:
            if i < 14:  # Need enough data for indicators
                daily_scores.iloc[i] = 50  # Neutral
                continue
                
            rsi_val = rsi_values.iloc[i] if i < len(rsi_values) else None
            mom_val = mom_values.iloc[i] if i < len(mom_values) else None
            macd_val = macd_values.iloc[i] if i < len(macd_values) else None
            
            if pd.isna(rsi_val) or pd.isna(mom_val) or pd.isna(macd_val):
                daily_scores.iloc[i] = 50
                continue
            
            # Calculate individual sell signals (0-100 scale)
            # RSI: >70 = high sell score, <30 = low sell score
            rsi_score = min(100, max(0, (rsi_val - 30) / 40 * 100))  # 30->0%, 70->100%
            
            # Momentum: positive = lower sell score, negative = higher sell score
            mom_score = min(100, max(0, (0.05 - mom_val) / 0.10 * 100))  # +5% -> 0%, -5% -> 100%
            
            # MACD: positive = lower sell score, negative = higher sell score
            macd_score = min(100, max(0, (0 - macd_val) / 2 * 100))  # +2 -> 0%, -2 -> 100%
            
            # Combined score (weighted average)
            combined_score = (rsi_score * 0.4 + mom_score * 0.3 + macd_score * 0.3)
            daily_scores.iloc[i] = combined_score
            
        except Exception:
            daily_scores.iloc[i] = 50
    
    return daily_scores


def calculate_daily_buy_scores(prices):
    """Calculate daily buy likelihood scores (0-100) based on indicators."""
    if prices is None or len(prices) < 30:
        return None
    
    # Calculate indicators
    rsi_values = calculate_rsi(prices, window=14)
    mom_values = calculate_momentum(prices, window=14)
    macd_values = calculate_macd(prices)
    
    daily_scores = pd.Series(index=prices.index, dtype=float)
    
    for i in range(len(prices)):
        try:
            if i < 14:  # Need enough data for indicators
                daily_scores.iloc[i] = 50  # Neutral
                continue
                
            rsi_val = rsi_values.iloc[i] if i < len(rsi_values) else None
            mom_val = mom_values.iloc[i] if i < len(mom_values) else None
            macd_val = macd_values.iloc[i] if i < len(macd_values) else None
            
            if pd.isna(rsi_val) or pd.isna(mom_val) or pd.isna(macd_val):
                daily_scores.iloc[i] = 50
                continue
            
            # Calculate individual buy signals (0-100 scale)
            # RSI: <30 = high buy score, >70 = low buy score
            rsi_score = min(100, max(0, (70 - rsi_val) / 40 * 100))  # 70->0%, 30->100%
            
            # Momentum: positive = higher buy score, negative = lower buy score
            mom_score = min(100, max(0, (mom_val + 0.05) / 0.10 * 100))  # -5% -> 0%, +5% -> 100%
            
            # MACD: positive = higher buy score, negative = lower buy score
            macd_score = min(100, max(0, (macd_val + 2) / 4 * 100))  # -2 -> 0%, +2 -> 100%
            
            # Combined score (weighted average)
            combined_score = (rsi_score * 0.4 + mom_score * 0.3 + macd_score * 0.3)
            daily_scores.iloc[i] = combined_score
            
        except Exception:
            daily_scores.iloc[i] = 50
    
    return daily_scores


def calculate_signal_trend(scores, window=5):
    """Calculate signal trend over recent days."""
    if scores is None or len(scores) < window:
        return "Insufficient Data"
    
    recent_scores = scores.tail(window)
    
    if len(recent_scores) < 2:
        return "Insufficient Data"
    
    # Calculate trend
    first_score = recent_scores.iloc[0]
    last_score = recent_scores.iloc[-1]
    
    if last_score > first_score + 5:
        return "Strongly Increasing"
    elif last_score > first_score + 2:
        return "Increasing"
    elif last_score < first_score - 5:
        return "Strongly Decreasing"
    elif last_score < first_score - 2:
        return "Decreasing"
    else:
        return "Stable"


def plot_sell_likelihood_buckets(daily_scores):
    """Create simplified chart showing stocks in three buckets: High Sell, Hold, Buy."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Categorize each stock based on its latest sell score
    high_sell = []
    hold = []
    buy = []
    
    for symbol, scores in daily_scores.items():
        if scores is not None and len(scores) > 0:
            latest_score = scores.iloc[-1]
            if latest_score > 70:
                high_sell.append((symbol, latest_score))
            elif latest_score < 30:
                buy.append((symbol, latest_score))
            else:
                hold.append((symbol, latest_score))
    
    # Sort each bucket by score
    high_sell.sort(key=lambda x: x[1], reverse=True)
    hold.sort(key=lambda x: x[1], reverse=True)
    buy.sort(key=lambda x: x[1], reverse=True)
    
    # Create horizontal bar chart
    y_positions = []
    labels = []
    colors = []
    
    current_y = 0
    
    # High Sell bucket (red)
    for symbol, score in high_sell:
        y_positions.append(current_y)
        labels.append(f"{symbol} ({score:.1f})")
        colors.append('#ff4444')
        current_y += 1
    
    # Hold bucket (gray)
    for symbol, score in hold:
        y_positions.append(current_y)
        labels.append(f"{symbol} ({score:.1f})")
        colors.append('#888888')
        current_y += 1
    
    # Buy bucket (green)
    for symbol, score in buy:
        y_positions.append(current_y)
        labels.append(f"{symbol} ({score:.1f})")
        colors.append('#44cc44')
        current_y += 1
    
    # Create horizontal bar chart
    if y_positions:
        ax.barh(y_positions, [score for _, score in high_sell + hold + buy], color=colors, alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Sell Likelihood Score (0-100)')
        ax.set_title('Current Stock Status - Sell Likelihood Buckets')
        
        # Add vertical lines for thresholds
        ax.axvline(x=70, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=30, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=50, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff4444', label='High Sell (>70)'),
            Patch(facecolor='#888888', label='Hold (30-70)'),
            Patch(facecolor='#44cc44', label='Buy (<30)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([0, 100])
    
    plt.tight_layout()
    plt.savefig('sell_likelihood_buckets.png', dpi=150, bbox_inches='tight')
    print("Sell likelihood buckets chart saved as sell_likelihood_buckets.png")
    plt.close()


def print_predictions(results_df, restrictions_df=None):
    """Print hybrid trading recommendations with restriction checks."""
    print("\n" + "=" * 140)
    print("HYBRID TRADING RECOMMENDATIONS FOR NEXT MONTH")
    print("=" * 140)
    
    # Format the recommendations table
    print(f"{'Symbol':<10} {'Buy Score':<12} {'Sell Score':<12} {'Manual':<12} {'Q-Learner':<12} {'RandomForest':<12} {'Final Rec':<15} {'Trade Size':<12} {'Restriction':<20}")
    print("-" * 150)
    
    for idx, row in results_df.iterrows():
        symbol = row['Symbol']
        buy_score = row['CurrentBuyScore']
        sell_score = row['CurrentSellScore']
        
        # Get manual strategy recommendation
        manual_rec = "HOLD"
        if row['Manual'] > 0.05:
            manual_rec = "BUY"
        elif row['Manual'] < -0.05:
            manual_rec = "SELL"
        
        # Get Q-Learner recommendation
        ql_rec = "HOLD"
        if pd.notna(row['QLearner']):
            if row['QLearner'] > 0.05:
                ql_rec = "BUY"
            elif row['QLearner'] < -0.05:
                ql_rec = "SELL"
        else:
            ql_rec = "N/A"
        
        # Get Random Forest recommendation
        rf_rec = "N/A"
        if pd.notna(row['RandomForest']):
            if row['RandomForest'] > 0.05:
                rf_rec = "BUY"
            elif row['RandomForest'] < -0.05:
                rf_rec = "SELL"
        
        # Use the pre-calculated recommendation
        final_rec = row.get('Recommendation', 'HOLD')
        
        # Get position sizing info
        shares_to_trade = row.get('SharesToTrade', 0)
        trade_action = row.get('TradeAction', 'HOLD')
        gradual_pct = row.get('GradualPercentage', 1.0)
        
        if shares_to_trade > 0:
            trade_size_str = f"{trade_action} {shares_to_trade} ({gradual_pct:.0%})"
        else:
            trade_size_str = "HOLD"
        
        # Check trading restrictions
        restriction_status = "OK"
        if restrictions_df is not None and final_rec in ["BUY", "SELL"]:
            # Get current price from market value and shares
            current_price = row.get('CurrentPrice', row['MarketValue'] / row['Shares'] if row['Shares'] > 0 else 0)
            allowed, message = check_trading_restrictions(symbol, final_rec, current_price, restrictions_df)
            if not allowed:
                restriction_status = "⚠️ VIOLATION"
                print(f"⚠️ RESTRICTION WARNING for {symbol}: {message}")
        
        print(f"{symbol:<10} {buy_score:>6.1f}/100   {sell_score:>6.1f}/100   {manual_rec:<12} {ql_rec:<12} {rf_rec:<12} {final_rec:<15} {trade_size_str:<12} {restriction_status:<20}")
    
    print("=" * 140)
    print("📊 HYBRID RECOMMENDATION LOGIC:")
    print("   - Buy Score: 0-100 based on RSI, Momentum, MACD indicators (high = buy signal)")
    print("   - Sell Score: 0-100 based on RSI, Momentum, MACD indicators (high = sell signal)")
    print("   - Manual Strategy: Based on recent 6-month performance")
    print("   - Q-Learner: ML-based strategy (Score >65 = SELL, Score <35 = BUY)")
    print("   - Random Forest: ML model trained on historical patterns (only if accuracy >55%)")
    print("   - Final Rec: Weighted combination considering all factors")
    print("⚠️ IMPORTANT: 60-day trading restrictions apply to all recommendations")
    print("   - Cannot buy and sell at higher price within 60 days")
    print("   - Cannot sell and buy at lower price within 60 days")
    print("   - Check trading_restrictions.csv for detailed restriction status")
    print("=" * 140)


def analyze_portfolio(csv_path, analysis_months=None):
    """Main analysis function for recent performance and predictions."""
    if analysis_months is None:
        analysis_months = config.ANALYSIS_MONTHS
        
    print("=" * 60)
    print("PORTFOLIO ANALYSIS - DAILY SELL LIKELIHOOD TRACKING")
    print("=" * 60)
    
    # Load portfolio
    portfolio = load_portfolio(csv_path)
    print(f"\nLoaded {len(portfolio)} positions from portfolio")
    print(f"Total Market Value: ${portfolio['MarketValue'].sum():,.2f}")
    print(f"Total Cost Basis: ${portfolio['CostBasis'].sum():,.2f}")
    
    # Load trading restrictions
    restrictions_path = csv_path.replace('portfolio.csv', 'trading_restrictions.csv')
    restrictions_df = load_trading_restrictions(restrictions_path)
    if restrictions_df is not None:
        print(f"Loaded trading restrictions for {len(restrictions_df)} symbols")
    else:
        print("No trading restrictions file found - all actions will be allowed")
    
    # Calculate actual performance
    total_cost, total_value, actual_return = calculate_actual_performance(portfolio, {})
    print(f"\nActual Portfolio Return: {actual_return:.2%}")
    
    # Set analysis period - last X months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=analysis_months * 30)
    
    print(f"\nAnalysis Period: {start_date.date()} to {end_date.date()} ({analysis_months} months)")
    print("=" * 60)
    
    # Calculate benchmark return
    benchmark_return = calculate_benchmark_return(start_date, end_date)
    if benchmark_return:
        print(f"Benchmark (SPY) Return: {benchmark_return:.2%}")
    
    # Store results
    results = []
    daily_scores = {}
    
    # Analyze each symbol
    for idx, row in portfolio.iterrows():
        symbol = row['Symbol']
        shares = row['Shares']
        
        print(f"\nAnalyzing {symbol}...")
        
        # Get historical data
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            days = (end_date - start_date).days
            period = f"{days}d" if days > 0 else "2mo"
            time.sleep(0.3)  # Delay to avoid rate limiting
            hist = ticker.history(period=period)
            
            if hist.empty:
                print(f"  No data found for {symbol}")
                continue
            
            prices = hist['Close']
            
            # Calculate buy & hold return
            buy_hold_return = calculate_buy_hold_return(prices)
            if buy_hold_return:
                print(f"  Buy & Hold Return: {buy_hold_return:.2%}")
            
            # Run Manual Strategy
            manual_return = None
            try:
                # Simple manual strategy: buy when RSI < 30, sell when RSI > 70
                rsi = calculate_rsi(prices, window=14)
                trades = pd.Series(index=prices.index, dtype=float)
                trades[:] = 0
                
                for i in range(len(prices)):
                    if i < 14:
                        continue
                    if rsi.iloc[i] < 30 and trades.iloc[i-1] == 0:
                        trades.iloc[i] = 1000
                    elif rsi.iloc[i] > 70 and trades.iloc[i-1] != 0:
                        trades.iloc[i] = -1000
                
                portfolio_value = compute_portfolio_value_from_trades(trades, prices)
                if portfolio_value is not None and len(portfolio_value) > 0:
                    # Get first and last values as scalars
                    first_value = portfolio_value.iloc[0].values[0] if isinstance(portfolio_value.iloc[0], pd.Series) else portfolio_value.iloc[0]
                    last_value = portfolio_value.iloc[-1].values[0] if isinstance(portfolio_value.iloc[-1], pd.Series) else portfolio_value.iloc[-1]
                    # Convert to float if needed
                    first_value = float(first_value) if not isinstance(first_value, (int, float)) else first_value
                    last_value = float(last_value) if not isinstance(last_value, (int, float)) else last_value
                    if first_value != 0:
                        manual_return = (last_value - first_value) / first_value
                    else:
                        manual_return = 0
                    print(f"  Manual Strategy Return: {manual_return:.2%}")
            except Exception as e:
                print(f"  Error in manual strategy: {e}")
            
            # Run Q-Learner Strategy
            ql_trades = run_qlearner_strategy(symbol, start_date, end_date)
            ql_return = None
            if ql_trades is not None:
                ql_portfolio = compute_portfolio_value_from_trades(ql_trades, prices)
                if ql_portfolio is not None and len(ql_portfolio) > 0:
                    # Get first and last values as scalars
                    first_value = ql_portfolio.iloc[0].values[0] if isinstance(ql_portfolio.iloc[0], pd.Series) else ql_portfolio.iloc[0]
                    last_value = ql_portfolio.iloc[-1].values[0] if isinstance(ql_portfolio.iloc[-1], pd.Series) else ql_portfolio.iloc[-1]
                    # Convert to float if needed
                    first_value = float(first_value) if not isinstance(first_value, (int, float)) else first_value
                    last_value = float(last_value) if not isinstance(last_value, (int, float)) else last_value
                    if first_value != 0:
                        ql_return = (last_value - first_value) / first_value
                    else:
                        ql_return = 0
                    print(f"  Q-Learner Strategy Return: {ql_return:.2%}")
            else:
                ql_return = None
            
            # Run Random Forest Strategy (only if accuracy > 55%)
            rf_trades, rf_accuracy, rf_model = run_randomforest_strategy(symbol, start_date, end_date)
            rf_model_for_enhancement = None
            
            if rf_trades is not None and rf_accuracy is not None:
                if rf_accuracy > config.ACCURACY_THRESHOLD:
                    rf_portfolio = compute_portfolio_value_from_trades(rf_trades, prices)
                    if rf_portfolio is not None and len(rf_portfolio) > 0:
                        # Get first and last values as scalars
                        first_value = rf_portfolio.iloc[0].values[0] if isinstance(rf_portfolio.iloc[0], pd.Series) else rf_portfolio.iloc[0]
                        last_value = rf_portfolio.iloc[-1].values[0] if isinstance(rf_portfolio.iloc[-1], pd.Series) else rf_portfolio.iloc[-1]
                        # Convert to float if needed
                        first_value = float(first_value) if not isinstance(first_value, (int, float)) else first_value
                        last_value = float(last_value) if not isinstance(last_value, (int, float)) else last_value
                        if first_value != 0:
                            rf_return = (last_value - first_value) / first_value
                        else:
                            rf_return = 0
                        print(f"  Random Forest Strategy Return: {rf_return:.2%} (Accuracy: {rf_accuracy:.2%})")
                        rf_model_for_enhancement = rf_model
                    else:
                        rf_return = None
                else:
                    print(f"  Random Forest skipped (Accuracy: {rf_accuracy:.2%} < {config.ACCURACY_THRESHOLD:.0%})")
                    rf_return = None
            else:
                rf_return = None
            
            # Calculate daily scores
            sell_scores = calculate_daily_sell_scores(prices)
            buy_scores = calculate_daily_buy_scores(prices)
            
            if sell_scores is not None:
                daily_scores[symbol] = sell_scores
                avg_sell_score = sell_scores.mean()
                current_sell_score = sell_scores.iloc[-1]
                print(f"  Average Sell Likelihood ({analysis_months}mo): {avg_sell_score:.1f}/100")
                print(f"  Current Sell Likelihood: {current_sell_score:.1f}/100")
            
            if buy_scores is not None:
                avg_buy_score = buy_scores.mean()
                current_buy_score = buy_scores.iloc[-1]
                print(f"  Average Buy Likelihood ({analysis_months}mo): {avg_buy_score:.1f}/100")
                print(f"  Current Buy Likelihood: {current_buy_score:.1f}/100")
            
            # Calculate position sizing recommendations
            portfolio_value = portfolio['MarketValue'].sum()
            current_price = row['MarketValue'] / row['Shares'] if row['Shares'] > 0 else 0
            
            # Determine recommendation based on scores
            if current_buy_score > config.STRONG_BUY_THRESHOLD:
                recommendation = 'STRONG BUY'
            elif current_buy_score > config.BUY_SCORE_THRESHOLD:
                recommendation = 'BUY'
            elif current_sell_score > config.STRONG_SELL_THRESHOLD:
                recommendation = 'STRONG SELL'
            elif current_sell_score > config.SELL_SCORE_THRESHOLD:
                recommendation = 'SELL'
            else:
                recommendation = 'HOLD'
            
            # Calculate trade size
            shares_to_trade, trade_action = calculate_trade_size(
                shares, row['MarketValue'], portfolio_value, 
                current_buy_score, current_sell_score, recommendation
            )
            
            # Calculate gradual entry/exit percentage
            gradual_percentage = calculate_gradual_entry_exit(
                current_buy_score, current_sell_score, shares, recommendation
            )
            
            # Adjust trade size based on gradual entry/exit
            shares_to_trade = int(shares_to_trade * gradual_percentage)
            
            # Store results
            results.append({
                'Symbol': symbol,
                'Shares': shares,
                'BuyHold': buy_hold_return,
                'Benchmark': benchmark_return,
                'Manual': manual_return,
                'QLearner': ql_return,
                'RandomForest': rf_return,
                'CurrentSellScore': current_sell_score if sell_scores is not None else None,
                'CurrentBuyScore': current_buy_score if buy_scores is not None else None,
                'MarketValue': row['MarketValue'],
                'Recommendation': recommendation,
                'SharesToTrade': shares_to_trade,
                'TradeAction': trade_action,
                'GradualPercentage': gradual_percentage,
                'CurrentPrice': current_price
            })
            
        except Exception as e:
            print(f"  Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"SUMMARY RESULTS - LAST {analysis_months} MONTHS PERFORMANCE")
    print("=" * 60)
    # Print summary without shares column for privacy
    summary_cols = ['Symbol', 'BuyHold', 'Benchmark', 'Manual', 'QLearner', 'RandomForest', 'CurrentSellScore', 'CurrentBuyScore']
    print(results_df[summary_cols].to_string(index=False))
    
    # Print predictions
    print_predictions(results_df, restrictions_df)
    
    # Plot sell likelihood buckets
    if daily_scores:
        plot_sell_likelihood_buckets(daily_scores)
    
    # Send email report
    print("\n" + "=" * 60)
    print("SENDING EMAIL REPORT")
    print("=" * 60)
    email_sent = send_email_report(results_df, subject=f"Portfolio Analysis Report - {datetime.now().strftime('%Y-%m-%d')}")
    
    if email_sent:
        print("✅ Portfolio analysis complete with email report sent!")
    else:
        print("❌ Portfolio analysis complete but email failed to send")
    
    # Maintain sold stocks in portfolio for future analysis
    if config.KEEP_SOLD_STOCKS:
        print("\n" + "=" * 60)
        print("MAINTAINING SOLD STOCKS FOR FUTURE ANALYSIS")
        print("=" * 60)
        maintain_sold_stocks(csv_path)
        print("✅ Sold stocks maintained in portfolio.csv")
    
    return results_df


if __name__ == "__main__":
    csv_path = "portfolio.csv"
    results = analyze_portfolio(csv_path, analysis_months=config.ANALYSIS_MONTHS)
