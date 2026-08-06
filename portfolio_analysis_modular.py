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
    check_trading_restrictions, update_trading_restriction
)
from strategies import run_qlearner_strategy, run_randomforest_strategy
from email_service import send_email_report
from technical_indicators import (
    calculate_rsi, calculate_momentum, calculate_macd
)


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


def compute_portfolio_value_from_trades(trades, prices, sv=100000):
    """Compute portfolio value from trades DataFrame."""
    if trades is None or prices is None:
        return None
    
    portfolio_values = pd.Series(index=prices.index, dtype=float)
    portfolio_values.iloc[0] = sv
    
    cash = sv
    shares = 0
    
    for date in prices.index:
        if date in trades.index:
            trade = trades.loc[date, 'Trades']
            if trade > 0:  # Buy
                cost = trade * prices.loc[date] * 1.005 + 9.95  # Include impact and commission
                cash -= cost
                shares += trade
            elif trade < 0:  # Sell
                proceeds = abs(trade) * prices.loc[date] * 0.995 - 9.95
                cash += proceeds
                shares += trade  # trade is negative
        
        portfolio_values.loc[date] = cash + shares * prices.loc[date]
    
    return portfolio_values


def calculate_position_size(portfolio_value, current_price, confidence, max_weight=0.10):
    """Calculate position size based on portfolio value, confidence, and weight limits."""
    if current_price <= 0 or portfolio_value <= 0:
        return 0
    
    # Calculate target position value based on confidence and max weight
    target_weight = confidence * max_weight
    target_value = portfolio_value * target_weight
    
    # Calculate number of shares
    shares = int(target_value / current_price)
    
    return shares


def check_trend_filter(prices, ma_period=200):
    """Check if price is above moving average (trend filter)."""
    if len(prices) < ma_period:
        return True  # Allow trades if not enough data
    
    ma = prices.rolling(window=ma_period).mean().iloc[-1]
    current_price = prices.iloc[-1]
    
    return current_price > ma


def detect_market_regime(spy_prices, vix_value=None):
    """Detect market regime using SPY and VIX."""
    if len(spy_prices) < 200:
        return "NEUTRAL"
    
    spy_ma200 = spy_prices.rolling(window=200).mean().iloc[-1]
    current_spy = spy_prices.iloc[-1]
    
    # Default VIX if not provided
    if vix_value is None:
        vix_value = 20
    
    # Determine regime
    if current_spy > spy_ma200 and vix_value < 25:
        return "BULL"
    elif current_spy < spy_ma200 and vix_value > 30:
        return "BEAR"
    else:
        return "NEUTRAL"


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
    slope = (recent_scores.iloc[-1] - recent_scores.iloc[0]) / len(recent_scores)
    
    if slope > 2:
        return "Strongly Increasing"
    elif slope > 0.5:
        return "Increasing"
    elif slope > -0.5:
        return "Stable"
    elif slope > -2:
        return "Decreasing"
    else:
        return "Strongly Decreasing"


def print_predictions(results_df, restrictions_df=None):
    """Print hybrid trading recommendations with restriction checks."""
    print("\n" + "=" * 140)
    print("HYBRID TRADING RECOMMENDATIONS FOR NEXT MONTH")
    print("=" * 140)
    
    # Format the recommendations table
    print(f"{'Symbol':<10} {'Shares':<10} {'Buy Score':<12} {'Sell Score':<12} {'Manual':<12} {'Q-Learner':<12} {'RandomForest':<12} {'Final Rec':<15} {'Conf':<5} {'Restriction':<20}")
    print("-" * 140)
    
    for idx, row in results_df.iterrows():
        symbol = row['Symbol']
        shares = row['Shares']
        buy_score = row['CurrentBuyScore']
        sell_score = row['CurrentSellScore']
        
        # Get manual strategy recommendation
        manual_rec = "HOLD"
        if pd.notna(row['Manual']):
            if row['Manual'] > 0.05:
                manual_rec = "BUY"
            elif row['Manual'] < -0.05:
                manual_rec = "SELL"
        else:
            manual_rec = "N/A"
        
        # Get Q-Learner recommendation
        ql_rec = "HOLD"
        if 'QLearnerSignal' in row and pd.notna(row['QLearnerSignal']):
            ql_rec = row['QLearnerSignal']
        else:
            ql_rec = "N/A"
        
        # Get Random Forest recommendation
        rf_rec = "N/A"
        if pd.notna(row['RandomForest']):
            if row['RandomForest'] > 0.05:
                rf_rec = "BUY"
            elif row['RandomForest'] < -0.05:
                rf_rec = "SELL"
        
        # Calculate final recommendation using signal spread
        buy_score_val = buy_score if pd.notna(buy_score) else 50
        sell_score_val = sell_score if pd.notna(sell_score) else 50
        
        # Use signal spread instead of independent scores
        signal_strength = buy_score_val - sell_score_val
        
        # Calculate confidence based on signal strength (0-1)
        confidence = min(abs(signal_strength) / 40, 1.0)
        
        # Adjust signal based on market regime
        if market_regime == "BEAR":
            # Weaken buy signals in bear market
            signal_strength *= 0.7
        elif market_regime == "BULL":
            # Strengthen buy signals in bull market
            signal_strength *= 1.2
        
        # Determine recommendation based on signal spread
        if signal_strength > 30:
            final_rec = "STRONG BUY"
            action_to_check = "BUY"
        elif signal_strength > 15:
            final_rec = "MODERATE BUY"
            action_to_check = "BUY"
        elif signal_strength < -30:
            final_rec = "STRONG SELL"
            action_to_check = "SELL"
        elif signal_strength < -15:
            final_rec = "MODERATE SELL"
            action_to_check = "SELL"
        else:
            final_rec = "HOLD"
            action_to_check = "HOLD"
        
        # Check trading restrictions
        restriction_status = "OK"
        if restrictions_df is not None and action_to_check in ["BUY", "SELL"]:
            # Get current price from market value and shares
            current_price = row['MarketValue'] / row['Shares'] if row['Shares'] > 0 else 0
            allowed, message = check_trading_restrictions(symbol, action_to_check, current_price, restrictions_df)
            if not allowed:
                restriction_status = "⚠️ VIOLATION"
                print(f"⚠️ RESTRICTION WARNING for {symbol}: {message}")
        
        print(f"{symbol:<10} {shares:<10.3f} {buy_score:>6.1f}/100   {sell_score:>6.1f}/100   {manual_rec:<12} {ql_rec:<12} {rf_rec:<12} {final_rec:<15} {confidence:.2f} {restriction_status:<20}")
    
    print("=" * 140)
    print("📊 HYBRID RECOMMENDATION LOGIC:")
    print("   - Buy Score: 0-100 based on RSI, Momentum, MACD indicators (high = buy signal)")
    print("   - Sell Score: 0-100 based on RSI, Momentum, MACD indicators (high = sell signal)")
    print("   - Signal Spread: buy_score - sell_score (measures buy/sell signal disagreement)")
    print("   - Confidence: abs(signal_strength) / 40 (0-1, higher = stronger signal)")
    print("   - Market Regime: BULL/BEAR/NEUTRAL based on SPY MA200 and VIX")
    print("   - Final Rec: Based on signal spread adjusted for market regime")
    print("   - Position Sizing: Gradual scaling with portfolio weight caps (max 10% per position)")
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
    
    # Detect market regime
    try:
        import yfinance as yf
        spy_ticker = yf.Ticker("SPY")
        spy_hist = spy_ticker.history(period="1y")
        if not spy_hist.empty:
            spy_prices = spy_hist['Close']
            market_regime = detect_market_regime(spy_prices)
            print(f"Market Regime: {market_regime}")
        else:
            market_regime = "NEUTRAL"
            print("Market Regime: NEUTRAL (unable to fetch SPY data)")
    except Exception as e:
        market_regime = "NEUTRAL"
        print(f"Market Regime: NEUTRAL (error: {e})")
    
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
                trades = pd.DataFrame(index=prices.index, columns=["Trades"])
                trades["Trades"] = 0.0
                
                for i in range(len(prices)):
                    if i < 14:
                        continue
                    if rsi.iloc[i] < 30 and trades.iloc[i-1, 0] == 0:
                        trades.iloc[i, 0] = 1000
                    elif rsi.iloc[i] > 70 and trades.iloc[i-1, 0] != 0:
                        trades.iloc[i, 0] = -1000
                
                portfolio_value = compute_portfolio_value_from_trades(trades, prices)
                if portfolio_value is not None:
                    manual_return = (portfolio_value.iloc[-1] - portfolio_value.iloc[0]) / portfolio_value.iloc[0]
                    print(f"  Manual Strategy Return: {manual_return:.2%}")
            except Exception as e:
                print(f"  Error in manual strategy: {e}")
            
            # Run Q-Learner Strategy
            ql_trades = run_qlearner_strategy(symbol, start_date, end_date)
            ql_return = None
            ql_signal = "HOLD"  # Current Q-Learner signal
            if ql_trades is not None:
                ql_portfolio = compute_portfolio_value_from_trades(ql_trades, prices)
                if ql_portfolio is not None:
                    ql_return = (ql_portfolio.iloc[-1] - ql_portfolio.iloc[0]) / ql_portfolio.iloc[0]
                    print(f"  Q-Learner Strategy Return: {ql_return:.2%}")
                    
                    # Get current signal from last trade
                    last_trade = ql_trades.iloc[-1]['Trades'] if len(ql_trades) > 0 else 0
                    if last_trade > 0:
                        ql_signal = "BUY"
                    elif last_trade < 0:
                        ql_signal = "SELL"
                    else:
                        ql_signal = "HOLD"
            else:
                ql_return = None
            
            # Run Random Forest Strategy (only if accuracy > 55%)
            rf_trades, rf_accuracy, rf_model = run_randomforest_strategy(symbol, start_date, end_date)
            rf_model_for_enhancement = None
            
            if rf_trades is not None and rf_accuracy is not None:
                if rf_accuracy > config.ACCURACY_THRESHOLD:
                    rf_portfolio = compute_portfolio_value_from_trades(rf_trades, prices)
                    if rf_portfolio is not None:
                        rf_return = (rf_portfolio.iloc[-1] - rf_portfolio.iloc[0]) / rf_portfolio.iloc[0]
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
            
            # Store results
            results.append({
                'Symbol': symbol,
                'Shares': shares,
                'BuyHold': buy_hold_return,
                'Benchmark': benchmark_return,
                'Manual': manual_return,
                'QLearner': ql_return,
                'QLearnerSignal': ql_signal,
                'RandomForest': rf_return,
                'CurrentSellScore': current_sell_score if sell_scores is not None else None,
                'CurrentBuyScore': current_buy_score if buy_scores is not None else None,
                'MarketValue': row['MarketValue']
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
    print(results_df.to_string(index=False))
    
    # Print predictions
    print_predictions(results_df, restrictions_df)
    
    # Send email report
    print("\n" + "=" * 60)
    print("SENDING EMAIL REPORT")
    print("=" * 60)
    email_sent = send_email_report(results_df, subject=f"Portfolio Analysis Report - {datetime.now().strftime('%Y-%m-%d')}")
    
    if email_sent:
        print("✅ Portfolio analysis complete with email report sent!")
    else:
        print("❌ Portfolio analysis complete but email failed to send")
    
    return results_df


if __name__ == "__main__":
    csv_path = "portfolio.csv"
    results = analyze_portfolio(csv_path, analysis_months=config.ANALYSIS_MONTHS)
