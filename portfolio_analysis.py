"""
Portfolio Analysis Script
Compares actual portfolio performance against Manual Strategy and Q-Learner Strategy
using real holdings data and yfinance for historical prices.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import getpass
import os
from dotenv import load_dotenv

# Add strategy_evaluation to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'strategy_evaluation'))

import ManualStrategy as ms
import StrategyLearner as sl
from util import get_data

def send_email_report(results_df, subject="Portfolio Analysis Report"):
    """Send portfolio analysis report via email with charts attached."""
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        # Email configuration
        sender_email = "demibotti2000@gmail.com"
        receiver_email = "demibotti2000@gmail.com"
        
        # Check if password is already stored in environment variables
        password = os.getenv('GMAIL_PASSWORD')
        
        if password:
            print("\n" + "=" * 60)
            print("EMAIL SETUP")
            print("=" * 60)
            print(f"Sending from: {sender_email}")
            print(f"Sending to: {receiver_email}")
            print("✅ Using saved password from environment variables")
        else:
            # Get password securely and offer to save it
            print("\n" + "=" * 60)
            print("EMAIL SETUP")
            print("=" * 60)
            print(f"Sending from: {sender_email}")
            print(f"Sending to: {receiver_email}")
            print("Note: You may need to use an App Password if 2FA is enabled")
            print("Get App Password: https://myaccount.google.com/apppasswords")
            password = getpass.getpass("Enter Gmail password or App Password: ")
            
            # Ask if user wants to save the password
            save_choice = input("Do you want to save this password for future use? (y/n): ").lower()
            if save_choice == 'y':
                # Save to .env file
                env_file = '.env'
                with open(env_file, 'a') as f:
                    f.write(f'\nGMAIL_PASSWORD={password}\n')
                print(f"✅ Password saved to {env_file}")
                print("⚠️  Make sure to add .env to your .gitignore file to keep it secure")
        
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        
        # Create email body
        body = f"""
        <html>
        <body>
            <h2>Portfolio Analysis Report - {datetime.now().strftime('%Y-%m-%d')}</h2>
            <p>Analysis period: Last 6 months (126 trading days)</p>
            <p>Training data: 5 years historical data</p>
            
            <h3>Summary Results</h3>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <th>Symbol</th>
                    <th>Shares</th>
                    <th>BuyHold</th>
                    <th>Benchmark</th>
                    <th>Manual</th>
                    <th>Q-Learner</th>
                    <th>RandomForest</th>
                    <th>CurrentSellScore</th>
                    <th>CurrentBuyScore</th>
                </tr>
        """
        
        # Add results to table
        for idx, row in results_df.iterrows():
            buyhold_val = f"{row['BuyHold']:.2%}" if pd.notna(row['BuyHold']) else 'N/A'
            benchmark_val = f"{row['Benchmark']:.2%}" if pd.notna(row['Benchmark']) else 'N/A'
            manual_val = f"{row['Manual']:.2%}" if pd.notna(row['Manual']) else 'N/A'
            ql_val = f"{row['QLearner']:.2%}" if pd.notna(row['QLearner']) else 'N/A'
            rf_val = f"{row['RandomForest']:.2%}" if pd.notna(row['RandomForest']) else 'N/A'
            sell_score_val = f"{row['CurrentSellScore']:.1f}/100" if pd.notna(row['CurrentSellScore']) else 'N/A'
            buy_score_val = f"{row['CurrentBuyScore']:.1f}/100" if pd.notna(row['CurrentBuyScore']) else 'N/A'
            
            body += f"""
                <tr>
                    <td>{row['Symbol']}</td>
                    <td>{row['Shares']}</td>
                    <td>{buyhold_val}</td>
                    <td>{benchmark_val}</td>
                    <td>{manual_val}</td>
                    <td>{ql_val}</td>
                    <td>{rf_val}</td>
                    <td>{sell_score_val}</td>
                    <td>{buy_score_val}</td>
                </tr>
            """
        
        body += """
            </table>
            
            <h3>Top 5 Sell Candidates</h3>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <th>Symbol</th>
                    <th>Sell Score</th>
                    <th>6-Month Return</th>
                    <th>Reason</th>
                </tr>
        """
        
        # Get top 5 sell candidates
        top_sells = results_df.nlargest(5, 'CurrentSellScore')
        for idx, row in top_sells.iterrows():
            sell_score = f"{row['CurrentSellScore']:.1f}/100"
            return_val = f"{row['BuyHold']:.2%}" if pd.notna(row['BuyHold']) else 'N/A'
            
            # Generate reason based on scores and performance
            reasons = []
            if row['CurrentSellScore'] > 65:
                reasons.append("High sell score (>65)")
            if row['CurrentBuyScore'] < 35:
                reasons.append("Low buy score (<35)")
            if pd.notna(row['BuyHold']) and row['BuyHold'] > 0.10:
                reasons.append("Strong 6-month performance")
            if pd.notna(row['Manual']) and row['Manual'] < 0:
                reasons.append("Manual strategy underperformance")
            if pd.notna(row['QLearner']) and row['QLearner'] < 0:
                reasons.append("Q-Learner negative signal")
            
            reason = "; ".join(reasons) if reasons else "Technical sell signal"
            
            body += f"""
                <tr>
                    <td>{row['Symbol']}</td>
                    <td>{sell_score}</td>
                    <td>{return_val}</td>
                    <td>{reason}</td>
                </tr>
            """
        
        body += """
            </table>
            
            <h3>Top 5 Buy Candidates</h3>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <th>Symbol</th>
                    <th>Buy Score</th>
                    <th>6-Month Return</th>
                    <th>Reason</th>
                </tr>
        """
        
        # Get top 5 buy candidates
        top_buys = results_df.nlargest(5, 'CurrentBuyScore')
        for idx, row in top_buys.iterrows():
            buy_score = f"{row['CurrentBuyScore']:.1f}/100"
            return_val = f"{row['BuyHold']:.2%}" if pd.notna(row['BuyHold']) else 'N/A'
            
            # Generate reason based on scores and performance
            reasons = []
            if row['CurrentBuyScore'] > 65:
                reasons.append("High buy score (>65)")
            if row['CurrentSellScore'] < 35:
                reasons.append("Low sell score (<35)")
            if pd.notna(row['BuyHold']) and row['BuyHold'] < -0.10:
                reasons.append("Oversold conditions")
            if pd.notna(row['Manual']) and row['Manual'] > 0:
                reasons.append("Manual strategy outperformance")
            if pd.notna(row['QLearner']) and row['QLearner'] > 0:
                reasons.append("Q-Learner positive signal")
            
            reason = "; ".join(reasons) if reasons else "Technical buy signal"
            
            body += f"""
                <tr>
                    <td>{row['Symbol']}</td>
                    <td>{buy_score}</td>
                    <td>{return_val}</td>
                    <td>{reason}</td>
                </tr>
            """
        
        body += """
            </table>
            
            <h3>Key Insights</h3>
            <ul>
                <li>Buy & Hold: Passive portfolio performance</li>
                <li>Benchmark: S&P 500 (SPY) performance</li>
                <li>Manual Strategy: Rule-based trading performance</li>
                <li>Q-Learner: ML-based strategy (Score >65 = SELL, Score <35 = BUY)</li>
                <li>Random Forest: ML model (only if accuracy >55%)</li>
            </ul>
            
            <p>Charts are attached to this email for detailed analysis.</p>
            <p><em>Generated by Portfolio Analysis System</em></p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Attach charts
        chart_files = [
            'portfolio_comparison.png',
            'sell_likelihood_tracking.png', 
            'sell_likelihood_buckets.png',
            'sell_likelihood_heatmap.png',
            'buy_likelihood_heatmap.png'
        ]
        
        for chart_file in chart_files:
            if os.path.exists(chart_file):
                with open(chart_file, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {chart_file}'
                    )
                    msg.attach(part)
                print(f"Attached: {chart_file}")
            else:
                print(f"Warning: {chart_file} not found, skipping attachment")
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        
        print("\n✅ Email sent successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error sending email: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_portfolio(csv_path):
    """Load portfolio from CSV file."""
    return pd.read_csv(csv_path)

def load_trading_restrictions(csv_path):
    """Load trading restrictions from CSV file."""
    try:
        restrictions = pd.read_csv(csv_path)
        # Convert date columns to datetime
        restrictions['LastActionDate'] = pd.to_datetime(restrictions['LastActionDate'])
        restrictions['RestrictionEndDate'] = pd.to_datetime(restrictions['RestrictionEndDate'], errors='coerce')
        return restrictions
    except FileNotFoundError:
        print("No trading restrictions file found, creating default...")
        return None

def check_trading_restrictions(symbol, action, current_price, restrictions_df):
    """Check if a trading action would violate 60-day restrictions."""
    if restrictions_df is None:
        return True, "No restrictions data available"
    
    symbol_restrictions = restrictions_df[restrictions_df['Symbol'] == symbol]
    
    if symbol_restrictions.empty:
        return True, "No previous restrictions found"
    
    restriction = symbol_restrictions.iloc[0]
    today = datetime.now()
    
    # Calculate days since last action
    days_since_action = (today - restriction['LastActionDate']).days
    
    # Update days since action
    restriction['DaysSinceAction'] = days_since_action
    
    # Check if restriction is still active
    if restriction['RestrictionType'] == 'NONE':
        return True, "No active restrictions"
    
    if pd.notna(restriction['RestrictionEndDate']) and today > restriction['RestrictionEndDate']:
        return True, "Restriction period has expired"
    
    # Check for violations
    last_action = restriction['LastAction']
    last_price = restriction['LastActionPrice']
    
    violation_reason = ""
    
    if last_action == 'BOUGHT' and action == 'SELL':
        if current_price > last_price and days_since_action < 60:
            violation_reason = f"VIOLATION: Bought at ${last_price:.2f} {days_since_action} days ago, cannot sell at higher price (${current_price:.2f}) within 60 days"
        else:
            return True, "Safe to sell"
    
    elif last_action == 'SOLD' and action == 'BUY':
        if current_price < last_price and days_since_action < 60:
            violation_reason = f"VIOLATION: Sold at ${last_price:.2f} {days_since_action} days ago, cannot buy at lower price (${current_price:.2f}) within 60 days"
        else:
            return True, "Safe to buy"
    
    else:
        return True, "Action allowed"
    
    return False, violation_reason

def update_trading_restriction(symbol, action, price, restrictions_df, csv_path):
    """Update trading restrictions after a trade is executed."""
    if restrictions_df is None:
        # Create new restrictions DataFrame
        restrictions_df = pd.DataFrame(columns=['Symbol', 'LastAction', 'LastActionDate', 'LastActionPrice', 'DaysSinceAction', 'RestrictionType', 'RestrictionEndDate', 'Notes'])
    
    today = datetime.now()
    
    # Calculate restriction end date (60 days from now)
    restriction_end_date = today + timedelta(days=60)
    
    # Determine restriction type
    if action == 'BOUGHT':
        restriction_type = 'CANNOT_SELL_HIGHER'
        restriction_end = restriction_end_date
        notes = f"Bought at ${price:.2f}, cannot sell at higher price until {restriction_end.date()}"
    elif action == 'SOLD':
        restriction_type = 'CANNOT_BUY_LOWER'
        restriction_end = restriction_end_date
        notes = f"Sold at ${price:.2f}, cannot buy at lower price until {restriction_end.date()}"
    else:
        restriction_type = 'NONE'
        restriction_end = None
        notes = "No restriction"
    
    # Update or add entry
    if symbol in restrictions_df['Symbol'].values:
        restrictions_df.loc[restrictions_df['Symbol'] == symbol, 'LastAction'] = action
        restrictions_df.loc[restrictions_df['Symbol'] == symbol, 'LastActionDate'] = today
        restrictions_df.loc[restrictions_df['Symbol'] == symbol, 'LastActionPrice'] = price
        restrictions_df.loc[restrictions_df['Symbol'] == symbol, 'DaysSinceAction'] = 0
        restrictions_df.loc[restrictions_df['Symbol'] == symbol, 'RestrictionType'] = restriction_type
        restrictions_df.loc[restrictions_df['Symbol'] == symbol, 'RestrictionEndDate'] = restriction_end
        restrictions_df.loc[restrictions_df['Symbol'] == symbol, 'Notes'] = notes
    else:
        new_row = {
            'Symbol': symbol,
            'LastAction': action,
            'LastActionDate': today,
            'LastActionPrice': price,
            'DaysSinceAction': 0,
            'RestrictionType': restriction_type,
            'RestrictionEndDate': restriction_end,
            'Notes': notes
        }
        restrictions_df = pd.concat([restrictions_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save to CSV
    restrictions_df.to_csv(csv_path, index=False)
    return restrictions_df

def get_historical_data(symbol, start_date, end_date):
    """Get historical price data for a symbol using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        # Use period instead of dates for more reliable data retrieval
        days = (end_date - start_date).days
        period = f"{days}d" if days > 0 else "1mo"
        time.sleep(0.3)  # Delay to avoid rate limiting
        hist = ticker.history(period=period)
        if hist.empty:
            print(f"Warning: No data found for {symbol}")
            return None
        return hist['Close']
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_actual_performance(portfolio, prices_dict):
    """Calculate actual portfolio performance based on cost basis vs current value."""
    total_cost_basis = portfolio['CostBasis'].sum()
    total_market_value = portfolio['MarketValue'].sum()
    actual_return = (total_market_value - total_cost_basis) / total_cost_basis
    return total_cost_basis, total_market_value, actual_return

def calculate_buy_hold_return(symbol, shares, start_price, end_price):
    """Calculate buy-and-hold return for a single stock."""
    initial_value = shares * start_price
    final_value = shares * end_price
    return (final_value - initial_value) / initial_value if initial_value > 0 else 0

def calculate_benchmark_return(start_date, end_date):
    """Calculate benchmark (SPY) return for the period."""
    try:
        ticker = yf.Ticker("SPY")
        days = (end_date - start_date).days
        period = f"{days}d" if days > 0 else "2mo"
        time.sleep(0.3)  # Delay to avoid rate limiting
        hist = ticker.history(period=period)
        if hist.empty:
            return 0.0
        prices = hist['Close']
        if len(prices) < 2:
            return 0.0
        return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    except Exception as e:
        print(f"Error calculating benchmark return: {e}")
        return 0.0

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
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
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
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_distance_from_ma(prices, ma_window):
    """Calculate distance from moving average as percentage."""
    ma = prices.rolling(window=ma_window).mean()
    distance = (prices - ma) / ma
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

def get_fundamental_data(symbol):
    """Get fundamental data for a stock."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        fundamentals = {
            'PE_Ratio': info.get('forwardPE', None),
            'Revenue_Growth': info.get('revenueGrowth', None),
            'EPS_Growth': info.get('earningsGrowth', None),
            'Operating_Margin': info.get('operatingMargins', None),
            'Debt_to_Equity': info.get('debtToEquity', None),
            'Profit_Margin': info.get('profitMargins', None),
            'ROE': info.get('returnOnEquity', None)
        }
        return fundamentals
    except Exception as e:
        print(f"Error getting fundamental data for {symbol}: {e}")
        return None

def get_earnings_data(symbol):
    """Get earnings calendar and surprise data."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        earnings = {
            'Next_Earnings_Date': info.get('nextEarningsDate', None),
            'EPS_TTM': info.get('trailingEps', None),
            'EPS_Forward': info.get('forwardEps', None)
        }
        return earnings
    except Exception as e:
        print(f"Error getting earnings data for {symbol}: {e}")
        return None

def run_qlearner_strategy(symbol, start_date, end_date, sv=100000):
    """Run Q-Learner Strategy on a symbol using yfinance data with score-based trading."""
    try:
        # Get extended historical data for better training (5 years)
        ticker = yf.Ticker(symbol)
        time.sleep(0.3)  # Delay to avoid rate limiting
        hist = ticker.history(period="5y")
        if hist.empty:
            print(f"  No data found for {symbol} for Q-Learner")
            return None
        
        # Get SPY data for relative strength calculation
        spy_ticker = yf.Ticker("SPY")
        time.sleep(0.3)
        spy_hist = spy_ticker.history(period="5y")
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
        
        if len(prices) < 252:  # Need at least 1 year of data for training
            print(f"  Insufficient data for Q-Learner {symbol} (got {len(prices)} days)")
            return None
        
        # Train on first 80% of data, test on last 20%
        split_point = int(len(prices) * 0.8)
        train_prices = prices.iloc[:split_point]
        test_prices = prices.iloc[split_point:]
        
        train_spy = spy_prices.iloc[:split_point]
        train_volume = volume.iloc[:split_point]
        train_high = high.iloc[:split_point]
        train_low = low.iloc[:split_point]
        
        # Calculate enhanced indicators for training data
        train_rsi = calculate_rsi(train_prices, window=14)
        train_mom = calculate_momentum(train_prices, window=14)
        train_macd_hist = calculate_macd_histogram(train_prices)
        train_rs_20d = calculate_relative_strength(train_prices, train_spy, window=20)
        train_rs_60d = calculate_relative_strength(train_prices, train_spy, window=60)
        train_rs_120d = calculate_relative_strength(train_prices, train_spy, window=120)
        train_vol_ratio = calculate_volume_ratio(train_volume, window=20)
        train_obv = calculate_obv(train_prices, train_volume)
        train_ad = calculate_accumulation_distribution(train_high, train_low, train_prices, train_volume)
        train_atr = calculate_atr(train_high, train_low, train_prices, window=14)
        train_dist_ma50 = calculate_distance_from_ma(train_prices, 50)
        train_dist_ma200 = calculate_distance_from_ma(train_prices, 200)
        train_bb_pos = calculate_bollinger_band_position(train_prices, window=20)
        
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
        
        if len(train_features) < 100:
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
        
        test_rsi = calculate_rsi(test_prices, window=14)
        test_mom = calculate_momentum(test_prices, window=14)
        test_macd_hist = calculate_macd_histogram(test_prices)
        test_rs_20d = calculate_relative_strength(test_prices, test_spy, window=20)
        test_rs_60d = calculate_relative_strength(test_prices, test_spy, window=60)
        test_rs_120d = calculate_relative_strength(test_prices, test_spy, window=120)
        test_vol_ratio = calculate_volume_ratio(test_volume, window=20)
        test_obv = calculate_obv(test_prices, test_volume)
        test_ad = calculate_accumulation_distribution(test_high, test_low, test_prices, test_volume)
        test_atr = calculate_atr(test_high, test_low, test_prices, window=14)
        test_dist_ma50 = calculate_distance_from_ma(test_prices, 50)
        test_dist_ma200 = calculate_distance_from_ma(test_prices, 200)
        test_bb_pos = calculate_bollinger_band_position(test_prices, window=20)
        
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
            if score > 65 and curr > -1000:  # Score > 65 = SELL signal
                trades.iloc[i] = -1000 - curr
                curr = -1000
            elif score < 35 and curr < 1000:  # Score < 35 = BUY signal
                trades.iloc[i] = 1000 - curr
                curr = 1000
            elif 35 <= score <= 65 and curr != 0:  # Neutral range = HOLD
                trades.iloc[i] = -curr
                curr = 0
        
        return trades
        
    except Exception as e:
        print(f"Error running Q-Learner for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_randomforest_strategy(symbol, start_date, end_date, sv=100000):
    try:
        # Get extended historical data for better training (5 years)
        ticker = yf.Ticker(symbol)
        time.sleep(0.3)  # Delay to avoid rate limiting
        hist = ticker.history(period="5y")
        if hist.empty:
            print(f"  No data found for {symbol} for Random Forest")
            return None, None, None
        
        # Get SPY data for relative strength calculation
        spy_ticker = yf.Ticker("SPY")
        time.sleep(0.3)
        spy_hist = spy_ticker.history(period="5y")
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
        
        if len(prices) < 252:  # Need at least 1 year of data for training
            print(f"  Insufficient data for Random Forest {symbol} (got {len(prices)} days)")
            return None, None, None
        
        # Train on first 80% of data, test on last 20%
        split_point = int(len(prices) * 0.8)
        train_prices = prices.iloc[:split_point]
        test_prices = prices.iloc[split_point:]
        
        train_spy = spy_prices.iloc[:split_point]
        train_volume = volume.iloc[:split_point]
        train_high = high.iloc[:split_point]
        train_low = low.iloc[:split_point]
        
        # Calculate enhanced indicators for training data
        train_rsi = calculate_rsi(train_prices, window=14)
        train_mom = calculate_momentum(train_prices, window=14)
        train_macd_hist = calculate_macd_histogram(train_prices)
        train_rs_20d = calculate_relative_strength(train_prices, train_spy, window=20)
        train_rs_60d = calculate_relative_strength(train_prices, train_spy, window=60)
        train_rs_120d = calculate_relative_strength(train_prices, train_spy, window=120)
        train_vol_ratio = calculate_volume_ratio(train_volume, window=20)
        train_obv = calculate_obv(train_prices, train_volume)
        train_ad = calculate_accumulation_distribution(train_high, train_low, train_prices, train_volume)
        train_atr = calculate_atr(train_high, train_low, train_prices, window=14)
        train_dist_ma50 = calculate_distance_from_ma(train_prices, 50)
        train_dist_ma200 = calculate_distance_from_ma(train_prices, 200)
        train_bb_pos = calculate_bollinger_band_position(train_prices, window=20)
        
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
        
        if len(train_features) < 100:
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
        
        test_rsi = calculate_rsi(test_prices, window=14)
        test_mom = calculate_momentum(test_prices, window=14)
        test_macd_hist = calculate_macd_histogram(test_prices)
        test_rs_20d = calculate_relative_strength(test_prices, test_spy, window=20)
        test_rs_60d = calculate_relative_strength(test_prices, test_spy, window=60)
        test_rs_120d = calculate_relative_strength(test_prices, test_spy, window=120)
        test_vol_ratio = calculate_volume_ratio(test_volume, window=20)
        test_obv = calculate_obv(test_prices, test_volume)
        test_ad = calculate_accumulation_distribution(test_high, test_low, test_prices, test_volume)
        test_atr = calculate_atr(test_high, test_low, test_prices, window=14)
        test_dist_ma50 = calculate_distance_from_ma(test_prices, 50)
        test_dist_ma200 = calculate_distance_from_ma(test_prices, 200)
        test_bb_pos = calculate_bollinger_band_position(test_prices, window=20)
        
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
            elif 0.4 <= pred <= 0.6 and curr != 0:  # Neutral signal
                trades.iloc[i] = -curr
                curr = 0
        
        print(f"  Random Forest Out-of-Sample Accuracy: {accuracy:.2%}")
        
        return trades, accuracy, model
        
    except Exception as e:
        print(f"Error running Random Forest for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def run_manual_strategy(symbol, start_date, end_date, sv=100000):
    """Run Manual Strategy on a symbol using yfinance data."""
    try:
        # Get price data using yfinance with period-based approach
        days = (end_date - start_date).days
        period = f"{days}d" if days > 0 else "2mo"
        ticker = yf.Ticker(symbol)
        time.sleep(0.3)  # Delay to avoid rate limiting
        hist = ticker.history(period=period)
        if hist.empty:
            print(f"  No data found for {symbol}")
            return None
        prices = hist['Close']
        
        if len(prices) < 30:
            print(f"  Insufficient data for {symbol} (got {len(prices)} days)")
            return None
        
        # Calculate indicators manually
        rsi_values = calculate_rsi(prices, window=14)
        mom_values = calculate_momentum(prices, window=14)
        macd_values = calculate_macd(prices)
        
        # Generate trades using Manual Strategy logic
        trades = pd.DataFrame(index=prices.index, columns=["Trades"])
        trades["Trades"] = 0.0
        
        curr = 0
        for i in range(len(prices)):
            try:
                rsi_val = rsi_values.iloc[i] if i < len(rsi_values) else None
                mom_val = mom_values.iloc[i] if i < len(mom_values) else None
                macd_val = macd_values.iloc[i] if i < len(macd_values) else None
                
                if pd.isna(rsi_val) or pd.isna(mom_val) or pd.isna(macd_val):
                    continue
                
                rsi_signal = 1 if rsi_val < 30 else -1 if rsi_val > 70 else 0
                mom_signal = 1 if mom_val > 0 else -1 if mom_val < 0 else 0
                macd_signal = 1 if macd_val > 0 else -1 if macd_val < 0 else 0
                
                signal = sum([rsi_signal, mom_signal, macd_signal])
                if signal >= 2:
                    signal = 1
                elif signal <= -2:
                    signal = -1
                else:
                    signal = 0
                
                if signal == 1 and curr < 1000:
                    trades.iloc[i] = 1000 - curr
                    curr = 1000
                elif signal == -1 and curr > -1000:
                    trades.iloc[i] = -1000 - curr
                    curr = -1000
                elif signal == 0 and curr != 0:
                    trades.iloc[i] = -curr
                    curr = 0
            except Exception as e:
                continue
        
        return trades
    except Exception as e:
        print(f"Error running Manual Strategy for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


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
            
            # Momentum: <0 = high sell score, >0 = low sell score  
            mom_score = min(100, max(0, 50 - mom_val * 100))  # Positive momentum reduces sell score
            
            # MACD: <0 = high sell score, >0 = low sell score
            macd_score = min(100, max(0, 50 - macd_val * 10))  # Positive MACD reduces sell score
            
            # Combined score (weighted average)
            combined_score = (rsi_score * 0.4 + mom_score * 0.3 + macd_score * 0.3)
            daily_scores.iloc[i] = combined_score
            
        except Exception:
            daily_scores.iloc[i] = 50
    
    return daily_scores

def calculate_ml_enhanced_scores(prices, model, score_type, symbol):
    """Calculate ML-enhanced buy/sell scores using trained model predictions."""
    try:
        # Get extended data for enhanced features
        spy_ticker = yf.Ticker("SPY")  # Need SPY for relative strength
        time.sleep(0.3)
        spy_hist = spy_ticker.history(period="5y")
        if spy_hist.empty:
            print("  No SPY data for ML enhancement")
            return None
        
        spy_prices = spy_hist['Close']
        
        # Align SPY prices with stock prices
        spy_aligned = spy_prices.reindex(prices.index, method='ffill')
        
        # Calculate enhanced indicators for entire price series
        rsi_values = calculate_rsi(prices, window=14)
        mom_values = calculate_momentum(prices, window=14)
        macd_hist_values = calculate_macd_histogram(prices)
        rs_20d_values = calculate_relative_strength(prices, spy_aligned, window=20)
        rs_60d_values = calculate_relative_strength(prices, spy_aligned, window=60)
        
        # For volume and high/low, we need to get the full history
        symbol_ticker = yf.Ticker(symbol)
        time.sleep(0.3)
        full_hist = symbol_ticker.history(period="5y")
        if full_hist.empty:
            print(f"  No full history for {symbol} ML enhancement")
            return None
        
        volume = full_hist['Volume'].reindex(prices.index, method='ffill')
        high = full_hist['High'].reindex(prices.index, method='ffill')
        low = full_hist['Low'].reindex(prices.index, method='ffill')
        
        vol_ratio_values = calculate_volume_ratio(volume, window=20)
        atr_values = calculate_atr(high, low, prices, window=14)
        dist_ma50_values = calculate_distance_from_ma(prices, 50)
        dist_ma200_values = calculate_distance_from_ma(prices, 200)
        bb_pos_values = calculate_bollinger_band_position(prices, window=20)
        obv_values = calculate_obv(prices, volume)
        ad_values = calculate_accumulation_distribution(high, low, prices, volume)
        rs_120d_values = calculate_relative_strength(prices, spy_aligned, window=120)
        
        # Market regime features
        spy_return_20d_values, spy_return_60d_values = calculate_market_regime(spy_aligned)
        
        # Get VIX for market sentiment
        current_vix = calculate_vix()
        if current_vix:
            vix_values = pd.Series([current_vix] * len(prices), index=prices.index)
        else:
            vix_values = pd.Series([20] * len(prices), index=prices.index)
        
        # Create feature DataFrame with all enhanced features
        features = pd.DataFrame({
            'RSI': rsi_values,
            'Momentum': mom_values,
            'MACD_Histogram': macd_hist_values,
            'RS_20D': rs_20d_values,
            'RS_60D': rs_60d_values,
            'RS_120D': rs_120d_values,
            'Volume_Ratio': vol_ratio_values,
            'OBV': obv_values,
            'Accum_Dist': ad_values,
            'ATR': atr_values,
            'Distance_MA50': dist_ma50_values,
            'Distance_MA200': dist_ma200_values,
            'BB_Position': bb_pos_values,
            'SPY_Return_20D': spy_return_20d_values,
            'SPY_Return_60D': spy_return_60d_values,
            'VIX': vix_values
        })
        
        # Remove NaN values
        valid_indices = ~features.isna().any(axis=1)
        features = features[valid_indices]
        
        if len(features) < 10:
            return None
        
        # Get ML predictions (probability of price going up)
        predictions = model.predict_proba(features)[:, 1]
        
        # Convert predictions to buy/sell scores (0-100 scale)
        ml_scores = pd.Series(index=features.index, dtype=float)
        
        for i in range(len(predictions)):
            pred = predictions[i]
            
            if score_type == 'buy':
                # Higher prediction = higher buy score
                ml_scores.iloc[i] = pred * 100
            elif score_type == 'sell':
                # Lower prediction = higher sell score
                ml_scores.iloc[i] = (1 - pred) * 100
        
        # Reindex to match original prices index
        ml_scores = ml_scores.reindex(prices.index, method='ffill')
        
        return ml_scores
        
    except Exception as e:
        print(f"Error calculating ML-enhanced scores: {e}")
        return None

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
            
            # Momentum: >0 = high buy score, <0 = low buy score  
            mom_score = min(100, max(0, 50 + mom_val * 100))  # Positive momentum increases buy score
            
            # MACD: >0 = high buy score, <0 = low buy score
            macd_score = min(100, max(0, 50 + macd_val * 10))  # Positive MACD increases buy score
            
            # Combined score (weighted average)
            combined_score = (rsi_score * 0.4 + mom_score * 0.3 + macd_score * 0.3)
            daily_scores.iloc[i] = combined_score
            
        except Exception:
            daily_scores.iloc[i] = 50
    
    return daily_scores

def generate_prediction(symbol, current_date):
    """Generate trading recommendation for next month considering 60-day restrictions."""
    try:
        # Get recent data for prediction (last 3 months)
        start_date = current_date - timedelta(days=90)
        # Use period-based approach for prediction
        ticker = yf.Ticker(symbol)
        time.sleep(0.3)  # Delay to avoid rate limiting
        hist = ticker.history(period="3mo")
        if hist.empty:
            return "INSUFFICIENT_DATA"
        prices = hist['Close']
        
        if prices is None or len(prices) < 30:
            return "INSUFFICIENT_DATA"
        
        # Calculate indicators
        rsi_values = calculate_rsi(prices, window=14)
        mom_values = calculate_momentum(prices, window=14)
        macd_values = calculate_macd(prices)
        
        # Get latest indicator values
        latest_rsi = rsi_values.iloc[-1]
        latest_mom = mom_values.iloc[-1]
        latest_macd = macd_values.iloc[-1]
        
        # Generate signals
        rsi_signal = 1 if latest_rsi < 30 else -1 if latest_rsi > 70 else 0
        mom_signal = 1 if latest_mom > 0 else -1 if latest_mom < 0 else 0
        macd_signal = 1 if latest_macd > 0 else -1 if latest_macd < 0 else 0
        
        signal = sum([rsi_signal, mom_signal, macd_signal])
        
        # Generate recommendation considering 60-day restrictions
        # If signal >= 2: Strong BUY signal
        # If signal <= -2: Strong SELL signal  
        # If -1 < signal < 1: HOLD signal
        
        if signal >= 2:
            recommendation = "BUY (Strong bullish signal)"
            confidence = "HIGH"
        elif signal <= -2:
            recommendation = "SELL (Strong bearish signal)"
            confidence = "HIGH"
        elif signal == 1:
            recommendation = "HOLD (Slightly bullish)"
            confidence = "MEDIUM"
        elif signal == -1:
            recommendation = "HOLD (Slightly bearish)"
            confidence = "MEDIUM"
        else:
            recommendation = "HOLD (Neutral)"
            confidence = "LOW"
        
        # Add 60-day restriction warning
        restriction_note = "⚠️ 60-day trading restriction applies"
        
        return f"{recommendation} | {confidence} | {restriction_note}"
        
    except Exception as e:
        return f"ERROR: {str(e)}"

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

def analyze_portfolio(csv_path, analysis_months=6):
    """Main analysis function for recent performance and predictions."""
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
    
    # Track daily sell likelihood scores
    daily_scores = {}
    
    # Analyze each position
    results = []
    
    for idx, row in portfolio.iterrows():
        symbol = row['Symbol']
        shares = row['Shares']
        cost_basis = row['CostBasis']
        market_value = row['MarketValue']
        
        print(f"\nAnalyzing {symbol}...")
        
        # Get historical data using period-based approach
        days = (end_date - start_date).days
        period = f"{days}d" if days > 0 else "2mo"
        ticker = yf.Ticker(symbol)
        time.sleep(1.0)  # Increased delay to avoid rate limiting
        try:
            hist = ticker.history(period=period)
        except:
            print(f"Skipping {symbol} - rate limited")
            continue
        if hist.empty:
            print(f"Skipping {symbol} - no data found")
            continue
        prices = hist['Close']
        
        if len(prices) < 30:
            print(f"Skipping {symbol} - insufficient data (got {len(prices)} days)")
            continue
        
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        
        # Buy and hold return
        buy_hold_return = calculate_buy_hold_return(symbol, shares, start_price, end_price)
        print(f"  Buy & Hold Return: {buy_hold_return:.2%}")
        
        # Benchmark return (SPY)
        benchmark_return = calculate_benchmark_return(start_date, end_date)
        print(f"  Benchmark (SPY) Return: {benchmark_return:.2%}")
        
        # Run Manual Strategy
        manual_trades = run_manual_strategy(symbol, start_date, end_date)
        if manual_trades is not None:
            manual_portfolio = compute_portfolio_value_from_trades(manual_trades, prices)
            if manual_portfolio is not None:
                manual_return = (manual_portfolio.iloc[-1] - manual_portfolio.iloc[0]) / manual_portfolio.iloc[0]
                print(f"  Manual Strategy Return: {manual_return:.2%}")
            else:
                manual_return = None
        else:
            manual_return = None
        
        # Run Q-Learner Strategy (score-based: >65 sell, <35 buy)
        ql_trades = run_qlearner_strategy(symbol, start_date, end_date)
        if ql_trades is not None:
            ql_portfolio = compute_portfolio_value_from_trades(ql_trades, prices)
            if ql_portfolio is not None:
                ql_return = (ql_portfolio.iloc[-1] - ql_portfolio.iloc[0]) / ql_portfolio.iloc[0]
                print(f"  Q-Learner Strategy Return: {ql_return:.2%}")
            else:
                ql_return = None
        else:
            ql_return = None
        
        # Run Random Forest Strategy (only if accuracy > 55%)
        rf_trades, rf_accuracy, rf_model = run_randomforest_strategy(symbol, start_date, end_date)
        rf_model_for_enhancement = None
        
        if rf_trades is not None and rf_accuracy is not None:
            if rf_accuracy > 0.55:  # Lowered threshold from 95% to 55% for more useful ML participation
                rf_portfolio = compute_portfolio_value_from_trades(rf_trades, prices)
                if rf_portfolio is not None:
                    rf_return = (rf_portfolio.iloc[-1] - rf_portfolio.iloc[0]) / rf_portfolio.iloc[0]
                    print(f"  Random Forest Strategy Return: {rf_return:.2%} (Accuracy: {rf_accuracy:.2%})")
                    rf_model_for_enhancement = rf_model  # Save model for score enhancement
                else:
                    rf_return = None
            else:
                print(f"  Random Forest skipped (Accuracy: {rf_accuracy:.2%} < 55%)")
                rf_return = None
        else:
            rf_return = None
        
        # Generate prediction for next month
        prediction = generate_prediction(symbol, end_date)
        
        # Calculate daily sell scores (optionally enhanced with ML if accuracy > 55%)
        sell_scores = calculate_daily_sell_scores(prices)
        buy_scores = calculate_daily_buy_scores(prices)
        
        # If ML model is accurate (>55%), enhance buy/sell scores with ML predictions
        if rf_model_for_enhancement is not None and rf_accuracy is not None:
            print(f"  Enhancing buy/sell scores with ML predictions (Accuracy: {rf_accuracy:.2%})")
            # Get ML predictions for the entire price series
            ml_enhanced_sell = calculate_ml_enhanced_scores(prices, rf_model_for_enhancement, 'sell', symbol)
            ml_enhanced_buy = calculate_ml_enhanced_scores(prices, rf_model_for_enhancement, 'buy', symbol)
            
            if ml_enhanced_sell is not None:
                sell_scores = (sell_scores * 0.7 + ml_enhanced_sell * 0.3)  # 70% technical + 30% ML
            if ml_enhanced_buy is not None:
                buy_scores = (buy_scores * 0.7 + ml_enhanced_buy * 0.3)  # 70% technical + 30% ML
        
        if sell_scores is not None:
            daily_scores[symbol] = sell_scores
            # Calculate average sell score over the period
            avg_sell_score = sell_scores.mean()
            recent_sell_score = sell_scores.iloc[-1]  # Most recent score
            print(f"  Average Sell Likelihood (2mo): {avg_sell_score:.1f}/100")
            print(f"  Current Sell Likelihood: {recent_sell_score:.1f}/100")
        else:
            avg_sell_score = None
            recent_sell_score = None
            
        if buy_scores is not None:
            avg_buy_score = buy_scores.mean()
            recent_buy_score = buy_scores.iloc[-1]  # Most recent score
            print(f"  Average Buy Likelihood (2mo): {avg_buy_score:.1f}/100")
            print(f"  Current Buy Likelihood: {recent_buy_score:.1f}/100")
        else:
            avg_buy_score = None
            recent_buy_score = None
        
        results.append({
            'Symbol': symbol,
            'Shares': shares,
            'BuyHold': buy_hold_return,
            'Benchmark': benchmark_return,
            'Manual': manual_return,
            'QLearner': ql_return,
            'RandomForest': rf_return,
            'Prediction': prediction,
            'AvgSellScore': avg_sell_score,
            'CurrentSellScore': recent_sell_score,
            'AvgBuyScore': avg_buy_score,
            'CurrentBuyScore': recent_buy_score,
            'MarketValue': row['MarketValue']
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS - LAST 2 MONTHS PERFORMANCE")
    print("=" * 60)
    # Print summary with sell and buy scores (without shares for privacy)
    summary_cols = ['Symbol', 'BuyHold', 'Benchmark', 'Manual', 'QLearner', 'RandomForest', 'CurrentSellScore', 'CurrentBuyScore']
    print(results_df[summary_cols].to_string(index=False))
    
    # Plot comparison
    if len(results_df) > 0:
        plot_comparison(results_df)
    
    # Plot daily sell likelihood tracking
    if daily_scores:
        plot_sell_likelihood_tracking(daily_scores)
        plot_sell_likelihood_buckets(daily_scores)
    
    # Plot sell likelihood heatmap
    if len(results_df) > 0:
        plot_sell_likelihood_heatmap(results_df)
    
    # Plot buy likelihood heatmap
    if len(results_df) > 0:
        plot_buy_likelihood_heatmap(results_df)
    
    # Calculate and plot portfolio cumulative returns (in-sample) - DISABLED due to rate limiting
    # This function makes too many API calls (18 stocks × 40 days = 720+ calls) and causes the program to hang
    # Individual stock analysis is already complete, so this is optional
    print("\nSkipping portfolio cumulative returns calculation (disabled due to rate limiting)")
    # Uncomment below if needed, but be aware it will take a long time
    # print("\nCalculating portfolio cumulative returns...")
    # try:
    #     result = calculate_portfolio_cumulative_returns(portfolio, start_date, end_date)
    #     if result is not None:
    #         portfolio_returns, benchmark_returns = result
    #         plot_portfolio_cumulative_returns(portfolio_returns, benchmark_returns, start_date, end_date)
    #     else:
    #         print("Portfolio cumulative returns calculation returned None")
    # except Exception as e:
    #     print(f"Error calculating portfolio cumulative returns: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # Calculate and plot out-of-sample performance (disabled due to rate limiting)
    # Uncomment if needed for detailed train/test analysis
    # out_of_sample_data, split_dates = calculate_out_of_sample_performance(portfolio, start_date, end_date)
    # if out_of_sample_data is not None:
    #     plot_out_of_sample_performance(out_of_sample_data, split_dates)
    
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
        print("⚠️ Portfolio analysis complete, but email failed to send.")
    
    return results_df

def plot_comparison(results_df):
    """Create comparison chart of strategy performance with benchmark, Q-Learner, and Random Forest."""
    fig, ax = plt.subplots(figsize=(18, 6))
    
    x = np.arange(len(results_df))
    width = 0.15
    
    buy_hold = results_df['BuyHold'].fillna(0)
    benchmark = results_df['Benchmark'].fillna(0)
    manual = results_df['Manual'].fillna(0)
    qlearner = results_df['QLearner'].fillna(0)
    randomforest = results_df['RandomForest'].fillna(0)
    
    # Ensure benchmark is consistent across all stocks
    if len(benchmark.unique()) > 1:
        print(f"Warning: Benchmark values vary across stocks. Using mean: {benchmark.mean():.2%}")
        benchmark = pd.Series([benchmark.mean()] * len(results_df))
    
    ax.bar(x - 2*width, buy_hold, width, label='Buy & Hold', color='blue', alpha=0.7)
    ax.bar(x - width, benchmark, width, label='Benchmark (SPY)', color='purple', alpha=0.7)
    ax.bar(x, manual, width, label='Manual Strategy', color='green', alpha=0.7)
    ax.bar(x + width, qlearner, width, label='Q-Learner Strategy', color='orange', alpha=0.7)
    ax.bar(x + 2*width, randomforest, width, label='Random Forest Strategy', color='red', alpha=0.7)
    
    ax.set_xlabel('Symbol')
    ax.set_ylabel('Return')
    ax.set_title('Hybrid Strategy Performance Comparison - Last 2 Months')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Symbol'])
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('portfolio_comparison.png', dpi=150)
    print("\nChart saved as portfolio_comparison.png")
    plt.close()

def plot_sell_likelihood_tracking(daily_scores):
    """Create chart showing daily sell likelihood tracking for all stocks."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each stock's sell likelihood over time
    for symbol, scores in daily_scores.items():
        if scores is not None and len(scores) > 0:
            # Normalize dates to days from start
            dates = range(len(scores))
            ax.plot(dates, scores.values, label=symbol, alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Days (Last 2 months)')
    ax.set_ylabel('Sell Likelihood Score (0-100)')
    ax.set_title('Daily Sell Likelihood Tracking - Last 2 Months')
    ax.axhline(y=70, color='red', linestyle='--', linewidth=1, label='High Sell Threshold (70)')
    ax.axhline(y=30, color='green', linestyle='--', linewidth=1, label='Low Sell Threshold (30)')
    ax.axhline(y=50, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, label='Neutral (50)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('sell_likelihood_tracking.png', dpi=150, bbox_inches='tight')
    print("Sell likelihood tracking chart saved as sell_likelihood_tracking.png")
    plt.close()

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

def plot_sell_likelihood_heatmap(results_df):
    """Create grayscale heatmap showing sell likelihood from light to dark."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract sell scores and symbols
    symbols = results_df['Symbol'].values
    sell_scores = results_df['CurrentSellScore'].values
    
    # Create a single row heatmap
    scores_2d = sell_scores.reshape(1, -1)
    
    # Create heatmap with grayscale colormap (light = low sell, dark = high sell)
    im = ax.imshow(scores_2d, cmap='Greys', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks(range(len(symbols)))
    ax.set_xticklabels(symbols, rotation=45, ha='right')
    ax.set_yticks([])
    ax.set_title('Sell Likelihood Heatmap - Lighter = Hold/Buy, Darker = Sell')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    cbar.set_label('Sell Likelihood Score (0-100)', rotation=0, labelpad=10)
    
    # Add threshold markers
    ax.axvline(x=-0.5, color='gray', linewidth=0.5)
    for i in range(len(symbols)):
        # Add score text on each cell
        score_text = f"{sell_scores[i]:.1f}"
        ax.text(i, 0, score_text, ha='center', va='center', 
                color='white' if sell_scores[i] > 50 else 'black', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sell_likelihood_heatmap.png', dpi=150, bbox_inches='tight')
    print("Sell likelihood heatmap saved as sell_likelihood_heatmap.png")
    plt.close()

def plot_buy_likelihood_heatmap(results_df):
    """Create grayscale heatmap showing buy likelihood from light to dark."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract buy scores and symbols
    symbols = results_df['Symbol'].values
    buy_scores = results_df['CurrentBuyScore'].values
    
    # Create a single row heatmap
    scores_2d = buy_scores.reshape(1, -1)
    
    # Create heatmap with grayscale colormap (light = low buy, dark = high buy)
    im = ax.imshow(scores_2d, cmap='Greys', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks(range(len(symbols)))
    ax.set_xticklabels(symbols, rotation=45, ha='right')
    ax.set_yticks([])
    ax.set_title('Buy Likelihood Heatmap - Lighter = Hold/Sell, Darker = Buy')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    cbar.set_label('Buy Likelihood Score (0-100)', rotation=0, labelpad=10)
    
    # Add threshold markers
    ax.axvline(x=-0.5, color='gray', linewidth=0.5)
    for i in range(len(symbols)):
        # Add score text on each cell
        score_text = f"{buy_scores[i]:.1f}"
        ax.text(i, 0, score_text, ha='center', va='center', 
                color='white' if buy_scores[i] > 50 else 'black', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('buy_likelihood_heatmap.png', dpi=150, bbox_inches='tight')
    print("Buy likelihood heatmap saved as buy_likelihood_heatmap.png")
    plt.close()

def calculate_portfolio_cumulative_returns(portfolio, start_date, end_date):
    """Calculate cumulative returns for the entire portfolio under different strategies."""
    try:
        print("  Getting SPY benchmark data...")
        # Get SPY benchmark data
        spy_ticker = yf.Ticker("SPY")
        days = (end_date - start_date).days
        period = f"{days}d" if days > 0 else "2mo"
        time.sleep(0.5)  # Delay to avoid rate limiting
        spy_hist = spy_ticker.history(period=period)
        if spy_hist.empty:
            print("  No SPY data found")
            return None
        spy_prices = spy_hist['Close']
        
        print(f"  SPY data loaded: {len(spy_prices)} days")
        
        # Calculate portfolio weights
        total_market_value = portfolio['MarketValue'].sum()
        portfolio['Weight'] = portfolio['MarketValue'] / total_market_value
        
        # Initialize cumulative returns series
        dates = spy_prices.index
        portfolio_returns = pd.Series(index=dates, dtype=float)
        benchmark_returns = pd.Series(index=dates, dtype=float)
        
        print(f"  Calculating returns for {len(portfolio)} stocks...")
        # Calculate portfolio buy & hold returns
        for date in dates:
            portfolio_return = 0.0
            for idx, row in portfolio.iterrows():
                symbol = row['Symbol']
                weight = row['Weight']
                
                try:
                    ticker = yf.Ticker(symbol)
                    time.sleep(0.3)  # Delay to avoid rate limiting
                    hist = ticker.history(period=period)
                    if hist.empty:
                        continue
                    prices = hist['Close']
                    
                    if date in prices.index:
                        stock_return = (prices.loc[date] / prices.iloc[0]) - 1
                        portfolio_return += weight * stock_return
                except Exception as e:
                    print(f"    Error processing {symbol}: {e}")
                    continue
            
            portfolio_returns.loc[date] = portfolio_return
            benchmark_returns.loc[date] = (spy_prices.loc[date] / spy_prices.iloc[0]) - 1
        
        print("  Portfolio returns calculation complete")
        return portfolio_returns, benchmark_returns
        
    except Exception as e:
        print(f"Error calculating portfolio cumulative returns: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_portfolio_cumulative_returns(portfolio_returns, benchmark_returns, start_date, end_date):
    """Plot cumulative returns for portfolio vs benchmark (in-sample style)."""
    if portfolio_returns is None or benchmark_returns is None:
        print("No data available for cumulative returns plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Normalize to start at 1.0
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    ax.plot(portfolio_cumulative.index, portfolio_cumulative.values, 
            label='Your Portfolio (Buy & Hold)', linewidth=2, color='blue')
    ax.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
            label='Benchmark (SPY)', linewidth=2, color='purple')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (Normalized)')
    ax.set_title(f'Portfolio vs Benchmark Cumulative Returns (In-Sample)\n{start_date.date()} to {end_date.date()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('portfolio_cumulative_returns.png', dpi=150)
    print("Portfolio cumulative returns chart saved as portfolio_cumulative_returns.png")
    plt.close()

def calculate_out_of_sample_performance(portfolio, start_date, end_date):
    """Calculate out-of-sample performance using train/test split like experiment 2."""
    try:
        # Split period into 70% training, 30% testing
        total_days = (end_date - start_date).days
        train_days = int(total_days * 0.7)
        train_end = start_date + timedelta(days=train_days)
        test_start = train_end + timedelta(days=1)
        
        if test_start >= end_date:
            print("Insufficient data for out-of-sample testing")
            return None, None
        
        print(f"\nOut-of-Sample Analysis:")
        print(f"Training Period: {start_date.date()} to {train_end.date()}")
        print(f"Testing Period: {test_start.date()} to {end_date.date()}")
        
        # Get SPY benchmark data for both periods
        spy_ticker = yf.Ticker("SPY")
        
        # Training period
        train_period_days = (train_end - start_date).days
        train_period = f"{train_period_days}d" if train_period_days > 0 else "1mo"
        
        # Add delay to avoid rate limiting
        import time
        time.sleep(1)
        spy_train = spy_ticker.history(period=train_period)
        
        # Testing period
        test_period_days = (end_date - test_start).days
        test_period = f"{test_period_days}d" if test_period_days > 0 else "1mo"
        
        time.sleep(1)
        spy_test = spy_ticker.history(period=test_period)
        
        if spy_train.empty or spy_test.empty:
            print("Insufficient SPY data for out-of-sample testing")
            return None, None
        
        spy_train_prices = spy_train['Close']
        spy_test_prices = spy_test['Close']
        
        # Calculate portfolio weights
        total_market_value = portfolio['MarketValue'].sum()
        portfolio['Weight'] = portfolio['MarketValue'] / total_market_value
        
        # Calculate training period returns
        train_dates = spy_train_prices.index
        train_portfolio_returns = pd.Series(index=train_dates, dtype=float)
        train_benchmark_returns = pd.Series(index=train_dates, dtype=float)
        
        for date in train_dates:
            portfolio_return = 0.0
            for idx, row in portfolio.iterrows():
                symbol = row['Symbol']
                weight = row['Weight']
                
                try:
                    ticker = yf.Ticker(symbol)
                    time.sleep(0.1)  # Small delay to avoid rate limiting
                    hist = ticker.history(period=train_period)
                    if hist.empty:
                        continue
                    prices = hist['Close']
                    
                    if date in prices.index:
                        stock_return = (prices.loc[date] / prices.iloc[0]) - 1
                        portfolio_return += weight * stock_return
                except:
                    continue
            
            train_portfolio_returns.loc[date] = portfolio_return
            train_benchmark_returns.loc[date] = (spy_train_prices.loc[date] / spy_train_prices.iloc[0]) - 1
        
        # Calculate testing period returns
        test_dates = spy_test_prices.index
        test_portfolio_returns = pd.Series(index=test_dates, dtype=float)
        test_benchmark_returns = pd.Series(index=test_dates, dtype=float)
        
        for date in test_dates:
            portfolio_return = 0.0
            for idx, row in portfolio.iterrows():
                symbol = row['Symbol']
                weight = row['Weight']
                
                try:
                    ticker = yf.Ticker(symbol)
                    time.sleep(0.1)  # Small delay to avoid rate limiting
                    hist = ticker.history(period=test_period)
                    if hist.empty:
                        continue
                    prices = hist['Close']
                    
                    if date in prices.index:
                        stock_return = (prices.loc[date] / prices.iloc[0]) - 1
                        portfolio_return += weight * stock_return
                except:
                    continue
            
            test_portfolio_returns.loc[date] = portfolio_return
            test_benchmark_returns.loc[date] = (spy_test_prices.loc[date] / spy_test_prices.iloc[0]) - 1
        
        return (train_portfolio_returns, train_benchmark_returns, test_portfolio_returns, test_benchmark_returns), (train_end, test_start)
        
    except Exception as e:
        print(f"Error calculating out-of-sample performance: {e}")
        print("Skipping out-of-sample analysis due to rate limiting or data issues")
        return None, None

def plot_out_of_sample_performance(train_data, split_dates):
    """Plot out-of-sample performance like experiment 2."""
    if train_data is None:
        print("No data available for out-of-sample plot")
        return
    
    train_portfolio, train_benchmark, test_portfolio, test_benchmark = train_data
    train_end, test_start = split_dates
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate cumulative returns
    train_portfolio_cum = (1 + train_portfolio).cumprod()
    train_benchmark_cum = (1 + train_benchmark).cumprod()
    test_portfolio_cum = (1 + test_portfolio).cumprod()
    test_benchmark_cum = (1 + test_benchmark).cumprod()
    
    # Plot training period
    ax.plot(train_portfolio_cum.index, train_portfolio_cum.values, 
            label='Portfolio (Training)', linewidth=2, color='blue')
    ax.plot(train_benchmark_cum.index, train_benchmark_cum.values, 
            label='Benchmark (Training)', linewidth=2, color='purple', linestyle='--')
    
    # Plot testing period (continue from training end)
    test_portfolio_normalized = test_portfolio_cum * train_portfolio_cum.iloc[-1]
    test_benchmark_normalized = test_benchmark_cum * train_benchmark_cum.iloc[-1]
    
    ax.plot(test_portfolio_cum.index, test_portfolio_normalized.values, 
            label='Portfolio (Testing)', linewidth=2, color='green')
    ax.plot(test_benchmark_cum.index, test_benchmark_normalized.values, 
            label='Benchmark (Testing)', linewidth=2, color='orange', linestyle='--')
    
    # Add vertical line at split point
    ax.axvline(x=train_end, color='red', linestyle=':', linewidth=2, label='Train/Test Split')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (Normalized)')
    ax.set_title('Portfolio vs Benchmark Out-of-Sample Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('portfolio_out_of_sample.png', dpi=150)
    print("Portfolio out-of-sample chart saved as portfolio_out_of_sample.png")
    plt.close()

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
    """Print hybrid trading recommendations with signal trends and restriction checks."""
    print("\n" + "=" * 140)
    print("HYBRID TRADING RECOMMENDATIONS FOR NEXT MONTH")
    print("=" * 140)
    
    # Format the recommendations table
    print(f"{'Symbol':<10} {'Buy Score':<12} {'Sell Score':<12} {'Manual':<12} {'Q-Learner':<12} {'RandomForest':<12} {'Final Rec':<15} {'Restriction':<20}")
    print("-" * 130)
    
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
        
        # Calculate final recommendation
        buy_signal = buy_score > 55
        sell_signal = sell_score > 55
        
        if manual_rec == "BUY" and ql_rec == "BUY":
            final_rec = "STRONG BUY"
            action_to_check = "BUY"
        elif manual_rec == "SELL" and ql_rec == "SELL":
            final_rec = "STRONG SELL"
            action_to_check = "SELL"
        elif buy_signal and manual_rec != "SELL":
            final_rec = "MODERATE BUY"
            action_to_check = "BUY"
        elif sell_signal and manual_rec != "BUY":
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
        
        print(f"{symbol:<10} {buy_score:>6.1f}/100   {sell_score:>6.1f}/100   {manual_rec:<12} {ql_rec:<12} {rf_rec:<12} {final_rec:<15} {restriction_status:<20}")
    
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

if __name__ == "__main__":
    csv_path = "/Users/king.botti/Documents/ML_for_trading/ML4T_2026Summer/portfolio.csv"
    results = analyze_portfolio(csv_path, analysis_months=6)
