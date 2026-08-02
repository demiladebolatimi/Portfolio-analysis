"""Portfolio data loading and management functions."""

import pandas as pd
from datetime import datetime, timedelta
import config


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
        import yfinance as yf
        import time
        ticker = yf.Ticker(symbol)
        # Use period instead of dates for more reliable data retrieval
        days = (end_date - start_date).days
        period = f"{days}d" if days > 0 else "1mo"
        time.sleep(0.3)  # Delay to avoid rate limiting
        hist = ticker.history(period=period)
        return hist
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


def update_portfolio_after_trade(symbol, new_shares, current_price, portfolio_path):
    """Update portfolio.csv after executing a trade."""
    try:
        portfolio = load_portfolio(portfolio_path)
        
        if symbol in portfolio['Symbol'].values:
            # Update existing position
            if new_shares == 0:
                # Keep stock with 0 shares for future analysis
                portfolio.loc[portfolio['Symbol'] == symbol, 'Shares'] = 0
                portfolio.loc[portfolio['Symbol'] == symbol, 'MarketValue'] = 0
                portfolio.loc[portfolio['Symbol'] == symbol, 'PortfolioWeight'] = 0
                print(f"✅ Updated {symbol} to 0 shares (kept for future analysis)")
            else:
                # Update with new shares
                portfolio.loc[portfolio['Symbol'] == symbol, 'Shares'] = new_shares
                new_market_value = new_shares * current_price
                portfolio.loc[portfolio['Symbol'] == symbol, 'MarketValue'] = new_market_value
                print(f"✅ Updated {symbol} to {new_shares} shares at ${current_price:.2f}")
        else:
            # Add new position
            new_row = {
                'Symbol': symbol,
                'Shares': new_shares,
                'PortfolioWeight': 0.0,  # Will be recalculated
                'CostBasis': new_shares * current_price,
                'MarketValue': new_shares * current_price
            }
            portfolio = pd.concat([portfolio, pd.DataFrame([new_row])], ignore_index=True)
            print(f"✅ Added new position {symbol} with {new_shares} shares at ${current_price:.2f}")
        
        # Recalculate portfolio weights
        total_value = portfolio['MarketValue'].sum()
        if total_value > 0:
            portfolio['PortfolioWeight'] = portfolio['MarketValue'] / total_value
        
        # Save updated portfolio
        portfolio.to_csv(portfolio_path, index=False)
        return portfolio
        
    except Exception as e:
        print(f"❌ Error updating portfolio: {e}")
        return None


def maintain_sold_stocks(portfolio_path, symbols_to_keep=None):
    """Ensure sold stocks remain in portfolio.csv for future analysis."""
    try:
        portfolio = load_portfolio(portfolio_path)
        
        if symbols_to_keep is None:
            # Keep all stocks currently in portfolio
            symbols_to_keep = portfolio['Symbol'].tolist()
        
        # If config says to keep sold stocks, ensure 0-share stocks remain
        if config.KEEP_SOLD_STOCKS:
            # This is a placeholder for future enhancement
            # Currently, the system keeps all stocks in portfolio.csv
            pass
        
        return portfolio
        
    except Exception as e:
        print(f"❌ Error maintaining sold stocks: {e}")
        return None
