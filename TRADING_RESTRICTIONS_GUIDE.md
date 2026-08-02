# Trading Restrictions System Guide

## Overview
This system enforces your 60-day trading rule: **Cannot buy and then sell at higher price, or sell and then buy at lower price within 60 days.**

## Files
- `trading_restrictions.csv` - Tracks all trading actions and restrictions
- `portfolio_analysis.py` - Contains restriction checking functions

## CSV Structure
```csv
Symbol,LastAction,LastActionDate,LastActionPrice,DaysSinceAction,RestrictionType,RestrictionEndDate,Notes
```

**Columns:**
- **Symbol**: Stock ticker
- **LastAction**: BOUGHT, SOLD, or HOLD
- **LastActionDate**: Date of last trade
- **LastActionPrice**: Price at last trade
- **DaysSinceAction**: Days since last trade (auto-calculated)
- **RestrictionType**: NONE, CANNOT_SELL_HIGHER, or CANNOT_BUY_LOWER
- **RestrictionEndDate**: When restriction expires (60 days after trade)
- **Notes**: Human-readable restriction details

## How It Works

### 1. Automatic Checking
When you get a trading recommendation, the system automatically:
- Checks if the action would violate 60-day rules
- Displays "⚠️ VIOLATION" in the recommendations table
- Prints detailed warning messages

### 2. Restriction Types

**CANNOT_SELL_HIGHER:**
- Triggered when you BUY a stock
- Prevents selling at higher price within 60 days
- Example: Bought NVDA at $200, cannot sell above $200 for 60 days

**CANNOT_BUY_LOWER:**
- Triggered when you SELL a stock  
- Prevents buying back at lower price within 60 days
- Example: Sold AAPL at $150, cannot buy below $150 for 60 days

### 3. Manual Updates
After executing a trade, manually update `trading_restrictions.csv`:

**For BUY trades:**
```csv
NVDA,BOUGHT,2024-08-01,200.00,0,CANNOT_SELL_HIGHER,2024-09-30,Bought at $200.00, cannot sell at higher price until 2024-09-30
```

**For SELL trades:**
```csv
AAPL,SOLD,2024-08-01,150.00,0,CANNOT_BUY_LOWER,2024-09-30,Sold at $150.00, cannot buy at lower price until 2024-09-30
```

## Example Workflow

### Scenario 1: Safe Trade
1. System recommends: BUY NVDA (current price $180)
2. Check restrictions: Last action was SOLD at $150, 90 days ago
3. Result: ✅ OK (restriction expired 30 days ago)

### Scenario 2: Violation Warning
1. System recommends: SELL GOOGL (current price $380)
2. Check restrictions: Last action was BOUGHT at $350, 15 days ago
3. Result: ⚠️ VIOLATION - "Bought at $350.00 15 days ago, cannot sell at higher price ($380.00) within 60 days"

### Scenario 3: Safe Trade Within Restriction
1. System recommends: SELL GOOGL (current price $340)
2. Check restrictions: Last action was BOUGHT at $350, 15 days ago
3. Result: ✅ OK (selling at lower price is allowed)

## Integration with Recommendations

The recommendations table now includes a **Restriction** column:
- **OK**: No violations
- **⚠️ VIOLATION**: Would break 60-day rule

## Functions Available

### `check_trading_restrictions(symbol, action, current_price, restrictions_df)`
Returns: `(allowed: bool, message: str)`

### `update_trading_restriction(symbol, action, price, restrictions_df, csv_path)`
Updates CSV after executing a trade

## Important Notes

1. **Manual Updates Required**: After executing trades, you must manually update the CSV
2. **Date Format**: Use YYYY-MM-DD format for dates
3. **Price Accuracy**: Use exact trade prices for accurate restriction checking
4. **Security**: `trading_restrictions.csv` is in `.gitignore` to protect your trading data

## Troubleshooting

### "No restrictions data available"
- Create `trading_restrictions.csv` with your current positions
- Set initial actions to "BOUGHT" with current purchase prices

### Incorrect restriction warnings
- Verify `LastActionDate` is correct
- Check that `LastActionPrice` matches your actual trade price
- Ensure `RestrictionEndDate` is 60 days after trade date

### Restriction not expiring
- Verify system date is correct
- Check `RestrictionEndDate` format is YYYY-MM-DD
- Ensure CSV is saved with correct data types
