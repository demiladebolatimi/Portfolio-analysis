"""Compare old live signal engine vs signal_engine v2 vs buy & hold (long-only, next-day execution)."""
import pickle, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from technical_indicators import calculate_rsi, calculate_momentum, calculate_macd
from signal_engine import compute_indicator_frame, compute_signal_scores

COST = 0.001  # 10 bps per unit of turnover
d = pickle.load(open('data_cache/ohlcv.pkl', 'rb'))
spy, vix, vix3m = d['SPY'].Close, d['^VIX'].Close, d['^VIX3M'].Close
syms = list(pd.read_csv('portfolio.csv').Symbol)


def old_strength(c, ma_window=200):
    """Vectorised replica of calculate_daily_buy/sell_scores + regime multiplier in portfolio_analysis_modular."""
    rsi, mom, macd = calculate_rsi(c), calculate_momentum(c), calculate_macd(c)
    # The live job only fetched ~6 months of prices, so its MA200 was always NaN (trend bias never applied).
    ma200 = c.rolling(ma_window).mean() if ma_window else pd.Series(np.nan, index=c.index)
    buy = ((70 - rsi) / 40 * 100).clip(0, 100) * .35 + ((mom + .05) / .1 * 100).clip(0, 100) * .25 \
        + ((macd + 2) / 4 * 100).clip(0, 100) * .25 + 15 * (c > ma200)
    sell = ((rsi - 30) / 40 * 100).clip(0, 100) * .35 + ((.05 - mom) / .1 * 100).clip(0, 100) * .25 \
        + ((0 - macd) / 2 * 100).clip(0, 100) * .25 + 15 * (c < ma200)
    s = spy.reindex(c.index).ffill(); v = vix.reindex(c.index).ffill()
    sma = s.rolling(200).mean()
    mult = pd.Series(1.0, index=c.index).mask((s > sma) & (v < 25), 1.2).mask((s < sma) & (v > 30), 0.7)
    return (buy.clip(0, 100) - sell.clip(0, 100)) * mult


def discrete_policy(strength):
    """STRONG BUY->1, MODERATE BUY->0.75, HOLD->keep, MODERATE SELL->0.5 cap, STRONG SELL->0."""
    pos, out = 1.0, []
    for x in strength.values:
        if np.isnan(x): out.append(np.nan); continue
        if x > 20: pos = 1.0
        elif x > 10: pos = max(pos, 0.75)
        elif x < -20: pos = 0.0
        elif x < -10: pos = min(pos, 0.5)
        out.append(pos)
    return pd.Series(out, index=strength.index)


def run(exposure, c):
    r = c.pct_change()
    e = exposure.shift(1)  # signal at close t -> held over t+1
    return (e * r - e.diff().abs() * COST).fillna(0)


def metrics(r):
    r = r.dropna()
    eq = (1 + r).cumprod(); yrs = len(r) / 252
    dd = (eq / eq.cummax() - 1).min()
    return {'CAGR': eq.iloc[-1] ** (1 / yrs) - 1, 'Vol': r.std() * np.sqrt(252),
            'Sharpe': r.mean() / r.std() * np.sqrt(252), 'MaxDD': dd,
            'Calmar': (eq.iloc[-1] ** (1 / yrs) - 1) / abs(dd)}


def main(start, end):
    sleeves = {k: {} for k in ['BuyHold', 'Old_live', 'Old', 'V2_discrete', 'V2_sized']}
    per_sym = []
    for s in syms:
        df = d[s]; c = df.Close
        frame = compute_indicator_frame(df, spy, vix, vix3m)
        sc = compute_signal_scores(frame)
        ex = {
            'BuyHold': pd.Series(1.0, index=c.index),
            'Old_live': discrete_policy(old_strength(c, None).where(c.rolling(200).count() == 200)),
            'Old': discrete_policy(old_strength(c).where(c.rolling(200).count() == 200)),
            'V2_discrete': discrete_policy(2 * sc['Composite'] - 100),
            'V2_sized': sc['TargetExposure'],
        }
        for k, e in ex.items():
            sleeves[k][s] = run(e, c).loc[start:end]
        row = {'Symbol': s}
        for k in ex:
            m = metrics(sleeves[k][s]) if len(sleeves[k][s]) > 252 else {'Sharpe': np.nan, 'CAGR': np.nan}
            row[f'{k}_Sharpe'] = m['Sharpe']
        per_sym.append(row)
    print(f'\n=== {start} -> {end}: equal-weight portfolio of {len(syms)} holdings ===')
    table = {}
    for k, v in sleeves.items():
        port = pd.DataFrame(v).mean(axis=1, skipna=True)
        table[k] = metrics(port)
    print(pd.DataFrame(table).T.round(3).to_string())
    ps = pd.DataFrame(per_sym).set_index('Symbol').round(2)
    print('\nPer-symbol Sharpe:'); print(ps.to_string())
    return pd.DataFrame(table).T, ps


if __name__ == '__main__':
    for a, b in [('2012-01-01', '2019-12-31'), ('2020-01-01', '2026-12-31'), ('2012-01-01', '2026-12-31')]:
        main(a, b)
