"""Walk-forward test of the live RandomForest 'Q-Learner' probability as a signal.

Retrains every 6 months on the trailing 5 years (purged 5-day gap), predicts the next 6 months.
Reports AUC / hit-rate and the effect of adding the +/-20 ML bias to the old heuristic signal.
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from technical_indicators import *
exec(open('research/compare.py').read().split("if __name__")[0])


def live_features(df, sp):
    c, h, l, v = df.Close, df.High, df.Low, df.Volume
    s20, s60 = calculate_market_regime(sp)
    return pd.DataFrame({
        'RSI': calculate_rsi(c), 'Momentum': calculate_momentum(c), 'MACD_Histogram': calculate_macd_histogram(c),
        'RS_20D': calculate_relative_strength(c, sp, 20), 'RS_60D': calculate_relative_strength(c, sp, 60),
        'RS_120D': calculate_relative_strength(c, sp, 120), 'Volume_Ratio': calculate_volume_ratio(v),
        'OBV': calculate_obv(c, v), 'Accum_Dist': calculate_accumulation_distribution(h, l, c, v),
        'ATR': calculate_atr(h, l, c), 'Distance_MA50': calculate_distance_from_ma(c, 50),
        'Distance_MA200': calculate_distance_from_ma(c, 200), 'BB_Position': calculate_bollinger_band_position(c),
        'SPY_Return_20D': s20, 'SPY_Return_60D': s60,
    }).replace([np.inf, -np.inf], np.nan)


def walk_forward_proba(X, y, start='2013-01-01'):
    out = pd.Series(np.nan, index=X.index)
    base = pd.Series(np.nan, index=X.index)
    dates = pd.date_range(start, X.index[-1] + pd.offsets.MonthBegin(), freq='6MS')
    for t0, t1 in zip(dates[:-1], dates[1:]):
        tr = (X.index < t0 - pd.Timedelta(days=8)) & (X.index >= t0 - pd.DateOffset(years=5))
        te = (X.index >= t0) & (X.index < t1)
        m = tr & X.notna().all(axis=1) & y.notna()
        if m.sum() < 500 or te.sum() == 0: continue
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=10, random_state=42, n_jobs=-1)
        rf.fit(X[m], y[m])
        ok = te & X.notna().all(axis=1)
        if ok.any():
            out[ok] = rf.predict_proba(X[ok])[:, 1]
            base[ok] = y[m].mean()
    return out, base


aucs, sl = [], {k: {} for k in ['BuyHold', 'Old', 'Old+ML', 'Old+MLcentered', 'V2', 'V2+MLcentered']}
for s in syms:
    df = d[s]; c = df.Close; sp = spy.reindex(c.index).ffill()
    X = live_features(df, sp)
    fwd = c.pct_change(5).shift(-5)
    y = (fwd > 0.02).astype(float).where(fwd.notna())
    p, base = walk_forward_proba(X, y)
    ok = p.notna() & y.notna()
    if ok.sum() > 200 and y[ok].nunique() == 2:
        aucs.append({'Symbol': s, 'AUC': roc_auc_score(y[ok], p[ok]), 'BaseRate': y[ok].mean(), 'MeanProba': p[ok].mean()})
    old = old_strength(c).where(c.rolling(200).count() == 200)
    centered = ((p - base) * 100).clip(-20, 20)
    v2 = 2 * compute_signal_scores(compute_indicator_frame(df, spy, vix, vix3m))['Composite'] - 100
    variants = {'BuyHold': None, 'Old': old, 'Old+ML': old + (p - 0.5) * 40, 'Old+MLcentered': old + centered,
                'V2': v2, 'V2+MLcentered': v2 + centered}
    for k, strength in variants.items():
        e = pd.Series(1.0, index=c.index) if strength is None else discrete_policy(strength.where(p.notna()))
        sl[k][s] = run(e, c).loc['2013-07-01':]
A = pd.DataFrame(aucs).set_index('Symbol').round(3)
print(A.to_string()); print('mean AUC', A.AUC.mean().round(3))
A.to_csv('data_cache/auc.csv')
print(pd.DataFrame({k: metrics(pd.DataFrame(v).mean(axis=1)) for k, v in sl.items()}).T.round(3))
