import pickle, numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
from technical_indicators import *
d = pickle.load(open('data_cache/ohlcv.pkl','rb'))
spy = d['SPY'].Close; vix = d['^VIX'].Close; vix3m = d['^VIX3M'].Close
syms = [s for s in pd.read_csv('portfolio.csv').Symbol]
H = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rows = []
for s in syms:
    df = d[s]; c, h, l, v = df.Close, df.High, df.Low, df.Volume
    sp = spy.reindex(c.index).ffill()
    adx, pdi, mdi = calculate_adx(h, l, c)
    f = pd.DataFrame({
        'rsi14': calculate_rsi(c), 'mom14': calculate_momentum(c),
        'macd_raw': calculate_macd(c), 'macd_hist_pct': calculate_macd_histogram_pct(c),
        'rs120': calculate_relative_strength(c, sp, 120), 'rs60': calculate_relative_strength(c, sp, 60),
        'dist_ma200': calculate_distance_from_ma(c, 200), 'dist_ma50': calculate_distance_from_ma(c, 50),
        'bb_pos': calculate_bollinger_band_position(c), 'vol_ratio': calculate_volume_ratio(v),
        'adx_dir': adx * np.sign(pdi - mdi), 'cmf': calculate_chaikin_money_flow(h, l, c, v),
        'obv_slope': calculate_obv_slope(c, v), 'mom12_1': calculate_momentum_12_1(c),
        'hi52': calculate_52w_high_proximity(c), 'ma200_slope': calculate_ma_slope(c),
        'rvol': calculate_realized_volatility(c), 'rsi2': calculate_rsi(c, 2),
        'vix': vix.reindex(c.index).ffill(), 'vix_ts': calculate_vix_term_structure(vix, vix3m).reindex(c.index).ffill(),
        'spy_dist200': calculate_distance_from_ma(sp, 200),
    })
    f['fwd'] = c.shift(-H) / c - 1
    f['sym'] = s
    rows.append(f.loc['2012-01-01':].dropna(subset=['fwd']))
F = pd.concat(rows)
# rank IC per date then average (cross-sectional) and pooled time-series IC per symbol
feats = [x for x in F.columns if x not in ('fwd','sym')]
ts_ic = F.groupby('sym').apply(lambda g: g[feats].corrwith(g['fwd'], method='spearman')).mean()
xs = F.reset_index().rename(columns={'index':'Date'})
xs = xs[xs.groupby('Date').sym.transform('count') >= 10]
xs_ic = xs.groupby('Date').apply(lambda g: g[feats].corrwith(g['fwd'], method='spearman')).mean()
print(pd.DataFrame({'timeseries_IC': ts_ic, 'cross_sectional_IC': xs_ic}).round(3).sort_values('timeseries_IC'))
