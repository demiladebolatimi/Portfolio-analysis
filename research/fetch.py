import os, pickle, yfinance as yf, pandas as pd
syms = list(pd.read_csv('portfolio.csv').Symbol) + ['SPY','^VIX','^VIX3M']
os.makedirs('data_cache', exist_ok=True)
out = {}
for s in syms:
    h = yf.Ticker(s).history(period='max', auto_adjust=True)
    h.index = h.index.tz_localize(None).normalize()
    out[s] = h[['Open','High','Low','Close','Volume']]
    print(s, h.index[0].date(), len(h))
pickle.dump(out, open('data_cache/ohlcv.pkl','wb'))
