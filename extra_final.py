import math
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt

tickers = ["AAPL", "META", "TSLA"]
start = "2018-01-01"
end = "2024-01-01"


def ma_strategy(price_series):
    df = price_series.to_frame(name="Close").copy()

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    df["Signal"] = 0
    df.loc[df["MA20"] > df["MA50"], "Signal"] = 1

    df["Return"] = df["Close"].pct_change()
    df["Strategy_return"] = (df["Return"] * df["Signal"].shift(1)).fillna(0)

    df["Equity"] = (1 + df["Strategy_return"]).cumprod()
    return df


results = {}
all_returns = []

for t in tickers:
    data = yf.download(t, start=start, end=end, auto_adjust=True)
    price = data["Close"].astype(float).squeeze()

    df = ma_strategy(price)
    results[t] = df
    all_returns.append(df["Strategy_return"].rename(t))


returns_df = pd.concat(all_returns, axis=1).dropna()

cols = 4
rows = 1 

plt.figure(figsize=(18, 4))
for i, t in enumerate(tickers):
    plt.subplot(rows, cols, i + 1)
    plt.plot(results[t]["Equity"])
    plt.title(f"{t}")
    plt.grid(True)

plt.tight_layout()



portfolio_returns = returns_df.mean(axis=1)
portfolio_equity = (1 + portfolio_returns).cumprod()

plt.subplot(rows, cols, len(tickers)+1)
plt.plot(portfolio_equity)
plt.title("Equal-Weight Portfolio Equity")
plt.grid(True)
plt.show()

corr = returns_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("SMA Strategy Return Correlation Matrix")
plt.show()
