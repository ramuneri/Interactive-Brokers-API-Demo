import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math

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
    df["Strategy_return"] = df["Signal"].shift(1) * df["Return"]
    df["Strategy_return"] = df["Strategy_return"].fillna(0)

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

# 4 plots total (3 price charts + 1 portfolio equity chart)
cols = 4
rows = 1

plt.figure(figsize=(18, 4))  # wider & shorter

# PRICE CHARTS
for i, t in enumerate(tickers):
    plt.subplot(rows, cols, i + 1)
    price_data = yf.download(t, start=start, end=end, auto_adjust=True)
    plt.plot(price_data["Close"])
    plt.title(t)
    plt.grid(True)
    plt.xticks(rotation=25)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))

# GOOD PORTFOLIO (equal-weight) in subplot #4
plt.subplot(rows, cols, len(tickers) + 1)  # this becomes position 4
plt.plot(portfolio_equity)
plt.title("Bendras GERAS")
plt.grid(True)

plt.tight_layout()
plt.show()


# CORRELATION MATRIX 
corr = returns_df.corr()

plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(tickers)), tickers)
plt.yticks(range(len(tickers)), tickers)
plt.title("Strategijų koreliacija")
plt.show()
