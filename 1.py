import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def momentum_strategy(price_series, n, commission):
    df = price_series.to_frame(name="Close").copy()

    df["Momentum"] = df["Close"] - df["Close"].shift(n)

    df["Signal"] = 0
    df.loc[df["Momentum"] > 0, "Signal"] = 1
    df.loc[df["Momentum"] < 0, "Signal"] = -1

    df["Trade_cost"] = 0.0
    df.loc[df["Signal"].diff() != 0, "Trade_cost"] = df["Close"] * commission

    df["Daily_change"] = df["Close"].diff().fillna(0)

    df["Daily_profit"] = (
        df["Daily_change"] * df["Signal"].shift(1).fillna(0)
        - df["Trade_cost"]
    )

    df["Strategy_return"] = df["Daily_profit"] / df["Close"].shift(1)
    df["Strategy_return"] = df["Strategy_return"].fillna(0)

    df["Equity"] = (1 + df["Strategy_return"]).cumprod()

    return df



def optimize_momentum_strategy(price_series, n_values, commission):
    best_sharpe = -999
    best_n = None
    best_result = None

    for n in n_values:
        df = momentum_strategy(price_series, n, commission)

        r = df["Strategy_return"]

        r = r.replace([np.inf, -np.inf], np.nan).dropna()

        if len(r) < 30:
            continue

        sharpe = (r.mean() / r.std()) * np.sqrt(252)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_n = n
            best_result = df

    return best_result, best_n, best_sharpe


# --- TESTING BLOCK ---
ticker = "AAPL"

data = yf.download(ticker, start="2020-01-01", end="2025-01-01", interval="1d", auto_adjust=True)
price = data["Close"].astype(float).squeeze()

n_values = [3, 5, 10, 20, 30]
commission = 0.001

test_df = momentum_strategy(price, n=5, commission=commission)
print("Momentum strategy test output:")
print(test_df.head())

best_df, best_n, best_sharpe = optimize_momentum_strategy(price, n_values, commission)

print("Optimization results:")
print("Best n =", best_n)
print("Best Sharpe =", best_sharpe)
print(best_df.tail())

plt.figure(figsize=(10,5))
plt.plot(best_df["Equity"])
plt.title(f"Momentum Strategy Equity Curve (Best n={best_n})")
plt.show()
