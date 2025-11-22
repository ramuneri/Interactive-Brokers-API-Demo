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


# ============================
# STEP 3 — Run for many tickers
# ============================

tickers = ["AAPL", "AMZN", "MSFT", "TSLA", "GOOGL"]

all_returns = []
results = {}

n_values = [3, 5, 10, 20, 30, 50]
commission = 0.001

for t in tickers:
    print(f"\n=== Optimizing {t} ===")
    data = yf.download(t, start="2020-01-01", end="2025-01-01", interval="1d", auto_adjust=True)
    price = data["Close"].astype(float)
    price = price.iloc[:, 0] if isinstance(price, pd.DataFrame) else price


    best_df, best_n, best_sharpe = optimize_momentum_strategy(price, n_values, commission)

    print(f"Best n for {t}: {best_n}, Sharpe={best_sharpe:.3f}")

    results[t] = best_df

    all_returns.append(best_df["Strategy_return"].rename(t))

returns_df = pd.concat(all_returns, axis=1).dropna()
print("\nCombined returns data:")
print(returns_df.head())

