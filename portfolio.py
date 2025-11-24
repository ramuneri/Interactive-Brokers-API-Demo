import math
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt


def momentum_strategy(price_series, n, commission):
    df = price_series.to_frame(name="Close").copy()

    df["Momentum"] = df["Close"] - df["Close"].shift(n)

    df["Signal"] = 0
    df.loc[df["Momentum"] > 0, "Signal"] = 1
    df.loc[df["Momentum"] < 0, "Signal"] = -1

    # Daily profit calculation
    df["Trade_cost"] = 0.0
    df.loc[df["Signal"].diff() != 0, "Trade_cost"] = df["Close"] * commission

    df["Daily_change"] = df["Close"].diff().fillna(0)
    df["Daily_profit"] = (df["Daily_change"] * df["Signal"].shift(1).fillna(0) - df["Trade_cost"])

    # Daily profit in % (needed for portfolio math)
    df["Return_pct"] = df["Daily_profit"] / df["Close"].shift(1)
    df["Return_pct"] = df["Return_pct"].replace([np.inf, -np.inf], 0).fillna(0)

    # Cumulative return curve
    df["Equity"] = (1 + df["Return_pct"]).cumprod()

    return df


def optimize_momentum_strategy(price_series, n_values, commission):
    best_sharpe = -999
    best_n = 0
    best_result = 0

    for n in n_values:
        df = momentum_strategy(price_series, n, commission)

        r = df["Return_pct"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) < 30:
            continue

        sharpe = (r.mean() / r.std()) * np.sqrt(252)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_n = n
            best_result = df

    return best_result, best_n, best_sharpe


tickers = ["AAPL", "AMZN", "MSFT", "TSLA", "GOOGL", "BP", "GLD", "SPOT", "BKNG", "META"]

start="2010-01-01"
end="2020-01-01"

n_values = list(range(5, 251, 5))
commission = 0.001

all_returns = []
results = {}

print("\n==================== OPTIMIZATION RESULTS ====================\n")
for t in tickers:

    data = yf.download(t, start=start, end=end, auto_adjust=True)
    price = data["Close"].astype(float).squeeze()

    best_df, best_n, best_sharpe = optimize_momentum_strategy(price, n_values, commission)

    print(f"{t}: best n = {best_n}, Sharpe = {best_sharpe:.3f}")

    results[t] = best_df
    all_returns.append(best_df["Return_pct"].rename(t))

# Combine daily strategy returns
returns_df = pd.concat(all_returns, axis=1).dropna()




portfolio_returns = returns_df.mean(axis=1)
portfolio_equity = (1 + portfolio_returns).cumprod()



# CORRELATION MATRIX
corr_matrix = returns_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Momentum Strategy Returns")
plt.show()


#  PRICE CHARTS and TOTALS
num_plots = len(tickers) + 2
cols = 4
rows = math.ceil(num_plots / cols)

plt.figure(figsize=(18, 12))

# 1. Plot each stock price
for i, t in enumerate(tickers):
    plt.subplot(rows, cols, i + 1)
    price_data = yf.download(t, start=start, end=end, auto_adjust=True)
    plt.plot(price_data["Close"])
    plt.title(t)
    plt.grid(True)
    plt.xticks(rotation=25)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))


# 2. BAD TOTAL (sum of strategy equity curves — WRONG METHOD)
bad_total = (1 + returns_df).cumprod().sum(axis=1)

plt.subplot(rows, cols, len(tickers) + 1)
plt.plot(bad_total)
plt.title("Bendras BLOGAS (neteisinga suma)")
plt.grid(True)

# 3. GOOD TOTAL
plt.subplot(rows, cols, len(tickers) + 2)
plt.plot(portfolio_equity)
plt.title("Bendras GERAS (lygios proporcijos)")
plt.grid(True)

plt.tight_layout()
plt.show()

