# Portfolio Creation Using Momentum Strategy

## Introduction

The goal of this assignment was to create a portfolio consisting of multiple financial instruments and evaluate its performance using a trading strategy developed in a previous task.
In this case, I reused my momentum strategy from Homework 3.

#### The assignment required:
Running the same strategy on multiple assets
Optimizing the parameters for each asset
Combining the strategies into a diversified portfolio
Plotting the combined performance
Calculating correlation between strategies

## Data and Instruments
#### I selected ten instruments from different sectors and asset classes:
   - AAPL — Apple Inc.
   - AMZN — Amazon.com, Inc.
   - MSFT — Microsoft Corporation
   - TSLA — Tesla, Inc.
   - GOOGL — Alphabet Inc.
   - BP — BP p.l.c.
   - GLD — SPDR Gold Shares ETF (Gold)
   - SPOT — Spotify Technology S.A.
   - BKNG — Booking Holdings Inc.
   - META — Meta Platforms

Historical price data was downloaded using Yahoo Finance (via yfinance library).

## Strategy Description (Momentum Strategy)
I reused the momentum strategy created in Homework 3:
Momentum = Close - Close.shift(n)

If Momentum > 0 → Signal = +1 (long)
If Momentum < 0 → Signal = -1 (short)

Daily Profit = Previous Signal × Daily Price Change – Trading Costs

#### The output of the strategy includes:
Daily returns
Daily profit
Equity curve
This logic was wrapped into a reusable function.

## Parameter Optimization

Each instrument can behave differently.
To find the best momentum window n, I used brute force optimization:
Tested multiple values and selected the parameters with the highest Sharpe ratio

Saved the best strategy results for each asset
Example output:
AAPL: best n = 20
AMZN: best n = 10
MSFT: best n = 30

This step ensures the strategy is tuned for each financial instrument.

---

## Portfolio Construction
   Two portfolio versions were created:
#### Bad Portfolio (“Bendras BLOGAS”)
   - This incorrectly sums equity curves in dollars.
   - This method is wrong because:

   - Expensive stocks dominate the result
   - Percentage performance is ignored
   - Capital allocation is unrealistic

   - The portfolio becomes price-weighted, not equal-weighted

#### Good Portfolio (“Bendras GERAS”)
   - An equal-weight portfolio using percentage returns:
   - portfolio_returns = mean of all strategy returns (each day)
   - portfolio_equity = cumulative product of (1 + portfolio_returns)

   - This method:
   - Allocates equal capital to each strategy
   - Represents real portfolio behavior
   - Produces fair and comparable performance
   - This is the correct way to combine strategies.

## Correlation Analysis
   I calculated the correlation matrix of daily strategy returns.
   This shows how similarly or differently strategies behave:
   Most correlations were low (–0.10 to +0.30) - this means good diversification.

Momentum strategies across different assets often have low correlation
The portfolio benefits from mixing independent strategies
Correlation was visualized using a heatmap.

## Visualizations
   The following charts were generated:
   Individual price charts for all 10 assets
   Bad portfolio total equity (incorrect)
   Good portfolio equity (correct)
   Correlation matrix heatmap
   Equity curves of individual strategies vs portfolio
   These help illustrate diversification, performance differences, and why equal-weighting matters.

## Conclusion
   This project demonstrates:
   Reuse of a strategy from previous work
   Ability to optimize strategy parameters
   Portfolio construction skills
   Understanding of correct vs. incorrect methods
   Interpretation of correlations
   Visualization of results
   The equal-weight momentum portfolio shows smoother growth and less risk compared to individual strategies, thanks to low correlation and diversified exposure.
