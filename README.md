# Stock_market_Analysis

Stock is something a person invest into and gains a share or own that company.
Some companies provide dividends but in stock there is no gaurantee of return.
Stocks of 4 companies MSFT, INTC, AAPL and GOOGL have been analysed.
The source of the data is Yahoo finance.
These are the steps which are taken in order to carry the analysis.

Resampled the data Quarterly and weekly
Analyzed open and close differences and percent change in adjusted close values for each stock to compare them
Computed variance for volatility, used rolling means (window size 42 and 252) for trends and seasonality, plotted all cases
Packages- Pandas, NumPy, Matplotlib, Seaborn

Resampling (Quarterly)

doesn't give good idea of which stock is doing better thats why  concept of moving averages is used.

The concept of moving window is to slide a window through the data to see long term fluctuations in data and ignore short term fluctuations.
Volatality of the stock is computed.
Volatility of stock is a measurement of the change in variance in the returns over a specific period of time.
More volatile more risk.

Rolling means is used to determine  trends (window size = 42) and seasonality (window size = 252)
Trends are for short time investment and seasonality is for long term investment.
Please refer to the word document for more information.




