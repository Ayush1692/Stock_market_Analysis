#Name: Ayush Kumar
#Stock market analysis


#step1: Import datasets

import pandas as pd
#intc = pd.read_csv("INTC.csv")


#parse date as index
intc = pd.read_csv('INTC.csv', header = 0, index_col = 'Date', parse_dates = True)

#msft = pd.read_csv("MSFT.csv")

msft = pd.read_csv('MSFT.csv', header = 0, index_col = 'Date', parse_dates = True)

#googl = pd.read_csv("GOOGL.csv")

googl = pd.read_csv('GOOGL.csv', header = 0, index_col = 'Date', parse_dates = True)

#aapl = pd.read_csv("AAPL.csv")

aapl = pd.read_csv('AAPL.csv', header = 0, index_col = 'Date', parse_dates = True)



intc.head()
intc.tail()

msft.head()
msft.tail()

googl.head()
googl.tail()

aapl.head()
aapl.tail()

intc.describe()
msft.describe()
googl.describe()
aapl.describe()


intc.columns

intc.index

intc.shape


import matplotlib.pyplot as plt
    


import matplotlib.dates as mdates

plt.plot(intc.index , intc['Adj Close'])

#Format yearly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#locate year
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
#shows grids
plt.grid(True)
#rotates the text to 90 degree
plt.xticks(rotation = 90)
plt.show()


#Subplots 

f, ax = plt.subplots(2,2, figsize = (10,10), sharex = True)
f.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
f.gca().xaxis.set_major_locator(mdates.YearLocator())

ax[0,0].plot(msft.index, msft['Adj Close'], color = 'r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation = 90)
ax[0,0].set_title('MSFT');

ax[0,1].plot(googl.index, googl['Adj Close'], color = 'g')
ax[0,1].grid(True)
#tick params for subplots
ax[0,1].tick_params(labelrotation = 90)
ax[0,1].set_title('GOOGL');

ax[1,0].plot(aapl.index, aapl['Adj Close'], color = 'b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation = 90)
ax[1,0].set_title('AAPL');

ax[1,1].plot(intc.index, intc['Adj Close'], color = 'y')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation = 90)
ax[1,1].set_title('INTC');




#Zooming-in

msft_18 = msft.loc[pd.Timestamp('2018-01-01'):pd.Timestamp('2018-12-31')]
plt.plot(msft_18.index, msft_18['Adj Close'])
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation = 90)
plt.show()

#Subplots

f, ax = plt.subplots(2,2, figsize = (10,10), sharex = True, sharey=True)
f.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
f.gca().xaxis.set_major_locator(mdates.YearLocator())

msft_18 = msft.loc[pd.Timestamp('2018-01-01'):pd.Timestamp('2018-12-31')]

#'.' delimiter
ax[0,0].plot(msft_18.index, msft_18['Adj Close'], '.' , color = 'r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation = 90)
ax[0,0].set_title('MSFT');

googl_18 = googl.loc[pd.Timestamp('2018-01-01'):pd.Timestamp('2018-12-31')]
ax[0,1].plot(googl_18.index, googl_18['Adj Close'], '.' , color = 'g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation = 90)
ax[0,1].set_title('GOOGL');


aapl_18 = aapl.loc[pd.Timestamp('2018-01-01'):pd.Timestamp('2018-12-31')]
ax[1,0].plot(aapl_18.index, aapl_18['Adj Close'], '.' , color = 'b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation = 90)
ax[1,0].set_title('AAPL');

intc_18 = intc.loc[pd.Timestamp('2018-01-01'):pd.Timestamp('2018-12-31')]
ax[1,1].plot(intc_18.index, intc_18['Adj Close'], '.' , color = 'y')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation = 90)
ax[1,1].set_title('INTC');


#Step2 :Resampling (Quaterly)

#mean of 4 months
monthly_msft_18 = msft_18.resample('4M').mean()
plt.scatter(monthly_msft_18.index, monthly_msft_18['Adj Close'])
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation = 90)
plt.show()

#Subplots

f, ax = plt.subplots(2,2, figsize = (10,10), sharex = True, sharey=True)


monthly_msft_18 = msft_18.resample('4M').mean()
ax[0,0].scatter(monthly_msft_18.index, monthly_msft_18['Adj Close'], color = 'r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation = 90)
ax[0,0].set_title('MSFT');

monthly_googl_18 = googl_18.resample('4M').mean()
ax[0,1].scatter(monthly_googl_18.index, monthly_googl_18['Adj Close'],  color = 'g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation = 90)
ax[0,1].set_title('GOOGL');


monthly_aapl_18 = aapl_18.resample('4M').mean()
ax[1,0].scatter(monthly_aapl_18.index, monthly_aapl_18['Adj Close'],  color = 'b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation = 90)
ax[1,0].set_title('AAPL');

monthly_intc_18 =  intc_18.resample('4M').mean()
ax[1,1].scatter(monthly_intc_18.index, monthly_intc_18['Adj Close'],  color = 'y')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation = 90)
ax[1,1].set_title('INTC');


#Resampling (Weekly)

intc_19 = intc.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]

weekly_intc_19 = intc_19.resample('W').mean()
weekly_intc_19.head()
#-o bigger notation for dot
plt.plot(weekly_intc_19.index, weekly_intc_19['Adj Close'], '-o')
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation = 90)
plt.show()


#Subplots
msft_19 = msft.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_msft_19 = msft_19.resample('W').mean()

googl_19 = googl.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_googl_19 = googl_19.resample('W').mean()

aapl_19 = aapl.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_aapl_19 = aapl_19.resample('W').mean()

intc_19 = intc.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_intc_19 = intc_19.resample('W').mean()



f, ax = plt.subplots(2,2, figsize = (10,10), sharex = True, sharey=True)

ax[0,0].plot(weekly_msft_19.index, weekly_msft_19['Adj Close'], '-o' , color = 'r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation = 90)
ax[0,0].set_title('MSFT');

ax[0,1].plot(weekly_googl_19.index, weekly_googl_19['Adj Close'], '-o' , color = 'g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation = 90)
ax[0,1].set_title('GOOGL');

ax[1,0].plot(weekly_aapl_19.index, weekly_aapl_19['Adj Close'], '-o' , color = 'b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation = 90)
ax[1,0].set_title('AAPL');

ax[1,1].plot(weekly_intc_19.index, weekly_intc_19['Adj Close'], '-o' , color = 'y')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation = 90)
ax[1,1].set_title('INTC');



#Analysing difference between levels(Resampling weekly)

msft['diff'] = msft['Open'] - msft['Close']
msft_diff = msft.resample('W').mean()
msft_diff.tail(10)

plt.scatter(msft_diff.loc['2019-01-15' : '2019-09-15'].index, msft_diff.loc['2019-01-15' : '2019-09-15']['diff'])

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation = 90)
plt.show()


#Subplots
msft['diff'] = msft['Open'] - msft['Close']
msft_diff = msft.resample('W').mean()

googl['diff'] = googl['Open'] - googl['Close']
googl_diff = googl.resample('W').mean()

aapl['diff'] = aapl['Open'] - aapl['Close']
aapl_diff = aapl.resample('W').mean()

intc['diff'] = intc['Open'] - intc['Close']
intc_diff = intc.resample('W').mean()

f, ax = plt.subplots(2,2, figsize = (10,10), sharex = True, sharey=True)

ax[0,0].scatter(msft_diff.loc['2019-01-15' : '2019-09-15'].index, msft_diff.loc['2019-01-15' : '2019-09-15']['diff'], color = 'r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation = 90)
ax[0,0].set_title('MSFT');

ax[0,1].scatter(googl_diff.loc['2019-01-15' : '2019-09-15'].index, googl_diff.loc['2019-01-15' : '2019-09-15']['diff'],  color = 'g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation = 90)
ax[0,1].set_title('GOOGL');

ax[1,0].scatter(aapl_diff.loc['2019-01-15' : '2019-09-15'].index, aapl_diff.loc['2019-01-15' : '2019-09-15']['diff'], color = 'b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation = 90)
ax[1,0].set_title('AAPL');

ax[1,1].scatter(intc_diff.loc['2019-01-15' : '2019-09-15'].index, intc_diff.loc['2019-01-15' : '2019-09-15']['diff'], color = 'r')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation = 90)
ax[1,1].set_title('INTC');


#Step 3 Moving windows
#Daily percentages

daily_close_msft = msft[['Adj Close']]

#Daily returns
daily_pct_change_msft = daily_close_msft.pct_change()

#Replace NA values with 0
daily_pct_change_msft.fillna(0, inplace = True)

daily_pct_change_msft.head()

daily_pct_change_msft.hist(bins = 50)

# Show the plot
plt.show()



daily_close_googl = googl[['Adj Close']]

# Daily returns
daily_pct_change_googl = daily_close_googl.pct_change()

# Replace NA values with 0
daily_pct_change_googl.fillna(0, inplace=True)

daily_close_intc = intc[['Adj Close']]

# Daily returns
daily_pct_change_intc = daily_close_intc.pct_change()

# Replace NA values with 0
daily_pct_change_intc.fillna(0, inplace=True)

daily_close_aapl = aapl[['Adj Close']]

# Daily returns
daily_pct_change_aapl = daily_close_aapl.pct_change()

# Replace NA values with 0
daily_pct_change_aapl.fillna(0, inplace=True)



import seaborn as sns
sns.set()

#import seaborn as sns
# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(12, 7))
#seaborn has distplot 
# Plot a simple histogram with binsize determined automatically
sns.distplot(daily_pct_change_msft['Adj Close'], color="b", ax=axes[0, 0], axlabel='MSFT');

# Plot a kernel density estimate and rug plot
sns.distplot(daily_pct_change_intc['Adj Close'], color="r", ax=axes[0, 1], axlabel='INTC');

# Plot a filled kernel density estimate
sns.distplot(daily_pct_change_googl['Adj Close'], color="g", ax=axes[1, 0], axlabel='GOOGL');

# Plot a historgram and kernel density estimate
sns.distplot(daily_pct_change_aapl['Adj Close'], color="m", ax=axes[1, 1], axlabel='AAPL');


#Step4 : Volatality

import numpy as np

#Window size
min_periods = 75

#calculate volatality
#Variance of daily percentage values
vol = daily_pct_change_msft.rolling(min_periods).std() * np.sqrt(min_periods)

vol.fillna(0, inplace = True)

vol.tail()

#plot the volatility
vol.plot(figsize = (10,8))

#Show the plot
plt.show()


#Rolling means (Trends and seasonality)

msft_adj_close_px = msft['Adj Close']
# Short moving window rolling mean
msft['42'] = msft_adj_close_px.rolling(window=42).mean()

# Long moving window rolling mean
msft['252'] = msft_adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
msft[['Adj Close', '42', '252']].plot(title="MSFT")

# Show plot
plt.show()

googl_adj_close_px = googl['Adj Close']
# Short moving window rolling mean
googl['42'] = googl_adj_close_px.rolling(window=42).mean()

# Long moving window rolling mean
googl['252'] = googl_adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
googl[['Adj Close', '42', '252']].plot(title="GOOGL")

# Show plot
plt.show()  

msft.loc['2019-01-01':'2019-09-15'][['Adj Close', '42', '252']].plot(title="MSFT in 2019");
googl.loc['2019-01-01':'2019-09-15'][['Adj Close', '42', '252']].plot(title="GOOGL in 2019");


aapl_adj_close_px = googl['Adj Close']
# Short moving window rolling mean
aapl['42'] = aapl_adj_close_px.rolling(window=42).mean()

# Long moving window rolling mean
aapl['252'] = aapl_adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
aapl[['Adj Close', '42', '252']].plot(title="AAPL")

# Show plot
plt.show()  



intc_adj_close_px = intc['Adj Close']
# Short moving window rolling mean
intc['42'] = intc_adj_close_px.rolling(window=42).mean()

# Long moving window rolling mean
intc['252'] = intc_adj_close_px.rolling(window=252).mean()

# Plot the adjusted closing price, the short and long windows of rolling means
intc[['Adj Close', '42', '252']].plot(title="INTC")

# Show plot
plt.show()  


aapl.loc['2019-01-01':'2019-09-15'][['Adj Close', '42', '252']].plot(title="AAPL in 2019");
intc.loc['2019-01-01':'2019-09-15'][['Adj Close', '42', '252']].plot(title="INTC in 2019");


#####End#####



 


























