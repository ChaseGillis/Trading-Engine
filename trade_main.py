import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller
import numpy as np
from pull_data import pull_data

pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 10)

stock_csv = r"stock_prices.csv"
def pick_pair(df):
    corr_matrix = df.corr().abs()  # absolute correlation matrix
    sn.heatmap(corr_matrix, annot=True)  # visualize the correlation matrix
    
    # Find the pair of stocks with maximum correlation less than 1
    max_corr_pair = corr_matrix[corr_matrix < 1].stack().idxmax()
    s1 = max_corr_pair[0]  # first element of the tuple (first stock)
    s2 = max_corr_pair[1]  # second element of the tuple (second stock)
    
    print(f"Stock1: {s1}\nStock2: {s2}")
    return s1, s2

def final_formatting():
    pull_data()
    # Read the CSV file
    df = pd.read_csv(stock_csv, parse_dates=['timestamp'])

    # Pivot the DataFrame to set dates as index, symbols as columns, and close prices as values
    df = df.pivot(index='timestamp', columns='symbol', values='close')

    return df

def backtest_strategy(ratio, zscore):
    # Convert ratio and zscore to pandas Series
    ratio = pd.Series(ratio)
    zscore = pd.Series(zscore)
    
    # Initialize positions
    long_positions = np.zeros(len(ratio))
    short_positions = np.zeros(len(ratio))
    
    # Generate signals
    long_positions[zscore > -1] = 1  # Buy signal
    short_positions[zscore < 1] = -1  # Sell signal
    
    # Convert long_positions and short_positions to pandas Series
    long_positions = pd.Series(long_positions, index=ratio.index)
    short_positions = pd.Series(short_positions, index=ratio.index)
    
    # Calculate daily returns
    daily_returns = ratio.diff() * long_positions.shift(1) + ratio.diff() * short_positions.shift(1)
    
    # Calculate cumulative returns
    cumulative_returns = daily_returns.cumsum()
    
    # Plot cumulative returns
    
    plt.plot(cumulative_returns, label='Cumulative Returns')
    plt.title('Cumulative Returns of Trading Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate performance metrics
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # Assuming 252 trading days in a year
    max_drawdown = np.min(cumulative_returns.cummin() - cumulative_returns)
    
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Maximum Drawdown: {max_drawdown}")
    
    return cumulative_returns

def main():
    df = final_formatting()    # Split the data into training and testing sets
    train_start = df.index[0]
    train_end = df.index[0] + pd.Timedelta(days=365)
    test_start = df.index[0] + pd.Timedelta(days=365)
    test_end = df.index[-1]
    
    train_data = df.loc[train_start:train_end]
    test_data = df.loc[test_start:test_end]


    s1,s2 = pick_pair(train_data)

    train_stock1 = train_data[s1]
    train_stock2 = train_data[s2]

    stock1 = test_data[s1]
    stock2 = test_data[s2]
    
    
    plt.plot(train_stock1, label = s1)
    plt.plot(train_stock2, label = s2)
    plt.legend()
    plt.show()

    result = ts.coint(train_stock1, train_stock2)
    cointigration_tstatistic = result[0]
    p_val = result[1]
    critical_values_test_statistic_at_1_5_10 = result[2]
    print("We want the P Value to be < 0.05 (Meaning Cointegration Exists)")
    print(f"P Value for the augmented Engle-Granger two-step cointegration test is {p_val}")


    # Compute the ADF test for Berkshire Hathaway and Microsoft
    # With all time series, you want to have stationary data otherwise our data will be very hard to predict.
    # ADF for Berkshire Hathaway Class B
    stock2_ADF = adfuller(train_stock2)
    print(f'P value for the Augmented Dickey-Fuller Test for {s2} is', stock2_ADF[1])
    stock1_ADF = adfuller(train_stock1)
    print(f'P value for the Augmented Dickey-Fuller Test for {s1} is', stock1_ADF[1])
    Spread_ADF = adfuller(train_stock2 - train_stock1)
    print(f'P value for the Augmented Dickey-Fuller Test for the spread is', Spread_ADF[1])
    Ratio_ADF = adfuller(train_stock2 / train_stock1)
    print(f'P value for the Augmented Dickey-Fuller Test for the ration is', Ratio_ADF[1])
    # Spread looks fine. If you'd want even better results, consider taking the difference (order 1) of Berkshire and MSFT

    # Results: can only claim stationary for the spread (since P value < 0.05). This suggests a constant mean over time.
    # Therefore, the two series are cointegrated.

    ratio = stock2 / stock1
    
    plt.plot(ratio, label = 'Price Ratio (stock2 / stock1))')
    plt.axhline(ratio.mean(), color='red')
    plt.legend()
    plt.title("Price Ratio between stock2 and stock1")
    plt.show()

    # note, here you can either use the spread OR the Price ratio approach. Anyways, let's standardize the ratio so we can have a 
    # upper and lower bound to help evaluate our trends.. Let's stick with the ratio data.

    # Calculate the Zscores of each row.
    df_zscore = (ratio - ratio.mean())/ratio.std()
    
    plt.plot(df_zscore, label = "Z Scores")
    plt.axhline(df_zscore.mean(), color = 'black')
    plt.axhline(1.0, color='red') # Setting the upper and lower bounds to be the z score of 1 and -1 (1/-1 standard deviation)
    plt.axhline(1.25, color='red') # 95% of our data will lie between these bounds.
    plt.axhline(-1.0, color='green') # 68% of our data will lie between these bounds.
    plt.axhline(-1.25, color='green') # 95% of our data will lie between these bounds.
    plt.legend(loc = 'best')
    plt.title('Z score of Ratio of stock2 to stock1')
    plt.show()
    # For the most part, the range that exists outside of these 'bands' must come converge back to the mean. Thus, you can 
    # determine when you can go long or short the pair (BRK_B to MSFT).

    # That's cool.. so when do we actually start trading? We need some form of 'signal' to trade (and to trade)
    # This is where it can become an 'artform' AND a probability game.
    # You could split 80 / 20 for train and test, BUT we are not going to be backtesting since we are just going over the model.
    # train = ratio[0:round(0.8*len(ratio))]
    # test = ratio[round(0.8*len(ratio)):]
    # print('Do the splits check out?',len(train) + len(test) == len(ratio))
    ratios_mavg5 = ratio.rolling(window=5, center=False).mean()
    ratios_mavg20 = ratio.rolling(window=20, center=False).mean()
    std_20 = ratio.rolling(window=20, center=False).std()
    zscore_20_5 = (ratios_mavg5 - ratios_mavg20)/std_20
    
    plt.plot(ratio.index, ratio.values)
    plt.plot(ratios_mavg5.index, ratios_mavg5.values)
    plt.plot(ratios_mavg20.index, ratios_mavg20.values)
    plt.legend(['Ratio', '5d Ratio MA', '20d Ratio MA'])
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.title('Ratio between stock2 and stock1 with 5 day and 20 day Moving Averages')
    plt.show()

    zscore_20_5.plot()
    
    plt.axhline(0, color='black')
    plt.axhline(1, color='red', linestyle='--')
    plt.axhline(1.25, color='red', linestyle='--')
    plt.axhline(-1, color='green', linestyle='--')
    plt.axhline(-1.25, color='green', linestyle='--')
    plt.legend(['Rolling Ratio z-score', 'Mean', '+1','+1.25','-1','-1.25'])
    plt.show()

    ratio.plot()
    buy = ratio.copy()
    sell = ratio.copy()
    buy[zscore_20_5>-1] = 0
    sell[zscore_20_5<1] = 0
    buy.plot(color='g', linestyle='None', marker='^')
    sell.plot(color='r', linestyle='None', marker='^')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, ratio.min(), ratio.max()))
    plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
    plt.title('Relationship stock2 to stock1')
    
    plt.show()
    
    # Perform backtest
    cumulative_returns = backtest_strategy(ratio, zscore_20_5)

if __name__ == '__main__':
    main()
