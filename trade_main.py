import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller
import numpy as np
from pull_data import pull_data  # assuming this function pulls data from somewhere

import streamlit as st

pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 10)

stock_csv = r"stock_prices.csv"


def calculate_coint_and_adf(stock1, stock2):
    coint_result = ts.coint(stock1, stock2)
    p_val_coint = coint_result[1]
    adf_stock1 = adfuller(stock1)
    adf_stock2 = adfuller(stock2)
    adf_spread = adfuller(stock2 - stock1)
    adf_ratio = adfuller(stock2 / stock1)
    return p_val_coint, adf_stock1, adf_stock2, adf_spread, adf_ratio

def check_stationarity(adf_results):
    return all(adf[1] < 0.05 for adf in adf_results)

def pick_pair(df):
    # Compute correlation matrix
    corr_matrix = df.corr()

    max_or_min = st.selectbox("Select correlation type:", ["Max", "Min", "Abs"])

    if max_or_min == "Max":
        best_corr = -np.inf
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if corr_matrix.iloc[i, j] > best_corr:
                    best_corr = corr_matrix.iloc[i, j]
                    s1 = df.columns[i]
                    s2 = df.columns[j]

    elif max_or_min == "Min":
        best_corr = np.inf
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if corr_matrix.iloc[i, j] < best_corr:
                    best_corr = corr_matrix.iloc[i, j]
                    s1 = df.columns[i]
                    s2 = df.columns[j]

    else:  # Absolute value
        best_corr = -np.inf
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i, j]) > best_corr:
                    best_corr = abs(corr_matrix.iloc[i, j])
                    s1 = df.columns[i]
                    s2 = df.columns[j]

    stock1 = df[s1]
    stock2 = df[s2]

    p_val_coint, *adf_results = calculate_coint_and_adf(stock1, stock2)

    if p_val_coint < 0.1 and check_stationarity(adf_results):
        st.write(f"Found suitable pair: {s1} - {s2}")
        st.write(f"correlation: {best_corr}")
        return s1, s2

    st.write("Couldn't find a suitable pair.")
    return None, None



def final_formatting():
    pull_data()
    df = pd.read_csv(stock_csv, parse_dates=['timestamp'])
    df = df.pivot(index='timestamp', columns='symbol', values='close')
    return df

def backtest_strategy(ratio, zscore, account_size=1000000, risk_percent=0.02, stop_loss_percent=0.07):
    ratio = pd.Series(ratio)
    zscore = pd.Series(zscore)
    long_positions = np.zeros(len(ratio))
    short_positions = np.zeros(len(ratio))
    
    # Implementing stop loss at 7%
    stop_loss = -stop_loss_percent  # Stop loss percentage
    
    # Dynamic moving average window
    def dynamic_moving_average(ratio, volatility_window=20):
        volatility = ratio.diff().rolling(window=volatility_window).std()
        return volatility
    
    volatility = dynamic_moving_average(ratio)
    volatility_mean = volatility.mean()
    
    # Adjust moving average window sizes dynamically based on volatility
    if volatility_mean > 0.01:
        mavg_window_1 = 5
        mavg_window_2 = 20
    else:
        mavg_window_1 = 10
        mavg_window_2 = 30

    ratios_mavg5 = ratio.rolling(window=mavg_window_1, center=False).mean()
    ratios_mavg20 = ratio.rolling(window=mavg_window_2, center=False).mean()
    std_20 = ratio.rolling(window=20, center=False).std()
    zscore_20_5 = (ratios_mavg5 - ratios_mavg20) / std_20
    
    # Adjust Z-score thresholds dynamically
    zscore_mean = zscore.mean()
    zscore_std = zscore.std()
    threshold_long = 1.0
    threshold_short = -1.0
    
    if zscore_std > 0:
        threshold_long *= zscore_std
        threshold_short *= zscore_std

    long_positions[zscore > threshold_short] = 1
    short_positions[zscore < threshold_long] = -1
    long_positions = pd.Series(long_positions, index=ratio.index)
    short_positions = pd.Series(short_positions, index=ratio.index)
    
    # Applying stop loss
    for i in range(1, len(ratio)):
        if long_positions[i-1] == 1 and ratio[i] - ratio[i-1] <= stop_loss:
            long_positions[i] = 0
        elif short_positions[i-1] == -1 and ratio[i-1] - ratio[i] <= stop_loss:
            short_positions[i] = 0

    # Position sizing based on account size and risk tolerance
    long_size = calculate_position_size(account_size, risk_percent, stop_loss_percent)
    short_size = calculate_position_size(account_size, risk_percent, stop_loss_percent)

    daily_returns = ratio.diff() * long_positions.shift(1) * long_size + ratio.diff() * short_positions.shift(1) * short_size
    cumulative_returns = daily_returns.cumsum()

    fig, ax = plt.subplots()
    ax.plot(cumulative_returns, label='Cumulative Returns')
    ax.set_title('Cumulative Returns of Trading Strategy')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)  # Explicitly close the figure to release resources

    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    max_drawdown = np.min(cumulative_returns.cummin() - cumulative_returns)
    st.write(f"Sharpe Ratio: {sharpe_ratio}")
    st.write(f"Maximum Drawdown: {max_drawdown}")

    return cumulative_returns

def calculate_position_size(account_size, risk_percent, stop_loss_percent):
    total_risk = account_size * risk_percent
    position_size = total_risk / stop_loss_percent
    return position_size





def app():
    st.title(':blue[Stock Pair Trading Strategy]')

    df = final_formatting()
    train_start = df.index[0]
    train_end = df.index[0] + pd.Timedelta(days=365)
    test_start = df.index[0] + pd.Timedelta(days=365)
    test_end = df.index[-1]
    train_data = df.loc[train_start:train_end]
    test_data = df.loc[train_end:test_end]

    s1, s2 = pick_pair(train_data)
    if not s1 or not s2:
        st.write("No pair passes all of the tests")
        return

    train_stock1 = train_data[s1]
    train_stock2 = train_data[s2]
    stock1 = test_data[s1]
    stock2 = test_data[s2]

    
    fig, ax = plt.subplots()
    ax.plot(train_stock1, label=s1)
    ax.plot(train_stock2, label=s2)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    ratio = stock2 / stock1
    fig, ax = plt.subplots()
    ax.plot(ratio, label='Price Ratio (stock2 / stock1))')
    ax.axhline(ratio.mean(), color='red')
    ax.legend()
    ax.set_title("Price Ratio between stock2 and stock1")
    st.pyplot(fig)
    plt.close(fig)

    df_zscore = (ratio - ratio.mean()) / ratio.std()
    fig, ax = plt.subplots()
    ax.plot(df_zscore, label="Z Scores")
    ax.axhline(df_zscore.mean(), color='black')
    ax.axhline(1.0, color='red')
    ax.axhline(1.25, color='red')
    ax.axhline(-1.0, color='green')
    ax.axhline(-1.25, color='green')
    ax.legend(loc='best')
    ax.set_title('Z score of Ratio of stock2 to stock1')
    st.pyplot(fig)
    plt.close(fig)

    ratios_mavg5 = ratio.rolling(window=5, center=False).mean()
    ratios_mavg20 = ratio.rolling(window=20, center=False).mean()
    std_20 = ratio.rolling(window=20, center=False).std()
    zscore_20_5 = (ratios_mavg5 - ratios_mavg20) / std_20
    
    fig, ax = plt.subplots()
    ax.plot(ratio.index, ratio.values)
    ax.plot(ratios_mavg5.index, ratios_mavg5.values)
    ax.plot(ratios_mavg20.index, ratios_mavg20.values)
    ax.legend(['Ratio', '5d Ratio MA', '20d Ratio MA'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Ratio')
    ax.set_title('Ratio between stock2 and stock1 with 5 day and 20 day Moving Averages')
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    zscore_20_5.plot(ax=ax)
    ax.axhline(0, color='black')
    ax.axhline(1, color='red', linestyle='--')
    ax.axhline(1.25, color='red', linestyle='--')
    ax.axhline(-1, color='green', linestyle='--')
    ax.axhline(-1.25, color='green', linestyle='--')
    ax.legend(['Rolling Ratio z-score', 'Mean', '+1', '+1.25', '-1', '-1.25'])
    ax.set_title('Ratio z-score with Rolling Mean and Thresholds')
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(ratio.index, ratio.values)
    buy = ratio.copy()
    sell = ratio.copy()
    buy[zscore_20_5 > -1] = 0
    sell[zscore_20_5 < 1] = 0
    buy = buy[buy.values!=0]
    sell = sell[sell.values!=0]
    ax.plot(buy.index, buy.values, color='g', linestyle='None', marker='^')
    ax.plot(sell.index, sell.values, color='r', linestyle='None', marker='^')
    ax.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
    ax.set_title('Relationship between stock2 and stock1 with Buy/Sell Signals')
    st.pyplot(fig)
    plt.close(fig)

    cumulative_returns = backtest_strategy(ratio, zscore_20_5)

if __name__ == '__main__':
    app()
