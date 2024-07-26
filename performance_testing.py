import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller
import numpy as np
from pull_data import pull_data  # assuming this function pulls data from somewhere

pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 10)

stock_csv = r"stock_prices.csv"


def calculate_coint_and_adf(stock1, stock2):
    if stock1.empty or stock2.empty:
        return None, None, None, None, None

    if stock1.isnull().any() or stock2.isnull().any():
        return None, None, None, None, None

    if np.any(np.isinf(stock1)) or np.any(np.isinf(stock2)):
        return None, None, None, None, None

    coint_result = ts.coint(stock1, stock2)
    p_val_coint = coint_result[1]
    adf_stock1 = adfuller(stock1)
    adf_stock2 = adfuller(stock2)
    adf_spread = adfuller(stock2 - stock1)
    adf_ratio = adfuller(stock2 / stock1)
    return p_val_coint, adf_stock1, adf_stock2, adf_spread, adf_ratio

def check_stationarity(adf_results):
    return all(adf[1] < 0.2 for adf in adf_results)

def pick_pair(df):
    corr_matrix = df.corr()
    cluster_map = sn.clustermap(corr_matrix, cmap='coolwarm', linewidths=.5, method='average')
    plt.close()

    max_or_min = "Abs"
    sort_order = True

    if max_or_min == "Max":
        sort_order = True
    elif max_or_min == "Min":
        sort_order = False
    else:
        corr_matrix = corr_matrix.abs()
        sort_order = True

    sorted_pairs = sorted(
        ((corr_matrix.iloc[i, j], df.columns[i], df.columns[j])
         for i in range(len(corr_matrix.columns))
         for j in range(i + 1, len(corr_matrix.columns))),
        reverse=sort_order
    )

    for corr_value, s1, s2 in sorted_pairs:
        stock1 = df[s1]
        stock2 = df[s2]

        p_val_coint, *adf_results = calculate_coint_and_adf(stock1, stock2)

        if not p_val_coint or not adf_results:
            continue

        if p_val_coint < 0.2 and check_stationarity(adf_results):
            return s1, s2

        if abs(corr_value) < 0.6:
            break

    return None, None

def final_formatting():
    pull_data()
    df = pd.read_csv(stock_csv, parse_dates=['timestamp'])
    df = df.pivot(index='timestamp', columns='symbol', values='close')
    return df

def backtest_strategy(ratio, zscore, account_size, risk_percent=0.02, stop_loss_percent=0.07):
    ratio = pd.Series(ratio)
    zscore = pd.Series(zscore)
    long_positions = np.zeros(len(ratio))
    short_positions = np.zeros(len(ratio))

    stop_loss = -stop_loss_percent

    def dynamic_moving_average(ratio, volatility_window=20):
        volatility = ratio.diff().rolling(window=volatility_window).std()
        return volatility

    volatility = dynamic_moving_average(ratio)
    volatility_mean = volatility.mean()

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

    for i in range(1, len(ratio)):
        if long_positions[i-1] == 1 and ratio[i] - ratio[i-1] <= stop_loss:
            long_positions[i] = 0
        elif short_positions[i-1] == -1 and ratio[i-1] - ratio[i] <= stop_loss:
            short_positions[i] = 0

    long_size = calculate_position_size(account_size, risk_percent, stop_loss_percent)
    short_size = calculate_position_size(account_size, risk_percent, stop_loss_percent)

    daily_returns = ratio.diff() * long_positions.shift(1) * long_size + ratio.diff() * short_positions.shift(1) * short_size
    cumulative_returns = daily_returns.cumsum()

    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    roi = cumulative_returns.iloc[-1] / account_size

    return sharpe_ratio, roi, cumulative_returns

def calculate_position_size(account_size, risk_percent, stop_loss_percent):
    total_risk = account_size * risk_percent
    position_size = total_risk / stop_loss_percent
    return position_size

def run_multiple_backtests(df, num_runs, account_size, risk_percent=0.02, stop_loss_percent=0.07):
    sharpe_ratios = []
    rois = []

    for _ in range(num_runs):
        train_start = df.index[0]
        train_end = df.index[0] + pd.Timedelta(days=365)
        test_start = df.index[0] + pd.Timedelta(days=365)
        test_end = df.index[-1]
        train_data = df.loc[train_start:train_end]
        test_data = df.loc[train_end:test_end]

        s1, s2 = pick_pair(train_data)
        if not s1 or not s2:
            continue

        stock1 = test_data[s1]
        stock2 = test_data[s2]
        ratio = stock2 / stock1
        df_zscore = (ratio - ratio.mean()) / ratio.std()

        sharpe_ratio, roi, _ = backtest_strategy(ratio, df_zscore, account_size, risk_percent, stop_loss_percent)
        sharpe_ratios.append(sharpe_ratio)
        rois.append(roi)

    avg_sharpe_ratio = np.mean(sharpe_ratios)
    avg_roi = np.mean(rois)

    return avg_sharpe_ratio, avg_roi

def main():
    df = final_formatting()

    num_runs = 10  # You can change this number to run more or fewer backtests
    account_size = 100000  # Example account size

    avg_sharpe_ratio, avg_roi = run_multiple_backtests(df, num_runs, account_size)

    print(f"Average Sharpe Ratio: {avg_sharpe_ratio}")
    print(f"Average ROI: {avg_roi * 100:.2f}%")

if __name__ == '__main__':
    main()
