import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import statsmodels.tsa.stattools as ts

pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 10)

stock_csv = r"Trading Engine/stock_prices.csv"

def og_index_change():
    df = pd.read_csv(stock_csv)
    df['timestamp'] = df['timestamp'].apply(lambda x: str(x.split(" ")[0]))
    df = df.set_index("timestamp")

    df.to_csv(stock_csv)

def final_formatting():
    # Read the CSV file
    df = pd.read_csv(stock_csv, parse_dates=['timestamp'])

    # Pivot the DataFrame to set dates as index, symbols as columns, and close prices as values
    df = df.pivot(index='timestamp', columns='symbol', values='close')

    return df

def correlation_finder():
    df = final_formatting()
    corr_matrix = df.corr()
    sn.heatmap(corr_matrix, annot=True)

def main():
    df = final_formatting()
    msft = df['MSFT']
    tsla = df['TSLA']
    '''plt.plot(msft, label = "Microsoft")
    plt.plot(tsla, label = "Tesla")
    plt.legend()
    plt.show()'''

    result = ts.coint(msft, tsla)
    cointigration_tstatistic = result[0]
    p_val = result[1]
    critical_values_test_statistic_at_1_5_10 = result[2]
    print("We want the P Value to be < 0.05 (Meaning Cointegration Exists)")
    print(f"P Value for the augmented Engle-Granger two-step cointegration test is {p_val}")

if __name__ == '__main__':
    main()