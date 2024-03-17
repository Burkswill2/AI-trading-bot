
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy


import pandas as pd
import pandas_ta as ta
from pandas import Timedelta

from tqdm import tqdm

import numpy as np
import plotly.graph_objects as go



from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient  # Create stock historical data client
from alpaca.data.timeframe import TimeFrame

from lumibot.traders import Trader

from alpaca_trade_api import REST

from datetime import datetime, timedelta

# Alpaca API Configuration Credentials
API_KEY = "PKHMP3QWCBKUYFU0G17O"
API_SECRET = "aD2btcQyPUo0qCDP55OIk7RwD6nSBeriJuZcKcho"
BASE_URL = "https://paper-api.alpaca.markets/v2"
FIG = go.Figure()
HISTORICAL_DATA = pd.DataFrame()

# API Credentials as a Dict
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,

    "PAPER": True  # Mark false when trading with real cash
}


class MLTrader(Strategy):

    def initialize(self, symbol: str = "SPY", cash_at_risk:float=0.5):

        self.symbol = symbol
        self.sleeptime = "1D"  # Dictates trading speed
        self.last_trade = None  # Statically set current trade as initial trade
        self.cash_at_risk = cash_at_risk
        self.api = REST(key_id=API_KEY, secret_key=API_SECRET, base_url=BASE_URL)

    def position_sizing(self):

        cash = self.get_cash()  # Get the current capital available
        last_price = self.get_last_price(self.symbol)  # Get the last price on the current symbol
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_ema_signal(self, df, current_candle, backcandles):

        snapshot = df.copy()  # Copy the DataFrame for state preservation

        # Assuming 'current_candle' is a Timestamp and 'backcandles' is an integer representing days
        start = current_candle - backcandles
        end = current_candle

        relevant_rows = snapshot.loc[start:end]

        if all(relevant_rows["EMA_fast"] < relevant_rows["EMA_slow"]):
            return 1
        elif all(relevant_rows["EMA_fast"] > relevant_rows["EMA_slow"]):
            return 2
        return 0

    def get_total_signal(self, df, current_candle, backcandles):

        ema_signal = self.get_ema_signal(df, current_candle, backcandles)

        if ema_signal == 2 and df['close'][current_candle] <= df['BBL_15_1.5'][current_candle]:
            return 2

        if (ema_signal == 1) and (df['close'][current_candle] >= df['BBU_15_1.5'][current_candle]):
            return 1

        return 0


    # def SIGNAL(self):
    #     return df.TotalSignal

    def pointpos(self, x):
        if x['TotalSignal'] == 2:
            return x['low'] - 1e3
        elif x['TotalSignal'] == 1:
            return x['low'] + 1e3
        else:
            return np.nan

    def get_historical_data(self):
        API_KEY = "PKHMP3QWCBKUYFU0G17O"
        API_SECRET = "aD2btcQyPUo0qCDP55OIk7RwD6nSBeriJuZcKcho"
        stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)

        # Fetch and dataframing
        request_params = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Hour,
            start=start_date,
            end=end_date
        )

        # print(historical_data.df)

        historical_data = stock_client.get_stock_bars(request_params)

        historical_data = historical_data.df
        # Reset the index to make both levels regular columns

        historical_data = historical_data.reset_index()
        # historical_data = historical_data.set_index('timestamp')

        # Set the desired level (now a regular column) as the new index
        # historical_data = historical_data.set_index('second')
        # historical_data = historical_data.set_index(historical_data.columns[1], drop=False)

        # print(historical_data)

        # # Format
        # historical_data["timestamp"] = historical_data.index.tz_convert('America/New_York')
        historical_data.ffill(inplace=True)
        historical_data.bfill(inplace=True)

        # Check for any remaining duplicates or null values
        # print(historical_data.index.is_unique)  # Should be True
        # print(historical_data.isnull().any())  # Should be False for all columns
        #
        # # Quick statistical summary to spot any anomalies
        # print(historical_data.describe())
        historical_data = historical_data[historical_data.high != historical_data.low]

        # Calculating Exponential Moving Averages (EMA)
        historical_data["EMA_slow"] = ta.ema(historical_data.close, length=50)
        historical_data["EMA_fast"] = ta.ema(historical_data.close, length=30)

        # Calculating Relative Strength Index (RSI)
        historical_data["RSI"] = ta.rsi(historical_data.close, length=10)

        # Calculating Average True Range (ATR)
        historical_data["ATR"] = ta.atr(historical_data.high, historical_data.low, historical_data.close, length=7)

        # Calculating Bollinger Bands
        my_bbands = ta.bbands(historical_data.close, length=15, std=1.5)
        historical_data = historical_data.join(my_bbands)

        # Concatenate data set to first 9999 rows
        # historical_data = historical_data[-10000:-1]
        # Loading bar for pandas apply methods
        tqdm.pandas()

        current_candle = pd.Timestamp.now(tz="UTC")

        # Getting EMA Signal
        historical_data['EMASignal'] = historical_data.apply(
            lambda row: self.get_ema_signal(historical_data, row.name, 7), axis=1)  # if row.name >= 20 else 0


        # Getting Total Signal
        historical_data['TotalSignal'] = historical_data.progress_apply(
            lambda row: self.get_total_signal(historical_data, row.name, 7), axis=1)

        # print(historical_data[historical_data.TotalSignal != 0].head(20))

        # Getting pointpos for plotting
        historical_data['pointpos'] = historical_data.apply(lambda row: self.pointpos(row), axis=1)
        # print(historical_data[historical_data.TotalSignal != 0].head(20))

        return historical_data

    def save_plot_to_html(self, fig, filename="plot.html"):
        # Save the Plotly figure to an HTML file
        fig.write_html(filename)

    def on_trading_iteration(self):

        """
        Todo

        Adjust logic so that graph is not generated on every iteration

        """
        cash, last_price, quantity = self.position_sizing()  # Get position sizing before executing a trade
        historical_data = self.get_historical_data()
        # print(historical_data)
        HISTORICAL_DATA = historical_data
        # self.save_plot_to_html(fig, filename="plot.html")

        # if cash > last_price:  # Check if there is enough cash to perform a transaction
        #         order = self.create_order(
        #             self.symbol,
        #             quantity,
        #             "buy",
        #             take_profit_price=last_price * 1.20,  # Setting a take profit 20% above the purchase price
        #             stop_loss_price=last_price * 0.95,  # Setting a stop loss 5% below the purchase price
        #         )
        #         self.submit_order(order)
        #         self.last_trade = "buy"


# Trading period configuration (3mo)
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 3, 15)


# Set up broker. Calling Alpaca broker and injecting API Config dict
broker = Alpaca(ALPACA_CREDS)

# Spin up an instance of the strategy
strategy = MLTrader(name='mlstrat', broker=broker, parmeters={"symbol": "SPY", "cash_at_risk": 0.5})

# Set up back testing
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parmeters={"symbol": "AAPL", "cash_at_risk": 0.5}
)


def get_plot(df):
    st = 100
    dfpl = df[st:st + 350]
    # dfpl.reset_index(inplace=True)
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                                         open=dfpl['open'],
                                         high=dfpl['high'],
                                         low=dfpl['low'],
                                         close=dfpl['close']),

                          go.Scatter(x=dfpl.index, y=dfpl['BBL_15_1.5'],
                                     line=dict(color='green', width=1),
                                     name="BBL"),
                          go.Scatter(x=dfpl.index, y=dfpl['BBU_15_1.5'],
                                     line=dict(color='green', width=1),
                                     name="BBU"),
                          go.Scatter(x=dfpl.index, y=dfpl['EMA_fast'],
                                     line=dict(color='black', width=1),
                                     name="EMA_fast"),
                          go.Scatter(x=dfpl.index, y=dfpl['EMA_slow'],
                                     line=dict(color='blue', width=1),
                                     name="EMA_slow")])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="entry")

    return fig


get_plot(HISTORICAL_DATA)
FIG.show()
