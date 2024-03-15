"""
A trading bot using the Alpaca API for executing trades and YahooDataBacktesting for backtesting its strategy.

This script initializes the trading bot with Alpaca brokerage credentials and sets up a strategy for trading.
It utilizes the lumibot framework for both the strategy implementation and backtesting against historical data
from Yahoo Finance.

Attributes:
    API_KEY (str): The API key for accessing Alpaca's trading services.
    API_SECRET (str): The API secret for accessing Alpaca's trading services.
    BASE_URL (str): The base URL for the Alpaca API.
    ALPACA_CREDS (dict): A dictionary containing the API credentials and specifying whether to use paper trading.
"""

from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader

from datetime import datetime

# API Config
API_KEY = "PKHMP3QWCBKUYFU0G17O"
API_SECRET = "aD2btcQyPUo0qCDP55OIk7RwD6nSBeriJuZcKcho"
BASE_URL = "https://paper-api.alpaca.markets/v2"

# API Config as a Dict
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True  # Mark false when trading with real cash
}


# Backbone of the trading bot

class MLTrader(Strategy):
    """
        The backbone of the trading bot.

        A trading strategy class for a machine learning-based trader.

        This class extends from lumibot's Strategy class, defining the initial setup and the trading
        actions for each iteration.

        Methods:
            initialize(symbol): Prepares the strategy for the trading symbol.
            on_trading_iteration(): Executes the trading strategy on each iteration.
    """
    def initialize(self, symbol: str = "SPY"):
        """
            Initializes the strategy with a trading symbol.

            Parameters:
                symbol (str): The trading symbol for which the strategy is applied.
        """

        self.symbol = symbol
        self.sleeptime = "24H"  # Dictates trading speed
        self.last_trade = None  # Let us track last trade

    def on_trading_iteration(self):
        """
            Executes the trading action for each iteration (buy 10 units).

            This method defines the baseline trade action if no previous trades have been made.
        """
        # Baseline trade
        if self.last_trade is None:
            order = self.create_order(
                self.symbol,
                10,
                "buy",
                # type = "market",
            )
            self.submit_order(order)
            self.last_trade = "buy"


start_date = datetime(2023, 12, 15)
end_date = datetime(2023, 12, 31)


# Set up broker
broker = Alpaca(ALPACA_CREDS)

# Spin up an instance of the strategy
strategy = MLTrader(name='mlstrat', broker=broker, parmeters={"symbol": "SPY"})

# Set up back testing
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parmeters={"symbol": "SPY"}
)

