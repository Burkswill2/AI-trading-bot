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

from alpaca_trade_api import REST

from datetime import datetime, timedelta
from yarl import URL


# Alpaca API Configuration Credentials
API_KEY = "PKHMP3QWCBKUYFU0G17O"
API_SECRET = "aD2btcQyPUo0qCDP55OIk7RwD6nSBeriJuZcKcho"
BASE_URL = "https://paper-api.alpaca.markets/v2"

# API Credentials as a Dict
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True  # Mark false when trading with real cash
}


class MLTrader(Strategy):
    """
    The backbone of the trading bot.

    A trading strategy class for a machine learning-based trader.

    This class extends from lumibot's Strategy class, defining the initial setup and the trading
    actions for each iteration.

    Attributes:
        symbol (str): The trading symbol for which the strategy is applied.
        sleeptime (str): Interval between trading actions.
        last_trade (str): Tracks the last trading action performed.
        cash_at_risk (float): Proportion of available cash to risk on each trade.

    Methods:
        initialize(symbol): Prepares the strategy for the trading symbol.
        position_sizing(): Calculates the amount of shares to buy based on available cash and cash at risk.
        on_trading_iteration(): Executes the trading strategy on each iteration.
    """
    def initialize(self, symbol: str = "SPY", cash_at_risk:float=0.5):
        """
        Initializes the strategy with a trading symbol.

         Parameters:
            symbol (str): The trading symbol for which the strategy is applied.
            cash_at_risk (float): Percentage of available cash risked in each trade.
        """
        self.symbol = symbol
        self.sleeptime = "24H"  # Dictates trading speed
        self.last_trade = None  # Statically set current trade as initial trade
        self.cash_at_risk = cash_at_risk
        self.api = REST(key_id=API_KEY, secret_key=API_SECRET, base_url=BASE_URL)

    # [CRITICAL] Set up position sizing
    def position_sizing(self):
        """
        Determines the size of the position to take based on cash at risk and current price of the symbol.

        Returns:
            tuple: Contains available cash, the last price of the symbol, and the quantity to trade.
        """
        cash = self.get_cash()  # Get the current capital available
        last_price = self.get_last_price(self.symbol)  # Get the last price on the current symbol
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        minus_three_days = today - timedelta(days=3)
        return today.strftime("%Y-%m-%d"), minus_three_days.strftime("%Y-%m-%d")

    def get_news(self):
        today, minus_three_days = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=minus_three_days, end=today)

        # Process news object into a list comprehension
        # Raw is the raw news data
        # Headline is the news headline
        news = [item.__dict__["_raw"]["headline"] for item in news]
        return news

    def on_trading_iteration(self):
        """
        Executes the trading action for each iteration based on position sizing and trading strategy.

        Buys a calculated quantity of the symbol if no trades have been made previously and sufficient cash is available.
        """
        cash, last_price, quantity = self.position_sizing()  # Get position sizing before executing a trade

        if cash > last_price:  # Check if there is enough cash to perform a transaction
            # Baseline trade
            if self.last_trade is None:
                news = self.get_news()  # Get news on trade target
                print(news)
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    take_profit_price=last_price * 1.20,  # Setting a take profit 20% above the purchase price
                    stop_loss_price=last_price * 0.95,  # Setting a stop loss 5% below the purchase price
                )
                self.submit_order(order)
                self.last_trade = "buy"


# Trading period configuration (5yr)
start_date = datetime(2019, 3, 15)
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
    parmeters={"symbol": "SPY", "cash_at_risk": 0.5}
)

