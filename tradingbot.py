# Alpaca is the broker
from lumibot.brokers import Alpaca
# Backtesting framework
from lumibot.backtesting import YahooDataBacktesting
# This is the actual training bot
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader

from datetime import datetime

#Variables to hold the API key to get data
API_KEY = "PKHMP3QWCBKUYFU0G17O"
API_SECRET = "aD2btcQyPUo0qCDP55OIk7RwD6nSBeriJuZcKcho"
BASE_URL = "https://paper-api.alpaca.markets/v2"

#Actual API keys for API requests
ALPACA_CREDS = {
    "API_KEY":API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True #Mark false when trading with real cash
}

# Backbone of the trading bot
# Initizalize will run once on bot start up
# on_trading_iteration will run on each trade
class MLTrader(Strategy):
    def initialize(self, symbol: str="SPY"):
        self.symbol = symbol
        self.sleeptime = "24H" # Dictates trading speed
        self.last_trade = None # Let us track last trade

    def on_trading_iteration(self):
        # Baseline trade
        if self.last_trade == None:
            order = self.create_order(
                self.symbol,
                10,
                "buy",
                type="market"
            )
            self.submit_order(order)
            self.last_trade = "buy"


start_date = datetime(2023,12,15),
end_date = datetime(2023,12,31),


#Set up broker
broker = ALPACA_CREDS

# Spin up an instance of the strategy
strategy = MLTrader(name='mlstrat', broker=broker, parmeters={"symbol":"SPY"})

# Set up back testing
# Helps allow me to evaluate hoe the bot is doing
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parmeters={"symbol":"SPY"}
)

