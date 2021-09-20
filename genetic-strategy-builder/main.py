import argparse
import numpy as np
import os
import random
import time

from GeneticStrategyBuilder import GeneticStrategyBuilder
from TickerData import TickerData














# parse arguments
parser = argparse.ArgumentParser(description='Find That Setup')
parser.add_argument('--tickers', nargs="+", dest="TICKERS", required=True,
                    help="Stock tickers to find trading setups for. Ex: --tickers AMD GOOGL INTC")
args = parser.parse_args()


# Save arguments from parser
TICKERS = args.TICKERS


# Set randomizer seeds for consistent results between runs
seed = 314
np.random.seed(seed)
random.seed(seed)

# Grab the current date
date_time_start = time.strftime("%Y-%m-%d_%H:%M:%S")

# Save the tickers to a list all uppercase
tickers = [ticker.upper() for ticker in TICKERS]


# Create the data/ subfolder if it does not already exist
if not os.path.isdir("data"):
    os.mkdir("data")


# Initialize TickerData, passing a list of tickers to load
ticker_data = TickerData(tickers=tickers)

# Create the GeneticStrategyBuilder object
genetic_strategy_builder = GeneticStrategyBuilder(ticker_data=ticker_data)










import pdb
pdb.set_trace()