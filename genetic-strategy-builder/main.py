import argparse

import multiprocessing as mp
import traceback
from pprint import pprint

import numpy as np
import os
import pdb
import random
import shutil
import time
import warnings

from util.Candidate import Candidate
from util.StrategyTester import StrategyTester, Result
from TickerData import TickerData
from util.TrendIndicators import TrendIndicators as ti
from util.MomentumIndicators import MomentumIndicators as mi



# parse arguments
parser = argparse.ArgumentParser(description='Find That Setup')
parser.add_argument('--tickers', nargs="+", dest="TICKERS", required=True,
                    help="Stock tickers to find trading setups for. Ex: --tickers AMD GOOGL INTC")
parser.add_argument('-u', dest='UPDATE', required=False, action='store_true',
                    help="Flag to remove old data files so they will be redownloaded.")
args = parser.parse_args()


# Save arguments from parser
TICKERS = args.TICKERS
UPDATE = args.UPDATE


# Set randomizer seeds for consistent results between runs
seed = 314
np.random.seed(seed)
random.seed(seed)

# Grab the current date
date_time_start = time.strftime("%Y-%m-%d_%H:%M:%S")

# Save the tickers to a list all uppercase
tickers = [ticker.upper() for ticker in TICKERS]

if UPDATE:
    data_path = os.path.dirname(os.path.realpath(__file__)) + '/data'
    shutil.rmtree(data_path)

# Create the data/ subfolder if it does not already exist
if not os.path.isdir("data"):
    os.mkdir("data")


# Initialize TickerData, passing a list of tickers to load
ticker_data = TickerData(tickers=tickers)
# # Add all technical indicators to dataset
#ticker_data.add_individual_indicators_to_dataset()
# ticker_data.add_technical_indicators_to_dataset()
# Cut down the data to only the timeframe being tested
num_days_to_train = 365
for ticker in ticker_data.data.keys():
    ticker_data.data[ticker] = ticker_data.data[ticker].iloc[-1 * num_days_to_train:-1]




tester = StrategyTester()

population_size = int(100)


manager = mp.Manager()

# Create a pool of size that matches the CPU core count
process_pool = mp.Pool(mp.cpu_count())

num_generations = 10000

best_candidate = None
population = []
for _ in range(population_size):
    # Mock data here for strategy tester
    population.append(Candidate())

best_performing_indicators = {}
for i in range(num_generations):
    # Add indicators to dataset with 10% randomization around default values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ticker_data.clear_ticker_data()
        ticker_data.add_individual_indicators_to_dataset(randomize=0.2)
    #pprint('Indicator Settings: ' + str(ticker_data.indicator_settings))

    # Create the shared dict and initialize with arrays
    threaded_results = manager.dict()
    for ticker in tickers:
        threaded_results[ticker] = []

    process_pool = mp.Pool(mp.cpu_count())

    for i in range(len(population)):
        for ticker in ticker_data.data.keys():
            result = process_pool.apply_async(tester.test_strategy, (threaded_results, ticker,
                                                                     ticker_data.data[ticker],
                                                                     population[i],))

    process_pool.close()
    process_pool.join()

    # Calculate average capital gain from each candidate for each ticker passed
    average_capital = [0] * population_size
    for ticker in tickers:
        ticker_results = threaded_results[ticker]
        for i in range(len(ticker_results)):
            average_capital[i] += ticker_results[i].capital

    for i in range(len(average_capital)):
        average_capital[i] = average_capital[i] / len(tickers)

    # Create new candidate list with the average capitals
    candidate_average = []

    # NOTE: Fix length mismatch here (should not vary from pop size). min() function is a temp fix
    for i in range(min(len(average_capital), len(threaded_results[tickers[0]]))):
        candidate_average.append(Result(average_capital[i], threaded_results[tickers[0]][i].candidate,
                                        threaded_results[tickers[0]][i].buys, threaded_results[tickers[0]][i].sells))

    # Sort candidate_average
    candidate_average = sorted(candidate_average, key=lambda x: x.capital)
    candidate_average.reverse()

    # Save best candidate
    if best_candidate is None or best_candidate.capital < candidate_average[0].capital:
        best_candidate = candidate_average[0]

    # Create new population, splicing top performers with the rest of the pop and filling out the rest with a randomized population
    num_elite = round(len(candidate_average) * 0.2)
    new_population = []
    for i in range(num_elite):
        elite = candidate_average[i].candidate
        # Mix an elite with another random member of the elite
        random_elite = candidate_average[random.randint(0, num_elite - 1)].candidate
        child = Candidate(dna_to_mix=[elite.DNA.copy(), random_elite.DNA.copy()])
        new_population.append(child)
    num_extra = round(population_size * 0.1)
    # Made 10% elites with non-elites
    for i in range(num_extra):
        elite = candidate_average[i].candidate
        # Mix an elite with another random member of the elite
        random_non_elite = candidate_average[random.randint(num_elite + 1, len(candidate_average) - 1)].candidate
        child = Candidate(dna_to_mix=[elite.DNA.copy(), random_non_elite.DNA.copy()])
        new_population.append(child)
    # Fill out the rest of the population with random candidates
    while len(new_population) < population_size:
        new_population.append(Candidate())

    # Store the frequencies of the indicators for the most elite population
    top_tier_elite = round(population_size * 0.02)
    for i in range(top_tier_elite):
        elite_dna = candidate_average[i].candidate.DNA
        for indicator in elite_dna:
            if str(indicator) not in best_performing_indicators:
                best_performing_indicators[str(indicator)] = 1
            else:
                best_performing_indicators[str(indicator)] += 1

    print('Best in Generation {}: ${:,.2f}  Buys: {}  Sells: {}  Best DNA: {}'.format(
          i + 1, candidate_average[0].capital, candidate_average[0].buys, candidate_average[0].sells,
          population[0].DNA))
    sorted_best = {k: v for k, v in reversed(sorted(best_performing_indicators.items(), key=lambda item: item[1]))}
    print('Most Frequent Indicators: {}'.format(str(sorted_best).replace('\'', '').replace('{', '(').replace('}', ')')))
    print('-Best Candidate- Earnings: ${:,.2f}  DNA: {}'.format(best_candidate.capital, best_candidate.candidate.DNA))
    settings_str = ""
    for dna in best_candidate.candidate.DNA:
        settings_str += ' [' + str(dna) + '] ' + str(ticker_data.indicator_settings[str(dna)]
                                                     ).replace('\'', '').replace('{', '(').replace('}', ')')
    print('-Best Candidate- Settings:' + str(settings_str))

    print('')

    population = new_population
