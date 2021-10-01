import argparse
import math

import multiprocessing as mp

import numpy as np
import os
import random
import shutil
import time
import warnings

from util.Candidate import Candidate
from util.StrategyTester import StrategyTester, Result
from util.TickerData import TickerData



# parse arguments
parser = argparse.ArgumentParser(description='Find That Setup')
parser.add_argument('--tickers', nargs="+", dest="TICKERS", required=True,
                    help="Stock tickers to find trading setups for. Ex: --tickers AMD GOOGL INTC")
parser.add_argument('--period', dest="TRAIN_PERIOD", required=False, type=int, default=1095,
                    help="Units of time to train on. Ex: --period 365")
parser.add_argument('--capital', dest="CAPITAL", required=False, type=int, default=10000,
                    help="Initial capital to start the trading algorithm with. Ex: --capital 10000")
parser.add_argument('--capital-normalization', dest="CAPITAL_NORMALIZATION", required=False, type=int, default=20,
                    help="Set to <=0 to disable. Sets a normalized cap for each result, to prevent outliers from "
                         "affecting the results negatively. Integer passed is the multiplier for the initial capital "
                         "/ year. So for a value of 20 and initial capital of 10,000, the yearly max would be 200,000."
                         "Ex: --capital-normalization 20")
parser.add_argument('--min-trades', dest="MIN_TRADES", required=False, type=int, default=3,
                    help="Min trades that should be executed. Values below this are removed. Ex: --min-trades 5")
parser.add_argument('--population', dest="POPULATION", required=False, type=int, default=100,
                    help="Number of member of each generation. Ex: --population 100")
parser.add_argument('--seed', dest="SEED", required=False, type=int, default=None,
                    help="Seed to use for random number generator for consistent results. Ex: --seed 314")
parser.add_argument('--randomize', dest="RANDOMIZE", required=False, type=float, default=0.2,
                    help="Percentage to randomize indicator settings. Ex: --randomize 0.25")
parser.add_argument('-no-filter', dest="FILTER_OUTLIERS", required=False, action='store_false', default=True,
                    help="Prevents filtering out of outliers from candidate list each generation. Filters for a "
                         "minimum number of trades and filters out top results that are a certain percentage away from "
                         "the norm. Ex: -filter")
parser.add_argument('-u', dest='UPDATE', required=False, action='store_true',
                    help="Flag to remove old data files so they will be redownloaded.")
args = parser.parse_args()


# Save arguments from parser
TICKERS = args.TICKERS
UPDATE = args.UPDATE
TRAIN_PERIOD = args.TRAIN_PERIOD
RANDOMIZE = args.RANDOMIZE
POPULATION = args.POPULATION
SEED = args.SEED
CAPITAL = args.CAPITAL
MIN_TRADES = args.MIN_TRADES
CAPITAL_NORMALIZATION = args.CAPITAL_NORMALIZATION
FILTER_OUTLIERS = args.FILTER_OUTLIERS
if CAPITAL_NORMALIZATION <= 0:
    CAPITAL_NORMALIZATION = None

# Set randomizer seeds for consistent results between runs
if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)

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

# Cut down the data to only the timeframe being tested
for ticker in ticker_data.data.keys():
    ticker_data.data[ticker] = ticker_data.data[ticker].iloc[-1 * (TRAIN_PERIOD + 500):-1]


MULTITHREAD_PROCESS_MULTIPLIER = 1
NUM_GENERATIONS = 100000
drop_threshold = 0.25


print('')
print('------------------------------------------------------------------')
print('')

tester = StrategyTester()

# Calculate the maximum multiplier per year, to help normalize extreme results
ceil = None
if CAPITAL_NORMALIZATION is not None:
    num_years = TRAIN_PERIOD / 365
    ceil = (CAPITAL * CAPITAL_NORMALIZATION) * num_years


best_candidate = None
best_settings_str = ""
best_buys = 0
best_sells = 0
population = []
for _ in range(POPULATION):
    # Mock data here for strategy tester
    population.append(Candidate())

best_performing_indicators = {}
for i in range(NUM_GENERATIONS):
    print('Adding technical indicators to the data...', end='')

    # Add indicators to dataset with 10% randomization around default values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ticker_data.clear_ticker_data()
        ticker_data.add_individual_indicators_to_dataset(randomize=RANDOMIZE)

        new_data = ticker_data.data.copy()
        # Trim data
        for ticker in new_data.keys():
            new_data[ticker] = new_data[ticker][-1 * TRAIN_PERIOD:-1]

    print('DONE')
    print('Testing every member of the population against each ticker passed...', end='')

    # Create the shared dict and initialize with arrays
    manager = mp.Manager()
    threaded_results = manager.dict()
    for ticker in tickers:
        threaded_results[ticker] = []

    process_pool = mp.Pool(mp.cpu_count() * MULTITHREAD_PROCESS_MULTIPLIER)

    for j in range(len(population)):
        for ticker in new_data.keys():
            process_pool.apply_async(tester.test_strategy, (threaded_results, ticker, new_data[ticker], population[j],
                                                            CAPITAL,))

    process_pool.close()
    process_pool.join()

    print('DONE')
    print('Sorting the candidates by performance and calculating results...', end='')

    # Calculate average capital gain from each candidate for each ticker passed
    average_capital = [0] * POPULATION
    average_buys = [0] * POPULATION
    average_sells = [0] * POPULATION
    for ticker in tickers:
        ticker_results = threaded_results[ticker]
        for j in range(len(ticker_results)):
            capital = ticker_results[j].capital

            if CAPITAL_NORMALIZATION is not None:
                if capital > ceil:
                    capital = ceil

            average_capital[j] += ticker_results[j].capital
            average_buys[j] += ticker_results[j].buys
            average_sells[j] += ticker_results[j].sells

    for j in range(len(average_capital)):
        average_capital[j] = average_capital[j] / len(tickers)
        average_buys[j] = round(average_buys[j] / len(tickers))
        average_sells[j] = round(average_sells[j] / len(tickers))

    # Create new candidate list with the average capitals
    candidate_average = []

    for j in range(min(len(average_capital), len(threaded_results[tickers[0]]))):
        candidate_average.append(Result(average_capital[j], threaded_results[tickers[0]][j].candidate,
                                 average_buys[j], average_sells[j]))

    # Sort candidate_average
    candidate_average = sorted(candidate_average, key=lambda x: x.capital)
    candidate_average.reverse()

    # Filter out candidates if they don't have enough buys or they are certain percentage more than previous ones
    if FILTER_OUTLIERS:
        filtered_candidate_average = []
        num_top = 5
        for j in range(len(candidate_average) - 1):
            if candidate_average[j].buys >= MIN_TRADES:
                filtered_candidate_average.append(candidate_average[j])
        for j in reversed(range(1, num_top)):
            if ((1 - drop_threshold) * candidate_average[j - 1].capital) > candidate_average[j].capital:
                filtered_candidate_average = filtered_candidate_average[j:]
                break
        candidate_average = filtered_candidate_average

    # Save best candidate
    if best_candidate is None or best_candidate.capital < candidate_average[0].capital:
        best_candidate = candidate_average[0]
        best_buys = candidate_average[0].buys
        best_sells = candidate_average[0].sells
        best_settings_str = ""
        for dna in best_candidate.candidate.DNA:
            cleaned_settings = ticker_data.indicator_settings[str(dna)].copy()
            if 'fillna' in cleaned_settings:
                cleaned_settings.pop('fillna')
            cleaned_settings = str(cleaned_settings).replace('\'', '').replace('{', '(').replace('}', ')')
            best_settings_str += ' [' + str(dna) + '] ' + str(cleaned_settings)

    # Create a list for the new population's candidates
    new_population = []

    # Save top maximum_elite percentage, passing them directly to the next generation
    maximum_elite = round(len(candidate_average) * 0.05)
    for j in range(maximum_elite):
        new_population.append(candidate_average[j].candidate)

    # Create new population, splicing top performers with the rest of the pop and filling out the rest with a randomized population
    num_elite = round(len(candidate_average) * 0.2)
    for j in range(num_elite):
        elite = candidate_average[j].candidate
        # Mix an elite with another random member of the elite
        random_elite = candidate_average[random.randint(0, num_elite - 1)].candidate
        child = Candidate(dna_to_mix=[elite.DNA.copy(), random_elite.DNA.copy()])
        new_population.append(child)
    num_extra = round(len(candidate_average) * 0.1)
    # Made 10% elites with non-elites
    for j in range(num_extra):
        elite = candidate_average[j].candidate
        # Mix an elite with another random member of the elite
        random_non_elite = candidate_average[random.randint(num_elite + 1, len(candidate_average) - 1)].candidate
        child = Candidate(dna_to_mix=[elite.DNA.copy(), random_non_elite.DNA.copy()])
        new_population.append(child)
    # Fill out the rest of the population with random candidates
    while len(new_population) < POPULATION:
        new_population.append(Candidate())

    # Store the frequencies of the indicators for the most elite population
    top_tier_elite = round(POPULATION * 0.02)
    for j in range(top_tier_elite):
        elite_dna = candidate_average[j].candidate.DNA
        for indicator in elite_dna:
            if str(indicator) not in best_performing_indicators:
                best_performing_indicators[str(indicator)] = 1
            else:
                best_performing_indicators[str(indicator)] += 1

    print('DONE')
    print('')

    # Output Section

    # Calculate top and low tier elite
    top_elite_print = []
    low_elite_print = []
    plebs = []
    for j in range(5):
        top_elite_print.append('${:,.2f}'.format(candidate_average[j].capital))
    for j in range(5):
        idx = num_elite - j
        low_elite_print.append('${:,.2f}'.format(candidate_average[idx].capital))
    for j in range(5):
        idx = round(len(candidate_average) * 0.5) - j
        plebs.append('${:,.2f}'.format(candidate_average[idx].capital))


    # Sort the indicators by their frequency
    sorted_best_ind = {k: v for k, v in reversed(sorted(best_performing_indicators.items(), key=lambda item: item[1]))}

    # Calculate output for the current settings
    curr_settings_str = ""
    for dna in candidate_average[0].candidate.DNA:
        cleaned_settings = ticker_data.indicator_settings[str(dna)].copy()
        if 'fillna' in cleaned_settings:
            cleaned_settings.pop('fillna')
        cleaned_settings = str(cleaned_settings).replace('\'', '').replace('{', '(').replace('}', ')')
        curr_settings_str += ' [' + str(dna) + '] ' + str(cleaned_settings)

    # Finally print the stuff I've been calculating forever
    print('Best in Generation {}: ${:,.2f}  Buys: {}  Sells: {}  DNA: {}'.format(
          i + 1, candidate_average[0].capital, candidate_average[0].buys, candidate_average[0].sells,
          population[0].DNA))
    print('-This Generation- Settings:' + str(curr_settings_str))
    print('-This Generation- Top Tier Elite: {}'.format(top_elite_print))
    print('-This Generation- Low Tier Elite: {}'.format(low_elite_print))
    print('-This Generation- Plebs: {}'.format(plebs))
    print('======================')
    print('Most Frequent Elite Indicators: {}'.format(str(sorted_best_ind
                                                          ).replace('\'', '').replace('{', '(').replace('}', ')')))
    print('======================')
    print('-Best Candidate- Earnings: ${:,.2f}  Buys: {}  Sells: {}  DNA: {}'.format(best_candidate.capital,
                                                                                     best_buys, best_sells,
                                                                                     best_candidate.candidate.DNA))
    print('-Best Candidate- Settings:' + str(best_settings_str))

    # Print individual results from each ticker for best candidate

    print('')
    print('------------------------------------------------------------------')
    print('')

    population = new_population
