import argparse
import collections
import copy
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import random
import shutil
import time

from util.Candidate import Candidate
from util.StrategyTester import StrategyTester, Result
from util.Threads import save_candidate_average
from util.TickerData import TickerData



# parse arguments
parser = argparse.ArgumentParser(description='Find That Setup')
parser.add_argument('--tickers', nargs="+", dest="TICKERS", required=True,
                    help="Stock tickers to find trading setups for. Ex: --tickers AMD GOOGL INTC")
parser.add_argument('--period', dest="TRAIN_PERIOD", required=False, type=int, default=252,
                    help="Number of days to train on (252 is 1 year). Less than 1 is no limit. Ex: --period 252")
parser.add_argument('--capital', dest="CAPITAL", required=False, type=int, default=10000,
                    help="Initial capital to start the trading algorithm with. Ex: --capital 10000")
parser.add_argument('--data-interval', dest="DATA_INTERVAL", required=False, type=str, default='1d',
                    help="Interval for the data to be downloaded and tested on. Can be from this list:"
                         "1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo"
                         "Ex: --data-interval 1m")
parser.add_argument('--capital-normalization', dest="CAPITAL_NORMALIZATION", required=False, type=int, default=10,
                    help="Set to <=0 to disable. Sets a normalized cap for each result, to prevent outliers from "
                         "affecting the results negatively. Integer passed is the multiplier for the initial capital "
                         "/ year. So for a value of 20 and initial capital of 10,000, the yearly max would be 200,000."
                         "Ex: --capital-normalization 20")
parser.add_argument('--commission', dest="COMMISSION", required=False, type=float, default=0.000,
                    help="Commission to take off the top for every buy order. Helps prevent strategies with a high"
                         "number of trades from zoning out the more efficient algorithms. Default is 0.000 (0.1%)."
                         "Ex (for 1%): --commission 0.01")
parser.add_argument('--min-trades', dest="MIN_TRADES", required=False, type=float, default=3,
                    help="Min trades that should be executed. Values below this are removed. Default: 1. "
                         "Ex: --min-trades 3")
parser.add_argument('--max-trades', dest="MAX_TRADES", required=False, type=float, default=float('inf'),
                    help="Max trades that should be executed. Values above this are removed. Default is disabled."
                         "Ex: --max-trades 40")
parser.add_argument('--population', dest="POPULATION", required=False, type=int, default=100,
                    help="Number of member of each generation. Ex: --population 100")
parser.add_argument('--randomize', dest="RANDOMIZE", required=False, type=float, default=0.2,
                    help="Percentage to randomize indicator settings. Ex: --randomize 0.25")
parser.add_argument('--pass-unaltered', dest="PASS_UNALTERED", required=False, type=int, default=0,
                    help="Number of each generation to be passed unaltered to the next. Ex: --pass-unaltered 1")
parser.add_argument('-remove-duplicates', dest="REMOVE_DUPLICATES", required=False, action='store_true', default=False,
                    help="Removes duplicate indicator types from the DNA strand randomly. Ex: -remove-duplicates")
parser.add_argument('-rng', dest="RNG", required=False, action='store_true', default=False,
                    help="Enables real RNG, which does not set RNG seeds for consistent results. Ex: -rng")
parser.add_argument('-no-filter', dest="FILTER_OUTLIERS", required=False, action='store_false', default=True,
                    help="Prevents filtering out of outliers from candidate list each generation. Filters for a "
                         "minimum number of trades and filters out top results that are a certain percentage away from "
                         "the norm. Ex: -no-filter")
parser.add_argument('-u', dest='UPDATE', required=False, action='store_true',
                    help="Flag to remove old data files so they will be redownloaded.")
args = parser.parse_args()


# Save arguments from parser
TICKERS = args.TICKERS
UPDATE = args.UPDATE
TRAIN_PERIOD = args.TRAIN_PERIOD
if TRAIN_PERIOD < 1:
    TRAIN_PERIOD = None
RANDOMIZE = args.RANDOMIZE
POPULATION = args.POPULATION
CAPITAL = args.CAPITAL
COMMISSION = args.COMMISSION
MIN_TRADES = args.MIN_TRADES
MAX_TRADES = args.MAX_TRADES
CAPITAL_NORMALIZATION = args.CAPITAL_NORMALIZATION
FILTER_OUTLIERS = args.FILTER_OUTLIERS
PASS_UNALTERED = args.PASS_UNALTERED
REMOVE_DUPLICATES = args.REMOVE_DUPLICATES
RNG = args.RNG
DATA_INTERVAL = args.DATA_INTERVAL
if CAPITAL_NORMALIZATION <= 0:
    CAPITAL_NORMALIZATION = None

MULTITHREAD_PROCESS_MULTIPLIER = 1
NUM_GENERATIONS = 9999999999999999
DROP_THRESHOLD = 0.2


print('')
print('------------------------------------------------------------------')
print('')

print('Initialization and setup...')

# Set randomizer seeds for consistent results between runs
if not RNG:
    np.random.seed(314)
    random.seed(314)

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

ticker_data = TickerData(tickers=tickers, interval=DATA_INTERVAL)

# Initialize TickerData, passing a list of tickers to load
indicator_gen_period = 25
if TRAIN_PERIOD is None:
    TRAIN_PERIOD = min([len(ticker_data.data[tickers[i]].index) for i in range(len(tickers))]) - indicator_gen_period

# Cut down the data to only the timeframe being tested
for ticker in ticker_data.data.keys():
    ticker_data.data[ticker] = ticker_data.data[ticker].iloc[-1 * (TRAIN_PERIOD + indicator_gen_period):-1]

tester = StrategyTester()

# Calculate the maximum multiplier per year, to help normalize extreme results
ceil = None
num_years = TRAIN_PERIOD / 252
if CAPITAL_NORMALIZATION is not None:
    ceil = (CAPITAL * CAPITAL_NORMALIZATION) * num_years

best_candidate = None
best_settings_str = ""
buy_and_hold_str = ""
best_ind_stock_performance = ""

best_buys = 0
best_sells = 0
population = []
for _ in range(POPULATION):
    # Mock data here for strategy tester
    population.append(Candidate(randomize=RANDOMIZE, remove_duplicates=REMOVE_DUPLICATES))

best_performing_indicators = {}

abc = None

# Stores a list of the top X candidates for voting
overall_best_candidates = []

print('DONE\n')

for generation in range(NUM_GENERATIONS):

    print('Adding indicator data and testing every member of the population against each ticker passed...', end='')

    #ticker_data.clear_ticker_data()
    new_data = ticker_data.data.copy()

    # Create the shared dict and initialize with arrays
    manager = mp.Manager()
    threaded_results = manager.dict()
    for ticker in tickers:
        threaded_results[ticker] = manager.list()

    process_pool = mp.Pool(mp.cpu_count() * MULTITHREAD_PROCESS_MULTIPLIER)

    tester.test_strategy(threaded_results, tickers[0], new_data[tickers[0]], population[0], 0, TRAIN_PERIOD, 0.01)

    for ticker in tickers:
        for j in range(len(population)):
            process_pool.apply_async(tester.test_strategy, (threaded_results, ticker, new_data[ticker], population[j],
                                                            j, TRAIN_PERIOD, COMMISSION, CAPITAL,))

    process_pool.close()
    process_pool.join()

    print('DONE')
    print('Sorting the candidates by performance and calculating results...', end='')

    # Sort by population ID so all threaded_results[ticker] lists are synced to the same index in each list
    for ticker in tickers:
        threaded_results[ticker] = sorted(threaded_results[ticker], key=lambda x: x.population_id)

    # Calculate average capital gain from each candidate for each ticker passed
    average_capital = [0] * POPULATION
    average_unadjusted_capital = [0] * POPULATION
    average_buys = [0] * POPULATION
    average_sells = [0] * POPULATION
    for ticker in tickers:
        ticker_results = threaded_results[ticker]
        for j in range(len(ticker_results)):
            capital = ticker_results[j].capital
            unadjusted_capital = ticker_results[j].unadjusted_capital

            if CAPITAL_NORMALIZATION is not None:
                if capital > ceil:
                    capital = ceil
                    unadjusted_capital = ceil

            average_capital[j] += capital
            average_unadjusted_capital[j] += unadjusted_capital
            average_buys[j] += ticker_results[j].buys
            average_sells[j] += ticker_results[j].sells

    for j in range(len(average_capital)):
        average_capital[j] = average_capital[j] / len(tickers)
        average_unadjusted_capital[j] = average_unadjusted_capital[j] / len(tickers)
        average_buys[j] = average_buys[j] / len(tickers)
        average_sells[j] = average_sells[j] / len(tickers)

    # Create new candidate list with the average capitals
    candidate_average = []

    # Copy threaded_results for faster performance in the following loop
    threaded_copy = dict(threaded_results)

    # Save candidate information to candidate_average so the results can be sorted by performance and kept in sync
    for j in range(min([len(threaded_copy[ticker]) for ticker in tickers])):
        save_candidate_average(threaded_copy, tickers, j, candidate_average,
                               average_capital[j], average_buys[j], average_sells[j], average_unadjusted_capital[j])

    # process_pool.close()
    # process_pool.join()

    # Sort candidate_average
    candidate_average = sorted(candidate_average, key=lambda x: x.capital)
    candidate_average.reverse()

    # Best not outlier sets the best result to another candidate if the candidate is not a passed percentage above the
    # X lower elements
    best_not_outlier = 0
    # Filter out candidates if they don't have enough buys or they are certain percentage more than previous ones
    num_top = 5
    if FILTER_OUTLIERS:
        filtered_candidate_average = []
        for j in range(len(candidate_average) - 1):
            add = True
            if MAX_TRADES < float('inf'):
                if candidate_average[j].buys > MAX_TRADES:
                    add = False
            if MIN_TRADES > 0:
                if candidate_average[j].buys < MIN_TRADES:
                    add = False
            if add:
                filtered_candidate_average.append(candidate_average[j])

        # for j in reversed(range(1, num_top)):
        #     if candidate_average[j - 1].capital > (candidate_average[j].capital * (1 + DROP_THRESHOLD)):
        #         best_not_outlier = j
        #         break
        # if len(filtered_candidate_average) > num_elite + num_extra:
        candidate_average = filtered_candidate_average
        # else:
        #     candidate_average = candidate_average[:num_elite + num_extra]

    # Save best candidate
    if best_candidate is None or best_candidate.capital < candidate_average[best_not_outlier].capital:
        best_candidate = candidate_average[best_not_outlier]
        best_buys = candidate_average[best_not_outlier].buys
        best_sells = candidate_average[best_not_outlier].sells

        if CAPITAL_NORMALIZATION is not None:
            # Set ceiling higher
            ceil = (best_candidate.capital * CAPITAL_NORMALIZATION) * num_years

        best_settings_str = ""
        best_ind_stock_performance = ""
        for ticker in tickers:
            best_ind_stock_performance += '{}: ${:,.2f}, '.format(
                ticker, candidate_average[best_not_outlier].ticker_capital[ticker])
        best_ind_stock_performance = best_ind_stock_performance[:-2]
        for j in range(len(best_candidate.candidate.DNA)):
            cleaned_settings = copy.deepcopy(best_candidate.candidate.DNA[j].get_settings())
            if 'fillna' in cleaned_settings:
                cleaned_settings.pop('fillna')
            cleaned_settings = str(collections.OrderedDict(sorted(cleaned_settings.items()))
                                   ).replace('OrderedDict(', '').replace('\',', ':').replace('\'', '').replace('[', ''
                                   ).replace(']', '').replace('(', '').replace(')', '')[:-1]
            best_settings_str += ' -[' + str(best_candidate.candidate.DNA[j]) + ']- ' + str(cleaned_settings)

    # Create a list for the new population's candidates
    new_population = []

    # Save top unaltered amount, passing them directly to the next generation

    for j in range(PASS_UNALTERED):
        new_population.append(copy.deepcopy(candidate_average[j].candidate))

    num_elite = int(len(candidate_average) * 0.2)
    num_extra = int(len(candidate_average) * 0.1)

    # Create new population, splicing first half of top performers with other elites
    for j in range(int(num_elite / 2)):
        elite = candidate_average[j].candidate
        # Mix an elite with another random member of the elite
        random_elite = candidate_average[random.randint(0, num_elite - 1)].candidate
        child = Candidate(dna_to_mix=[elite.DNA.copy(), random_elite.DNA.copy()], remove_duplicates=REMOVE_DUPLICATES)
        new_population.append(child)
    # Splice second half of elites with non-elites
    for j in range(int(num_elite / 2), num_elite + num_extra):
        elite = candidate_average[j].candidate
        # Mix an elite with another random non-elite
        random_non_elite = candidate_average[random.randint(num_elite + 1, len(candidate_average) - 1)].candidate
        child = Candidate(dna_to_mix=[elite.DNA.copy(), random_non_elite.DNA.copy()],
                          remove_duplicates=REMOVE_DUPLICATES)
        new_population.append(child)
    # Fill out the rest of the population with random candidates
    while len(new_population) < POPULATION:
        new_population.append(Candidate(randomize=RANDOMIZE, remove_duplicates=REMOVE_DUPLICATES))

    # Store the frequencies of the indicators for the most elite population
    top_tier_elite = round(POPULATION * 0.02)
    for j in range(top_tier_elite):
        elite_dna = candidate_average[j].candidate.DNA
        for indicator in elite_dna:
            if str(indicator) not in best_performing_indicators:
                best_performing_indicators[str(indicator)] = 1
            else:
                best_performing_indicators[str(indicator)] += 1

    top_vote = 30
    # Initialize
    if len(overall_best_candidates) == 0:
        for j in range(top_vote):
            overall_best_candidates.append(candidate_average[j])
    else:
        # Otherwise, calculate the top 10 overall best
        overall_best_candidates += candidate_average[:top_vote]
        overall_best_candidates = sorted(overall_best_candidates, key=lambda x: x.capital)
        overall_best_candidates.reverse()
        overall_best_candidates = overall_best_candidates[:top_vote]

    # Count vote for best and deliver the message...
    num_vote_buy = 0
    num_vote_sell = 0
    for j in range(top_vote):
        if overall_best_candidates[j].buy_position is True:
            num_vote_buy += 1
        elif overall_best_candidates[j].buy_position is False:
            num_vote_sell += 1

    if num_vote_buy > num_vote_sell:
        vote = 'BUY'
    elif num_vote_buy < num_vote_sell:
        vote = 'SELL'
    else:
        vote = 'NEUTRAL'


    overall_best_candidate_str = "Top {} Candidate Capital: ".format(top_vote)
    for result in overall_best_candidates:
        overall_best_candidate_str += '${:,.2f}, '.format(result.capital)

    overall_best_candidate_str = overall_best_candidate_str[:-2]

    print('DONE')
    print('')

    # Output Section

    plt.clf()

    buy_coords = copy.deepcopy(new_data[tickers[0]])
    sell_coords = copy.deepcopy(new_data[tickers[0]])

    for timestamp, row in new_data[tickers[0]].iterrows():
        if timestamp not in best_candidate.buy_list:
            buy_coords = buy_coords.drop(index=timestamp)
        if timestamp not in best_candidate.sell_list:
            sell_coords = sell_coords.drop(index=timestamp)

    # Create a new series with them in order
    buy_sell_index_combined = buy_coords.index.union(sell_coords.index)
    buy_sell_close_combined = pd.concat([buy_coords.close, sell_coords.close]).sort_index()

    plt.subplot(211)
    plt.plot(new_data[tickers[0]]['close'], label="close", color='black', zorder=2)
    plt.subplot(211)
    plt.plot(buy_sell_index_combined, buy_sell_close_combined, color='b', zorder=1)
    plt.subplot(211)
    plt.scatter(buy_coords.index, buy_coords.close, color='lime', zorder=3, edgecolors='black')
    plt.subplot(211)
    plt.scatter(sell_coords.index, sell_coords.close, color='r', zorder=3, edgecolors='black')

    ax = plt.gca()
    ax.set_facecolor('ghostwhite')

    plt.xlabel("date")
    plt.ylabel("$ price")
    plt.title("{}: Overall Best".format(tickers[0]))

    buy_coords = copy.deepcopy(new_data[tickers[0]])
    sell_coords = copy.deepcopy(new_data[tickers[0]])

    for timestamp, row in new_data[tickers[0]].iterrows():
        if timestamp not in candidate_average[0].buy_list:
            buy_coords = buy_coords.drop(index=timestamp)
        if timestamp not in candidate_average[0].sell_list:
            sell_coords = sell_coords.drop(index=timestamp)

    # Create a new series with them in order
    buy_sell_index_combined = buy_coords.index.union(sell_coords.index)
    buy_sell_close_combined = pd.concat([buy_coords.close, sell_coords.close]).sort_index()

    plt.subplot(212)
    plt.plot(new_data[tickers[0]]['close'], label="close", color='black', zorder=2)
    plt.subplot(212)
    plt.plot(buy_sell_index_combined, buy_sell_close_combined, color='b', zorder=1)
    plt.subplot(212)
    plt.scatter(buy_coords.index, buy_coords.close, color='lime', zorder=3, edgecolors='black')
    plt.subplot(212)
    plt.scatter(sell_coords.index, sell_coords.close, color='r', zorder=3, edgecolors='black')

    plt.xlabel("date")
    plt.ylabel("$ price")
    plt.title("{}: Generation {}'s Best".format(tickers[0], generation + 1))

    ax = plt.gca()
    ax.set_facecolor('ghostwhite')

    plt.tight_layout()

    plt.draw()
    plt.pause(0.2)


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
    for j in range(len(candidate_average[0].candidate.DNA)):
        cleaned_settings = copy.deepcopy(candidate_average[0].candidate.DNA[j].get_settings())
        if 'fillna' in cleaned_settings:
            cleaned_settings.pop('fillna')
        cleaned_settings = str(collections.OrderedDict(sorted(cleaned_settings.items()))
                               ).replace('OrderedDict(', '').replace('\',', ':').replace('\'', '').replace('[', ''
                               ).replace(']', '').replace('(', '').replace(')', '')[:-1]
        curr_settings_str += ' -[' + str(candidate_average[0].candidate.DNA[j]) + ']- ' + str(cleaned_settings)

    # Calculate the amount if you were to just have bought and held
    if buy_and_hold_str == "":
        avg = 0
        for ticker in tickers:
            idx = (-1 * TRAIN_PERIOD) if TRAIN_PERIOD < len(new_data[ticker]) else 0
            buy_hold_earnings = CAPITAL / new_data[ticker].iloc[idx]['adjclose']
            buy_hold_earnings = buy_hold_earnings * new_data[ticker].iloc[-1]['adjclose']
            avg += buy_hold_earnings
            buy_and_hold_str += '{}: ${:,.2f}, '.format(ticker, buy_hold_earnings)
        buy_and_hold_str = '{}: ${:,.2f} - '.format('Average', avg / len(tickers)) + buy_and_hold_str
        buy_and_hold_str = buy_and_hold_str[:-2]

    individual_stock_performance_str = ""
    for ticker in tickers:
        individual_stock_performance_str += '{}: ${:,.2f}, '.format(ticker, candidate_average[0].ticker_capital[ticker])
    individual_stock_performance_str = individual_stock_performance_str[:-2]

    generation_gains_losses_str = ""
    for gain_loss in candidate_average[0].trade_gains_losses:
        generation_gains_losses_str += '${:,.2f}, '.format(gain_loss)
    generation_gains_losses_str = generation_gains_losses_str[:-2]

    best_gains_losses_str = ""
    for gain_loss in best_candidate.trade_gains_losses:
        best_gains_losses_str += '${:,.2f}, '.format(gain_loss)
    best_gains_losses_str = best_gains_losses_str[:-2]


    # Finally print the stuff I've been calculating for forever it seems like
    idx = (-1 * TRAIN_PERIOD) if TRAIN_PERIOD < len(new_data[tickers[0]]) else 0
    print('Time Range: {} -> {}'.format(str(new_data[tickers[0]].iloc[idx].name),
                                        str(new_data[tickers[0]].iloc[-1].name)))
    print('-Best in Generation- {}: ${:,.2f} (Unadjusted: ${:,.2f})  Avg Trades: {}  DNA: {}'
          ''.format(generation + 1, candidate_average[0].capital, candidate_average[0].unadjusted_capital,
                    candidate_average[0].buys, str(list(population[0].DNA))))
    print('-Best in Generation- Trade Gains/Losses: {}'.format(generation_gains_losses_str))
    print('-Best in Generation- Settings:' + str(curr_settings_str))
    print('-Best in Generation- Stock Performance: {}'.format(individual_stock_performance_str))
    print('======================')
    print('-Best Candidate- Earnings: ${:,.2f} (Unadjusted: ${:,.2f})  Avg Trades: {}  DNA: {}'
          ''.format(best_candidate.capital, best_candidate.unadjusted_capital, best_buys, best_candidate.candidate.DNA))
    print('-Best Candidate- Trade Gains/Losses: {}'.format(best_gains_losses_str))
    print('-Best Candidate- Settings:' + str(best_settings_str))
    print('-Best Candidate- Stock Performance: {}'.format(best_ind_stock_performance))
    print('======================')
    print(overall_best_candidate_str)
    print('======================')
    print('Top {} candidate votes - avg: {} -  buy: {},  sell: {}'.format(top_vote, vote, num_vote_buy, num_vote_sell))
    print('======================')
    print('-This Generation- Top Tier Elite: {}'.format(top_elite_print))
    print('-This Generation- Low Tier Elite: {}'.format(low_elite_print))
    print('-This Generation- Plebs: {}'.format(plebs))
    print('======================')
    print('Buy+Hold Earnings: - {}'.format(buy_and_hold_str))
    print('Most Frequent Elite Indicators: {}'.format(str(sorted_best_ind
                                                          ).replace('\'', '').replace('{', '(').replace('}', ')')))

    # Print individual results from each ticker for best candidate

    print('')
    print('------------------------------------------------------------------')
    print('')

    population = new_population
