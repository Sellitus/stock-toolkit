
import copy
import math
import warnings


class Result:
    def __init__(self, capital, candidate, buys, sells, buy_list, sell_list, ticker_capital, population_id):
        self.capital = capital
        self.candidate = candidate
        self.buys = buys
        self.sells = sells
        self.buy_list = buy_list
        self.sell_list = sell_list
        self.ticker_capital = ticker_capital
        self.population_id = population_id

class StrategyTester():

    def test_strategy(self, threaded_results, ticker, data, candidate, population_id, train_period,
                      initial_capital=10000):
        buy_position = False

        capital = initial_capital
        last_capital = capital

        purchase_amount = 0
        buys = []
        sells = []
        profitable = []
        unprofitable = []

        data = copy.deepcopy(data)

        price = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Add the candidate's DNA to the dataset
            for strategy in candidate.DNA:
                strategy.add_indicator(data)

        # Trim data
        data = data[-1 * train_period:-1]

        for idx, row in data.iterrows():
            # Calculate votes of all signals
            votes = []
            for strategy in candidate.DNA:
                votes.append(strategy.signal(row))

            # Calculate result
            buy = 0
            sell = 0
            for result in votes:
                if result == 'BUY':
                    buy += 1
                elif result == 'SELL':
                    sell += 1

            # Buy as much stock as possible
            price = row['close']
            if buy_position is False and purchase_amount == 0 and buy > sell and price != 0:
                # Conduct the buy transaction
                purchase_amount = capital / price
                # If you can't purchase anymore...the game is over
                # NOTE: (change later to stick around and see if the price gets within buying range)
                if purchase_amount < 1:
                    break

                capital -= purchase_amount * price
                buy_position = True

                # Log the buy
                buys.append(row.name)

            elif buy_position is True and purchase_amount > 0 and sell > buy:
                # Conduct the sale transaction
                capital += purchase_amount * price

                purchase_amount = 0
                buy_position = False

                # Log the sale
                sells.append(row.name)

        # If it ends with stock purchased, sell the stock
        if purchase_amount > 0:
            # Log the sale
            #sells.append(data[-1:].copy())
            # Conduct the sale transaction
            capital += purchase_amount * price
            purchase_amount = 0
            buy_position = False

        threaded_results[ticker] += [Result(capital, candidate, len(buys), len(sells), list(buys), list(sells),
                                            None, population_id)]
        return capital
