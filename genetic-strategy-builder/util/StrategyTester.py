
import copy
import math
import warnings


class Result:
    def __init__(self, capital, candidate, buys, sells, buy_list, sell_list, buy_position, ticker_capital,
                 population_id):
        self.capital = capital
        self.candidate = candidate
        self.buys = buys
        self.sells = sells
        self.buy_list = buy_list
        self.sell_list = sell_list
        self.buy_position = buy_position
        self.ticker_capital = ticker_capital
        self.population_id = population_id

class StrategyTester():

    def test_strategy(self, threaded_results, ticker, data, candidate, population_id, train_period, commission=0.001,
                      initial_capital=10000):
        # commission charges a 1% fee per buy, which is used to affect a strategy that trades too much negatively.
        # Commission is only charged on the buy order since 1% is high

        buy_position = False

        capital = initial_capital
        last_capital = capital

        # Stores the last x rows for when an indicator requires history data
        x = 10
        last_x_rows = []

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
            last_x_rows.append(row)
            if len(last_x_rows) > x:
                last_x_rows.pop(0)

            # Calculate votes of all signals
            votes = []
            for strategy in candidate.DNA:
                votes.append(strategy.signal([row, last_x_rows]))

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
                # Charge the commission fee from the top
                capital = capital * (1.0 - commission)

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
            # Since this is not a real sale, do not log it, only adding to the capital
            # Conduct the sale transaction
            capital += purchase_amount * price
            # purchase_amount = 0
            # buy_position = False

        threaded_results[ticker] += [Result(capital, candidate, len(buys), len(sells), list(buys), list(sells),
                                            buy_position, None, population_id)]
        return capital
