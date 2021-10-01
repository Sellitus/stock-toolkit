
import math
import multiprocessing as mp


class Result:
    def __init__(self, capital, candidate, buys, sells):
        self.capital = capital
        self.candidate = candidate
        self.buys = buys
        self.sells = sells

class StrategyTester():

    def test_strategy(self, threaded_results, ticker, data, candidate, initial_capital=10000):
        buy_position = False

        capital = initial_capital
        purchase_amount = 0

        buys = []
        sells = []

        price = 0

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
            if buy_position is False and purchase_amount == 0 and buy > sell:
                # Log the buy
                buys.append(row.name)
                # Conduct the buy transaction
                if price == 0:
                    price = 0.00000000000000000001
                purchase_amount = math.floor(capital / price)
                capital -= purchase_amount * price
                buy_position = True

                # If you can't purchase anymore...the game is over
                # NOTE: (change later to stick around and see if the price gets within buying range)
                if purchase_amount < 1:
                    break
                    # return 'BANKRUPT'

                #print('BUY p: {} c: {}'.format(price, capital))

            elif buy_position is True and purchase_amount > 0 and sell > buy:
                # Log the sale
                sells.append(row.name)
                # Conduct the sale transaction
                capital += purchase_amount * price
                purchase_amount = 0
                buy_position = False

                if capital < price:
                    break
                    # return 'BANKRUPT'

                #print('SELL p: {} c: {}'.format(price, capital))

        # If it ends with stock purchased, sell the stock
        if purchase_amount > 0:
            # Log the sale
            sells.append(data[-1:].copy())
            # Conduct the sale transaction
            capital += purchase_amount * price
            purchase_amount = 0
            buy_position = False

        threaded_results[ticker] += [Result(capital, candidate, buys, sells)]
        return capital
