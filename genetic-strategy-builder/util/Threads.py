
from .StrategyTester import Result





def save_candidate_average(threaded_results, tickers, j, candidate_average, average_capital, average_buys,
                           average_sells, average_unadjusted_capital):
    ticker_capital = {}
    for ticker in tickers:
        ticker_capital[ticker] = threaded_results[ticker][j].capital

    candidate = threaded_results[tickers[0]][j].candidate

    candidate_average.append(Result(average_capital, candidate, average_buys, average_sells,
                                    threaded_results[tickers[0]][j].buy_list, threaded_results[tickers[0]][j].sell_list,
                                    threaded_results[tickers[0]][j].buy_position,
                                    average_unadjusted_capital, ticker_capital, None))
