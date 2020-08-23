""":"

# Find a suitable python interpreter (adapt for your specific needs)
for cmd in ~/anaconda3/envs/stock-toolkit/bin/python ; do
   command -v > /dev/null $cmd && exec $cmd $0 "$@"
done

echo "Python not found!" >2

exit 2

":"""

# Import libraries
from bs4 import BeautifulSoup
from get_all_tickers import get_tickers as gt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen, Request

import matplotlib.pyplot as plt
import pandas as pd

import nltk
import pickle
import requests


def scrapeFinviz(tickers):
    # Get Data
    finviz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    for ticker in tickers:
        try:
            url = finviz_url + ticker
            print('Fetching {} news from: {}'.format(ticker, url))
            req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
            resp = urlopen(req)
            html = BeautifulSoup(resp, features="lxml")
            news_table = html.find(id='news-table')
            news_tables[ticker] = news_table
        except Exception as e:
            print('Unable to fetch data for: {}'.format(ticker))
    return news_tables


def printFinviz(tickers, news_tables):
    try:
        for ticker in tickers:
            df = news_tables[ticker]
            df_tr = df.findAll('tr')

            print('\n')
            print('Recent News Headlines for {}: '.format(ticker))

            for i, table_row in enumerate(df_tr):
                a_text = table_row.a.text
                td_text = table_row.td.text
                td_text = td_text.strip()
                print(a_text, '(', td_text, ')')
                if i == num_headlines - 1:
                    break
    except KeyError:
        pass


def parseFinviz(news_tables):
    parsed_news = []
    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text()
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]

            else:
                date = date_scrape[0]
                time = date_scrape[1]

            ticker = file_name.split('_')[0]

            parsed_news.append([ticker, date, time, text])
    return parsed_news


def analyzeSentiment(parsed_news):
    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()

    columns = ['Ticker', 'Date', 'Time', 'Headline']
    news = pd.DataFrame(parsed_news, columns=columns)
    scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

    df_scores = pd.DataFrame(scores)
    news = news.join(df_scores, rsuffix='_right')

    return news


def getSp500Tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    # with open("sp500tickers.pickle", "wb") as f:
    #     pickle.dump(tickers, f)

    return tickers




# Download vader_lexicon which is required by the sentiment analyzer
nltk.download('vader_lexicon')

# Parameters
num_headlines = 10  # the # of article headlines displayed per ticker
tickers = ['AMD', 'SNE', 'ATVI', 'EBAY', 'NFLX', 'TTWO', 'EA', 'NVDA', 'WORK', 'INTC', 'DXCM', 'ZM', 'INTU', 'ASML']
tickers = getSp500Tickers()
tickers = [ticker.strip() for ticker in tickers]

news_tables = scrapeFinviz(tickers)

printFinviz(tickers, news_tables)

parsed_news = parseFinviz(news_tables)

news = analyzeSentiment(parsed_news)

# View Data
news['Date'] = pd.to_datetime(news.Date).dt.date

unique_ticker = news['Ticker'].unique().tolist()
news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

values = []
for ticker in tickers:
    if ticker not in news_dict:
        continue
    dataframe = news_dict[ticker]
    dataframe = dataframe.set_index('Ticker')
    dataframe = dataframe.drop(columns=['Headline'])
    print('\n')
    print(dataframe.head())

    mean = round(dataframe['compound'].mean(), 2)
    values.append(mean)

df = pd.DataFrame(list(zip(tickers, values)), columns=['Ticker', 'Mean Sentiment'])
df = df.set_index('Ticker')
df = df.sort_values('Mean Sentiment', ascending=False)
print('\n')
print(df)