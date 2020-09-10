
Install Conda and Required Packages:
- First install Anaconda
conda create --name stock-toolkit python
conda activate stock-toolkit
conda install pandas matplotlib
conda install -c anaconda sqlalchemy
pip install backtrader requests pandas numpy matplotlib yahoo_fin sklearn beautifulsoup4 nltk lxml requests_html ta get-all-tickers fastquant

- For CPU Tensorflow:
pip install tensorflow
- For GPU Tensorflow:
pip install tensorflow-gpu

- How to Watch Training: 
tensorboard --logdir="logs"




Optimizing:

Underfitting – Validation and training error high

Overfitting – Validation error is high, training error low

Good fit – Validation error low, slightly higher than the training error

Unknown fit - Validation error low, training error 'high'





















Backtrader Reference: https://analyzingalpha.com/backtrader-backtesting-trading-strategies

Neural Reference: https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras

Sentiment Reference: https://towardsdatascience.com/stock-news-sentiment-analysis-with-python-193d4b4378d4
