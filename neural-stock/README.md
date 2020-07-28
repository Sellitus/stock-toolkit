Setup:

(Install Anaconda)

conda create -n neural-stock python

conda activate neural-stock

- If you want to use your GPU, install CUDA first and run:

pip install tensorflow-gpu

pip install tensorflow pandas numpy matplotlib yahoo_fin sklearn


How to Watch Training:
tensorboard --logdir="logs"

Sentiment Analysis Setup:

pip install beautifulsoup4

pip install nltk


















Neural Reference: https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras

Sentiment Reference: https://towardsdatascience.com/stock-news-sentiment-analysis-with-python-193d4b4378d4



