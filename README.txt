# Stock-Toolkit (legacy)

Setup Guide:


--- A ---

Either install a VM in any hypervisor OR install linux on native hardware (only necessary if running neural-stock with GPU accelerated tensorflow)



--- B ---

Only if you are running GPU accelerated tensorflow, look up a guide on how to install the most recent version of CUDA, drivers and other supporting packages to prepare your environment for running tensorflow-gpu



--- C ---

Run the setup_stock-toolkit.sh bash file as a non-root user:
bash setup_stock-toolkit.sh

Activate after installation:
source ~/.bashrc
conda activate stock-toolkit

# Install TA lib Linux library:
https://github.com/mrjbq7/ta-lib#dependencies

conda create --name stock-toolkit python
conda activate stock-toolkit
conda install pandas matplotlib
conda install -c anaconda sqlalchemy
pip install backtrader requests pandas numpy matplotlib yahoo_fin sklearn beautifulsoup4 nltk lxml requests_html ta get-all-tickers fastquant ta-lib



##### IF tensorflow is needed (for running neural-stock only), install tensorflow within the conda environment:

# Update all pip packages in conda env
pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U

- For GPU Tensorflow (extremely fast):
pip install tensorflow-gpu==2.6 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
- For CPU Tensorflow (much slower):
pip install tensorflow==2.6 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com


- How to Watch Training: 
tensorboard --logdir="logs"









Optimizing:

Underfitting – Validation and training error high

Overfitting – Validation error is high, training error low

Good fit – Validation error low, slightly higher than the training error

Unknown fit - Validation error low, training error 'high'













TROUBLESHOOTING:

If Yahoo Finance is having issues, run this with conda activated:
pip install yahoo_fin --update







Backtrader Reference: https://analyzingalpha.com/backtrader-backtesting-trading-strategies

Neural Reference: https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras

Sentiment Reference: https://towardsdatascience.com/stock-news-sentiment-analysis-with-python-193d4b4378d4
