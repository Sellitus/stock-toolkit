
Install Conda and Required Packages:
- First install Anaconda
conda create --name backtrader
conda activate backtrader
conda install pandas matplotlib
conda install -c anaconda sqlalchemy
pip install backtrader requests

Reference: https://analyzingalpha.com/backtrader-backtesting-trading-strategies

Install Bulbea:
git clone https://github.com/achillesrasquinha/bulbea.git && cd bulbea
conda activate backtrader
pip install -i requirements.txt
python setup.py install
pip install tensorflow



Activate Conda Env:
conda activate backtrader