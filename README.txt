Setup Guide:
  A) Install Hyper-V VM with GPU Passthrough
  B) Install NVIDIA CUDA
  C) Install all necessary packages for running scripts on the Ubuntu machine

--- A ---

1) Create Ubuntu VM in Hyper-V.
2) Enable GPU passthrough:

!!! NOTE !!! : Do not do this on a GPU you are using to see the screen, as it will be disabled.

-Set the $VM variable to the name of the VM:

$VM='ML CUDA Mate 20.04'
Set-VM $VM -GuestControlledCacheTypes $true

-Open device manager, right click on GPU to be dedicated, properties. Go to the 'Details' tab, and select 'Location paths' from the dropdown box. Copy the PCIROOT location (Example: PCIROOT(0)#PCI(0100)#PCI(0000) ) and set the $Location to this value as shown below.

$Location = 'PCIROOT(0)#PCI(0100)#PCI(0000)'

- Set memory limits, based on the size of the GPU's memory
Set-VM $VM -LowMemoryMappedIoSpace 512MB 
Set-VM $VM -HighMemoryMappedIoSpace 6GB

- In device manager, right click the GPU and select 'disable' and wait for it to fully disable.

Dismount-VMHostAssignableDevice -force -LocationPath $Location
Add-VMAssignableDevice -LocationPath $Location -VMName VMName


--- B ---

(Look up updated guide)





--- C ---

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
