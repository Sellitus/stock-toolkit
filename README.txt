Setup Guide:
  A1) Install CUDA in Windows Subsystem for Linux
  A2) Install Hyper-V VM with GPU Passthrough
  B) Install NVIDIA CUDA
  C) Install all necessary packages for running scripts on the Ubuntu machine
  D) EXTRA: Disable GPU Passthrough


--- A1 ---

- Install Windows drivers from here: https://developer.nvidia.com/cuda/wsl/download
- Install Windows Subsystem for Linux Windows 10 feature
- Install a flavor of Ubuntu from the Windows Store
- Launch and setup Ubuntu for WSL
- Install Windows drivers from here: https://developer.nvidia.com/cuda/wsl/download
- Install appropriate repo and packages within Ubuntu

sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo apt update

- Check the repo URL in /etc/apt/sources.list.d/cuda.list and get newest version for the following command

sudo apt install -y cuda-toolkit-11-4







--- A2 ---

1) Create Ubuntu VM in Hyper-V.
2) Fully install Ubuntu OS within VM and shut the VM down.
3) Enable GPU passthrough:

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
- First install Anaconda (don't use sudo), get the newest install script from here: https://www.anaconda.com/products/individual

curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
(make sure to say yes to running conda init during install)
source ~/.bashrc

conda create --name stock-toolkit python
conda activate stock-toolkit
conda install pandas matplotlib
conda install -c anaconda sqlalchemy
# keras-buoy removed from pip install list
pip install backtrader requests pandas numpy matplotlib yahoo_fin sklearn beautifulsoup4 nltk lxml requests_html ta get-all-tickers fastquant
# Update all pip packages in conda env
pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U

- For GPU Tensorflow (extremely fast):
pip install tensorflow-gpu==2.6 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
- For CPU Tensorflow (much slower):
pip install tensorflow==2.6 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com


- How to Watch Training: 
tensorboard --logdir="logs"


--- D ---

-Disable GPU Passthrough

(Use the same values used for $VM and $Location as in step A3

Remove-VMAssignableDevice -LocationPath $Location -VMName $VM
Enable-PnpDevice $Location
Mount-VMHostAssignableDevice -LocationPath $Location




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
