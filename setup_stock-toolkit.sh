


# Settings
INSTALL_DIR=$HOME/anaconda3
CONDA_EXE=$HOME/anaconda3/condabin/conda
PIP_EXE=$HOME/anaconda3/envs/stock-toolkit/bin/pip
ENVIRONMENT_DIR=$INSTALL_DIR/envs/stock-toolkit/


# Install necessary prereqs
sudo apt install -y gfortran llvm build-essential postfix mailutils


if [ ! -d "$INSTALL_DIR" ]; then
  # Scrape the Anaconda install page and grab download link
  download_link=$(wget -qO- https://www.anaconda.com/products/individual | grep -io -m 1 'https://repo.anaconda.com/archive/Anaconda3-.*-Linux-x86_64.sh')
  # Download the Anaconda install file and get the filename
  wget "$download_link"
  filename=$(ls Anaconda*-Linux-x86_64.sh)
  # Make sure the Anaconda install file is removed on error or exit
  trap "rm $filename" EXIT
  # Install Anaconda silently
  bash "$filename" -b -p "$INSTALL_DIR"

  # Add conda to the $PATH, init and update conda
  echo 'export PATH=$HOME/anaconda3/bin:$PATH' >> "$HOME/.bashrc"
  yes | $CONDA_EXE init bash
fi



# Update Anaconda
yes | $CONDA_EXE update -n base -c defaults conda

# Remove any existing stock-toolkit conda environment if it exists
if [ -d "$ENVIRONMENT_DIR" ]; then rm -rf $ENVIRONMENT_DIR; fi

# Create the conda environment and install packages to it
# NOTE: Current version of Python (3.10) does not currently work
yes | $CONDA_EXE create --name stock-toolkit python=3.9
# Activate the conda-forge channel
yes | $CONDA_EXE install -c anaconda sqlalchemy
yes | $CONDA_EXE install pandas matplotlib
yes | $PIP_EXE install --upgrade pip
yes | $PIP_EXE install backtrader requests pandas numpy matplotlib yahoo_fin sklearn beautifulsoup4 nltk lxml requests_html ta get-all-tickers fastquant schedule --no-input


echo "
Execute these commands and you're ready to run stock-toolkit projects:

source ~/.bashrc
conda activate stock-toolkit"


