
# Make sure the script is run as root
if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "ERROR: Script was not run under the root user. Use sudo"
    exit 1
fi

if [ "$#" -ne 1 ]
then
  echo "ERROR: The machine's ID must be passed at the first argument"
  echo "Usage: sudo bash install_pycron.sh 2"
  exit 1
fi

echo "
[Unit]
Description=PyCron
After=multi-user.target
[Service]
Type=simple
Restart=always
ExecStart=/home/sellitus/anaconda3/envs/stock-toolkit/bin/python /home/sellitus/PythonProjects/stock-toolkit/genetic-strategy-builder/pycron.py --id $1
[Install]
WantedBy=multi-user.target
" > /etc/systemd/system/pycron.service

sudo systemctl enable pycron
sudo systemctl start pycron
