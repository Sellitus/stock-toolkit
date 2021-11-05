

# Make sure the script is run as root
if [[ $(/usr/bin/id -u) -ne 0 ]]; then
    echo "ERROR: Script was not run under the root user. Use sudo"
    exit 1
fi

if [ $# -lt 2 ]
then
  echo "ERROR: The machine's ID must be passed at the first argument, and the non-root user to use for running
  algen.py must be passed as the second argument"
  echo "Usage: sudo bash install_pycron.sh 2 sellitus"
  exit 1
fi

SYSTEM_ID=$1
PYCRON_USER=$2

echo "
[Unit]
Description=PyCron
After=multi-user.target
[Service]
User=$PYCRON_USER
Type=simple
Restart=always
ExecStart=/home/$PYCRON_USER/anaconda3/envs/stock-toolkit/bin/python /home/$PYCRON_USER/PythonProjects/stock-toolkit/genetic-strategy-builder/pycron.py --id $SYSTEM_ID
[Install]
WantedBy=multi-user.target
" > /etc/systemd/system/pycron.service

sudo systemctl enable pycron
sleep 1
sudo systemctl restart pycron

EMAIL_FILE=".email_pw"
if [ ! -f "$EMAIL_FILE" ]; then
  echo "WARNING: Email password file $EMAIL_FILE does not exist. Create this file with the email on the first line and password on the second"
else
  echo "Found email password file $EMAIL_FILE ! Setup should now be complete"
fi
