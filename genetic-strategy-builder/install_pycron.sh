

echo "
[Unit]
Description=PyCron
After=multi-user.target
[Service]
Type=simple
Restart=always
ExecStart=/home/sellitus/anaconda3/envs/stock-toolkit/bin/python /home/sellitus/PythonProjects/stock-toolkit/genetic-strategy-builder/pycron.py
[Install]
WantedBy=multi-user.target
" > /etc/systemd/system/pycron.service

sudo systemctl enable pycron
sudo systemctl start pycron
