rsync -a ~/PycharmProjects/stock-toolkit/ sellitus@192.168.50.105:~/RSYNC/stock-toolkit/ && ssh sellitus@192.168.50.105 -t "tmux new -d -s main1 & tmux send-keys -t main1 \"~/anaconda3/envs/stock-toolkit/bin/python ~/PycharmProjects/stock-toolkit/neural-stock/main.py $@\" ENTER"
