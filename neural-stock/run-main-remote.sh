# --ignore-existing and --delete
rsync -a ~/PycharmProjects/stock-toolkit/ sellitus@192.168.50.105:~/RSYNC/stock-toolkit/ && ssh sellitus@192.168.50.105 -t "tmux new -d -s main1 2>&1 >/dev/null & tmux send-keys -t main1 \"~/anaconda3/envs/stock-toolkit/bin/python ~/RSYNC/stock-toolkit/neural-stock/main.py $@\" ENTER"
