""":"

# Find a suitable python interpreter (adapt for your specific needs)
for cmd in ~/anaconda3/envs/stock-toolkit/bin/python /usr/bin/python /usr/bin/python3 ; do
   command -v > /dev/null $cmd && exec $cmd $0 "$@"
done

echo "Python not found!" >2

exit 2

":"""

import shutil


try:
    shutil.rmtree("logs")
except Exception:
    pass

try:
    shutil.rmtree("results")
except Exception:
    pass

try:
    shutil.rmtree("data")
except Exception:
    pass
