import run_random
import run_simple
import run_rl
import os

for id in range(8):
    os.system("python3 run_random.py %d".format(id))

for id in range(8):
    os.system("python3 run_simple.py %d".format(id))

for id in range(8):
    os.system("python3 run_rl.py %d".format(id))
