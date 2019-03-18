import run_random
import run_simple
import run_rl
import os, sys
print("Working dir:"+os.getcwd())
print("Python version:"+sys.version)

randDict = []
simpleDict = []
rlDict = []

for id in range(0, 8):
    randDict.append(run_random.main(id))


randSuc = 0
randFail = 0
randPc = 0
randRuns = 0
for r in randDict:
    randSuc += r.get("Success", "none")
    randFail += r.get("Failures", "none")
    randRuns += r.get("Episodes", "none")
    randPc += r.get("SuccessRate", "none")
randPc /= 8

print("Successes for random agent: ", randSuc)
print("Failures by random agent: ", randFail)
print("Number of attempts: ", randRuns)
print("Rate of success for random agent: ", randPc, "%")

for id in range(0, 8):
    simpleDict.append(run_simple.main(id))

simpSuc = 0
simpFail = 0
simpPc = 0
simpRuns = 0
for s in simpleDict:
    simpSuc += s.get("Success", "none")
    simpFail += s.get("Failures", "none")
    simpRuns += s.get("Episodes", "none")
    simpPc += s.get("SuccessRate", "none")
simpPc /= 8

print("Successes for simple agent: ", simpSuc)
print("Failures by simple agent: ", simpFail)
print("Number of attempts: ", simpRuns)
print("Rate of success for simple agent: ", simpPc, "%")

for id in range(0, 8):
    rlDict.append(run_rl.main(id))

rlSuc = 0
rlFail = 0
rlPc = 0
rlRuns = 0
for r in rlDict:
    rlSuc += r.get("Success", "none")
    rlFail += r.get("Failures", "none")
    rlRuns += r.get("Episodes", "none")
    rlPc += r.get("SuccessRate", "none")
rlPc /= 8

print("Successes for RL agent: ", rlSuc)
print("Failures by RL agent: ", rlFail)
print("Number of attempts: ", rlRuns)
print("Rate of success for RL agent: ", rlPc, "%")
