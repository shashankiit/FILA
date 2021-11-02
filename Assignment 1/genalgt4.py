from bandit import SupBandit, alg_t4
import numpy as np
seeds = list(range(50))
ins = 2
th = 0.2

file = f"../instances/instances-task4/i-{ins}.txt"
hz = [100, 400, 1600, 6400, 25600, 102400]
# alg = "thompson-sampling-t1"
# alg = "ucb-t1"
alg = "alg-t4"
print(f"instance {ins} :: threshold {th}")

f = open(file, "r")
supportrewards = f.readlines()
f.close()
suprt = list(map(float, supportrewards[0].strip().split()))
rewards = [list(map(float, x.strip().split()))
           for x in supportrewards[1:]]

num_hands = len(rewards)
# Calculate best of rewards ----------------------------------------------

a = np.array(suprt)
ind = a > th
maxhigh = []
for i in range(num_hands):
    b = np.array(rewards[i])
    maxhigh.append(np.around(np.sum(b[ind]), 4))
best = max(maxhigh)
ep = 0.02
c = 2.0

high = 0

# --------------------------------------------------------------------------

f1 = open(f"Task4/i-{ins}-threshhold-{th}-results.txt", "w")
toplot = []
for h in hz:
    print(h)
    a = []
    for s in seeds:
        print(f"{s}", end="\r", flush=True)
        np.random.seed(s)
        multi_band = []
        for i in range(num_hands):
            multi_band.append(SupBandit(suprt, rewards[i]))
        highreg, high = alg_t4(multi_band, h, best, th)
        f1.write(f"{file}, alg-t4, {s}, {ep}, {c}, {th}, {h}, 0, {high}\n")
        a.append(highreg)
    y = np.mean(a)
    toplot.append(y)

f1.close()
np.save(f"Task4/i-{ins}-threshhold-{th}-plot", toplot)
print(toplot)
