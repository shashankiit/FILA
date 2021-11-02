from bandit import SupBandit, alg_t3
import numpy as np
seeds = list(range(50))
ins = 2
file = f"../instances/instances-task3/i-{ins}.txt"
hz = [100, 400, 1600, 6400, 25600, 102400]
# alg = "thompson-sampling-t1"
# alg = "ucb-t1"
alg = "epsilon-greedy-t1"
print(alg, f"instance {ins}")

f = open(file, "r")
supportrewards = f.readlines()
f.close()
suprt = list(map(float, supportrewards[0].strip().split()))
rewards = [list(map(float, x.strip().split()))
           for x in supportrewards[1:]]

# Calculate best of rewards ----------------------------------------------
a = np.array(suprt)
b = np.array(rewards)
temp = b * a
d = np.sum(temp, axis=1)
best = np.around(max(d), 4)
ep = 0.02
c = 2.0
th = 0.0
high = 0
num_hands = len(rewards)
# --------------------------------------------------------------------------

f1 = open(f"Task3/i-{ins}-{alg}-results.txt", "w")
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
        reg = np.around(alg_t3(multi_band, h, best, alg), 4)
        f1.write(f"{file}, alg-t3, {s}, {ep}, {c}, {th}, {h}, {reg}, {high}\n")
        a.append(reg)
    y = np.mean(a)
    toplot.append(y)

f1.close()
np.save(f"Task3/i-{ins}-{alg} plot", toplot)
print(toplot)
