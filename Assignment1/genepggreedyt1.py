from bandit import Bandit, epsilon_greedy_t1
import numpy as np
seeds = list(range(50))
ins = 3
file = f"../instances/instances-task1/i-{ins}.txt"
hz = [100, 400, 1600, 6400, 25600, 102400]
alg = "epsilon-greedy-t1"


f = open(file, "r")
rewards = [float(x.strip()) for x in list(f.readlines())]
f.close()
ep = 0.02
c = 2
th = 0
high = 0
num_hands = len(rewards)


f1 = open(f"Task1/{alg}/i-{ins}-results.txt", "w")
toplot = []
for h in hz:
    print(h)
    a = []
    for s in seeds:
        print(s, end="\r", flush=True)
        np.random.seed(s)
        multi_band = []
        for i in range(num_hands):
            multi_band.append(Bandit(rewards[i]))
        reg = epsilon_greedy_t1(multi_band, h, ep, max(rewards))
        f1.write(f"{file}, {alg}, {s}, {ep}, {c}, {th}, {h}, {reg}, {high}\n")
        a.append(reg)
    y = np.mean(a)
    toplot.append(y)

f1.close()
np.save(f"Task1/{alg}/i-{ins} plot", toplot)
print(toplot)
