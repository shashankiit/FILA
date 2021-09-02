from bandit import Bandit, ucb_t2
import numpy as np
seeds = list(range(50))
ins = 5
file = f"../instances/instances-task2/i-{ins}.txt"
hz = 10000
alg = "ucb-t2"
scale = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14,
         0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]

f = open(file, "r")
rewards = [float(x.strip()) for x in list(f.readlines())]
f.close()
ep = 0.02
th = 0
high = 0
num_hands = len(rewards)


f1 = open(f"Task2/task2-results-i-{ins}.txt", "w")
toplot = []
for c in scale:
    print(f"c = {c}\n")
    a = []
    for s in seeds:
        print(f"{s}", end="\r", flush=True)
        np.random.seed(s)
        multi_band = []
        for i in range(num_hands):
            multi_band.append(Bandit(rewards[i]))
        reg = ucb_t2(multi_band, hz, max(rewards), c)
        f1.write(f"{file}, {alg}, {s}, {ep}, {c}, {th}, {hz}, {reg}, {high}\n")
        a.append(reg)
    y = np.mean(a)
    toplot.append(y)

f1.close()
np.save(f"Task2/i-{ins} plot", toplot)
print(toplot)
