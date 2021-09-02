import argparse
import numpy as np
import math


class Bandit:
    def __init__(self, mean_rew):
        self.mean_rew = mean_rew
        self.num_pulls = 0
        self.reward = 0

    def pull(self):
        self.num_pulls += 1
        reward = np.random.binomial(1, self.mean_rew)
        self.reward += reward
        return reward

    def getempmean(self):
        if self.num_pulls > 0:
            return self.reward/self.num_pulls
        else:
            return 0

    def getucb(self, step):
        ucb = self.getempmean() + math.sqrt(2*math.log(step)/self.num_pulls)
        return ucb

    def getucb2(self, c, step):
        ucb = self.getempmean() + math.sqrt(c*math.log(step)/self.num_pulls)
        return ucb


class SupBandit:
    def __init__(self, support, prob) -> None:
        self.support = support
        self.prob = prob
        self.reward = 0
        self.num_pulls = 0

    def pull(self):
        self.num_pulls += 1
        rew = np.random.choice(self.support, p=self.prob)
        self.reward += rew
        return rew


def ucb_t2(multi_band, hz, best, c):
    num_hand = len(multi_band)
    # indexes = np.arange(num_hand)
    initstep = np.arange(num_hand)
    rew = 0
    reg = 0
    maxatt = best * hz
    np.random.shuffle(initstep)
    for i in range(hz):
        if i < num_hand:
            ind = initstep[i]
            t = multi_band[ind].pull()
        else:
            ucblist = [x.getucb2(c, i) for x in multi_band]
            opt_ind = ucblist.index(max(ucblist))
            t = multi_band[opt_ind].pull()
        rew += t
        reg = maxatt - rew
    return reg


def thompson_sampling_t1(multi_band, hz, best):
    rew, reg = 0, 0
    maxatt = hz*best
    for i in range(hz):
        beta_sample_list = [np.random.beta(
            x.reward+1, x.num_pulls-x.reward+1) for x in multi_band]
        ind = beta_sample_list.index(max(beta_sample_list))
        t = multi_band[ind].pull()
        rew += t
        reg = maxatt - rew
    return reg


def kl_ucb_t1(multi_band, hz, best):
    return 4


def ucb_t1(multi_band, hz, best):
    num_hand = len(multi_band)
    # indexes = np.arange(num_hand)
    initstep = np.arange(num_hand)
    rew = 0
    reg = 0
    t = 0
    maxatt = best * hz
    np.random.shuffle(initstep)
    for i in range(hz):
        if i < num_hand:
            ind = initstep[i]
            t = multi_band[ind].pull()
        else:
            ucblist = [x.getucb(i+1) for x in multi_band]
            opt_ind = ucblist.index(max(ucblist))
            t = multi_band[opt_ind].pull()
        rew += t
        reg = maxatt - rew
    return reg


def epsilon_greedy_t1(multi_band, hz, ep, best):
    '''
    Returns regret with eG3 algorithm
    multi_band: list of bandit classes
    hz: horizon
    ep: epsilon
    best: mean of optimal arm instance
    '''
    num_hand = len(multi_band)
    indexes = list(range(num_hand))
    opt_ind = 0
    rew = 0
    t = 0
    maxatt = best*hz
    reg = 0
    for i in range(hz):
        dec = np.random.binomial(1, ep)
        if dec == 1:  # sample uniformly
            ind = np.random.choice(indexes)
            t = multi_band[ind].pull()
        else:
            emp_mean_list = [x.getempmean() for x in multi_band]
            opt_ind = emp_mean_list.index(max(emp_mean_list))
            t = multi_band[opt_ind].pull()
        rew += t
        reg = maxatt - rew
    return reg


def alg_t3():
    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, metavar="in",
                    help="path to the instance file")
    ap.add_argument("--algorithm", required=True, metavar="al",
                    help="algorithm to use", choices=["epsilon-greedy-t1", "ucb-t1", "kl-ucb-t1", "thompson-sampling-t1", "ucb-t2", "alg-t3", "alg-t4"])
    ap.add_argument("--randomSeed", required=True, metavar="rs",
                    help="non-negative integer")
    ap.add_argument("--epsilon", required=True, metavar="ep",
                    help="ep is a number in [0, 1]")
    ap.add_argument("--scale", default=2, metavar="c",
                    help="c is a positive real number")
    ap.add_argument("--threshold", default=0, metavar="th",
                    help="th is a number in [0, 1]")
    ap.add_argument("--horizon", required=True, metavar="hz",
                    help="non-negative integer")

    args = vars(ap.parse_args())

    inspath = args["instance"]
    alg = args["algorithm"]
    rs = int(args["randomSeed"])
    ep = float(args["epsilon"])
    c = float(args["scale"])
    th = float(args["threshold"])
    hz = int(args["horizon"])

    np.random.seed(int(rs))

    reg, high = 0, 0
    if alg == 'epsilon-greedy-t1':
        f = open(inspath, "r")
        rewards = [float(x.strip()) for x in list(f.readlines())]
        f.close()
        num_hands = len(rewards)
        multi_band = []
        for i in range(num_hands):
            multi_band.append(Bandit(rewards[i]))
        reg = epsilon_greedy_t1(multi_band, hz, ep, max(rewards))

    elif alg == 'ucb-t1':
        f = open(inspath, "r")
        rewards = [float(x.strip()) for x in list(f.readlines())]
        f.close()
        num_hands = len(rewards)
        multi_band = []
        for i in range(num_hands):
            multi_band.append(Bandit(rewards[i]))
        reg = ucb_t1(multi_band, hz, max(rewards))

    elif alg == 'kl-ucb-t1':
        f = open(inspath, "r")
        rewards = [float(x.strip()) for x in list(f.readlines())]
        f.close()
        num_hands = len(rewards)
        multi_band = []
        for i in range(num_hands):
            multi_band.append(Bandit(rewards[i]))
        reg = kl_ucb_t1(multi_band, hz, max(rewards))

    elif alg == 'thompson-sampling-t1':
        f = open(inspath, "r")
        rewards = [float(x.strip()) for x in list(f.readlines())]
        f.close()
        num_hands = len(rewards)
        multi_band = []
        for i in range(num_hands):
            multi_band.append(Bandit(rewards[i]))
        reg = thompson_sampling_t1(multi_band, hz, max(rewards))

    elif alg == 'ucb-t2':
        f = open(inspath, "r")
        rewards = [float(x.strip()) for x in list(f.readlines())]
        f.close()
        num_hands = len(rewards)
        multi_band = []
        for i in range(num_hands):
            multi_band.append(Bandit(rewards[i]))
        reg = ucb_t2(multi_band, hz, max(rewards), c)

    elif alg == 'alg-t3':
        f = open(inspath, "r")
        supportrewards = f.readlines()
        f.close()
        suprt = list(map(float, supportrewards[0].strip().split()))
        rewards = [list(map(float, x.strip().split()))
                   for x in supportrewards[1:]]
        num_hands = len(rewards)
        # Calculate best of rewards
        a = np.array(suprt)
        b = np.array(rewards)
        temp = b * a
        d = np.sum(temp, axis=1)
        best = np.max(d)
        print(best)
        multi_band = []
        for i in range(num_hands):
            multi_band.append(SupBandit(suprt, rewards[i]))

        # reg = alg_t3(multi_band, hz, findmax(rewards), c)

    print(f"{inspath}, {alg}, {rs}, {ep}, {c}, {th}, {hz}, {reg}, {high}\n")
