import argparse
# import os
# import psutil
import numpy as np
import pulp
# import time
# import json


class MDPState:
    def __init__(self, id):
        self.id = id
        self.data = {}
        self.policy = 0  # action to take according to policy
        self.value_func = 0
        pass

    def add_transition(self, ac, s2, r, p):
        if ac not in self.data.keys():
            self.data[ac] = {}

        self.data[ac][s2] = {"reward": r, "prob": p}
        pass

    def get_transprobs(numStates):
        pass


def value_iteration1(mdp_trans, gamma, set_tak_act, end):
    ns = len(mdp_trans)
    na = len(mdp_trans[0])
    curr_val = [1]*ns
    value_func = [0]*ns
    policy = [0]*ns
    i = 0
    while True:
        i += 1
        curr_val = value_func.copy()
        for state in range(ns):
            if state in end:
                continue

            optimal_action = 0
            val_max = float('-inf')
            for action in sorted(set_tak_act[state]):
                st_ac_trans = mdp_trans[state][action]
                val_s1 = 0
                for s2, rew, prob in st_ac_trans:
                    val_s1 += prob*(rew + gamma*curr_val[s2])
                if val_s1 > val_max:
                    val_max = val_s1
                    optimal_action = action
            if len(set_tak_act[state]) == 0:
                val_max = 0
            value_func[state] = val_max
            policy[state] = optimal_action
        if np.linalg.norm(np.array(curr_val)-np.array(value_func)) <= 1e-10:
            # print("Total: ", i)
            for s in range(ns):
                if len(set_tak_act[s]) == 0 or policy[s] in set_tak_act[s]:
                    continue
                else:
                    for x in set_tak_act[s]:
                        policy[s] = x
                        break
            return value_func, policy


def how_pol_itr1(mdp_trans, gamma, set_tak_act, end):
    ns = len(mdp_trans)
    na = len(mdp_trans[0])
    policy = []
    for s in range(ns):
        if len(set_tak_act[s]) == 0:
            policy.append(0)
        else:
            for x in sorted(set_tak_act[s]):
                policy.append(x)
                break
    i = 0  # TODO check this
    while i < 10:
        i += 1
        curr_val = [0]*ns
        value_func = [1]*ns
        # evaluate value function by iteration
        while True:
            curr_val = value_func.copy()
            for state in range(ns):
                if state in end:
                    value_func[state] = 0
                    continue
                st_ac_trans = mdp_trans[state][policy[state]]
                val_s1 = 0
                for s2, rew, prob in st_ac_trans:
                    val_s1 += prob*(rew + gamma*curr_val[s2])
                value_func[state] = val_s1
            if np.linalg.norm(np.array(curr_val)-np.array(value_func)) <= 1e-10:
                break

        new_policy = policy.copy()
        for state in range(ns):
            if state in end:
                continue
            curr_Q = value_func[state]
            max_Q = curr_Q
            optimal_action = policy[state]

            for action in sorted(set_tak_act[state]):
                Q_state_action = 0
                st_ac_trans = mdp_trans[state][action]
                for s2, rew, prob in st_ac_trans:
                    Q_state_action += prob*(rew + gamma*value_func[s2])

                if abs(Q_state_action - curr_Q) >= 1e-10 and Q_state_action > max_Q:
                    optimal_action = action
                    max_Q = Q_state_action
            new_policy[state] = optimal_action
        if np.array_equal(policy, new_policy):
            for s in range(ns):
                if len(set_tak_act[s]) == 0 or policy[s] in set_tak_act[s]:
                    continue
                else:
                    for x in set_tak_act[s]:
                        policy[s] = x
                        break
            return value_func, policy
        else:
            policy = new_policy
    return value_func, policy


def linear_prog(mdp_trans, set_tak_act, gamma):
    ns = len(mdp_trans)
    na = len(mdp_trans[0])

    prob = pulp.LpProblem("ValueFn", pulp.LpMinimize)
    value_function = [pulp.LpVariable('V'+str(i)) for i in range(ns)]

    prob += pulp.lpSum(value_function)

    for state in range(ns):
        for action in set_tak_act[state]:
            qsaprob = [prob * (rew + gamma*value_function[s2])
                       for s2, rew, prob in mdp_trans[state][action]]

            if len(mdp_trans[state][action]) > 0:
                prob += value_function[state] >= pulp.lpSum(qsaprob)
        if len(set_tak_act[state]) == 0:
            prob += value_function[state] >= 0

    optimization_result = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    num_value_function = [pulp.value(x) for x in value_function]

    policy = [0 for i in range(ns)]
    for state in range(ns):
        # value = V0[state]
        # actVal = [actVlEval(V0, state, action, mdp_trans, gamma)
        #           for action in range(na)]
        # action = (np.abs(np.asarray(actVal) - value)).argmin()
        # policy[state] = action
        Qs = [float("-inf")] * na
        for action in set_tak_act[state]:
            st_ac_trans = mdp_trans[state][action]
            Qs[action] = 0
            for s2, rew, prob in st_ac_trans:
                Qs[action] += prob * (rew + gamma*num_value_function[s2])
        policy[state] = np.argmax(Qs)
    for s in range(ns):
        if len(set_tak_act[s]) == 0 or policy[s] in set_tak_act[s]:
            continue
        else:
            for x in set_tak_act[s]:
                policy[s] = x
                break
    return num_value_function, policy


def eval_inp(mdp, algorithm):
    f = open(mdp, 'r')
    lines = f.readlines()
    f.close()
    state_list = []
    set_takeable_actions = {}

    for line in lines:
        x = line.split()

        if x[0] == "numStates":
            numStates = int(x[1])

            for i in range(numStates):
                # Initialize state classes
                set_takeable_actions[i] = set()
                state_list.append(MDPState(i))
                pass
        elif x[0] == "numActions":
            numActions = int(x[1])
            mdp_trans = [[[] for a in range(numActions)]
                         for s in range(numStates)]
        elif x[0] == "start":
            start = int(x[1])
        elif x[0] == "end":
            end = list(map(int, x[1:]))
        elif x[0] == 'mdptype':
            mdptype = x[1]
        elif x[0] == "discount":
            discount = float(x[1])
        elif x[0] == 'transition':
            s1 = int(x[1])
            ac = int(x[2])
            s2 = int(x[3])
            r = float(x[4])
            p = float(x[5])

            mdp_trans[s1][ac].append((s2, r, p))
            set_takeable_actions[s1].add(ac)

    alg = algorithm

    if alg == "vi":
        return value_iteration1(mdp_trans, discount, set_takeable_actions, end)
    elif alg == "hpi":
        return how_pol_itr1(mdp_trans, discount, set_takeable_actions, end)
    else:
        return linear_prog(mdp_trans,  set_takeable_actions, discount)
    # print(numStates, numActions, start, end, mdptype, discount)


if __name__ == "__main__":
    # begin = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mdp", help="path to the input MDP file", required=True)
    parser.add_argument("--algorithm", choices=["vi", "hpi", "lp"],
                        help="Specify algorithm from vi, hpi, and lp", default="hpi")
    args = parser.parse_args()
    val, policy = eval_inp(args.mdp, args.algorithm)
    for i in range(len(val)):
        print("{:.6f} {}".format(val[i], policy[i]))

    # end = time.time()
    # f = open("record.txt", 'a')
    # f.write(str(end-begin)+'\n')
    # f.close()
    # print(end-begin)
