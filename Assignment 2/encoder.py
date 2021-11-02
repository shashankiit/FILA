import argparse
# data/attt/states/states_file_p1.txt
# data/attt/policies/p2_policy2.txt
# python encoder.py  --policy data/attt/policies/p2_policy2.txt --states data/attt/states/states_file_p1.txt
BLANK = '0'


def check_agent_won(testst, agent_id):
    a = [testst[0], testst[1], testst[2]]
    b = [testst[3], testst[4], testst[5]]
    c = [testst[6], testst[7], testst[8]]
    grid = [a, b, c]
    for i in range(3):
        if grid[i][0] == grid[i][1] and grid[i][1] == grid[i][2] and grid[i][0] == agent_id:
            return 1
    for j in range(3):
        if grid[0][j] == grid[1][j] and grid[1][j] == grid[2][j] and grid[0][j] == agent_id:
            return 1
    if grid[0][0] == grid[1][1] and grid[1][1] == grid[2][2] and grid[1][1] == agent_id:
        return 1
    if grid[2][0] == grid[1][1] and grid[1][1] == grid[0][2] and grid[1][1] == agent_id:
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", help="path to the policy file of player 2", required=True)
    parser.add_argument("--states",
                        help="path to the state file of player 1", required=True)
    args = parser.parse_args()

    polfile = open(args.policy, 'r')
    pollines = polfile.readlines()
    polfile.close()

    statefile = open(args.states, 'r')
    states = statefile.read().split()  # states of player 1
    statefile.close()
    states.append("win")
    states.append("loss")
    states.append("tie")

    map_state_mdpstate = {}
    for i, s in enumerate(states):
        map_state_mdpstate[s] = i

    print(f"numStates {len(states)}")
    print("numActions 9")
    p2_policy = {}
    p2_id = pollines[0].strip()
    agent_id = '1' if p2_id == '2' else '2'

    for line in pollines[1:]:  # policy of player 2
        elems = line.split()
        st = elems[0]
        p2_policy[st] = list(map(float, elems[1:]))

    end = [map_state_mdpstate["win"],
           map_state_mdpstate["loss"], map_state_mdpstate["tie"]]
    print("end {} {} {}".format(end[0], end[1], end[2]))
    for s in states:
        # print("Player 1 state: ", s)
        if s in ["win", "loss", "tie"]:
            continue
        for action in range(9):
            # print(f"Action {action}")
            if s[action] == BLANK:
                snew = list(s)
                snew[action] = agent_id  # Player 1 (agent) takes action
                snew = "".join(snew)
                # print("State after player 1 action :", snew)
                try:
                    prob = p2_policy[snew]
                    for i, p in enumerate(prob):
                        if p == 0:
                            continue
                        s1new = list(snew)
                        s1new[i] = p2_id
                        # Player 2 has made a move. Possible state of agent
                        s1new = "".join(s1new)
                        # print("Action by player 2: ", i, " with prob :", p)
                        # print("State after player 2 action :", s1new)
                        if s1new in states:
                            mdps2 = map_state_mdpstate[s1new]
                            print(
                                f"transition {map_state_mdpstate[s]} {action} {mdps2} 0 {p}")
                        # this means p2 has lost( won in normal ttt) or game is over
                        else:
                            if check_agent_won(s1new, p2_id):
                                winst = map_state_mdpstate["win"]
                                print(
                                    f"transition {map_state_mdpstate[s]} {action} {winst} 1 {p}")
                            else:
                                tiest = map_state_mdpstate["tie"]
                                print(
                                    f"transition {map_state_mdpstate[s]} {action} {tiest} 0 {p}")
                            pass

                # either game is over or agent has lost (won in normal ttt and lost in attt)
                except:
                    testst = "".join(snew)
                    # Check if this state agent has won
                    if check_agent_won(testst, agent_id):
                        lossst = map_state_mdpstate["loss"]
                        print(
                            f"transition {map_state_mdpstate[s]} {action} {lossst} 0 1")
                    else:
                        tiest = map_state_mdpstate["tie"]
                        print(
                            f"transition {map_state_mdpstate[s]} {action} {tiest} 0 1")
    print("mdptype episodic")
    print("discount 0.9")
