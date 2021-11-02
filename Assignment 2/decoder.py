import argparse
from sys import set_asyncgen_hooks

if __name__ == "__main__":
    # begin = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--value-policy", help="path to the value policy of p1", required=True)
    parser.add_argument(
        "--player-id", help="player id of player 2 to optimize", required=True)
    parser.add_argument("--states",
                        help="statesfilepath of p2", required=True)
    args = parser.parse_args()

    print(args.player_id)
    f = open(args.value_policy)
    val_pol = f.readlines()
    f.close()

    f = open(args.states)
    states = f.read()
    states = states.split()
    f.close()

    for i, s in enumerate(states):
        pol = int(val_pol[i].split()[1])
        probs = [0]*9
        probs[pol] = 1
        print("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(
            s, probs[0], probs[1], probs[2], probs[3], probs[4], probs[5], probs[6], probs[7], probs[8]))
