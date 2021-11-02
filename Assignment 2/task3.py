import os
import subprocess
import shutil


def difference(p1, p2):
    p1l = open(p1, 'r').readlines()[1:]
    p2l = open(p2, 'r').readlines()[1:]
    diff = 0
    for i in range(len(p1l)):
        if p1l[i] != p2l[i]:
            diff += 1
    return diff


if __name__ == "__main__":
    start_with = 2
    mdp_file_path = "task3_results/mdpfile.txt"
    num_iter = 10
    if os.path.exists('task3_results'):
        shutil.rmtree('task3_results')

    os.makedirs('task3_results')
    states_p1_path = "data/attt/states/states_file_p1.txt"
    states_p2_path = "data/attt/states/states_file_p2.txt"
    f = open(states_p1_path)
    p1_states = f.read().split()
    f.close()

    f = open(states_p2_path)
    p2_states = f.read().split()
    f.close()

    if start_with == 1:
        print("Initialized player 2 policy")
        f = open("task3_results/p2_init_pol.txt", 'w')
        f.write("2\n")
        for state in p2_states:
            f.write(state)
            num_tak_act = state.count('0')
            prob = 1/num_tak_act
            for ac in range(9):
                if state[ac] == '0':
                    f.write(f" {prob}")
                else:
                    f.write(f" {0}")
            f.write("\n")
        f.close()

        for i in range(num_iter):
            print("Iteration :", i+1)

            val_pol_p1 = f"task3_results/p1_value_and_policy_file_iter_{i}.txt"
            opt_pol_p1 = f"task3_results/p1_policy_iter_{i}.txt"
            val_pol_p2 = f"task3_results/p2_value_and_policy_file_iter_{i}.txt"
            opt_pol_p2 = f"task3_results/p2_policy_iter_{i}.txt"

            print("\nOptimizing Player 1")

            if i == 0:
                pol_p2_path = "task3_results/p2_init_pol.txt"
            else:
                pol_p2_path = f"task3_results/p2_policy_iter_{i-1}.txt"

            cmd_output = subprocess.check_output(
                "python encoder.py --policy "+pol_p2_path+" --states "+states_p1_path, universal_newlines=True)
            f = open(mdp_file_path, "w")
            f.write(cmd_output)
            f.close()
            cmd_output = subprocess.check_output(
                "python planner.py --mdp "+mdp_file_path, universal_newlines=True)
            f = open(val_pol_p1, "w")
            f.write(cmd_output)
            f.close()
            cmd_output = subprocess.check_output(
                "python decoder.py --value-policy "+val_pol_p1+" --states "+states_p1_path + " --player-id 1", universal_newlines=True)
            f = open(opt_pol_p1, "w")
            f.write(cmd_output)
            f.close()

            if i == 0:
                print("No previous policy to compare for player 1\n")
            else:
                print("No. of different policies for states in player 1 : ",
                      difference(f"task3_results/p1_policy_iter_{i-1}.txt", opt_pol_p1), "\n")

            print("Optimizing Player 2")

            pol_p1_path = opt_pol_p1

            cmd_output = subprocess.check_output(
                "python encoder.py --policy "+pol_p1_path+" --states "+states_p2_path, universal_newlines=True)
            f = open(mdp_file_path, "w")
            f.write(cmd_output)
            f.close()
            cmd_output = subprocess.check_output(
                "python planner.py --mdp "+mdp_file_path, universal_newlines=True)
            f = open(val_pol_p2, "w")
            f.write(cmd_output)
            f.close()
            cmd_output = subprocess.check_output(
                "python decoder.py --value-policy "+val_pol_p2+" --states "+states_p2_path + " --player-id 2", universal_newlines=True)
            f = open(opt_pol_p2, "w")
            f.write(cmd_output)
            f.close()

            print("No. of different policies for states in player 2 : ",
                  difference(pol_p2_path, opt_pol_p2), "\n")

    else:  # player 1 optimal
        print("Initialized player 1 policy")
        f = open("task3_results/p1_init_pol.txt", 'w')
        f.write("1\n")
        for state in p1_states:
            f.write(state)
            num_tak_act = state.count('0')
            prob = 1/num_tak_act
            for ac in range(9):
                if state[ac] == '0':
                    f.write(f" {prob}")
                else:
                    f.write(f" {0}")
            f.write("\n")
        f.close()

        for i in range(num_iter):
            print("Iteration :", i+1)

            val_pol_p1 = f"task3_results/p1_value_and_policy_file_iter_{i}.txt"
            opt_pol_p1 = f"task3_results/p1_policy_iter_{i}.txt"
            val_pol_p2 = f"task3_results/p2_value_and_policy_file_iter_{i}.txt"
            opt_pol_p2 = f"task3_results/p2_policy_iter_{i}.txt"

            print("\nOptimizing Player 2")

            if i == 0:
                pol_p1_path = "task3_results/p1_init_pol.txt"
            else:
                pol_p1_path = f"task3_results/p1_policy_iter_{i-1}.txt"

            cmd_output = subprocess.check_output(
                "python encoder.py --policy "+pol_p1_path+" --states "+states_p2_path, universal_newlines=True)
            f = open(mdp_file_path, "w")
            f.write(cmd_output)
            f.close()
            cmd_output = subprocess.check_output(
                "python planner.py --mdp "+mdp_file_path, universal_newlines=True)
            f = open(val_pol_p2, "w")
            f.write(cmd_output)
            f.close()
            cmd_output = subprocess.check_output(
                "python decoder.py --value-policy "+val_pol_p2+" --states "+states_p2_path + " --player-id 2", universal_newlines=True)
            f = open(opt_pol_p2, "w")
            f.write(cmd_output)
            f.close()

            if i == 0:
                print("No previous policy to compare for player 2\n")
            else:
                print("No. of different policies for states in player 2 : ",
                      difference(f"task3_results/p2_policy_iter_{i-1}.txt", opt_pol_p2), "\n")

            print("Optimizing Player 1")

            pol_p2_path = opt_pol_p2

            cmd_output = subprocess.check_output(
                "python encoder.py --policy "+pol_p2_path+" --states "+states_p1_path, universal_newlines=True)
            f = open(mdp_file_path, "w")
            f.write(cmd_output)
            f.close()
            cmd_output = subprocess.check_output(
                "python planner.py --mdp "+mdp_file_path, universal_newlines=True)
            f = open(val_pol_p1, "w")
            f.write(cmd_output)
            f.close()
            cmd_output = subprocess.check_output(
                "python decoder.py --value-policy "+val_pol_p1+" --states "+states_p1_path + " --player-id 1", universal_newlines=True)
            f = open(opt_pol_p1, "w")
            f.write(cmd_output)
            f.close()

            print("No. of different policies for states in player 1 : ",
                  difference(pol_p1_path, opt_pol_p1), "\n")
