"""
Grid search on s and k
"""

from main import main

if __name__ == "__main__":

    ######################
    # Problem parameters #
    ######################

    S = 2000 # Number of samples
    ns = 200 # Number of simulations
    s_list = [10, 20, 50, 100, 250, 500, 750, 1000] # Matrix size
    k_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 750, 1_000] # Increment size

    for s in s_list:
        main(S, ns, s, k_list)
