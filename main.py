"""
Generate results/result_1287.csv which contains the data in Table 3 and Table 4
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.inversion import fpd_inv
from utils.update_methods import di_method, ism_method, wmi_method


def main(S: int, ns: int, s: int, k_list: list[int]):
    """Compare the DI, ISM, and WMI methods in terms of execution time and norm of the error
    Save the results in f"results/result_{s}.csv"

    Parameters
    ----------
    S : int
        Number of samples
    ns : int
        Number of simulations for each method and k value
    s : int
        Matrix size
    k_list : list[int]
        The number of new data points to increment
    """

    result_list = [] # To construct a DataFrame with columns ["method", "s", "N", "k", "iter", "error", "diff_max", "time"])

    #####################
    #  Data generation  #
    #####################

    np.random.seed(42)
    data = np.random.normal(size=(S, s))
    full_mm = data.T @ data / S # Make it spd and normalize it
    eye = np.eye(s) # Initialize the eye matrix to compute the inversion error


    ##########################
    # Loop on increment size #
    ##########################


    for k in (k_bar := tqdm(k_list)):
        # Can't run if the number of new data is greater than the number of samples
        if k > S:
            print(f"Impossible to execute the function if k={k} > {S}=S")

        N = S - k # Number of training data

        # Can't run if the number of training data is lower than the size of the matrix
        if N < s:
            print(f"Impossible to execute the function if N={N} < {s}=s")

        k_bar.set_description(f"Moment matrix of size ({s}, {s}) with {N} data points and adding {k} new data points")
        mm = data[:N].T @ data[:N] / N
        mi = fpd_inv(mm)

        # Simulations loop
        for i in tqdm(range(1, ns+1), leave=False, desc="Simulation"):
            # Direct Inverion (DI) method
            result_di, time_di = di_method(mm, mi, data[N:], N)
            diff_di = np.abs(eye - result_di @ full_mm)
            result_list.append({"method": "DI", "s": s, "N": N, "k":k, "iter": i, "error": np.linalg.norm(diff_di, "fro"), "diff_max": np.max(diff_di), "time": time_di})

            # Woodbury Matrix Identity (WMI) method
            # Doing WMI before ISM to not compute ISM if i > 50 when k > 50
            result_wmi, time_wmi = wmi_method(mm, mi, data[N:], N)
            diff_wmi = np.abs(eye - result_wmi @ full_mm)
            result_list.append({"method": "WMI", "s": s, "N": N, "k":k, "iter": i, "error": np.linalg.norm(diff_wmi, "fro"), "diff_max": np.max(diff_wmi), "time": time_wmi})

            # Make ns = 50 for ISM if k > 50
            if k > 50 and i > 50:
                continue

            # Iterative Sherman-Morrison (ISM) method
            result_ism, time_ism = ism_method(mm, mi, data[N:], N)
            diff_ism = np.abs(eye - result_ism @ full_mm)
            result_list.append({"method": "ISM", "s": s, "N": N, "k":k, "iter": i, "error": np.linalg.norm(diff_ism, "fro"), "diff_max": np.max(diff_ism), "time": time_ism})

        # Save result for every k (just in case)
        result_df = pd.DataFrame(result_list)
        result_df.to_csv(f"results_15000/result_{s}.csv")


if __name__ == "__main__":
    ######################
    # Problem parameters #
    ######################

    S = 2000 # Number of samples
    ns = 200 # Number of simulations
    s = 1287 # Matrix size
    k_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 750, 1_000] # Increment size

    main(S, ns, s, k_list)
