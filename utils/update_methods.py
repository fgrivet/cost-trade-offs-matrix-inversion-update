"""
This file implement the update methods of the article: the DI method (Algorithm 1), the ISM method (Algorithm 2), and the WMI method (Algorithm 3)
"""

import time

import numpy as np

from .inversion import fpd_inv


def di_method(mm: np.ndarray, mm_inv: np.ndarray, X: np.ndarray, N: int) -> tuple[np.ndarray, float]:
    """Implementation of the Direct Inversion (DI) method: Algorithm 1

    Parameters
    ----------
    mm : np.ndarray
        The moment matrix to update, spd matrix of size (s, s)
    mm_inv : np.ndarray
        The inverse of the moment matrix to update, spd matrix of size (s, s)
    X : np.ndarray
        The design matrix containing the k new vectors to add, matrix of size (k, s)
    N : int
        The number of training data

    Returns
    -------
    tuple[np.ndarray, float]
        The inverse of the moment matrix updated with size (s, s) and the execution time of the DI method
    """
    # Step 1: denormalization
    M = mm.copy() * N

    # Step 2: DI method (algorithm 1)
    time_start = time.perf_counter()
    k = X.shape[0]
    M_updated = M + X.T @ X
    M_updated_inv = fpd_inv(M_updated)
    time_end = time.perf_counter()

    # Step 3: renormalization
    result = M_updated_inv * (N + k)

    return result, time_end - time_start


def ism_method(mm: np.ndarray, mm_inv: np.ndarray, X: np.ndarray, N: int) -> tuple[np.ndarray, float]:
    """Implementation of the Iterative Sherman-Morrison (ISM) method: Algorithm 2

    Parameters
    ----------
    mm : np.ndarray
        The moment matrix to update, spd matrix of size (s, s)
    mm_inv : np.ndarray
        The inverse of the moment matrix to update, spd matrix of size (s, s)
    X : np.ndarray
        The design matrix containing the k new vectors to add, matrix of size (k, s)
    N : int
        The number of training data

    Returns
    -------
    tuple[np.ndarray, float]
        The inverse of the moment matrix updated with size (s, s) and the execution time of the DI method
    """
    # Step 1: denormalization
    M_inv = mm_inv.copy() / N

    # Step 2: ISM method (algorithm 2)
    time_start = time.perf_counter()
    k = X.shape[0]
    for i in range(k):
        v_x = X[i, :].reshape(-1, 1)
        l = M_inv @ v_x
        d = 1 + v_x.T @ l
        l_d = l / d
        M_updated_inv = M_inv - l_d @ l.T
    time_end = time.perf_counter()

    # Step 3: renormalization
    result = M_updated_inv * (N + k)

    return result, time_end - time_start


def wmi_method(mm: np.ndarray, mm_inv: np.ndarray, X: np.ndarray, N: int) -> tuple[np.ndarray, float]:
    """Implementation of the Woodbury Matrix Identity (WMI) method: Algorithm 3

    Parameters
    ----------
    mm : np.ndarray
        The moment matrix to update, spd matrix of size (s, s)
    mm_inv : np.ndarray
        The inverse of the moment matrix to update, spd matrix of size (s, s)
    X : np.ndarray
        The design matrix containing the k new vectors to add, matrix of size (k, s)
    N : int
        The number of training data

    Returns
    -------
    tuple[np.ndarray, float]
        The inverse of the moment matrix updated with size (s, s) and the execution time of the DI method
    """
    # Step 1: denormalization
    M_inv = mm_inv.copy() / N

    # Step 2: WMI method (algorithm 3)
    time_start = time.perf_counter()
    k = X.shape[0]
    I = np.eye(k)
    R = X @ M_inv
    S = I + R @ X.T
    if k == 1:
        S_inv = 1 / S
    else:
        S_inv = fpd_inv(S)
    Q = R.T @ S_inv
    M_updated_inv = M_inv - Q @ R
    time_end = time.perf_counter()

    # Step 3: renormalization
    result = M_updated_inv * (N + k)

    return result, time_end - time_start
