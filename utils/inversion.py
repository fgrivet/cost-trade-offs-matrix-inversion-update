"""
This file implement the fpd_inv function that inverse an spd matrix using Cholesky decomposition from lapack
"""

import numpy as np
from scipy.linalg import lapack

# according to: https://stackoverflow.com/a/58719188

inds_cache = {}


def upper_triangular_to_symmetric(ut: np.ndarray):
    """Make a symetric matrix from its upper triangular part

    Parameters
    ----------
    ut : np.ndarray
        The matrix containing only its upper triangular part
    """
    n = ut.shape[0]
    try:
        inds = inds_cache[n]
    except KeyError:
        inds = np.tri(n, k=-1, dtype=bool)
        inds_cache[n] = inds
    ut[inds] = ut.T[inds]


def fpd_inv(M: np.ndarray) -> np.ndarray:
    """Inverse the matrix M using Cholesky decomposition from lapack

    Parameters
    ----------
    M : np.ndarray
        The spd matrix to inverse

    Returns
    -------
    np.ndarray
        The inverse of M (forced to be symmetric)

    Raises
    ------
    ValueError
        If M is not spd and can't be made pd with regularization
    """
    cholesky, info = lapack.dpotrf(M)
    if info != 0:
        print("Error in dpotrf: ", info)
        # Probably not spd, trying to make it pd by adding regularization from 1e-10 to 1e-4
        for eps in range(10, 3, -1):
            cholesky, info = lapack.dpotrf(M + 10**-eps * np.eye(M.shape[0]))
            # Stop if the decomposition worked
            if info == 0:
                break
            else:
                print(f"Error in dpotrf: {info} for eps = {eps}")
        # Raise an error in non worked
        if info != 0:
            raise ValueError("dpotrf failed on input {}".format(M))
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError("dpotri failed on input {}".format(cholesky))
    upper_triangular_to_symmetric(inv)
    return inv
