"""Module for Spherical Harmonics

Author: Opal Issan (PhD student @ UCSD). contact: oissan@ucsd.edu
Last modified: July 21st, 2024
"""

import numpy as np
import scipy.special


def get_spherical_harmonic_basis_matrix(
    latitude: list | np.ndarray, longitude: list | np.ndarray, ell=10
):
    """
    A function to construct the spherical harmonic evaluation at GSE locations (latitude, longitude)

    Parameters
    ----------
    latitude: float or na
        latitude location in radians [0, pi]
    longitude: ndarray
        longitude location in radians [0, 2pi]
    ell: ndarray
        integer of the higher order of spherical harmonic fit

    Returns
    -------
    Y: ndarray matrix dimensions = ((ell+1)^2, N)
        [Y_{0 ,0}, Y_{-1, 1}, Y_{0, 1}, Y_{1, 1} ... Y_{ell, ell}]
    """
    # number of sample points
    N = len(latitude)
    # running index
    ii = 0
    # vector with all basis functions evaluated at the [latitude, longitude] location
    Y = np.zeros((N, ell**2 + 2 * ell + 1))
    for n in range(0, ell + 1):
        for m in range(-n, n + 1):
            Y[:, ii] = scipy.special.sph_harm(m, n, longitude, latitude).real
            # update running index
            ii += 1
    return Y


def ridge_regression(basis_matrix: np.ndarray, data: np.ndarray, lambda_: float):
    """
    A function to perform ridge regression (L2 regularization of the coefficient vector)

    Parameters
    ----------
    basis_matrix: ndarray (matrix)
        matrix dimensions =  (N x (ell+1)^2)
    data: ndarray (vector)
        vector dimensions = (N x 1)
    lambda_: float
        regularization coefficient

    Returns
    -------
    coeff: ndarray (vector)
        vector with coefficients
    """
    N, len_a = np.shape(basis_matrix)
    return (
        np.linalg.inv(basis_matrix.T @ basis_matrix + lambda_ * np.eye(len_a))
        @ basis_matrix.T
        @ data
    )


def construct_global_view(
    coeff: np.ndarray, longitude: np.ndarray, latitude: np.ndarray
):
    """A function to perform ridge regression (L2 regularization of the coefficient vector)


    Parameters
    ----------
    coeff: ndarray (vector)
        vector with coefficients
    longitude: float or vector (ndarray)
        longitude location in radians [0, 2pi]
    latitude: float or vector (ndarray)
        latitude location in radians [0, pi]

    Returns
    -------
    a: ndarray (vector)
        vector with coefficients

    """
    # length of coefficient vector (ell+1)^2
    ell_total = len(coeff)
    # get ell (order of spherical harmonic approximation)
    ell = int(np.sqrt(ell_total) - 1)
    # running index
    ii = 0
    # initialize prediction
    G = np.zeros(len(latitude))
    for n in range(ell + 1):
        for m in range(-n, n + 1):
            # spherical harmonic linear combination
            G += coeff[ii] * scipy.special.sph_harm(m, n, longitude, latitude).real
            # update running index
            ii += 1
    return G
