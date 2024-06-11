'''
ABC DOC STRING
'''

import numpy as np


def crps_neighbor_checks(ens_shape: tuple, obs_shape: tuple) -> tuple:
    """
    Check if the dimensions of the ensemble and observation matrix are valid
    for the neighborhood-based CRPS calculation.

    Parameters
    ----------
    ens_shape : tuple
        The shape of the ensemble matrix.

    obs_shape : tuple
        The shape of the observation matrix.

    Returns
    -------
    tuple
        A tuple with two elements. The first element is a boolean indicating
        if the dimensions are valid. The second element is a string with an
        error message if the dimensions are not valid.
    """

    # check if the ensemble and observation matrix are 2D
    if len(ens_shape) != 2 or len(obs_shape) != 2:
        return (False, "The ensemble and observation matrix must be 2D.")

    # check if the length of the ensemble and the observations are the same
    if ens_shape[0] != obs_shape[0]:
        return (False, "The length of the ensemble and the observations must be the same.")

    return (True, None)


def remove_nanrows(yobs: np.ndarray, ens: np.ndarray) -> np.ndarray:
    """
    Remove rows with all NaNs from the observation matrix.

    Parameters
    ----------
    yobs : 2D np.ndarray
        The observation matrix. Assumes that the first dimension is the time
        and the second dimension is the observations.

    ens : 2D np.ndarray
        The ensemble of simulations. Assumes that the first dimension is the time
        and the second dimension is the ensemble members.

    Returns
    -------
    yobs : 2D np.ndarray
        The observation matrix with the rows with all NaNs removed.

    ens : 2D np.ndarray
        The ensemble of simulations with the rows with all NaNs removed.

    """

    # find the rows with all NaNs
    nanrows = np.isnan(yobs).all(axis=1)

    # remove the rows with all NaNs
    yobs = yobs[~nanrows]
    ens = ens[~nanrows]

    return yobs, ens
