'''
ABC DOC STRING
'''

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .utils.crps_helpers import crps_neighbor_checks, remove_nanrows


def crps_neighborhood_chunk(ens_chunk: np.ndarray, yobs_chunk: np.ndarray) -> np.ndarray:
    r"""
    Calculate the neighborhood-based CRPS for a subset of time-steps.

    Parameters
    ----------
    ens_chunk : 2D np.ndarray
        The ensemble of simulations. Assumes that the first dimension is the time
        and the second dimension is the ensemble members.

    yobs_chunk : 2D np.ndarray
        The observation matrix. Assumes that the first dimension is the time
        and the second dimension is the observations.

    Returns
    -------
    crps_neighbour_vec : np.ndarray
        The neighborhood-based CRPS for the subset of time-steps.

    References
    ----------
    - Neighborhood-based ensemble evaluation using the CRPS
        (https://doi.org/10.1175/MWR-D-21-0224.1)
        Equation 5, 14.

    **Equation 5**

    .. math::
        CRPS(F_x, G_y) = \mathbb{E}_{X,Y}(|X - Y|) - \frac{1}{2}[\mathbb{E}_{X,X'}(|X - X'|) + \mathbb{E}_{Y,Y'}(|Y - Y'|)]

    **Equation 14**

    .. math::
        CRPS(F_x, G_y) = \frac{1}{N} \frac{1}{M} \sum_{i=1}^{M} \sum_{j=1}^{N} |x_i - y_j| - \frac{1}{2M^2} \sum_{j=1}^{M} \sum_{k=1}^{M} |x_j - x_k| -  \frac{1}{2N^2} \sum_{j=1}^{N} \sum_{k=1}^{N} |y_j - y_k|

    """

    # use shorter variable names
    ens = ens_chunk
    yobs = yobs_chunk

    # remove rows with all NaNs for the observation matrix
    # the same rows are removed from the ensemble matrix
    yobs, ens = remove_nanrows(yobs=yobs, ens=ens)

    # get the dimensions
    ENSIZE = ens.shape[1]
    OSIZE = yobs.shape[1]

    # calculate the first term of Eq. 14 for all time-steps of this chunk
    # related to absolute error btw simulation and observation
    term1s_vec = np.nansum(
        np.abs(ens[:, :, None] - yobs[:, None, :]), axis=(1, 2),
    ) / (ENSIZE * OSIZE)

    # calculate the second term of Eq. 14 for all time-steps of this chunk
    # related to the spread of the ensemble
    term2s_vec = np.nansum(
        np.abs(ens[:, :, None] - ens[:, None, :]), axis=(1, 2),
    ) / (2 * (ENSIZE ** 2))

    # calculate the third term of Eq. 14 for all time-steps of this chunk
    # related to the spread of the observations
    term3s_vec = np.nansum(
        np.abs(yobs[:, :, None] - yobs[:, None, :]), axis=(1, 2),
    ) / (2 * (OSIZE ** 2))

    # note that this is not averaged. Averaging is done in the main function
    crps_values_chunk = term1s_vec - term2s_vec - term3s_vec

    return crps_values_chunk


def crps_neighboor(ens: np.ndarray, yobs: np.ndarray, chunk_size=5, njobs=2) -> np.ndarray:
    """
    Calculate the neighborhood-based CRPS for a given ensemble simulation
    and ensemble of observations.

    Please refer to the documentation of the crps_neighborhood_chunk function
    for the details on the calculation.

    Parameters
    ----------
    ens : 2D np.ndarray
        The ensemble of forecasts

    yobs : 2D np.ndarray
        The observation matrix. Assumes that the first dimension is the time
        and the second dimension is the observations.

    chunk_size : int
        The size of the chunks to split the data into.
        large datasets.

    njobs : int
        The number of jobs to run in parallel.

    Returns
    -------
    crps_n : float
        The neighborhood-based CRPS averaged over all time-steps.

    Notes
    -----
    For large datasets, it is recommended to use a chunk_size and njobs
    to speed up the calculation. Please note that running large chunks in
    parallel may require a lot of memory. Therefore, it is recommended to
    test the chunk_size and njobs for your specific use case.

    """

    # assertions
    check = crps_neighbor_checks(ens.shape, yobs.shape)
    assert check[0], check[1]

    # split the data into chunks
    ens_chunks = np.array_split(ens, ens.shape[0] // chunk_size)
    y_obs_chunks = np.array_split(yobs, yobs.shape[0] // chunk_size)

    # process the chunks in parallel
    with ProcessPoolExecutor(max_workers=njobs) as executor:
        crps_values = np.concatenate(list(
            executor.map(crps_neighborhood_chunk, ens_chunks, y_obs_chunks))
        )

    # average the CRPS values over all time-steps
    crps_n = np.nanmean(crps_values)

    return crps_n


def crps_neighborhood_chunk_nonvectorized(
    ens_chunk: np.ndarray, yobs_chunk: np.ndarray
) -> np.ndarray:
    r"""
    Calculate the neighborhood-based CRPS for a subset of time-steps without
    broadcasting (non-vectorized).

    Parameters
    ----------
    ens_chunk : 2D np.ndarray
        The ensemble of simulations. Assumes that the first dimension is the time
        and the second dimension is the ensemble members.

    yobs_chunk : 2D np.ndarray
        The observation matrix. Assumes that the first dimension is the time
        and the second dimension is the observations.

    Returns
    -------
    crps_neighbour_vec : float
        The neighborhood-based CRPS for the subset of time-steps.
    """

    # use shorter variable names
    ens = ens_chunk
    yobs = yobs_chunk

    # remove rows with all NaNs for the observation matrix
    # the same rows are removed from the ensemble matrix
    yobs, ens = remove_nanrows(yobs, ens)

    # get the dimensions
    ENS_SIZE = ens.shape[1]
    OBS_SIZE = yobs.shape[1]
    T_STEPS = ens.shape[0]

    term1s_vec = np.zeros(T_STEPS)
    term2s_vec = np.zeros(T_STEPS)
    term3s_vec = np.zeros(T_STEPS)

    for t in range(T_STEPS):

        for i in range(ENS_SIZE):
            for j in range(OBS_SIZE):
                term1s_vec[t] += np.abs(ens[t, i] - yobs[t, j])

        term1s_vec[t] /= (ENS_SIZE * OBS_SIZE)

        for i in range(ENS_SIZE):
            for j in range(ENS_SIZE):
                term2s_vec[t] += np.abs(ens[t, i] - ens[t, j])

        term2s_vec[t] /= (2 * (ENS_SIZE ** 2))

        for i in range(OBS_SIZE):
            for j in range(OBS_SIZE):
                term3s_vec[t] += np.abs(yobs[t, i] - yobs[t, j])

        term3s_vec[t] /= (2 * (OBS_SIZE ** 2))

    crps_values_chunk = term1s_vec - term2s_vec - term3s_vec

    return crps_values_chunk
