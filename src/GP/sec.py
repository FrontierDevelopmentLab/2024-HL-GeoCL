"""This is a module to compute Spherical Elementary Currents via superMAG magnetometer observations

references:
[1] O. Amm. Ionospheric elementary current systems in spherical coordinates and their application. Journal of
geomagnetism and geoelectricity, 49(7):947–955, 1997.
[2] Amm, O., and A. Viljanen. "Ionospheric disturbance magnetic field continuation from the ground to the ionosphere
using spherical elementary current systems." Earth, Planets and Space 51.6: 431-440, 1999.

Author: Opal Issan (PhD student @ucsd). email: oissan@ucsd.edu.
Last Modified: July 23st, 2024
"""
import numpy as np


def T_df(obs_loc, sec_loc):
    """calculates the divergence free (df) magnetic field transfer function in Eq. (14) [2]

    Parameters
    ----------
    obs_loc : ndarray (nobs, 3 [lat, lon, r]) in [deg (-90, 90), deg (0, 360), km]
        locations of the observation points

    sec_loc : ndarray (nsec, 3 [lat, lon, r]) in [deg (-90, 90), deg (0, 360), km]
        locations of the SEC points

    Returns
    -------
    ndarray (nobs, 3, nsec)
        T transfer matrix in Eq. (14) [2] assuming mu_{0}/4pi is absorbed in I0
    """
    # angular distance from observation location and sec location
    # theta prime in Eq. (9) & (10) [2]
    theta = calc_angular_distance(obs_loc[:, :2], sec_loc[:, :2])

    # takes into account the change in coordinate system from sec to obs
    alpha = calc_bearing(obs_loc[:, :2], sec_loc[:, :2])

    # r / R ratio in Eq. (9) & (10) [2]
    r_ratio = obs_loc[0, 2] / sec_loc[0, 2]

    # first term in the parenthesis in Eq. (9) [2]
    factor_term = 1. / np.sqrt(1 - 2 * r_ratio * np.cos(theta) + (r_ratio ** 2))

    # Eq. (9) [2]
    Br = (factor_term - 1) / obs_loc[0, 2]

    # Eq. (10) [2]
    Bt = np.divide(-(factor_term * (r_ratio - np.cos(theta)) + np.cos(theta)) / obs_loc[0, 2], np.sin(theta),
                   out=np.zeros_like(np.sin(theta)), where=np.sin(theta) != 0)

    # transform back to Bx, By, Bz at each local point
    T = np.zeros((len(obs_loc)*3, len(sec_loc)))
    # alpha == angle (from cartesian x-axis (By), going towards y-axis (Bx))
    T[:len(obs_loc), :] = -Bt * np.sin(alpha)
    T[len(obs_loc):2*len(obs_loc), :] = -Bt * np.cos(alpha)
    T[2*len(obs_loc):, :] = -Br
    return T


def calc_angular_distance(latlon1, latlon2):
    """Calculate the angular distance between a set of points.

    This function calculates the angular distance in radians
    between any number of latitude and longitude points.

    Parameters
    ----------
    latlon1 : ndarray (n x 2 [lat, lon]) in [deg (-90, 90), deg (0, 360)]
        An array of n (latitude, longitude) points.

    latlon2 : ndarray (m x 2 [lat, lon]) in [deg (-90, 90), deg (0, 360)]
        An array of m (latitude, longitude) points.

    Returns
    -------
    ndarray (n x m)
        The array of distances between the input arrays.
    """
    lat1 = np.deg2rad(latlon1[:, 0])[:, np.newaxis]
    lon1 = np.deg2rad(latlon1[:, 1])[:, np.newaxis]
    lat2 = np.deg2rad(latlon2[:, 0])[np.newaxis, :]
    lon2 = np.deg2rad(latlon2[:, 1])[np.newaxis, :]

    # angular distance between two points
    return np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))


def calc_bearing(latlon1, latlon2):
    """calculate the bearing (direction) between a set of latitude and longitude points.

    Parameters
    ----------
    latlon1 : ndarray (n x 2 [lat, lon]) in [deg (-90, 90), deg (0, 360)]
        An array of n (latitude, longitude) points.

    latlon2 : ndarray (m x 2 [lat, lon]) in [deg (-90, 90), deg (0, 360)]
        An array of m (latitude, longitude) points.

    Returns
    -------
    ndarray (n x m)
        The array of bearings between the input arrays.
    """
    lat1 = np.deg2rad(latlon1[:, 0])[:, np.newaxis]
    lon1 = np.deg2rad(latlon1[:, 1])[:, np.newaxis]
    lat2 = np.deg2rad(latlon2[:, 0])[np.newaxis, :]
    lon2 = np.deg2rad(latlon2[:, 1])[np.newaxis, :]

    # used to rotate the SEC coordinate frame into the observation coordinate frame
    # SEC coordinates are: theta (+ north), phi (+ east), r (+ out)
    # observation coordinates are: X (+ north), Y (+ east), Z (+ down)
    return np.pi / 2 - np.arctan2(np.sin(lon2 - lon1) * np.cos(lat2),
                                  np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))