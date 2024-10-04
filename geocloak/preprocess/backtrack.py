"""
The Backtrack module will provides various (at least 3) methods
of backtracking from L1 time stamp to SDO time stamp.
"""

import warnings

import astropy.constants as const
import numpy as np
import pandas as pd
import sunpy.coordinates.sun as sun

warnings.filterwarnings("ignore")


def ballistic(times: pd.Timestamp | str, velocity: float):
    """
    Thi function will map the give time and velocity to SDO time
    using the ballistic back propogation approach.

    Parameters
    ----------
    times : pd.Timestamp | str | datetime
        The time for which the SDO time needs to be calculated.
    velocity : float
        The velocity of the solar wind at the given time.

    Returns
    -------
    newtime : datetime
        The Solar observation (SDO) time corresponding to the given time and velocity.
    """
    try:
        times = pd.to_datetime(times)
    except TypeError:
        raise TypeError("Unable to Conver time to datetime.")

    dt = (const.au.to("km").value / velocity) / (3600.0 * 24)
    newtime = (times - pd.to_timedelta(dt, unit="day")).round("min")
    return newtime


def HUX(times: list | np.ndarray, vr: list | np.ndarray, backward=True):
    """
    This function will map the give time and velocity to SDO time
    using the HUX back propogation approach.
    Reference: Riley P and Issan O (2021) Using a Heliospheric
      Upwinding eXtrapolation Technique to Magnetically Connect
      Different Regions of the Heliosphere. Front. Phys. 9:679497.
      https://doi.org/10.3389/fphy.2021.679497

    Parameters
    ----------
    times : list | np.ndarray
        The time for which the SDO time needs to be calculated.
    vr : list | np.ndarray
        Radial velocity at the given time at L1 (OMNI/ACE/DSCOVR).
    backward: bool
        Whether to calculate time in backward direction or not.


    Returns
    -------
    newtime: list | np.ndarray
        The SDO time corresponding to the given time and velocity.

    Raises
    ------
    ValueError
        Time is not in correct format.
        CFL condition Violated.
    """
    # Radius of the Sun in Km
    rsun = const.R_sun.to("km").value

    # Distance of AU in km
    au = const.au.to("km").value

    # Scale length over which accelarion span
    rh = 50.0 * rsun

    # Factor by which intial speed will increase
    alpha = 0.15

    # Rotation rate of the sun in radian/sec
    omega_rot = (2 * np.pi) / (25.38 * 86400)

    # dr vector for integration
    dr_vec = np.ones(500) * (au - rsun) / 500

    # Get phi vector and calculate dphi vector from it after sorting
    l0 = np.deg2rad(sun.L0(times).value)
    ind = np.argsort(l0)
    dp_vec = l0[ind][1:] - l0[ind][:-1]

    # The real HUX start here
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))
    v[-1, :] = vr[ind]

    for i in range(len(dr_vec)):
        for j in range(len(dp_vec) + 1):
            if j != len(dp_vec):  # courant condition
                if (omega_rot * dr_vec[i]) / (dp_vec[j] * v[-(i + 1), j]) > 1:
                    print(
                        "CFL violated",
                        dr_vec[i] - dp_vec[j] * v[-(i + 1), j] / omega_rot,
                    )
                    raise ValueError("CFL violated")

                frac2 = (omega_rot * dr_vec[i]) / dp_vec[j]
            else:
                frac2 = (omega_rot * dr_vec[i]) / dp_vec[0]

            frac1 = (v[-(i + 1), j - 1] - v[-(i + 1), j]) / v[-(i + 1), j]
            v[-(i + 2), j] = v[-(i + 1), j] + frac1 * frac2

    # Add acceleration after upwind.
    v_acc = alpha * (v[0, :] * (1 - np.exp(-rsun / rh)))
    v[0, :] = -v_acc + v[0, :]

    # Calculate shift in phi from Earth (mean L1) to Sun
    vv = v.T
    p, r = l0[ind], np.linspace(rsun, au, 500)
    phi_shift_mat = np.zeros((len(r), len(p)))
    phi_shift_mat[0, :] = p
    dr = np.mean(r[1:] - r[:-1])
    for ii in range(len(r) - 1):
        phi_shift = -(omega_rot / vv[:, -ii - 1]) * dr
        phi_shift_mat[ii + 1, :] = phi_shift_mat[ii, :] + phi_shift

    # Calculate total shift
    dphi = phi_shift_mat[0, :] - phi_shift_mat[-1, :]

    # Unsort it again to get back in to input order
    dphi = dphi[np.argsort(ind)][-1]

    # get time based on fractiona carrington rotation number
    crn = sun.carrington_rotation_number(times[-1])

    direction = -1 if backward else 1
    crnf = crn + (direction * (dphi / (2.0 * np.pi)))
    newtime = sun.carrington_rotation_time(crnf)
    return newtime
