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
from spherical_harmonics import ridge_regression


class SECS:
    """Spherical Elementary Current System (SECS).

    The algorithm is implemented directly in spherical coordinates
    from the equations of the 1999 Amm & Viljanen paper.

    Parameters
    ----------

    sec_df_loc : ndarray (nsec x 3 [lat, lon, r])
        The latitude, longiutde, and radius of the divergence free (df) SEC locations.

    sec_cf_loc : ndarray (nsec x 3 [lat, lon, r])
        The latitude, longiutde, and radius of the curl free (cf) SEC locations.

    References
    ----------
    .. [1] Amm, O., and A. Viljanen. "Ionospheric disturbance magnetic field continuation
           from the ground to the ionosphere using spherical elementary current systems."
           Earth, Planets and Space 51.6 (1999): 431-440. doi:10.1186/BF03352247
    """

    def __init__(self, sec_df_loc=None, sec_cf_loc=None):

        if sec_df_loc is None and sec_cf_loc is None:
            raise ValueError("Must initialize the object with SEC locations")

        self.sec_df_loc = sec_df_loc
        self.sec_cf_loc = sec_cf_loc

        if self.sec_df_loc is not None:
            self.sec_df_loc = np.asarray(sec_df_loc)
            if self.sec_df_loc.shape[-1] != 3:
                raise ValueError("SEC DF locations must have 3 columns (lat, lon, r)")
            if self.sec_df_loc.ndim == 1:
                # Add an empty dimension if only one SEC location is passed in
                self.sec_df_loc = self.sec_df_loc[np.newaxis, ...]

        if self.sec_cf_loc is not None:
            self.sec_cf_loc = np.asarray(sec_cf_loc)
            if self.sec_cf_loc.shape[-1] != 3:
                raise ValueError("SEC CF locations must have 3 columns (lat, lon, r)")
            if self.sec_cf_loc.ndim == 1:
                # Add an empty dimension if only one SEC location is passed in
                self.sec_cf_loc = self.sec_cf_loc[np.newaxis, ...]

        # Storage of the scaling factors
        self.sec_amps = None
        self.sec_amps_var = None

    @property
    def has_df(self):
        """Whether this system has any divergence free currents."""
        return self.sec_df_loc is not None

    @property
    def has_cf(self):
        """Whether this system has any curl free currents."""
        return self.sec_cf_loc is not None

    @property
    def nsec(self):
        """The number of elementary currents in this system."""
        nsec = 0
        if self.has_df:
            nsec += len(self.sec_df_loc)
        if self.has_cf:
            nsec += len(self.sec_cf_loc)
        return nsec

    def fit(self, obs_loc, obs_B, epsilon=0.05):
        """Fits the SECS to the given observations.

        Given a number of observation locations and measurements,
        this function fits the SEC system to them. It uses singular
        value decomposition (SVD) to fit the SEC amplitudes with the
        `epsilon` parameter used to regularize the solution.

        Parameters
        ----------
        obs_locs : ndarray (nobs x 3 [lat, lon, r])
            Contains latitude, longitude, and radius of the observation locations
            (place where the measurements are made)

        obs_B: ndarray (ntimes x nobs x 3 [Bx, By, Bz])
            An array containing the measured/observed B-fields.

        obs_std : ndarray (ntimes x nobs x 3 [varX, varY, varZ]), optional
            Standard error of vector components at each observation location.
            This can be used to weight different observations more/less heavily.
            An infinite value eliminates the observation from the fit.
            Default: ones(nobs x 3) equal weights

        epsilon : float
            Value used to regularize/smooth the SECS amplitudes. Multiplied by the
            largest singular value obtained from SVD.
            Default: 0.05
        """
        if obs_loc.shape[-1] != 3:
            raise ValueError("Observation locations must have 3 columns in spherical coordinates (lat, lon, r)")

        # calculate the transfer functions
        T_obs = T_df(obs_loc=obs_loc, sec_loc=self.sec_df_loc)
        T = T_obs.reshape(-1, self.nsec)

        # calculate the singular value decomposition (SVD)
        # NOTE: T_obs has shape (nobs, 3, nsec) to (nobs*3, nsec)
        self.sec_amps = ridge_regression(basis_matrix=T, data=obs_B, lambda_=epsilon)

        return self

    def fit_unit_currents(self):
        """Sets all SECs to a unit current amplitude."""
        self.sec_amps = np.ones((1, self.nsec))

        return self

    def predict(self, pred_loc, J=False):
        """Calculate the predicted magnetic field or currents.

        After a set of observations has been fit to this system we can
        predict the magnetic fields or currents at any other location. This
        function uses those fit amplitudes to predict at the requested locations.

        Parameters
        ----------
        pred_loc: ndarray (npred x 3 [lat, lon, r])
            An array containing the locations where the predictions are desired.

        J: boolean
            Whether to predict currents (J=True) or magnetic fields (J=False)
            Default: False (magnetic field prediction)

        Returns
        -------
        ndarray (ntimes x npred x 3 [lat, lon, r])
            The predicted values calculated from the current amplitudes that were
            fit to this system.
        """
        if pred_loc.shape[-1] != 3:
            raise ValueError("Prediction locations must have 3 columns (lat, lon, r)")

        if self.sec_amps is None:
            raise ValueError("There are no currents associated with the SECs," +
                             "you need to call .fit() first to fit to some observations.")

        # T_pred shape=(npred x 3 x nsec)
        # sec_amps shape=(nsec x ntimes)
        if J:
            # Predicting currents
            T_pred = self._calc_J(pred_loc)
        else:
            # Predicting magnetic fields
            T_pred = self._calc_T(pred_loc)

        # NOTE: dot product is slow on multi-dimensional arrays (i.e. > 2 dimensions)
        #       Therefore this is implemented as tensordot, and the arguments are
        #       arranged to eliminate needs of transposing things later.
        #       The dot product is done over the SEC locations, so the final output
        #       is of shape: (ntimes x npred x 3)

        return np.squeeze(np.tensordot(self.sec_amps, T_pred, (1, 2)))

    def predict_B(self, pred_loc):
        """Calculate the predicted magnetic fields.

        After a set of observations has been fit to this system we can
        predict the magnetic fields or currents at any other location. This
        function uses those fit amplitudes to predict at the requested locations.

        Parameters
        ----------
        pred_loc: ndarray (npred x 3 [lat, lon, r])
            An array containing the locations where the predictions are desired.

        Returns
        -------
        ndarray (ntimes x npred x 3 [lat, lon, r])
            The predicted values calculated from the current amplitudes that were
            fit to this system.
        """
        return self.predict(pred_loc)

    def predict_J(self, pred_loc):
        """Calculate the predicted currents.

        After a set of observations has been fit to this system we can
        predict the magnetic fields or currents at any other location. This
        function uses those fit amplitudes to predict at the requested locations.

        Parameters
        ----------
        pred_loc: ndarray (npred x 3 [lat, lon, r])
            An array containing the locations where the predictions are desired.

        Returns
        -------
        ndarray (ntimes x npred x 3 [lat, lon, r])
            The predicted values calculated from the current amplitudes that were
            fit to this system.
        """
        return self.predict(pred_loc, J=True)

    def _calc_T(self, obs_loc):
        """Calculates the T transfer matrix.

        The magnetic field transfer matrix to go from SEC locations to observation
        locations. It assumes unit current amplitudes that will then be
        scaled with the proper amplitudes later.
        """
        T = T_df(obs_loc=obs_loc, sec_loc=self.sec_df_loc)

        return T

    def _calc_J(self, obs_loc):
        """Calculates the J transfer matrix.

        The current transfer matrix to go from SEC locations to observation
        locations. It assumes unit current amplitudes that will then be
        scaled with the proper amplitudes later.
        """
        if self.has_df:
            J = J_df(obs_loc=obs_loc, sec_loc=self.sec_df_loc)

        if self.has_cf:
            J1 = J_cf(obs_loc=obs_loc, sec_loc=self.sec_cf_loc)
            # df is already present in T
            if self.has_df:
                J = np.concatenate([J, J1], axis=2)
            else:
                J = J1

        return J


def T_df(obs_loc, sec_loc):
    """calculates the divergence free magnetic field transfer function.

    The transfer function goes from SEC location to observation location
    and assumes unit current SECs at the given locations.

    Parameters
    ----------
    obs_loc : ndarray (nobs, 3 [lat, lon, r])
        The locations of the observation points.

    sec_loc : ndarray (nsec, 3 [lat, lon, r])
        The locations of the SEC points.

    Returns
    -------
    ndarray (nobs, 2, nsec)
        The T transfer matrix (Eq. (14) in Amm 1999)
    """
    nobs = len(obs_loc)  # number of magnetometer stations
    nsec = len(sec_loc)  # number of spherical elementary currents

    # radial component of the spherical currents and observations
    obs_r = obs_loc[:, 2]
    sec_r = sec_loc[:, 2]

    theta = calc_angular_distance(obs_loc[:, :2], sec_loc[:, :2])
    alpha = calc_bearing(obs_loc[:, :2], sec_loc[:, :2])

    # simplify calculations by storing this ratio
    # r / R_{I} in Eq. (9) and Eq. (10) in Amm 1999
    ratio = obs_r / sec_r

    sin_theta = np.sin(theta)
    factor = 1. / np.sqrt(1 - 2 * ratio * np.cos(theta) + ratio ** 2)

    # Eq. (9) in Amm 1999
    Br = (factor - 1) / ratio

    # Eq. (10) in Amm 1999
    Bt = np.divide(-(factor * (ratio - np.cos(theta)) + np.cos(theta)) / ratio, np.sin(theta),
                   out=np.zeros_like(np.sin(theta)), where=np.sin(theta) != 0)

    # transform back to Bx, By, Bz at each local point
    T = np.zeros((nobs, 3, nsec))
    # alpha == angle (from cartesian x-axis (By), going towards y-axis (Bx))
    T[:, 0, :] = -Bt * np.sin(alpha)
    T[:, 1, :] = -Bt * np.cos(alpha)
    T[:, 2, :] = -Br
    return T


def J_df(obs_loc, sec_loc):
    """Calculates the divergence free current density transfer function.

    The transfer function goes from SEC location to observation location
    and assumes unit current SECs at the given locations.

    Parameters
    ----------
    obs_loc : ndarray (nobs, 3 [lat, lon, r])
        The locations of the observation points.

    sec_loc : ndarray (nsec, 3 [lat, lon, r])
        The locations of the SEC points.

    Returns
    -------
    ndarray (nobs, 3, nsec)
        The J transfer matrix.
    """
    nobs = len(obs_loc)
    nsec = len(sec_loc)

    obs_r = obs_loc[:, 2][:, np.newaxis]
    sec_r = sec_loc[:, 2][np.newaxis, :]

    # Input to the distance calculations is degrees, output is in radians
    theta = calc_angular_distance(obs_loc[:, :2], sec_loc[:, :2])
    alpha = calc_bearing(obs_loc[:, :2], sec_loc[:, :2])

    # Amm & Viljanen: Equation 6
    tan_theta2 = np.tan(theta / 2.)

    J_phi = 1. / (4 * np.pi * sec_r)
    J_phi = np.divide(J_phi, tan_theta2, out=np.ones_like(tan_theta2) * np.inf,
                      where=tan_theta2 != 0.)
    # Only valid on the SEC shell
    J_phi[sec_r != obs_r] = 0.

    # Transform back to Bx, By, Bz at each local point
    J = np.empty((nobs, 3, nsec))
    # alpha == angle (from cartesian x-axis (By), going towards y-axis (Bx))
    J[:, 0, :] = -J_phi * np.cos(alpha)
    J[:, 1, :] = J_phi * np.sin(alpha)
    J[:, 2, :] = 0.

    return J


def calc_angular_distance(latlon1, latlon2):
    """Calculate the angular distance between a set of points.

    This function calculates the angular distance in radians
    between any number of latitude and longitude points.

    Parameters
    ----------
    latlon1 : ndarray (n x 2 [lat, lon])
        An array of n (latitude, longitude) points.

    latlon2 : ndarray (m x 2 [lat, lon])
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
    """Calculate the bearing (direction) between a set of points.

    This function calculates the bearing in radians
    between any number of latitude and longitude points.
    It is the direction from point 1 to point 2 going from the
    cartesian x-axis towards the cartesian y-axis.

    Parameters
    ----------
    latlon1 : ndarray (n x 2 [lat, lon])
        An array of n (latitude, longitude) points.

    latlon2 : ndarray (m x 2 [lat, lon])
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

    dlon = lon2 - lon1

    # used to rotate the SEC coordinate frame into the observation coordinate frame
    # SEC coordinates are: theta (+ north), phi (+ east), r (+ out)
    # observation coordinates are: X (+ north), Y (+ east), Z (+ down)
    return np.pi / 2 - np.arctan2(np.sin(dlon) * np.cos(lat2),
                                   np.cos(lat1) * np.sin(lat2) -
                                   np.sin(lat1) * np.cos(lat2) * np.cos(dlon))