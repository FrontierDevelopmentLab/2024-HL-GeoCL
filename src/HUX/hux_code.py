"""HUX-b propagation implemented. """
import numpy as np
import copy


def apply_hux_b_model(r_final, dr_vec, dp_vec, alpha=0.15, rh=50 * 695700, add_v_acc=True,
                      r0=1 * 695700, omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """ Apply 1d HUX backwards propagation.

    :param r_final: 1d array, initial velocity for backward propagation. units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r.
    :param dp_vec: 1d array, mesh spacing in p.
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh:  float, hyper parameter for acceleration (default r=50 rs). units: (km)
    :param add_v_acc: bool, True will add acceleration boost.
    :param r0: float, initial radial location. units = (km).
    :param omega_rot: differential rotation.
    :return: velocity matrix dimensions (nr x np) """

    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[-1, :] = r_final

    for i in range(len(dr_vec)):
        for j in range(len(dp_vec) + 1):

            if j != len(dp_vec):
                # courant condition
                if (omega_rot * dr_vec[i]) / (dp_vec[j] * v[-(i + 1), j]) > 1:
                    print("CFL violated", dr_vec[i] - dp_vec[j] * v[-(i + 1), j] / omega_rot)
                    raise ValueError('CFL violated')

                frac2 = (omega_rot * dr_vec[i]) / dp_vec[j]
            else:
                frac2 = (omega_rot * dr_vec[i]) / dp_vec[0]

            frac1 = (v[-(i + 1), j - 1] - v[-(i + 1), j]) / v[-(i + 1), j]
            v[-(i + 2), j] = v[-(i + 1), j] + frac1 * frac2

    # add acceleration after upwind.
    if add_v_acc:
        v_acc = alpha * (v[0, :] * (1 - np.exp(-r0 / rh)))
        v[0, :] = -v_acc + v[0, :]

    return v


def apply_ballistic_approximation(v_initial, dr, phi_vec, omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """ Apply the ballistic model for mapping solar wind streams to
    different locations in the heliosphere is the simplest possible approximation

    :param phi_vec: mesh carrington longitude spacing (radians).
    :param v_initial: 1d array, initial velocity for ballistic. units = (km/sec).
    :param dr: delta r. units = (km).
    :param omega_rot: differential rotation. (1/secs)
    :return: shifted phi coordinates. """

    # delta_phi = -omega * delta_r / (vr0)
    delta_phi = (omega_rot * dr) / v_initial
    phi_shifted = phi_vec - delta_phi
    # force periodicity
    return phi_shifted % (2 * np.pi)


def apply_modified_ballistic_approximation(v_initial, r0_vec, rf_vec, phi_vec, omega_rot=(2 * np.pi) / (25.38 * 86400)):
    """ Apply the ballistic model for mapping solar wind streams to
    different locations in the heliosphere is the simplest possible approximation

    :param rf_vec: r final 1d array. units = (km).
    :param phi_vec: mesh carrington longitude spacing (radians).
    :param v_initial: 1d array, initial velocity for ballistic. units = (km/sec).
    :param r0_vec: r0, 1d array. units = (km).
    :param omega_rot: differential rotation. (1/secs)
    :return: shifted phi coordinates. """
    dr = rf_vec - r0_vec
    delta_phi = (omega_rot * dr) / v_initial
    phi_shifted = phi_vec - delta_phi
    # force periodicity
    return phi_shifted % (2 * np.pi)


def compute_phi_shift_backward(p, r, v, omega=2 * np.pi / 25.38, method="hux"):
    # initialize phi shift matrix.
    phi_shift_mat = np.zeros((len(r), len(p)))

    # phi at last index is original phi grid
    phi_shift_mat[0, :] = p

    # delta r.
    dr = np.mean(r[1:] - r[:-1])

    # compute the phi shift for each idx in r.
    for ii in range(len(r) - 1):
        # delta r. (r_next-r_prev)
        if method == "ballistic":
            phi_shift = -(omega / v[:, -1]) * dr
        if method == "hux":
            phi_shift = -(omega / v[:, -ii - 1]) * dr
        phi_shift_mat[ii + 1, :] = phi_shift_mat[ii, :] + phi_shift

    return phi_shift_mat