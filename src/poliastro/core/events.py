import numpy as np
from numba import njit as jit
from numpy.linalg import norm


@jit
def in_umbral_shadow(r_sat, r_sec, R_s, R_p):
    r"""Calculate whether a satellite is in umbra or not, follows Algorithm 34 from Vallado.

    Parameters
    ----------
    r_sat: numpy.ndarray
        Position of the satellite in the frame of the attractor (km).
    r_sec: numpy.ndarray
        Position of the secondary body in the frame of the attractor (km).
    R_s: float
        Radius of the secondary body (km).
    R_p: float
        Radius of the primary body (attractor) responsible for the umbral shadow (km).

    Returns
    -------
    eclipse: float
        -1 if satellite is in umbra, else +1.

    """
    def angl(r1, r2):
        mag1 = np.linalg.norm(r1)
        mag2 = np.linalg.norm(r2)
        temp = np.dot(r1, r2) / (mag1 * mag2)
        if abs(temp) > 1.0:
            temp = np.sign(temp) * 1.0
        theta = np.arccos(temp)
        return theta

    R = norm(r_sec)  # Distance between the primary and the secondary body.
    alpha_um = np.arcsin((R_s - R_p) / R)
    alpha_pen = np.arcsin((R_s + R_p) / R)

    dot_sec_sat = np.dot(r_sat, r_sec)

    r_sat_norm = np.linalg.norm(r_sat)
    r_sec_norm = np.linalg.norm(r_sec)

    if dot_sec_sat > 0:
        eclipse = 1
    if dot_sec_sat < 0:
        angle = angl(-r_sec, r_sat)
        sat_horiz = r_sat_norm * np.cos(angle)
        sat_vert = r_sat_norm * np.sin(angle)
        x = R_p / np.sin(alpha_pen)
        pen_vert = np.tan(alpha_pen) * (x + sat_horiz)
        if sat_vert <= pen_vert:
            y = R_p / np.sin(alpha_um)
            umb_vert = np.tan(alpha_um) * (y - sat_horiz)
            # Edge condition for entering umbra: sat_vert - pen_vert = 0.
            # +ve to -ve direction means entering umbra.
            if sat_vert <= umb_vert:
                eclipse = -1
            else:
                eclipse = 1
        else:
            eclipse = 1

    return eclipse


@jit
def in_penumbral_shadow(r_sat, r_sec, R_s, R_p):
    r"""Calculate whether a satellite is in penumbra or not, follows Algorithm 34 from Vallado.

    Parameters
    ----------
    r_sat: numpy.ndarray
        Position of the satellite in the frame of the attractor (km).
    r_sec: numpy.ndarray
        Position of the secondary body in the frame of the attractor (km).
    R_s: float
        Radius of the secondary body.
    R_p: float
        Radius of the primary body (attractor) responsible for the umbral shadow (km).

    Returns
    -------
    eclipse: float
        -1 if satellite is in penumbra, else +1.

    """
    def angl(r1, r2):
        mag1 = np.linalg.norm(r1)
        mag2 = np.linalg.norm(r2)
        temp = np.dot(r1, r2) / (mag1 * mag2)
        if abs(temp) > 1.0:
            temp = np.sign(temp) * 1.0
        theta = np.arccos(temp)
        return theta
    R = norm(r_sec)
    alpha_pen = np.arcsin((R_s + R_p) / R)

    dot_sec_sat = np.dot(r_sat, r_sec)

    r_sat_norm = np.linalg.norm(r_sat)
    r_sec_norm = np.linalg.norm(r_sec)

    if dot_sec_sat > 0: # Not in shadow surely
        eclipse = 1

    if dot_sec_sat < 0:
        angle = angl(-r_sec, r_sat)
        sat_horiz = r_sat_norm * np.cos(angle)
        sat_vert = r_sat_norm * np.sin(angle)
        x = R_p / np.sin(alpha_pen)
        pen_vert = np.tan(alpha_pen) * (x + sat_horiz)
        # Edge condition for entering penumbra: sat_vert - pen_vert = 0.
        # +ve to -ve direction means entering penumbra.
        if sat_vert <= pen_vert:
            eclipse = -1
        elif sat_vert > pen_vert:
            eclipse = 1

    return eclipse
