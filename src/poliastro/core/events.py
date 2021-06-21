import numpy as np
from numba import njit as jit


@jit
def line_of_sight(r1, r2, R, R_polar, ellipsoid=True):
    """Calculate whether there exists a line-of-sight (LOS) between two
    position vectors, r1 and r2.

    Parameters
    ----------
    r1: ~np.array
        Position vector.
    r2: ~np.array
        Position vector.
    R: float
        Equatorial radius of the attractor.
    R_polar: float
        Polar radius of the attractor.
    ellipsoid: bool, optional
        Whether the attractor shape is taken as ellipsoid or not, defaults to True.

    Returns
    -------
    los: bool
        True if there exists a LOS, else False.

    """
    los = False
    # Create temp variables to prevent overwriting r1 and r2.
    temp_r1 = r1
    temp_r2 = r2

    r1_sqrd = np.dot(temp_r1, temp_r1)
    r2_sqrd = np.dot(temp_r2, temp_r2)

    if ellipsoid:
        ecc = np.sqrt(1 - (R_polar / R) ** 2)
        # Account for ellipsoidal Earth.
        scale_factor = 1 / np.sqrt(1 - ecc ** 2)
        # Scale k-component of the vectors.
        temp_r1[-1] = temp_r1[-1] * scale_factor
        temp_r2[-1] = temp_r2[-1] * scale_factor

    dot_temp_r1_r2 = np.dot(temp_r1, temp_r2)

    t_min = (r1_sqrd - dot_temp_r1_r2) / (r1_sqrd + r2_sqrd - 2 * dot_temp_r1_r2)

    if t_min < 0 or 1 < t_min:
        los = True
        return los
    else:
        d_sqrd = ((1 - t_min) * r1_sqrd + dot_temp_r1_r2 * t_min) / R ** 2
        if d_sqrd > 1:
            los = True
            return los
        return los
