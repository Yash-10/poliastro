import numpy as np
from numba import njit as jit
from numpy.linalg import norm


@jit
def line_of_sight(r1, r2, R, R_polar):
    """Calculate whether there exists a line-of-sight (LOS) between two
    position vectors, r1 and r2. Follows algorithm 35 (SIGHT) from Vallado.

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

    Returns
    -------
    los: bool
        True if there exists a LOS, else False.

    """
    if norm(r1) <= R or norm(r2) <= R:
        raise ValueError("Both r1 and r2 must be above the surface of the attractor.")

    los = False
    # Create temp variables to prevent overwriting r1 and r2.
    temp_r1 = r1
    temp_r2 = r2

    r1_sqrd = np.dot(temp_r1, temp_r1)
    r2_sqrd = np.dot(temp_r2, temp_r2)

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
