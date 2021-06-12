from astropy import units as u_
from numpy.linalg import norm

from poliastro.core.perturbations import (
    penumbral_shadow as penumbral_shadow_fast,
    umbral_shadow as umbral_shadow_fast,
)


class Event:
    """Base class for event functionalities.

    Parameters
    ----------
    terminal: bool
        Whether to terminate integration if this event occurs.
    direction: float
        Handle triggering of event.

    """

    def __init__(self, terminal, direction):
        self._terminal, self._direction = terminal, direction
        self._last_t = None

    @property
    def terminal(self):
        return self._terminal

    @property
    def direction(self):
        return self._direction

    @property
    def last_t(self):
        return self._last_t * u_.s

    def __call__(self, t, u, k):
        raise NotImplementedError


class AltitudeCrossEvent(Event):
    """Detect if a satellite crosses a specific threshold altitude.

    Parameters
    ----------
    alt: float
        Threshold altitude (km).
    R: float
        Radius of the attractor (km).
    terminal: bool
        Whether to terminate integration if this event occurs.
    direction: float
        Handle triggering of event based on whether altitude is crossed from above
        or below, defaults to -1, i.e., event is triggered only if altitude is
        crossed from above (decreasing altitude).

    """

    def __init__(self, alt, R, terminal=True, direction=-1):
        super().__init__(terminal, direction)
        self._R = R
        self._alt = alt  # Threshold altitude from the ground.

    def __call__(self, t, u, k):
        self._last_t = t
        H = norm(u[:3])
        # SciPy will search for H - R = 0
        return H - self._R


class PenumbraEvent:
    """Detect whether a satellite is in penumbra or not.

    Parameters
    ----------
    r_sun: ~astropy.units.Quantity
        Position vector of the Sun with respect to the Earth.
    R: ~astropy.units.Quantity
        Radius of the attractor.

    """

    def __init__(self, r_sun, R):
        self._r_sun = r_sun
        self._R = R
        self._last_t = None

    @property
    def terminal(self):
        return False  # Satellite keeps on moving, unlike the LithobrakeEvent.

    @property
    def last_t(self):
        return self._last_t * u_.s

    def __call__(self, t, u, k):
        self._last_t = t
        r_sun = self._r_sun.to(u_.km).value
        R = self._R.to(u_.km).value
        r_sat = (u[:3] * u_.km).value
        shadow = penumbral_shadow_fast(r_sat, r_sun, R)

        return shadow


class UmbraEvent:
    """Detect whether a satellite is in umbra or not.

    Parameters
    ----------
    r_sun: ~astropy.units.Quantity
        Position vector of the Sun with respect to the Earth in the ECI frame.
    R: ~astropy.units.Quantity
        Radius of the attractor.

    """

    def __init__(self, r_sun, R):
        self._r_sun = r_sun
        self._R = R
        self._last_t = None

    @property
    def terminal(self):
        return False

    @property
    def last_t(self):
        return self._last_t * u_.s

    def __call__(self, t, u, k):
        self._last_t = t
        r_sun = self._r_sun.to(u_.km).value
        R = self._R.to(u_.km).value
        r_sat = (u[:3] * u_.km).value
        shadow = umbral_shadow_fast(r_sat, r_sun, R)

        return shadow
