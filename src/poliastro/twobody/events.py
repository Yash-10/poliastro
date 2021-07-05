from astropy import units as u
from numpy.linalg import norm

from astropy.coordinates import get_body_barycentric_posvel

from poliastro.core.events import in_umbral_shadow as in_umbral_shadow_fast, in_penumbral_shadow as in_penumbral_shadow_fast

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
        return self._last_t * u.s

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
        r_norm = norm(u[:3])

        return (
            r_norm - self._R - self._alt
        )  # If this goes from +ve to -ve, altitude is decreasing.


class LithobrakeEvent(AltitudeCrossEvent):
    """Terminal event that detects impact with the attractor surface.
    Parameters
    ----------
    R : float
        Radius of the attractor (km).
    terminal: bool
        Whether to terminate integration if this event occurs.
    """

    def __init__(self, R, terminal=True):
        super().__init__(0, R, terminal, direction=-1)


class PenumbraEvent(Event):
    """Detect whether a satellite is in penumbra or not.

    Parameters
    ----------
    r_sec: ~astropy.units.Quantity
        Position vector of the seconday body with respect to the primary body.
    orbit: poliastro.twobody.orbit.Orbit
        Orbit of the satellite.
    terminal: bool
        Whether to terminate integration when the event occurs, defaults to False.

    """

    def __init__(self, orbit, terminal=False, direction=0):
        super().__init__(terminal, direction)
        self._primary_body = orbit.attractor
        self._secondary_body = orbit.attractor.parent
        self._epoch = orbit.epoch

    def __call__(self, t, u_, k):
        self._last_t = t

        # Solve for primary and secondary bodies position w.r.t. solar system
        # barycenter at a particular epoch.
        (r_primary_wrt_ssb, _), (r_secondary_wrt_ssb, _) = [
            get_body_barycentric_posvel(body.name, self._epoch + t * u.s)
            for body in (self._primary_body, self._secondary_body)
        ]
        r_sec = (r_secondary_wrt_ssb - r_primary_wrt_ssb).xyz.to(u.km).value
        r_sat = (u_[:3] << u.km).value

        in_penumbra = in_penumbral_shadow_fast(r_sat, r_sec, self._primary_body.R, self._secondary_body.R)

        return in_penumbra


class UmbraEvent(Event):
    """Detect whether a satellite is in umbra or not.

    Parameters
    ----------
    r_sec: ~astropy.units.Quantity
        Position vector of the secondary body with respect to the Earth.
    orbit: poliastro.twobody.orbit.Orbit
        Orbit of the satellite.
    terminal: bool
        Whether to terminate integration when the event occurs, defaults to False.

    """

    def __init__(self, orbit, terminal=False, direction=0):
        super().__init__(terminal, direction)
        self._primary_body = orbit.attractor
        self._secondary_body = orbit.attractor.parent
        self._epoch = orbit.epoch

    def __call__(self, t, u_, k):
        self._last_t = t

        # Solve for primary and secondary bodies position w.r.t. solar system
        # barycenter at a particular epoch.
        (r_primary_wrt_ssb, _), (r_secondary_wrt_ssb, _) = [
            get_body_barycentric_posvel(body.name, self._epoch + t * u.s)
            for body in (self._primary_body, self._secondary_body)
        ]
        r_sec = (r_secondary_wrt_ssb - r_primary_wrt_ssb).xyz.to(u.km).value
        r_sat = (u_[:3] << u.km).value

        in_umbra = in_umbral_shadow_fast(r_sat, r_sec, self._primary_body.R, self._secondary_body.R)

        return in_umbra
