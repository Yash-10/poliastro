from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel
from numpy.linalg import norm

from poliastro.core.events import (
    in_penumbral_shadow as in_penumbral_shadow_fast,
    in_umbral_shadow as in_umbral_shadow_fast,
)
from poliastro.core.perturbations import shadow_function as shadow_function_fast


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


class EclipseEvent(Event):
    """Base class for the eclipse event.

    Parameters
    ----------
    orbit: poliastro.twobody.orbit.Orbit
        Orbit of the satellite.
    terminal: bool, optional
        Whether to terminate integration when the event occurs, defaults to False.
    direction: float, optional
        Specify which direction must the event trigger, defaults to 0.

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
        r_sec = ((r_secondary_wrt_ssb - r_primary_wrt_ssb).xyz << u.km).value
        r_sat = (u_[:3] << u.km).value

        return r_sat, r_sec, self._primary_body.R, self._secondary_body.R


class PenumbraEvent(EclipseEvent):
    """Detect whether a satellite is in penumbra or not.

    Parameters
    ----------
    orbit: poliastro.twobody.orbit.Orbit
        Orbit of the satellite.
    terminal: bool, optional
        Whether to terminate integration when the event occurs, defaults to False.
    direction: float, optional
        Handle triggering of event based on whether entry is into or out of
        penumbra, defaults to 0, i.e., event is triggered at both, entry and exit points.

    """

    def __init__(self, orbit, terminal=False, direction=0):
        super().__init__(orbit, terminal, direction)
        self._total_eclipse = False  # Check for penumbra i.e. partial eclipse.

    def __call__(self, t, u_, k):
        r_sat, r_sec, primary_body_R, secondary_body_R = super().__call__(t, u_, k)
        r_sat = (r_sat << u.km).value
        r_sec = (r_sec << u.km).value
        primary_body_R = (primary_body_R << u.km).value
        # secondary_body_R = (secondary_body_R << u.km).value
        angle_diff = shadow_function_fast(
            r_sat, r_sec, primary_body_R, total_eclipse=self._total_eclipse
        )
        return angle_diff


class UmbraEvent(EclipseEvent):
    """Detect whether a satellite is in umbra or not.

    Parameters
    ----------
    orbit: poliastro.twobody.orbit.Orbit
        Orbit of the satellite.
    terminal: bool, optional
        Whether to terminate integration when the event occurs, defaults to False.
    direction: float, optional
        Handle triggering of event based on whether entry is into or out of
        umbra, defaults to 0, i.e., event is triggered at both, entry and exit points.

    """

    def __init__(self, orbit, terminal=False, direction=0):
        super().__init__(orbit, terminal, direction)
        self._total_eclipse = True  # Check for umbra.

    def __call__(self, t, u_, k):
        r_sat, r_sec, primary_body_R, _ = super().__call__(t, u_, k)
        r_sat = (r_sat << u.km).value
        r_sec = (r_sec << u.km).value
        primary_body_R = (primary_body_R << u.km).value
        angle_diff = shadow_function_fast(
            r_sat, r_sec, primary_body_R, total_eclipse=self._total_eclipse
        )
        return angle_diff
