import numpy as np
import pytest
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from numpy.linalg import norm

from poliastro.bodies import Earth
from poliastro.constants import H0_earth, rho0_earth
from poliastro.core.perturbations import atmospheric_drag_exponential
from poliastro.core.propagation import func_twobody
from poliastro.twobody import Orbit
from poliastro.twobody.events import AltitudeCrossEvent, PenumbraEvent, UmbraEvent
from poliastro.twobody.propagation import cowell


@pytest.mark.slow
def test_altitude_crossing():
    # Test decreasing altitude cross over Earth. No analytic solution.
    R = Earth.R.to(u.km).value

    orbit = Orbit.circular(Earth, 230 * u.km)
    t_flight = 48.209538 * u.d

    # Parameters of a body
    C_D = 2.2  # Dimensionless (any value would do)
    A_over_m = ((np.pi / 4.0) * (u.m ** 2) / (100 * u.kg)).to_value(
        u.km ** 2 / u.kg
    )  # km^2/kg

    # Parameters of the atmosphere
    rho0 = rho0_earth.to(u.kg / u.km ** 3).value  # kg/km^3
    H0 = H0_earth.to(u.km).value  # km

    tofs = [50] * u.d

    thresh_alt = 50  # km
    altitude_cross_event = AltitudeCrossEvent(thresh_alt, R)
    events = [altitude_cross_event]

    def f(t0, u_, k):
        du_kep = func_twobody(t0, u_, k)
        ax, ay, az = atmospheric_drag_exponential(
            t0, u_, k, R=R, C_D=C_D, A_over_m=A_over_m, H0=H0, rho0=rho0
        )
        du_ad = np.array([0, 0, 0, ax, ay, az])
        return du_kep + du_ad

    rr, _ = cowell(
        Earth.k,
        orbit.r,
        orbit.v,
        tofs,
        events=events,
        f=f,
    )

    assert_quantity_allclose(norm(rr[0].to(u.km).value) - thresh_alt, R)
    assert_quantity_allclose(altitude_cross_event.last_t, t_flight, rtol=1e-2)


def test_altitude_cross_not_happening_is_ok():
    R = Earth.R.to(u.km).value

    orbit = Orbit.circular(Earth, 230 * u.km)

    tofs = [25] * u.d

    thresh_alt = 50  # km
    altitude_cross_event = AltitudeCrossEvent(thresh_alt, R)
    events = [altitude_cross_event]

    rr, _ = cowell(
        Earth.k,
        orbit.r,
        orbit.v,
        tofs,
        events=events,
    )

    assert altitude_cross_event.last_t == tofs[-1]


def test_umbra():
    tofs = np.array([0, 100, 1000]) << u.d
    # Data for `r_sun` and `r0` taken from Howard Curtis (Example 10.8)
    r_sun = np.array([-11747041, 139486985, 60472278]) << u.km
    r0 = np.array([2817.899, -14110.473, -7502.672]) << u.km
    v0 = np.array([736.138, 298.997, 164.354]) << u.km / u.s
    orbit = Orbit.from_vectors(Earth, r0, v0)

    umbra_event = UmbraEvent(orbit)
    events = [umbra_event]

    rr, _ = cowell(
        Earth.k,
        orbit.r,
        orbit.v,
        tofs,
        events=events,
    )

    # assert ?


def test_umbra_event_terminal_set_to_true():
    tofs = np.array([0, 100, 1000]) << u.d
    r0 = np.array([2817.899, -14110.473, -7502.672]) << u.km
    v0 = np.array([736.138, 298.997, 164.354]) << u.km / u.s
    orbit = Orbit.from_vectors(Earth, r0, v0)

    # Check terminal works
    umbra_event = UmbraEvent(orbit, terminal=True)
    events = [umbra_event]

    rr, _ = cowell(
        Earth.k,
        orbit.r,
        orbit.v,
        tofs,
        events=events,
    )

    # assert ?

def test_umbra_event_not_firing_is_ok():
    # Check umbra event not firing is ok.
    tofs = [1000] * u.s
    r0 = np.array([281.89, 1411.473, 750.672]) << u.km
    v0 = np.array([7.36138, 2.98997, 1.64354]) << u.km / u.s
    orbit = Orbit.from_vectors(Earth, r0, v0)

    umbra_event = UmbraEvent(orbit)
    events = [umbra_event]

    rr, _ = cowell(
        Earth.k,
        orbit.r,
        orbit.v,
        tofs,
        events=events,
    )

    assert umbra_event.last_t == tofs[-1]


def test_penumbra():
    tofs = np.array([0, 100, 1000]) << u.d
    # Data for `r_sun` and `r0` taken from Howard Curtis (Example 10.8)
    r_sun = np.array([-11747041, 139486985, 60472278]) << u.km
    r0 = np.array([2817.899, -14110.473, -7502.672]) << u.km
    v0 = np.array([736.138, 298.997, 164.354]) << u.km / u.s
    orbit = Orbit.from_vectors(Earth, r0, v0)

    penumbra_event = PenumbraEvent(orbit)
    events = [penumbra_event]

    rr, _ = cowell(
        Earth.k,
        orbit.r,
        orbit.v,
        tofs,
        events=events,
    )

    # assert ?


def test_penumbra_event_terminal_set_to_true():
    tofs = np.array([0, 100, 1000]) << u.d
    r0 = np.array([2817.899, -14110.473, -7502.672]) << u.km
    v0 = np.array([736.138, 298.997, 164.354]) << u.km / u.s
    orbit = Orbit.from_vectors(Earth, r0, v0)

    # Check terminal works
    penumbra_event = PenumbraEvent(orbit, terminal=True)
    events = [penumbra_event]

    rr, _ = cowell(
        Earth.k,
        orbit.r,
        orbit.v,
        tofs,
        events=events,
    )

    # assert ?


def test_penumbra_event_not_firing_is_ok():
    # Check penumbra event not firing is ok.
    tofs = [1000] * u.s
    r0 = np.array([281.89, 1411.473, 750.672]) << u.km
    v0 = np.array([7.36138, 2.98997, 1.64354]) << u.km / u.s
    orbit = Orbit.from_vectors(Earth, r0, v0)

    penumbra_event = PenumbraEvent(orbit)
    events = [penumbra_event]

    rr, _ = cowell(
        Earth.k,
        orbit.r,
        orbit.v,
        tofs,
        events=events,
    )

    assert penumbra_event.last_t == tofs[-1]
