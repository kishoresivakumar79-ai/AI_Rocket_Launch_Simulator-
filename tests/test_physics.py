import pytest
import numpy as np
from simulation.gravity_model import GravityModel
from simulation.atmosphere_model import AtmosphereModel
from simulation.physics_engine import SimulationEngine
from config.rocket_config import ROCKET_PRESETS

def test_gravity_decay():
    grav = GravityModel(g0=9.80665, earth_radius=6371000.0)
    g_surface = grav.get_gravity(0.0)
    assert pytest.approx(g_surface, rel=1e-4) == 9.80665

    # At 100 km altitude, gravity should be slightly lower (~9.5 m/s^2)
    g_100k = grav.get_gravity(100000.0)
    assert 9.4 < g_100k < 9.6

def test_atmosphere_lapse():
    atm = AtmosphereModel()
    sea_props = atm.get_properties(0.0)
    assert pytest.approx(sea_props["temperature_k"], rel=1e-3) == 288.15
    assert pytest.approx(sea_props["pressure_pa"], rel=1e-3) == 101325.0

    # At 10,000m altitude (troposphere), temperature should drop: 288.15 - 0.0065*10000 = 223.15 K
    trop_props = atm.get_properties(10000.0)
    assert pytest.approx(trop_props["temperature_k"], rel=1e-2) == 223.15
    assert trop_props["density_kg_m3"] < sea_props["density_kg_m3"]

def test_transonic_drag_rise():
    atm = AtmosphereModel()
    cd_subsonic = atm.get_drag_coefficient(0.3, mach=0.5)
    cd_transonic = atm.get_drag_coefficient(0.3, mach=1.05)
    cd_supersonic = atm.get_drag_coefficient(0.3, mach=3.0)

    # Transonic wave drag peak should exceed subsonic drag
    assert cd_transonic > cd_subsonic
    assert cd_supersonic < cd_transonic

