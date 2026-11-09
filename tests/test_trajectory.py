import pytest
import numpy as np
from simulation.trajectory import Trajectory3DOF
from simulation.atmosphere_model import AtmosphereModel
from simulation.gravity_model import GravityModel
from simulation.physics_engine import simulate_rocket

def test_vertical_attitude_low_altitude():
    traj = Trajectory3DOF(AtmosphereModel(), GravityModel())
    # Below pitch kick alt (1200m), attitude should be pure vertical [0, 0, 1]
    att = traj.compute_flight_attitude(altitude=500.0, velocity_vector=np.array([0.0, 0.0, 50.0]))
    assert np.allclose(att, np.array([0.0, 0.0, 1.0]))

def test_rk4_step_motion():
    traj = Trajectory3DOF(AtmosphereModel(), GravityModel())
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])
    thrust_dir = np.array([0.0, 0.0, 1.0])
    wind = np.array([0.0, 0.0, 0.0])

    new_pos, new_vel, aux = traj.rk4_step(
        position=pos,
        velocity=vel,
        dt=1.0,
        mass=1000.0,
        thrust_magnitude=20000.0, # TWR = 2.0 approx
        thrust_direction=thrust_dir,
        cross_sectional_area=1.0,
        base_drag_cd=0.3,
        wind_vector=wind
    )

    # Rocket should move upward
    assert new_pos[2] > 0
    assert new_vel[2] > 0
    assert aux["thrust_force_n"] == 20000.0

def test_simulate_rocket_prototype():
    data = simulate_rocket(initial_mass=10000, dry_mass=4000, fuel_mass=6000, thrust=200000, burn_time=120, dt=0.5)
    assert isinstance(data, np.ndarray)
    assert data.shape[1] == 5 # time, altitude, velocity, acceleration, fuel
    assert data[:, 1].max() > 10000.0 # Reaches significant altitude

