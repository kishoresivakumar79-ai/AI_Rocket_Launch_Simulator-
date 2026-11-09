import pytest
from simulation.fuel_manager import FuelManager
from config.rocket_config import ROCKET_PRESETS

def test_fuel_consumption():
    profile = ROCKET_PRESETS["Educational Sounding Rocket"]
    mgr = FuelManager(profile)

    initial_fuel = mgr.total_current_fuel
    initial_mass = mgr.total_current_mass

    assert initial_fuel == 6000.0
    assert initial_mass == 4000.0 + 6000.0 + 250.0

    # Step 10 seconds
    res = mgr.step_fuel_consumption(dt=10.0, throttle=1.0)
    assert res["thrust"] == 220000.0
    assert mgr.total_current_fuel < initial_fuel
    assert mgr.total_current_mass < initial_mass

def test_staging_event():
    # Falcon 9 has 2 stages
    profile = ROCKET_PRESETS["Falcon 9 Medium-Lift (2-Stage)"]
    mgr = FuelManager(profile)

    assert mgr.current_stage_idx == 0
    assert len(mgr.stages) == 2

    # Burn out stage 1 artificially by requesting huge dt
    res = mgr.step_fuel_consumption(dt=300.0)
    assert res["staging_occurred"] is True
    assert mgr.current_stage_idx == 1
    assert mgr.current_stage["name"] == "Stage 2 (Vacuum Upper)"

