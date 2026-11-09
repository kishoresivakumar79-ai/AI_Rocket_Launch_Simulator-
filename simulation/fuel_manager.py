# Multi-stage Fuel & Mass Management
from typing import List, Dict, Any, Optional
from config.rocket_config import RocketProfile, StageConfig

class FuelManager:
    """
    Manages active stage burning, fuel consumption rate, staging transitions,
    and instantaneous vehicle mass.
    """
    def __init__(self, rocket_profile: RocketProfile):
        self.profile = rocket_profile
        self.stages: List[Dict[str, Any]] = []
        for idx, s in enumerate(rocket_profile.stages):
            self.stages.append({
                "index": idx,
                "name": s.name,
                "dry_mass": s.dry_mass,
                "initial_fuel": s.fuel_mass,
                "current_fuel": s.fuel_mass,
                "thrust": s.thrust,
                "isp": s.isp,
                "diameter": s.diameter,
                "drag_cd": s.drag_cd,
                "burn_time": s.burn_time,
                "ignited": False,
                "burned_out": False,
                "jettisoned": False
            })
        self.payload_mass = rocket_profile.payload_mass
        self.current_stage_idx: int = 0
        if self.stages:
            self.stages[0]["ignited"] = True

    @property
    def current_stage(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    @property
    def total_current_mass(self) -> float:
        """Total current mass = payload + non-jettisoned stages (dry + remaining fuel)"""
        m = self.payload_mass
        for s in self.stages:
            if not s["jettisoned"]:
                m += s["dry_mass"] + s["current_fuel"]
        return m

    @property
    def total_current_fuel(self) -> float:
        """Sum of remaining fuel across all non-jettisoned stages."""
        return sum(s["current_fuel"] for s in self.stages if not s["jettisoned"])

    @property
    def total_initial_fuel(self) -> float:
        """Sum of initial fuel capacity across all stages."""
        return sum(s["initial_fuel"] for s in self.stages)

    @property
    def overall_fuel_percentage(self) -> float:
        init = self.total_initial_fuel
        if init <= 0:
            return 0.0
        return max(0.0, min(100.0, (self.total_current_fuel / init) * 100.0))

    def get_stage_fuel_percentage(self, stage_idx: int) -> float:
        if 0 <= stage_idx < len(self.stages):
            s = self.stages[stage_idx]
            if s["initial_fuel"] <= 0:
                return 0.0
            return max(0.0, min(100.0, (s["current_fuel"] / s["initial_fuel"]) * 100.0))
        return 0.0

    def step_fuel_consumption(self, dt: float, throttle: float = 1.0) -> Dict[str, Any]:
        """
        Consumes fuel for active stage over time interval dt at given throttle (0.0 to 1.0).
        Returns status dict containing thrust, mass_flow, burned_out flag, and staging flag.
        """
        stage = self.current_stage
        if stage is None or stage["burned_out"]:
            return {
                "thrust": 0.0,
                "mass_flow": 0.0,
                "stage_active": False,
                "staging_occurred": False,
                "current_stage_name": "Coast / No Stage Active"
            }

        g0 = 9.80665
        nominal_thrust = stage["thrust"] * throttle
        
        # Mass flow rate dot{m} = Thrust / (Isp * g0)
        if stage["isp"] > 0:
            mdot = nominal_thrust / (stage["isp"] * g0)
        else:
            mdot = stage["initial_fuel"] / max(1.0, stage["burn_time"])

        consumed = mdot * dt
        staging_occurred = False

        if stage["current_fuel"] > consumed:
            stage["current_fuel"] -= consumed
            effective_thrust = nominal_thrust
        else:
            # Stage runs out of fuel
            consumed = stage["current_fuel"]
            fraction = consumed / max(1e-6, mdot * dt)
            effective_thrust = nominal_thrust * fraction
            stage["current_fuel"] = 0.0
            stage["burned_out"] = True
            
            # Transition to next stage if available
            stage["jettisoned"] = True
            staging_occurred = True
            self.current_stage_idx += 1
            if self.current_stage_idx < len(self.stages):
                self.stages[self.current_stage_idx]["ignited"] = True

        return {
            "thrust": effective_thrust,
            "mass_flow": mdot if effective_thrust > 0 else 0.0,
            "stage_active": True,
            "staging_occurred": staging_occurred,
            "current_stage_name": stage["name"],
            "current_stage_index": stage["index"],
            "remaining_fuel_kg": stage["current_fuel"],
            "total_mass_kg": self.total_current_mass
        }

