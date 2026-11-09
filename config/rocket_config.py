# Configuration for Rocket Presets and Specs
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class StageConfig:
    name: str
    dry_mass: float  # kg
    fuel_mass: float  # kg
    thrust: float  # N (vacuum or sea-level average)
    isp: float  # Specific impulse in seconds (e.g., 280-450 s)
    burn_time: float  # Calculated or prescribed in seconds
    diameter: float  # meters (for frontal area calculation)
    drag_cd: float = 0.35  # Subsonic drag coefficient

    @property
    def total_mass(self) -> float:
        return self.dry_mass + self.fuel_mass

    @property
    def mass_flow_rate(self) -> float:
        """Mass flow rate dot{m} = Thrust / (Isp * g0)"""
        g0 = 9.80665
        if self.isp > 0:
            return self.thrust / (self.isp * g0)
        return self.fuel_mass / max(self.burn_time, 1.0)


@dataclass
class RocketProfile:
    name: str
    description: str
    stages: List[StageConfig]
    payload_mass: float  # kg
    target_pitch_over_alt: float = 1200.0  # Altitude (m) to start gravity turn pitch kick
    pitch_kick_angle_deg: float = 8.0     # Initial pitch kick tilt (degrees from vertical)
    final_pitch_deg: float = 75.0         # Final tilt angle (degrees from vertical)

    @property
    def total_initial_mass(self) -> float:
        return sum(s.total_mass for s in self.stages) + self.payload_mass

    @property
    def total_dry_mass(self) -> float:
        return sum(s.dry_mass for s in self.stages) + self.payload_mass

    @property
    def total_fuel_mass(self) -> float:
        return sum(s.fuel_mass for s in self.stages)


ROCKET_PRESETS: Dict[str, RocketProfile] = {
    "Educational Sounding Rocket": RocketProfile(
        name="Educational Sounding Rocket",
        description="Single-stage solid/liquid sounding rocket for suborbital atmospheric research.",
        stages=[
            StageConfig(
                name="Booster Stage",
                dry_mass=4000.0,
                fuel_mass=6000.0,
                thrust=220000.0,
                isp=265.0,
                burn_time=72.0,
                diameter=1.2,
                drag_cd=0.38
            )
        ],
        payload_mass=250.0,
        target_pitch_over_alt=1000.0,
        pitch_kick_angle_deg=5.0,
        final_pitch_deg=45.0
    ),
    "Falcon 9 Medium-Lift (2-Stage)": RocketProfile(
        name="Falcon 9 Medium-Lift (2-Stage)",
        description="Two-stage orbital rocket with reusable booster recovery and high efficiency upper stage.",
        stages=[
            StageConfig(
                name="Stage 1 (Booster)",
                dry_mass=25000.0,
                fuel_mass=395000.0,
                thrust=7607000.0,
                isp=300.0,
                burn_time=162.0,
                diameter=3.7,
                drag_cd=0.32
            ),
            StageConfig(
                name="Stage 2 (Vacuum Upper)",
                dry_mass=4500.0,
                fuel_mass=92500.0,
                thrust=934000.0,
                isp=348.0,
                burn_time=397.0,
                diameter=3.7,
                drag_cd=0.30
            )
        ],
        payload_mass=15000.0,
        target_pitch_over_alt=1500.0,
        pitch_kick_angle_deg=10.0,
        final_pitch_deg=82.0
    ),
    "Saturn V Heavy-Lift (3-Stage)": RocketProfile(
        name="Saturn V Heavy-Lift (3-Stage)",
        description="Massive Apollo lunar launch vehicle with three powerful liquid propellant stages.",
        stages=[
            StageConfig(
                name="S-IC (Stage 1)",
                dry_mass=130000.0,
                fuel_mass=2160000.0,
                thrust=35100000.0,
                isp=263.0,
                burn_time=168.0,
                diameter=10.1,
                drag_cd=0.35
            ),
            StageConfig(
                name="S-II (Stage 2)",
                dry_mass=36000.0,
                fuel_mass=450000.0,
                thrust=5140000.0,
                isp=421.0,
                burn_time=360.0,
                diameter=10.1,
                drag_cd=0.33
            )
        ],
        payload_mass=45000.0,
        target_pitch_over_alt=2000.0,
        pitch_kick_angle_deg=8.0,
        final_pitch_deg=85.0
    ),
    "Artemis SLS Deep Space": RocketProfile(
        name="Artemis SLS Deep Space",
        description="NASA Artemis lunar exploration launch vehicle with solid rocket boosters and core stage.",
        stages=[
            StageConfig(
                name="Core Stage + SRBs",
                dry_mass=85000.0,
                fuel_mass=980000.0,
                thrust=39100000.0,
                isp=310.0,
                burn_time=126.0,
                diameter=8.4,
                drag_cd=0.34
            ),
            StageConfig(
                name="ICPS Upper Stage",
                dry_mass=3500.0,
                fuel_mass=29000.0,
                thrust=110000.0,
                isp=462.0,
                burn_time=1120.0,
                diameter=5.0,
                drag_cd=0.28
            )
        ],
        payload_mass=27000.0,
        target_pitch_over_alt=1800.0,
        pitch_kick_angle_deg=9.0,
        final_pitch_deg=80.0
    )
}

