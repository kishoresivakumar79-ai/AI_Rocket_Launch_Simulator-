# Mission Flight Phase and Event Controller
from enum import Enum
from typing import List, Dict, Any, Optional

class FlightPhase(str, Enum):
    COUNTDOWN = "COUNTDOWN"
    LIFTOFF = "LIFTOFF"
    POWERED_ASCENT = "POWERED_ASCENT"
    TRANSONIC = "TRANSONIC PASSING"
    MAX_Q = "MAX DYNAMIC PRESSURE (MAX-Q)"
    MECO = "MAIN ENGINE CUTOFF (MECO)"
    STAGE_SEPARATION = "STAGE SEPARATION"
    STAGE_2_BURNING = "STAGE 2 POWERED FLIGHT"
    SECO = "SECOND ENGINE CUTOFF (SECO)"
    BALLISTIC_COAST = "BALLISTIC COASTING"
    APOGEE = "APOGEE (MAX ALTITUDE)"
    ATMOSPHERIC_REENTRY = "ATMOSPHERIC RE-ENTRY"
    SPLASHDOWN_OR_TOUCHDOWN = "MISSION COMPLETE / TOUCHDOWN"

class FlightController:
    """
    Monitors vehicle trajectory, identifies telemetry threshold events,
    records mission timeline, and drives the active mission phase.
    """
    def __init__(self):
        self.current_phase = FlightPhase.COUNTDOWN
        self.events: List[Dict[str, Any]] = []
        self.max_q_val: float = 0.0
        self.max_q_time: float = 0.0
        self.max_q_alt: float = 0.0
        self.max_velocity: float = 0.0
        self.max_vel_time: float = 0.0
        self.max_altitude: float = 0.0
        self.apogee_time: float = 0.0
        self.prev_vz: float = 0.0
        self.transonic_logged: bool = False
        self.meco_logged: bool = False
        self.seco_logged: bool = False
        self.apogee_logged: bool = False
        self.max_q_confirmed: bool = False
        self.touchdown_logged: bool = False

    def log_event(self, time_s: float, event_name: str, details: str, telemetry: Dict[str, Any]):
        self.events.append({
            "time_s": round(time_s, 2),
            "event": event_name,
            "details": details,
            "altitude_m": round(telemetry.get("altitude", 0.0), 1),
            "velocity_mps": round(telemetry.get("velocity", 0.0), 1),
            "downrange_km": round(telemetry.get("downrange_km", 0.0), 2)
        })

    def update(
        self,
        time_s: float,
        position: List[float],
        velocity: List[float],
        acceleration: List[float],
        fuel_status: Dict[str, Any],
        aux_data: Dict[str, float]
    ) -> FlightPhase:
        alt = position[2]
        speed = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**0.5
        vz = velocity[2]
        mach = aux_data.get("mach", 0.0)
        q_dyn = aux_data.get("q_dyn", 0.0)
        downrange_km = (position[0]**2 + position[1]**2)**0.5 / 1000.0

        current_telem = {
            "altitude": alt,
            "velocity": speed,
            "downrange_km": downrange_km
        }

        # Liftoff trigger
        if time_s == 0.0:
            self.current_phase = FlightPhase.LIFTOFF
            self.log_event(time_s, "LIFTOFF", "Liftoff confirmed! Clock is running.", current_telem)

        # Track Max Velocity
        if speed > self.max_velocity:
            self.max_velocity = speed
            self.max_vel_time = time_s

        # Track Max Altitude (Apogee)
        if alt > self.max_altitude:
            self.max_altitude = alt
            self.apogee_time = time_s

        # Track Max-Q dynamic pressure
        if q_dyn > self.max_q_val:
            self.max_q_val = q_dyn
            self.max_q_time = time_s
            self.max_q_alt = alt
        elif not self.max_q_confirmed and self.max_q_val > 10000.0 and q_dyn < self.max_q_val * 0.95 and alt > 8000:
            self.max_q_confirmed = True
            self.log_event(
                self.max_q_time,
                "MAX-Q",
                f"Maximum aerodynamic dynamic pressure {self.max_q_val/1000:.1f} kPa reached at {self.max_q_alt:.0f}m",
                {"altitude": self.max_q_alt, "velocity": speed, "downrange_km": downrange_km}
            )

        # Transonic passing
        if not self.transonic_logged and mach >= 1.0:
            self.transonic_logged = True
            self.current_phase = FlightPhase.TRANSONIC
            self.log_event(time_s, "MACH 1 (TRANSONIC)", f"Rocket has exceeded Mach 1.0 at {alt:.0f} m", current_telem)

        # Stage separation & Cutoff checks
        staging_flag = fuel_status.get("staging_occurred", False)
        stage_idx = fuel_status.get("current_stage_index", 0)

        if staging_flag:
            if stage_idx == 0:
                self.meco_logged = True
                self.current_phase = FlightPhase.MECO
                self.log_event(time_s, "MECO", "Main Engine Cutoff. Booster burnout.", current_telem)
                self.current_phase = FlightPhase.STAGE_SEPARATION
                self.log_event(time_s + 0.5, "STAGE SEPARATION", "Stage 1 separated successfully. Upper stage primed.", current_telem)
            elif stage_idx == 1:
                self.seco_logged = True
                self.current_phase = FlightPhase.SECO
                self.log_event(time_s, "SECO", "Second Stage Engine Cutoff. Payload in target trajectory.", current_telem)

        # Active powered flight phases
        if fuel_status.get("stage_active", False) and fuel_status.get("thrust", 0.0) > 0:
            if stage_idx == 0 and not staging_flag:
                self.current_phase = FlightPhase.POWERED_ASCENT
            elif stage_idx > 0 and not staging_flag:
                self.current_phase = FlightPhase.STAGE_2_BURNING
        elif not fuel_status.get("stage_active", False) or fuel_status.get("thrust", 0.0) == 0:
            if not self.apogee_logged and vz > 0:
                self.current_phase = FlightPhase.BALLISTIC_COAST

        # Apogee detection: vz crosses from positive to negative
        if not self.apogee_logged and self.prev_vz > 0.0 and vz <= 0.0 and alt > 500.0:
            self.apogee_logged = True
            self.current_phase = FlightPhase.APOGEE
            self.log_event(time_s, "APOGEE", f"Apogee reached: {alt:.1f} m ({alt/1000:.2f} km) at T+{time_s:.1f}s", current_telem)

        # Descent & Touchdown
        if self.apogee_logged and alt < 80000.0 and vz < -10.0:
            self.current_phase = FlightPhase.ATMOSPHERIC_REENTRY

        if not self.touchdown_logged and self.apogee_logged and alt <= 5.0 and time_s > 10.0:
            self.touchdown_logged = True
            self.current_phase = FlightPhase.SPLASHDOWN_OR_TOUCHDOWN
            self.log_event(time_s, "TOUCHDOWN", f"Flight concluded. Distance: {downrange_km:.2f} km downrange.", current_telem)

        self.prev_vz = vz
        return self.current_phase

