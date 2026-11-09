# Standard Atmospheric Model (US Standard Atmosphere 1976 Approximation)
import math

class AtmosphereModel:
    """
    US Standard Atmosphere (1976) multi-layer model.
    Calculates temperature, pressure, density, speed of sound, and transonic drag rise.
    """
    R_SPECIFIC = 287.05287  # Specific gas constant for dry air (J/(kg*K))
    GAMMA = 1.4             # Ratio of specific heats for air
    G0 = 9.80665            # Standard gravity (m/s^2)

    # Layer boundaries (geopotential altitude in meters), base temp (K), base pressure (Pa), lapse rate (K/m)
    LAYERS = [
        (0.0, 11000.0, 288.15, 101325.0, -0.0065),      # Troposphere
        (11000.0, 20000.0, 216.65, 22632.1, 0.0),       # Tropopause (isothermal)
        (20000.0, 32000.0, 216.65, 5474.89, 0.0010),     # Stratosphere 1
        (32000.0, 47000.0, 228.65, 868.019, 0.0028),     # Stratosphere 2
        (47000.0, 51000.0, 270.65, 110.906, 0.0),        # Stratopause (isothermal)
        (51000.0, 71000.0, 270.65, 66.9389, -0.0028),    # Mesosphere 1
        (71000.0, 84852.0, 214.65, 3.95642, -0.0020),    # Mesosphere 2
    ]

    def __init__(self, sea_level_temp: float = 288.15, sea_level_pressure: float = 101325.0):
        self.t0 = sea_level_temp
        self.p0 = sea_level_pressure

    def get_properties(self, altitude_m: float):
        """
        Returns a dict of atmospheric properties at altitude_m:
        temperature (K), pressure (Pa), density (kg/m^3), speed_of_sound (m/s)
        """
        h = max(0.0, float(altitude_m))
        
        # Beyond ~85km, treat as near-vacuum exosphere
        if h >= 84852.0:
            # Exosphere exponential decay
            scale_height = 7000.0
            density = 1.225 * math.exp(-h / scale_height)
            temp = 186.87
            pressure = density * self.R_SPECIFIC * temp
            sound_speed = math.sqrt(self.GAMMA * self.R_SPECIFIC * temp)
            return {
                "temperature_k": temp,
                "pressure_pa": max(0.0, pressure),
                "density_kg_m3": max(0.0, density),
                "speed_of_sound_mps": sound_speed
            }

        # Find matching atmospheric layer
        for h_base, h_top, t_base, p_base, lapse in self.LAYERS:
            if h <= h_top:
                delta_h = h - h_base
                if lapse != 0.0:
                    temp = t_base + lapse * delta_h
                    pressure = p_base * ((temp / t_base) ** (-self.G0 / (lapse * self.R_SPECIFIC)))
                else:
                    temp = t_base
                    pressure = p_base * math.exp(-self.G0 * delta_h / (self.R_SPECIFIC * t_base))
                
                density = pressure / (self.R_SPECIFIC * temp)
                sound_speed = math.sqrt(self.GAMMA * self.R_SPECIFIC * temp)
                return {
                    "temperature_k": temp,
                    "pressure_pa": max(0.0, pressure),
                    "density_kg_m3": max(0.0, density),
                    "speed_of_sound_mps": sound_speed
                }

        # Fallback default
        return {"temperature_k": 216.65, "pressure_pa": 1e-6, "density_kg_m3": 1e-9, "speed_of_sound_mps": 295.0}

    def get_mach_number(self, velocity_mps: float, altitude_m: float) -> float:
        """Calculate Mach number given true airspeed and altitude."""
        props = self.get_properties(altitude_m)
        c = props["speed_of_sound_mps"]
        return abs(velocity_mps) / max(c, 1.0)

    def get_dynamic_pressure(self, velocity_mps: float, altitude_m: float) -> float:
        """Calculates dynamic pressure Q = 0.5 * rho * v^2 in Pascals."""
        props = self.get_properties(altitude_m)
        rho = props["density_kg_m3"]
        return 0.5 * rho * (velocity_mps ** 2)

    def get_drag_coefficient(self, base_cd: float, mach: float) -> float:
        """
        Wave drag model across subsonic, transonic, and supersonic regimes:
        - Subsonic (Mach < 0.8): base_cd
        - Transonic (0.8 <= Mach <= 1.2): peak wave drag rise up to ~2.0 * base_cd
        - Supersonic (Mach > 1.2): decays gradually as 1 / sqrt(M^2 - 1)
        """
        if mach < 0.8:
            return base_cd
        elif mach <= 1.05:
            # Steep wave drag rise
            fraction = (mach - 0.8) / 0.25
            return base_cd * (1.0 + 1.1 * (math.sin(fraction * math.pi / 2) ** 2))
        elif mach <= 1.4:
            # Peak transonic plateau
            fraction = (mach - 1.05) / 0.35
            return base_cd * (2.1 - 0.5 * fraction)
        else:
            # Supersonic decay towards asymptotic limit
            decay = 1.0 / math.sqrt(max(0.1, mach ** 2 - 1.0))
            return max(base_cd * 0.9, base_cd * (1.1 + 0.5 * decay))

