# Environmental and Launch Pad Configuration
from dataclasses import dataclass
from typing import Dict

@dataclass
class LaunchSite:
    name: str
    latitude_deg: float  # Degrees north/south
    longitude_deg: float # Degrees east/west
    elevation_m: float   # Altitude above sea level
    earth_rotation_boost_mps: float # Eastward tangential surface speed boost (m/s)

@dataclass
class EnvironmentConfig:
    site: LaunchSite
    sea_level_pressure_pa: float = 101325.0
    sea_level_temp_k: float = 288.15
    gravity_sea_level: float = 9.80665
    earth_radius_m: float = 6371000.0
    wind_speed_mps: float = 5.0
    wind_direction_deg: float = 90.0 # From East (0=North, 90=East)
    wind_shear_factor: float = 0.0015 # Wind speed growth per meter of altitude up to tropopause

LAUNCH_SITES: Dict[str, LaunchSite] = {
    "Kennedy Space Center (LC-39A)": LaunchSite(
        name="Kennedy Space Center (LC-39A)",
        latitude_deg=28.5721,
        longitude_deg=-80.6480,
        elevation_m=3.0,
        earth_rotation_boost_mps=408.0
    ),
    "Vandenberg Space Force Base (SLC-4E)": LaunchSite(
        name="Vandenberg Space Force Base (SLC-4E)",
        latitude_deg=34.6321,
        longitude_deg=-120.6106,
        elevation_m=112.0,
        earth_rotation_boost_mps=382.0
    ),
    "Guiana Space Centre, Kourou (CSG)": LaunchSite(
        name="Guiana Space Centre, Kourou (CSG)",
        latitude_deg=5.2374,
        longitude_deg=-52.7685,
        elevation_m=15.0,
        earth_rotation_boost_mps=463.0
    ),
    "Baikonur Cosmodrome (Site 1/5)": LaunchSite(
        name="Baikonur Cosmodrome (Site 1/5)",
        latitude_deg=45.9646,
        longitude_deg=63.3052,
        elevation_m=100.0,
        earth_rotation_boost_mps=323.0
    )
}

