"""Physical and geodetic constants shared across the package."""

#: Speed of light in vacuum, in metres per second.
SPEED_OF_LIGHT_M_S: float = 299792458.0

#: WGS-84 semi-major axis, in metres.
WGS84_SEMI_MAJOR_AXIS_M: float = 6378137.0

#: WGS-84 flattening.
WGS84_FLATTENING: float = 1 / 298.257223563

#: Margin added to the light-travel time from the network centre, as in
#: ``ligo.skymap.bayestar.condition``.
ARRIVAL_TIME_MARGIN_S: float = 0.005
