import pytest
from dataclasses import dataclass
from shapely.geometry import Polygon, MultiPolygon
from dem_handler.utils.spatial import (
    get_bounds_for_shape_crossing_antimeridian,
    check_shape_crosses_antimeridian,
    check_bounds_cross_antimeridian,
    check_dem_type_in_bounds,
)


@dataclass
class CheckCrossAM:
    test_name: str
    shape: Polygon | MultiPolygon
    bounds: tuple
    max_antimeridian_crossing_degrees: float
    crosses_AM: bool


# crosses the AM
poly1 = Polygon(
    [
        (178.576126, -71.618423),
        (-178.032867, -70.167343),
        (176.938004, -68.765106),
        (173.430893, -70.119957),
        (178.576126, -71.618423),
    ],
)

# crosses the AM
poly2 = Polygon(
    [
        (179.5, -10.0),  # Just west of the antimeridian
        (-179.8, 5.0),  # Just east of the antimeridian
        (-178.5, 15.0),  # Further east
        (178.0, 12.0),  # West again
        (179.5, -10.0),  # Close the ring
    ]
)

# crosses the AM
poly3 = Polygon(
    [
        (170.0, -55.0),  # Western point
        (-175.0, -50.0),  # Just east of antimeridian
        (-160.0, -45.0),  # Farther east
        (175.0, -40.0),  # Back west
        (170.0, -55.0),  # Close the loop
    ]
)

# crosses the AM
poly4 = MultiPolygon([poly1, poly2])

# polygon on the regular meridian
poly5 = Polygon(
    [
        (-2.5, -5.0),  # Southwest corner
        (2.5, -5.0),  # Southeast corner
        (2.5, 5.0),  # Northeast corner
        (-2.5, 5.0),  # Northwest corner
        (-2.5, -5.0),  # Close the ring
    ]
)

# Polygon fully in the western hemisphere (centered around -5° longitude, 5° wide)
poly6 = Polygon(
    [
        (-7.5, -5.0),  # Southwest corner
        (-2.5, -5.0),  # Southeast corner
        (-2.5, 5.0),  # Northeast corner
        (-7.5, 5.0),  # Northwest corner
        (-7.5, -5.0),  # Close the ring
    ]
)

# the bounds corresponding to the shapes
b1 = (-178.032867, -71.618423, 173.430893, -68.765106)
b2 = (-178.5, -10, 178.0, 15)
b3 = (-160, -55, 170, -40)
b4 = (-178.032867, -71.618423, 173.430893, 15)
b5 = (-2.5, -5, 2.5, 5)
b6 = (-7.5, -5, -2.5, 5)

t1a = CheckCrossAM(
    test_name="t1a",
    shape=poly1,
    bounds=b1,
    max_antimeridian_crossing_degrees=20,
    crosses_AM=True,
)
t1b = CheckCrossAM(
    test_name="t1b",
    shape=poly1,
    bounds=b1,
    max_antimeridian_crossing_degrees=1,
    crosses_AM=False,  # max_antimeridian_crossing_degrees not big enough,
)

t2 = CheckCrossAM(
    test_name="t2",
    shape=poly2,
    bounds=b2,
    max_antimeridian_crossing_degrees=20,
    crosses_AM=True,
)
t3a = CheckCrossAM(
    test_name="t3a",
    shape=poly3,
    bounds=b3,
    max_antimeridian_crossing_degrees=20,
    crosses_AM=False,  # max_antimeridian_crossing_degrees not big enough,
)
t3b = CheckCrossAM(
    test_name="t3b",
    shape=poly3,
    bounds=b3,
    max_antimeridian_crossing_degrees=35,
    crosses_AM=True,
)
t4 = CheckCrossAM(
    test_name="t4",
    shape=poly4,
    bounds=b4,
    max_antimeridian_crossing_degrees=20,
    crosses_AM=True,
)
t5 = CheckCrossAM(
    test_name="t5",
    shape=poly5,
    bounds=b5,
    max_antimeridian_crossing_degrees=20,
    crosses_AM=False,
)
t6 = CheckCrossAM(
    test_name="t6",
    shape=poly6,
    bounds=b6,
    max_antimeridian_crossing_degrees=20,
    crosses_AM=False,
)


# Combine for parameterization
test_cases = [t1a, t1b, t2, t3a, t3b, t4, t5, t6]


@pytest.mark.parametrize("test_case", test_cases)
def test_get_bounds_for_shape_crossing_antimeridian(test_case):
    # assert the incorrect bounds still cross the AM
    shape_crosses_AM = check_shape_crosses_antimeridian(
        test_case.shape, test_case.max_antimeridian_crossing_degrees
    )
    assert shape_crosses_AM == test_case.crosses_AM
    bounds_cross_AM = check_bounds_cross_antimeridian(
        test_case.bounds, test_case.max_antimeridian_crossing_degrees
    )
    assert bounds_cross_AM == test_case.crosses_AM
    if shape_crosses_AM:
        # get the bounds for the shape that crosses
        bounds = get_bounds_for_shape_crossing_antimeridian(test_case.shape)
        assert bounds == test_case.bounds


@dataclass
class BoundsDEMCheckCase:
    dem_type: str
    resolution: int
    bounds: tuple[float, float, float, float]
    in_bounds: bool
    is_error: bool


test_invalid_dem_type = BoundsDEMCheckCase(
    dem_type="spaghetti",
    resolution=90,
    bounds=(161.96252, -70.75924, 162.10388, -70.72293),
    in_bounds=False,
    is_error=True,
)

test_invalid_dem_resolution = BoundsDEMCheckCase(
    dem_type="REMA",
    resolution=15,
    bounds=(161.96252, -70.75924, 162.10388, -70.72293),
    in_bounds=False,
    is_error=True,
)

test_antarctic_bounds_with_rema = BoundsDEMCheckCase(
    dem_type="REMA",
    resolution=32,
    bounds=(161.96252, -70.75924, 162.10388, -70.72293),
    in_bounds=True,
    is_error=False,
)

test_antarctic_bounds_with_cop30 = BoundsDEMCheckCase(
    dem_type="copernicus",
    resolution=30,
    bounds=(161.96252, -70.75924, 162.10388, -70.72293),
    in_bounds=True,
    is_error=False,
)

test_australia_bounds_with_rema = BoundsDEMCheckCase(
    dem_type="REMA",
    resolution=10,
    bounds=(133.0, -25.0, 134.0, -24.0),
    in_bounds=False,
    is_error=False,
)

test_heard_island_bounds_with_rema = BoundsDEMCheckCase(
    dem_type="rema_2",
    resolution=32,
    bounds=(73.2144, -53.224, 73.87, -52.962),
    in_bounds=False,
    is_error=False,
)

test_heard_island_bounds_with_cop30 = BoundsDEMCheckCase(
    dem_type="cop_glo30",
    resolution=30,
    bounds=(73.2144, -53.224, 73.87, -52.962),
    in_bounds=True,
    is_error=False,
)

test_antimeridian_with_cop30 = BoundsDEMCheckCase(
    dem_type="cop_glo30",
    resolution=30,
    bounds=(-178.032867, -80.618423, 173.430893, -68.765106),
    in_bounds=True,
    is_error=False,
)

test_antimeridian_with_rema = BoundsDEMCheckCase(
    dem_type="rema",
    resolution=2,
    bounds=(-178.032867, -80.618423, 173.430893, -68.765106),
    in_bounds=True,
    is_error=False,
)

test_cases = [
    test_invalid_dem_type,
    test_invalid_dem_resolution,
    test_antarctic_bounds_with_rema,
    test_antarctic_bounds_with_cop30,
    test_australia_bounds_with_rema,
    test_heard_island_bounds_with_rema,
    test_heard_island_bounds_with_cop30,
    test_antimeridian_with_cop30,
    test_antimeridian_with_rema,
]


@pytest.mark.parametrize("test_case", test_cases)
def test_check_dem_type_in_bounds(test_case: BoundsDEMCheckCase):

    if test_case.is_error:
        with pytest.raises(ValueError):
            check_dem_type_in_bounds(
                test_case.dem_type, test_case.resolution, test_case.bounds
            )

    else:
        assert (
            check_dem_type_in_bounds(
                test_case.dem_type, test_case.resolution, test_case.bounds
            )
            == test_case.in_bounds
        )
