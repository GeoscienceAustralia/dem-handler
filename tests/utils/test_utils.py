import pytest
from dataclasses import dataclass
from shapely.geometry import Polygon, MultiPolygon
from dem_handler.utils.spatial import (
    BoundingBox,
    get_correct_bounds_from_shape_at_antimeridian,
    check_s1_bounds_cross_antimeridian,
    check_dem_type_in_bounds,
)

# Separate lists for clarity
poly1 = Polygon(
    [
        (178.576126, -71.618423),
        (-178.032867, -70.167343),
        (176.938004, -68.765106),
        (173.430893, -70.119957),
        (178.576126, -71.618423),
    ],
)
poly2 = Polygon(
    [
        (179.5, -10.0),  # Just west of the antimeridian
        (-179.8, 5.0),  # Just east of the antimeridian
        (-178.5, 15.0),  # Further east
        (178.0, 12.0),  # West again
        (179.5, -10.0),  # Close the ring
    ]
)
poly3 = Polygon(
    [
        (170.0, -55.0),  # Western point
        (-175.0, -50.0),  # Just east of antimeridian
        (-160.0, -45.0),  # Farther east
        (175.0, -40.0),  # Back west
        (170.0, -55.0),  # Close the loop
    ]
)

poly4 = MultiPolygon([poly1, poly2])

shapes = [poly1, poly2, poly3, poly4]

expected_bounds = [
    BoundingBox(-178.032867, -71.618423, 173.430893, -68.765106),
    BoundingBox(-178.5, -10, 178.0, 15),
    BoundingBox(-160, -55, 170, -40),
    BoundingBox(-178.032867, -71.618423, 173.430893, 15),
]

# Combine for parameterization
test_cases = list(zip(shapes, expected_bounds))


@pytest.mark.parametrize("shape, expected_bound", test_cases)
def test_get_correct_bounds_from_shape_at_antimeridian(shape, expected_bound):
    # assert the incorrect bounds still cross the AM
    assert check_s1_bounds_cross_antimeridian(shape.bounds)
    result = get_correct_bounds_from_shape_at_antimeridian(shape)
    assert result == expected_bound


@dataclass
class BoundsDEMCheckCase:
    dem_type: str
    bounds: tuple[float, float, float, float]
    in_bounds: bool
    is_error: bool


test_invalid_dem_type = BoundsDEMCheckCase(
    dem_type="spaghetti",
    bounds=(161.96252, -70.75924, 162.10388, -70.72293),
    in_bounds=False,
    is_error=True,
)

test_antarctic_bounds_with_rema = BoundsDEMCheckCase(
    dem_type="REMA",
    bounds=(161.96252, -70.75924, 162.10388, -70.72293),
    in_bounds=True,
    is_error=False,
)

test_antarctic_bounds_with_cop30 = BoundsDEMCheckCase(
    dem_type="cop_30",
    bounds=(161.96252, -70.75924, 162.10388, -70.72293),
    in_bounds=True,
    is_error=False,
)

test_australia_bounds_with_rema = BoundsDEMCheckCase(
    dem_type="REMA_32",
    bounds=(133.0, -25.0, 134.0, -24.0),
    in_bounds=False,
    is_error=False,
)

test_heard_island_bounds_with_rema = BoundsDEMCheckCase(
    dem_type="rema_2",
    bounds=(73.2144, -53.224, 73.87, -52.962),
    in_bounds=False,
    is_error=False,
)

test_heard_island_bounds_with_cop30 = BoundsDEMCheckCase(
    dem_type="cop_glo30",
    bounds=(73.2144, -53.224, 73.87, -52.962),
    in_bounds=True,
    is_error=False,
)

test_antimeridian_with_cop30 = BoundsDEMCheckCase(
    dem_type="cop_glo30",
    bounds=(-178.032867, -80.618423, 173.430893, -68.765106),
    in_bounds=True,
    is_error=False,
)

test_antimeridian_with_rema = BoundsDEMCheckCase(
    dem_type="rema",
    bounds=(-178.032867, -80.618423, 173.430893, -68.765106),
    in_bounds=True,
    is_error=False,
)

test_cases = [
    test_invalid_dem_type,
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
            check_dem_type_in_bounds(test_case.dem_type, test_case.bounds)

    else:
        assert (
            check_dem_type_in_bounds(test_case.dem_type, test_case.bounds)
            == test_case.in_bounds
        )
