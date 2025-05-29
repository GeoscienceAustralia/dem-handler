import pytest
from shapely.geometry import Polygon
from dem_handler.utils.spatial import (
    BoundingBox,
    get_correct_bounds_from_shape_at_antimeridian,
    check_s1_bounds_cross_antimeridian,
)

# Separate lists for clarity
shapes = [
    Polygon(
        [
            (178.576126, -71.618423),
            (-178.032867, -70.167343),
            (176.938004, -68.765106),
            (173.430893, -70.119957),
            (178.576126, -71.618423),
        ],
    ),
    Polygon(
        [
            (179.5, -10.0),  # Just west of the antimeridian
            (-179.8, 5.0),  # Just east of the antimeridian
            (-178.5, 15.0),  # Further east
            (178.0, 12.0),  # West again
            (179.5, -10.0),  # Close the ring
        ]
    ),
    Polygon(
        [
            (170.0, -55.0),  # Western point
            (-175.0, -50.0),  # Just east of antimeridian
            (-160.0, -45.0),  # Farther east
            (175.0, -40.0),  # Back west
            (170.0, -55.0),  # Close the loop
        ]
    ),
]

expected_bounds = [
    BoundingBox(-178.032867, -71.618423, 173.430893, -68.765106),
    BoundingBox(-178.5, -10, 178.0, 15),
    BoundingBox(-160, -55, 170, -40),
]

# Combine for parameterization
test_cases = list(zip(shapes, expected_bounds))


@pytest.mark.parametrize("shape, expected_bound", test_cases)
def test_get_correct_bounds_from_shape_at_antimeridian(shape, expected_bound):
    # assert the incorrect bounds still cross the AM
    assert check_s1_bounds_cross_antimeridian(shape.bounds)
    result = get_correct_bounds_from_shape_at_antimeridian(shape)
    assert result == expected_bound
