import pytest
from dataclasses import dataclass
from shapely.geometry import Polygon, MultiPolygon
from dem_handler.utils.spatial import (
    BoundingBox,
    check_bounds_likely_cross_antimeridian,
    check_dem_type_in_bounds,
    resize_bounds,
)


@dataclass
class TestValidBounds:
    bounds: BoundingBox
    lat_tol: float
    lon_tol: float
    is_valid: int


# Test cases for bounding box validity
TEST_VALID_BOUNDS = [
    # Fully valid within standard ranges
    TestValidBounds(
        BoundingBox(-170, 0, -165, 5), lat_tol=0.0, lon_tol=0.0, is_valid=True
    ),
    TestValidBounds(
        BoundingBox(0, 10, 160, 50), lat_tol=0.0, lon_tol=0.0, is_valid=True
    ),
    # Crosses antimeridian but still valid logically
    TestValidBounds(
        BoundingBox(170, -70, -170, -65), lat_tol=0.0, lon_tol=0.0, is_valid=True
    ),
    # Longitude slightly out of range, invalid without tolerance
    TestValidBounds(
        BoundingBox(-181, 0, -170, 15), lat_tol=0.0, lon_tol=0.0, is_valid=False
    ),
    # Latitude slightly out of range, invalid without tolerance
    TestValidBounds(
        BoundingBox(0, 0, 30, 92), lat_tol=0.0, lon_tol=0.0, is_valid=False
    ),
    # Becomes valid with longitude tolerance
    TestValidBounds(
        BoundingBox(-181, 0, -170, 15), lat_tol=0.0, lon_tol=2.0, is_valid=True
    ),
    # ✅ Becomes valid with latitude tolerance
    TestValidBounds(BoundingBox(0, 0, 30, 92), lat_tol=3.0, lon_tol=0.0, is_valid=True),
]


@pytest.mark.parametrize("case", TEST_VALID_BOUNDS)
def test_check_valid_bounding_box(case: TestValidBounds):
    """Test BoundingBox validity logic with and without tolerance."""
    result = case.bounds._check_valid(lat_tol=case.lat_tol, lon_tol=case.lon_tol)
    assert result == case.is_valid, (
        f"Expected {case.is_valid} for bounds {case.bounds} "
        f"with lat_tol={case.lat_tol}, lon_tol={case.lon_tol}, got {result}"
    )


@dataclass
class TestResizeBounds:
    bounds: BoundingBox
    scale_factor: BoundingBox
    resized_bounds: BoundingBox


TEST_RESIZE_BOUNDS = [
    # normal bounds
    TestResizeBounds(
        BoundingBox(-170, 0, -165, 5), 2, BoundingBox(-172.5, -2.5, -162.5, 7.5)
    ),
    TestResizeBounds(
        BoundingBox(-170, 0, -166, 6),
        0.5,
        BoundingBox(-169, 1.5, -167, 4.5),
    ),
    # antimeridian bounds
    TestResizeBounds(
        BoundingBox(178, -54, -178, -50),
        0.5,
        BoundingBox(179, -53, -179, -51),
    ),
    # scale by too much, ensure set to max allowable extents
    TestResizeBounds(
        BoundingBox(-170, 0, -165, 5), 8, BoundingBox(-180, -17.5, -147.5, 22.5)
    ),
]


@pytest.mark.parametrize("case", TEST_RESIZE_BOUNDS)
def test_resize_bounds(case: TestResizeBounds):
    resized_bounds = resize_bounds(case.bounds, scale_factor=case.scale_factor)
    assert resized_bounds == case.resized_bounds


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


@pytest.mark.parametrize("shape", shapes)
def test_get_correct_bounds_from_shape_at_antimeridian(shape):
    # assert the incorrect bounds still cross the AM
    assert check_bounds_likely_cross_antimeridian(shape.bounds)


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
    bounds=(173.430893, -80.618423, -178.032867, -68.765106),
    in_bounds=True,
    is_error=False,
)

test_antimeridian_with_rema = BoundsDEMCheckCase(
    dem_type="rema",
    resolution=2,
    bounds=(173.430893, -80.618423, -178.032867, -68.765106),
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
