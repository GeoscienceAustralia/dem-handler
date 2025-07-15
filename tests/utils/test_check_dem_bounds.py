import pytest
from dem_handler.utils.spatial import check_dem_type_in_bounds
from dataclasses import dataclass


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


test_cases = [
    test_invalid_dem_type,
    test_antarctic_bounds_with_rema,
    test_antarctic_bounds_with_cop30,
    test_australia_bounds_with_rema,
    test_heard_island_bounds_with_rema,
    test_heard_island_bounds_with_cop30,
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
