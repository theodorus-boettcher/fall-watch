"""Tests for turning every source into our four classes."""

import pytest

from door_counter.mix import (DROP, METAL, RIGID, SOFT, class_map_from_names, clean_home_lines, drop_duplicates,
                              remap_lines, source_stem)


def square(*, cls: int, x1: float, y1: float, x2: float, y2: float) -> str:
    return f"{cls} {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"


def test_roboflow_suffix_is_stripped() -> None:
    assert source_stem(name="cardboard109_jpg.rf.27fa7c403165b9f609f6bd2fa926e53e.jpg") == "cardboard109"
    assert source_stem(name="1792445680002_00002842_jpg.rf.0a1b.jpg") == "1792445680002_00002842"


def test_universe_names_map_to_ours() -> None:
    assert class_map_from_names(names=["POLYSTYRENE", "bottle", "can"]) == {0: RIGID, 1: RIGID, 2: METAL}
    with pytest.raises(KeyError):
        class_map_from_names(names=["banana"])


def test_background_objects_go_and_mixed_pictures_are_left_out() -> None:
    lines = [square(cls=0, x1=0.1, y1=0.1, x2=0.2, y2=0.2), square(cls=1, x1=0.5, y1=0.5, x2=0.6, y2=0.6)]
    assert remap_lines(lines=lines, class_map={0: SOFT, 1: None}) == [f"{SOFT} " + lines[0].split(" ", 1)[1]]
    assert remap_lines(lines=lines, class_map={0: SOFT, 1: DROP}) is None


def test_home_cleaning_drops_the_support_and_duplicates() -> None:
    support = square(cls=METAL, x1=0.0, y1=0.0, x2=0.12, y2=1.0)
    can = square(cls=METAL, x1=0.4, y1=0.4, x2=0.6, y2=0.7)
    same_can = square(cls=METAL, x1=0.41, y1=0.41, x2=0.6, y2=0.69)
    other_can = square(cls=METAL, x1=0.7, y1=0.1, x2=0.9, y2=0.3)
    assert clean_home_lines(lines=[support, same_can, can, other_can]) == [can, other_can]


def test_a_lid_inside_its_can_is_the_same_item() -> None:
    can = square(cls=METAL, x1=0.3, y1=0.3, x2=0.7, y2=0.6)
    lid = square(cls=METAL, x1=0.55, y1=0.32, x2=0.68, y2=0.58)
    bottle_on_the_can = square(cls=RIGID, x1=0.55, y1=0.32, x2=0.68, y2=0.58)
    assert drop_duplicates(lines=[lid, can, bottle_on_the_can]) == [can, bottle_on_the_can]
