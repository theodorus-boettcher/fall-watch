"""Tests for matching outlines against human ones."""

from door_counter.mix import METAL, RIGID
from door_counter.teacher_check import match_counts


def square(*, cls: int, x1: float, y1: float, x2: float, y2: float) -> str:
    return f"{cls} {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"


def test_each_human_outline_matches_once_and_only_the_same_material() -> None:
    can = square(cls=METAL, x1=0.1, y1=0.1, x2=0.4, y2=0.4)
    bottle = square(cls=RIGID, x1=0.6, y1=0.6, x2=0.9, y2=0.9)
    found = [can, square(cls=METAL, x1=0.11, y1=0.1, x2=0.4, y2=0.4),  # second outline of the same can
             square(cls=METAL, x1=0.6, y1=0.6, x2=0.9, y2=0.9)]  # right place, wrong material
    counts = match_counts(found=found, truth=[can, bottle])
    assert counts[METAL].tolist() == [1, 3, 1]
    assert counts[RIGID].tolist() == [0, 0, 1]
