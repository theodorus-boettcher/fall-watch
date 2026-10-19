"""Tests for picking the pictures the hand filter drops."""

import pytest

pytest.importorskip("ultralytics")  # hand_filter needs the train dependency group

from door_counter.hand_filter import pictures_to_drop, touches_edge  # noqa: E402


def test_a_hand_drops_its_neighbours_too() -> None:
    has_hand = [False] * 10
    has_hand[0] = has_hand[6] = True
    assert pictures_to_drop(has_hand=has_hand, neighbours=2) == {0, 1, 2, 4, 5, 6, 7, 8}


def test_no_hands_drops_nothing() -> None:
    assert pictures_to_drop(has_hand=[False] * 5, neighbours=2) == set()


@pytest.mark.parametrize(("box", "expected"), [
    ((0.0, 100.0, 50.0, 200.0), True),  # left side
    ((300.0, 400.0, 350.0, 479.0), True),  # bottom
    ((200.0, 100.0, 400.0, 300.0), False),  # middle
])
def test_only_boxes_at_the_edge_count(box: tuple[float, float, float, float], expected: bool) -> None:
    assert touches_edge(box=box, width_px=640, height_px=480) is expected
