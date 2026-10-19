"""Tests for the home count scoring."""

import pytest

pytest.importorskip("ultralytics")  # eval_home needs the train dependency group

from door_counter import eval_home  # noqa: E402


def test_truth_fills_missing_materials_with_zero() -> None:
    assert eval_home.truth_vector(contents={"metal": 2, "cardboard": 1}) == (1, 0, 0, 2)
    with pytest.raises(ValueError):
        eval_home.truth_vector(contents={"glass": 1})


def test_scores_count_errors_per_material() -> None:
    truth = (1, 0, 0, 2)
    counted = [(1, 0, 0, 2), (1, 1, 0, 1)]  # right, then one bag too many and one can missing
    score = eval_home.score_counts(model="v1", scene="test-1", counted=counted, truth=truth)
    assert score.pictures == 2
    assert score.exact_pct == 50.0
    assert score.mean_abs_error == pytest.approx(2 / 8)
    assert score.mean_abs_error_per_material == (0.0, 0.5, 0.0, 0.5)
    assert score.mean_true_per_material == (1.0, 0.0, 0.0, 2.0)


def test_pooled_scenes_use_one_truth_row_per_picture() -> None:
    score = eval_home.score_counts(model="v1", scene="all", counted=[(1, 0, 0, 0), (0, 0, 0, 3)],
                                   truth=[(1, 0, 0, 0), (0, 0, 0, 2)])
    assert score.exact_pct == 50.0
    assert score.mean_true_per_material == (0.5, 0.0, 0.0, 1.0)
