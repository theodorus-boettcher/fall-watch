"""Tests for scoring fall alerts on clips."""

import pytest

pytest.importorskip("ultralytics", reason="needs the train group: uv run --group train pytest")
from door_counter.eval_falls import ClipResult, summarise  # noqa: E402


def test_summary_keeps_sleeping_out_of_false_alarms() -> None:
    def result(*, fall: bool, sleeping: bool = False, alerted: bool, delay: float | None = None) -> ClipResult:
        return ClipResult(model="m", clip="c", is_fall=fall, has_sleeping=sleeping, alerted=alerted,
                          seconds_to_alert=delay)

    results = [result(fall=True, alerted=True, delay=2.0), result(fall=True, alerted=False),
               result(fall=False, alerted=True), result(fall=False, alerted=False),
               result(fall=False, sleeping=True, alerted=True)]
    s = summarise(model="m", clip_set="x", results=results)
    assert (s.falls, s.falls_caught, s.normal_clips, s.false_alarms) == (2, 1, 2, 1)
    assert (s.sleeping_clips, s.sleeping_alerts, s.mean_seconds_to_alert) == (1, 1, 2.0)
