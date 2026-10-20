"""Tests for scoring fall alerts on clips."""

from door_counter.eval_falls import ClipResult, parse_intervals, summarise


def test_parse_intervals_reads_gmdcsa_ranges() -> None:
    line = ("05.mp4,05, Day (Light On),Full T-shirt & Trousers, Walking to falling. Right side fall, "
            "Falling (SW)[1.8 to 5]:; Walking[0 to 1.8]")
    assert parse_intervals(text=line) == {"Falling": (1.8, 5.0), "Walking": (0.0, 1.8)}
    assert parse_intervals(text="Sitting[0 to 1]; Sleeping[2 to 8]")["Sleeping"] == (2.0, 8.0)


def test_summary_keeps_sleeping_out_of_false_alarms() -> None:
    def result(*, fall: bool, sleeping: bool = False, alerted: bool, delay: float | None = None) -> ClipResult:
        return ClipResult(model="m", clip="c", is_fall=fall, has_sleeping=sleeping, alerted=alerted,
                          seconds_to_alert=delay)

    s = summarise(model="m", results=[result(fall=True, alerted=True, delay=2.0), result(fall=True, alerted=False),
                                      result(fall=False, alerted=True), result(fall=False, alerted=False),
                                      result(fall=False, sleeping=True, alerted=True)])
    assert (s.falls, s.falls_caught, s.normal_clips, s.false_alarms) == (2, 1, 2, 1)
    assert (s.sleeping_clips, s.sleeping_alerts, s.mean_seconds_to_alert) == (1, 1, 2.0)
