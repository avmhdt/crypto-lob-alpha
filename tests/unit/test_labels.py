"""Labels: values, latency shifts and nulls across gaps (AC-W-13)."""

from __future__ import annotations

import numpy as np

from lob_alpha.labels import clock_labels, event_labels, stretch_end

MS = 1_000_000


def test_clock_labels_values_latency_and_gap_nulls() -> None:
    ts = np.array([0, 100, 200, 300, 400, 500], np.int64) * MS
    mid = np.array([10.0, 10.5, 11.0, 11.0, 12.0, 13.0])
    seg = np.zeros(6, np.int64)
    t = np.array([0, 100, 200], np.int64) * MS
    gaps = [(350 * MS, 360 * MS)]  # a gap starts at 350 ms
    lab = clock_labels(ts, mid, seg, t, [100, 200], [0, 100], gaps, 10_000 * MS)
    assert lab["dmid_ticks_time_100"].tolist() == [0.5, 0.5, 0.0]
    got = lab["dmid_ticks_time_200"]
    assert got[0] == 1.0 and got[1] == 0.5 and np.isnan(got[2])  # 200+200 crosses the gap at 350
    lat = lab["dmid_ticks_time_100_lat100"]
    assert lat[0] == 0.5 and lat[1] == 0.0 and np.isnan(lat[2])


def test_clock_labels_null_past_day_end_and_across_resets() -> None:
    ts = np.array([0, 100, 200, 300], np.int64) * MS
    mid = np.array([1.0, 2.0, 50.0, 51.0])
    seg = np.array([0, 0, 2, 2], np.int64)  # a book reset at index 2
    lab = clock_labels(
        ts, mid, seg, np.array([0, 200], np.int64) * MS, [100, 250], [0], [], 400 * MS
    )
    assert lab["dmid_ticks_time_100"].tolist() == [1.0, 1.0]
    assert np.isnan(lab["dmid_ticks_time_250"][0])  # lands in the next segment
    assert np.isnan(lab["dmid_ticks_time_250"][1])  # past the day end


def test_event_labels_stay_in_segment() -> None:
    mid = np.array([1.0, 2.0, 4.0, 7.0, 7.5])
    seg = np.array([0, 0, 0, 3, 3], np.int64)
    lab = event_labels(mid, seg, [1, 2])
    one = lab["dmid_ticks_event_1"]
    assert (
        one[:2].tolist() == [1.0, 2.0] and np.isnan(one[2]) and one[3] == 0.5 and np.isnan(one[4])
    )
    two = lab["dmid_ticks_event_2"]
    assert two[0] == 3.0 and np.isnan(two[1]) and np.isnan(two[3])


def test_stretch_end() -> None:
    t = np.array([0, 5, 20], np.int64)
    assert stretch_end(t, [(10, 15)], 100).tolist() == [10, 10, 100]
    assert stretch_end(t, [], 50).tolist() == [50, 50, 50]
