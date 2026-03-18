"""
test_combine_tsv.py
-------------------
Unit and integration tests for combine_tsv.py.
Run with: python -m pytest test_combine_tsv.py -v
  or:     python -m unittest test_combine_tsv -v
"""

import unittest
import os
import json
from datetime import datetime, timezone

import combine_tsv


class TestParseRow(unittest.TestCase):

    def _make_row(self, **overrides):
        """Return a TSV line with sensible defaults."""
        defaults = {
            "timestamp":     "2026-03-14T17:23:01.761Z",
            "up_bid":        "55.00",
            "up_ask":        "56.00",
            "down_bid":      "44.00",
            "down_ask":      "45.00",
            "price_to_beat": "70679.78",
            "current_price": "70685.94",
            "diff_pct":      "0.008715",
            "diff_usd":      "6.16",
            "time_to_close": "119733",
        }
        defaults.update(overrides)
        return "\t".join(defaults.values())

    def test_parse_row_basic(self):
        row = self._make_row()
        result = combine_tsv.parse_row(row)
        self.assertEqual(result["timestamp"], "2026-03-14T17:23:01.761Z")
        self.assertAlmostEqual(result["up_bid"], 55.0)
        self.assertAlmostEqual(result["price_to_beat"], 70679.78)
        self.assertAlmostEqual(result["current_price"], 70685.94)
        self.assertEqual(result["time_to_close"], 119733)  # int

    def test_parse_row_empty_field_is_none(self):
        # Simulate a row where up_ask is blank (market nearly resolved)
        row = "2026-03-14T17:24:59.792Z\t99.00\t\t1.00\t70679.78\t70694.50\t0.020826\t14.72\t1592"
        result = combine_tsv.parse_row(row)
        self.assertIsNone(result["up_ask"])

    def test_parse_row_time_to_close_truncated_to_int(self):
        row = self._make_row(time_to_close="119733.9")
        result = combine_tsv.parse_row(row)
        self.assertIsInstance(result["time_to_close"], int)
        self.assertEqual(result["time_to_close"], 119733)


class TestGetWindowKey(unittest.TestCase):

    def test_window_key_rounds_minute_down(self):
        # 17:23 → window starts at 17:20
        key = combine_tsv.get_window_key("2026-03-14T17:23:01.761Z")
        self.assertEqual(key, (2026, 3, 14, 17, 20))

    def test_window_key_at_boundary(self):
        # 17:25:00 → new window starting at 17:25
        key = combine_tsv.get_window_key("2026-03-14T17:25:01.837Z")
        self.assertEqual(key, (2026, 3, 14, 17, 25))

    def test_window_key_hour_rollover(self):
        # 18:00:xx → window starts at 18:00
        key = combine_tsv.get_window_key("2026-03-14T18:00:05.000Z")
        self.assertEqual(key, (2026, 3, 14, 18, 0))

    def test_different_windows_give_different_keys(self):
        key1 = combine_tsv.get_window_key("2026-03-14T17:24:59.792Z")
        key2 = combine_tsv.get_window_key("2026-03-14T17:25:01.837Z")
        self.assertNotEqual(key1, key2)

    def test_same_window_gives_same_key(self):
        key1 = combine_tsv.get_window_key("2026-03-14T17:23:01.761Z")
        key2 = combine_tsv.get_window_key("2026-03-14T17:24:59.792Z")
        self.assertEqual(key1, key2)


def _make_parsed_row(timestamp, price_to_beat=70000.0, current_price=70010.0, time_to_close=100000):
    """Helper: return a minimal parsed row dict."""
    return {
        "timestamp": timestamp,
        "up_bid": 55.0, "up_ask": 56.0,
        "down_bid": 44.0, "down_ask": 45.0,
        "price_to_beat": price_to_beat,
        "current_price": current_price,
        "diff_pct": 0.01, "diff_usd": 10.0,
        "time_to_close": time_to_close,
    }


class TestSegmentRows(unittest.TestCase):

    def test_single_segment(self):
        rows = [
            _make_parsed_row("2026-03-14T17:23:01Z"),
            _make_parsed_row("2026-03-14T17:23:03Z"),
            _make_parsed_row("2026-03-14T17:24:59Z"),
        ]
        segments = combine_tsv.segment_rows(rows)
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]), 3)

    def test_two_segments_at_clock_boundary(self):
        rows = [
            _make_parsed_row("2026-03-14T17:24:59Z"),  # last row of 17:20 window
            _make_parsed_row("2026-03-14T17:25:01Z"),  # first row of 17:25 window
        ]
        segments = combine_tsv.segment_rows(rows)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0][0]["timestamp"], "2026-03-14T17:24:59Z")
        self.assertEqual(segments[1][0]["timestamp"], "2026-03-14T17:25:01Z")

    def test_boundary_row_belongs_to_new_segment(self):
        rows = [
            _make_parsed_row("2026-03-14T17:24:55Z"),
            _make_parsed_row("2026-03-14T17:24:59Z"),  # last of old window
            _make_parsed_row("2026-03-14T17:25:01Z"),  # first of new window
            _make_parsed_row("2026-03-14T17:25:03Z"),
        ]
        segments = combine_tsv.segment_rows(rows)
        self.assertEqual(len(segments), 2)
        self.assertEqual(len(segments[0]), 2)  # rows at :55 and :59
        self.assertEqual(len(segments[1]), 2)  # rows at :01 and :03

    def test_same_price_to_beat_across_boundary_still_splits(self):
        # Even if price_to_beat is identical, the clock boundary splits segments
        rows = [
            _make_parsed_row("2026-03-14T17:24:59Z", price_to_beat=70000.0),
            _make_parsed_row("2026-03-14T17:25:01Z", price_to_beat=70000.0),
        ]
        segments = combine_tsv.segment_rows(rows)
        self.assertEqual(len(segments), 2)

    def test_empty_input(self):
        self.assertEqual(combine_tsv.segment_rows([]), [])


class TestReassignStrayCloseRows(unittest.TestCase):

    def test_stray_close_row_moved_to_prev_segment(self):
        # First row of seg2 has time_to_close=200ms → belongs to seg1
        seg1 = [_make_parsed_row("2026-03-14T17:24:55Z", time_to_close=8000)]
        seg2 = [
            _make_parsed_row("2026-03-14T17:25:00Z", time_to_close=200),   # stray close
            _make_parsed_row("2026-03-14T17:25:03Z", time_to_close=297000),
        ]
        result = combine_tsv.reassign_stray_close_rows([seg1, seg2])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)   # stray row moved here
        self.assertEqual(result[0][-1]["timestamp"], "2026-03-14T17:25:00Z")
        self.assertEqual(len(result[1]), 1)   # remaining row stays

    def test_normal_segments_unchanged(self):
        seg1 = [_make_parsed_row("2026-03-14T17:24:55Z", time_to_close=8000)]
        seg2 = [_make_parsed_row("2026-03-14T17:25:03Z", time_to_close=297000)]
        result = combine_tsv.reassign_stray_close_rows([seg1, seg2])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 1)

    def test_stray_row_is_only_row_segment_removed(self):
        # seg2 consists solely of a stray close row → segment is dropped entirely
        seg1 = [_make_parsed_row("2026-03-14T17:24:55Z", time_to_close=8000)]
        seg2 = [_make_parsed_row("2026-03-14T17:25:00Z", time_to_close=200)]
        seg3 = [_make_parsed_row("2026-03-14T17:30:03Z", time_to_close=297000)]
        result = combine_tsv.reassign_stray_close_rows([seg1, seg2, seg3])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][-1]["timestamp"], "2026-03-14T17:25:00Z")
        self.assertEqual(result[1][0]["timestamp"], "2026-03-14T17:30:03Z")

    def test_first_segment_stray_row_deleted(self):
        # No previous segment to receive it — row is discarded
        seg1 = [
            _make_parsed_row("2026-03-14T17:25:00Z", time_to_close=200),
            _make_parsed_row("2026-03-14T17:25:03Z", time_to_close=297000),
        ]
        seg2 = [_make_parsed_row("2026-03-14T17:30:03Z", time_to_close=297000)]
        result = combine_tsv.reassign_stray_close_rows([seg1, seg2])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0]["timestamp"], "2026-03-14T17:25:03Z")

    def test_first_segment_only_stray_row_segment_removed(self):
        # Stray row is the only row in the first segment → segment dropped entirely
        seg1 = [_make_parsed_row("2026-03-14T17:25:00Z", time_to_close=200)]
        seg2 = [_make_parsed_row("2026-03-14T17:30:03Z", time_to_close=297000)]
        result = combine_tsv.reassign_stray_close_rows([seg1, seg2])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0]["timestamp"], "2026-03-14T17:30:03Z")

    def test_single_segment_stray_row_deleted(self):
        # Single segment with a stray row — row is discarded, segment removed
        seg1 = [_make_parsed_row("2026-03-14T17:25:00Z", time_to_close=200)]
        result = combine_tsv.reassign_stray_close_rows([seg1])
        self.assertEqual(len(result), 0)

    def test_none_time_to_close_not_treated_as_stray(self):
        seg1 = [_make_parsed_row("2026-03-14T17:24:55Z", time_to_close=8000)]
        seg2 = [
            _make_parsed_row("2026-03-14T17:25:00Z", time_to_close=None),
            _make_parsed_row("2026-03-14T17:25:03Z", time_to_close=297000),
        ]
        # Patch None into the row
        seg2[0]["time_to_close"] = None
        result = combine_tsv.reassign_stray_close_rows([seg1, seg2])
        self.assertEqual(len(result[0]), 1)  # nothing moved
        self.assertEqual(len(result[1]), 2)


class TestFilterSegments(unittest.TestCase):

    def test_drops_segment_with_no_close_row(self):
        segment = [
            _make_parsed_row("2026-03-14T17:21:00Z", time_to_close=200000),
            _make_parsed_row("2026-03-14T17:22:00Z", time_to_close=150000),
        ]
        kept, dropped = combine_tsv.filter_segments([segment])
        self.assertEqual(len(kept), 0)
        self.assertEqual(dropped, 1)

    def test_keeps_segment_with_close_row(self):
        segment = [
            _make_parsed_row("2026-03-14T17:21:00Z", time_to_close=200000),
            _make_parsed_row("2026-03-14T17:24:59Z", time_to_close=8000),
        ]
        kept, dropped = combine_tsv.filter_segments([segment])
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 0)

    def test_keeps_segment_with_close_row_at_boundary(self):
        # Exactly at threshold: 14999 is kept, 15000 is dropped
        segment_kept = [_make_parsed_row("2026-03-14T17:24:59Z", time_to_close=14999)]
        segment_drop = [_make_parsed_row("2026-03-14T17:24:59Z", time_to_close=15000)]
        kept, dropped = combine_tsv.filter_segments([segment_kept, segment_drop])
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 1)

    def test_partial_start_segment_kept_if_close_confirmed(self):
        # Segment starts mid-period (only a few rows) but reaches the close
        segment = [
            _make_parsed_row("2026-03-14T17:23:00Z", time_to_close=120000),
            _make_parsed_row("2026-03-14T17:24:58Z", time_to_close=2000),
        ]
        kept, dropped = combine_tsv.filter_segments([segment])
        self.assertEqual(len(kept), 1)


class TestAnnotateSegment(unittest.TestCase):

    def test_outcome_up_when_current_greater(self):
        rows = [_make_parsed_row("2026-03-14T17:24:59Z",
                                  price_to_beat=70000.0, current_price=70010.0,
                                  time_to_close=500)]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["outcome"], "UP")

    def test_outcome_up_on_exact_tie(self):
        rows = [_make_parsed_row("2026-03-14T17:24:59Z",
                                  price_to_beat=70000.0, current_price=70000.0,
                                  time_to_close=500)]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["outcome"], "UP")

    def test_outcome_down_when_current_less(self):
        rows = [_make_parsed_row("2026-03-14T17:24:59Z",
                                  price_to_beat=70000.0, current_price=69990.0,
                                  time_to_close=500)]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["outcome"], "DOWN")

    def test_start_price_is_price_to_beat(self):
        rows = [_make_parsed_row("2026-03-14T17:24:59Z", price_to_beat=70679.78)]
        episode = combine_tsv.annotate_segment(rows)
        self.assertAlmostEqual(episode["start_price"], 70679.78)

    def test_end_price_is_current_price_of_last_row(self):
        # Two rows with DIFFERENT current_price — ensures end_price is from last, not first
        rows = [
            _make_parsed_row("2026-03-14T17:23:00Z", current_price=70685.94),
            _make_parsed_row("2026-03-14T17:24:59Z", current_price=70694.50),
        ]
        episode = combine_tsv.annotate_segment(rows)
        self.assertAlmostEqual(episode["end_price"], 70694.50)   # last row, not 70685.94

    def test_hour_annotation(self):
        rows = [_make_parsed_row("2026-03-14T17:23:01Z")]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["hour"], 17)

    def test_day_annotation_saturday(self):
        # 2026-03-14 is a Saturday → weekday() == 5
        rows = [_make_parsed_row("2026-03-14T17:23:01Z")]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["day"], 5)

    def test_hour_and_day_from_first_row_not_last(self):
        # Segment spanning midnight: first row at 23:59 (hour=23, day=5 Saturday),
        # last row at 00:00 next day (would be hour=0, day=6 Sunday).
        # hour and day must come from the FIRST row.
        rows = [
            _make_parsed_row("2026-03-14T23:59:01Z"),   # Saturday 23:59
            _make_parsed_row("2026-03-14T23:59:57Z"),   # still Saturday (same window)
        ]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["hour"], 23)
        self.assertEqual(episode["day"], 5)   # Saturday, not Sunday

    def test_rows_are_preserved_in_order(self):
        rows = [
            _make_parsed_row("2026-03-14T17:23:01Z"),
            _make_parsed_row("2026-03-14T17:23:03Z"),
        ]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(len(episode["rows"]), 2)
        self.assertEqual(episode["rows"][0]["timestamp"], "2026-03-14T17:23:01Z")
        self.assertEqual(episode["rows"][1]["timestamp"], "2026-03-14T17:23:03Z")

    def test_session_id_is_window_start(self):
        # 17:23 UTC → window starts at 17:20 → session_id = "2026-03-14T17:20:00Z"
        rows = [_make_parsed_row("2026-03-14T17:23:01Z")]
        episode = combine_tsv.annotate_segment(rows)
        self.assertEqual(episode["session_id"], "2026-03-14T17:20:00Z")


class TestAnnotateCrossEpisode(unittest.TestCase):

    def _make_episode(self, session_id, start_price=80000.0, diff_pct_last_row=0.01):
        return {
            "session_id":  session_id,
            "outcome":     "UP",
            "hour":        17,
            "day":         0,
            "start_price": start_price,
            "end_price":   start_price + 10,
            "rows": [
                _make_parsed_row("2026-03-14T17:23:01Z"),
                {**_make_parsed_row("2026-03-14T17:24:59Z", time_to_close=1592),
                 "diff_pct": diff_pct_last_row},
            ],
        }

    def test_first_episode_prev_session_is_none(self):
        episodes = [self._make_episode("2026-03-14T17:20:00Z")]
        result = combine_tsv.annotate_cross_episode(episodes)
        self.assertIsNone(result[0]["diff_pct_prev_session"])

    def test_prev_session_diff_pct_from_last_row(self):
        episodes = [
            self._make_episode("2026-03-14T17:20:00Z", diff_pct_last_row=0.0123),
            self._make_episode("2026-03-14T17:25:00Z"),
        ]
        result = combine_tsv.annotate_cross_episode(episodes)
        self.assertAlmostEqual(result[1]["diff_pct_prev_session"], 0.0123)

    def test_diff_pct_hour_none_when_no_prior_session(self):
        episodes = [self._make_episode("2026-03-14T17:20:00Z", start_price=80000.0)]
        result = combine_tsv.annotate_cross_episode(episodes)
        self.assertIsNone(result[0]["diff_pct_hour"])

    def test_diff_pct_hour_computed_when_prior_session_exists(self):
        episodes = [
            self._make_episode("2026-03-14T16:20:00Z", start_price=80000.0),
            self._make_episode("2026-03-14T17:20:00Z", start_price=80800.0),
        ]
        result = combine_tsv.annotate_cross_episode(episodes)
        self.assertAlmostEqual(result[1]["diff_pct_hour"], 1.0)  # (80800-80000)/80000 * 100

    def test_diff_pct_hour_none_when_gap_is_not_exactly_one_hour(self):
        # 17:25 is 1h05m after 16:20 — not exactly 1 hour
        episodes = [
            self._make_episode("2026-03-14T16:20:00Z", start_price=80000.0),
            self._make_episode("2026-03-14T17:25:00Z", start_price=80800.0),
        ]
        result = combine_tsv.annotate_cross_episode(episodes)
        self.assertIsNone(result[1]["diff_pct_hour"])

    def test_avg_pct_variance_hour_none_when_no_prior_slots_exist(self):
        episodes = [self._make_episode("2026-03-14T17:20:00Z")]
        result = combine_tsv.annotate_cross_episode(episodes)
        self.assertIsNone(result[0]["avg_pct_variance_hour"])

    def test_avg_pct_variance_hour_uses_absolute_value(self):
        # Prior slot at T-5m has negative diff_pct — should be treated as positive
        episodes = [
            self._make_episode("2026-03-14T17:00:00Z", diff_pct_last_row=-0.05),
            self._make_episode("2026-03-14T17:05:00Z"),
        ]
        result = combine_tsv.annotate_cross_episode(episodes)
        self.assertAlmostEqual(result[1]["avg_pct_variance_hour"], 0.05)

    def test_avg_pct_variance_hour_averages_present_prior_slots(self):
        # Episode at 17:10 has two prior slots: 17:05 (diff=-0.04) and 17:00 (diff=0.02)
        # avg = mean(0.04, 0.02) = 0.03
        episodes = [
            self._make_episode("2026-03-14T17:00:00Z", diff_pct_last_row=0.02),
            self._make_episode("2026-03-14T17:05:00Z", diff_pct_last_row=-0.04),
            self._make_episode("2026-03-14T17:10:00Z"),
        ]
        result = combine_tsv.annotate_cross_episode(episodes)
        self.assertAlmostEqual(result[2]["avg_pct_variance_hour"], 0.03)

    def test_avg_pct_variance_hour_is_rolling(self):
        # Two episodes in the same clock hour must get different values as the window shifts
        episodes = [
            self._make_episode("2026-03-14T17:00:00Z", diff_pct_last_row=0.02),
            self._make_episode("2026-03-14T17:05:00Z", diff_pct_last_row=-0.04),
            self._make_episode("2026-03-14T17:10:00Z"),
        ]
        result = combine_tsv.annotate_cross_episode(episodes)
        # 17:05 sees only 17:00 (T-5m) → 0.02
        self.assertAlmostEqual(result[1]["avg_pct_variance_hour"], 0.02)
        # 17:10 sees 17:05 (T-5m) and 17:00 (T-10m) → mean(0.04, 0.02) = 0.03
        self.assertAlmostEqual(result[2]["avg_pct_variance_hour"], 0.03)

    def test_avg_pct_variance_hour_excludes_slots_beyond_60_minutes(self):
        # Episode at 17:05 has a prior episode at 16:00 (65 min back — outside window)
        episodes = [
            self._make_episode("2026-03-14T16:00:00Z", diff_pct_last_row=9999.0),
            self._make_episode("2026-03-14T17:05:00Z"),
        ]
        result = combine_tsv.annotate_cross_episode(episodes)
        self.assertIsNone(result[1]["avg_pct_variance_hour"])


class TestWriteOutput(unittest.TestCase):

    def _make_episode(self, outcome="UP", hour=17, day=5,
                      session_id="2026-03-14T17:20:00Z"):
        return {
            "session_id":            session_id,
            "outcome":               outcome,
            "hour":                  hour,
            "day":                   day,
            "start_price":           70679.78,
            "end_price":             70694.50,
            "diff_pct_prev_session": None,
            "diff_pct_hour":         None,
            "avg_pct_variance_hour": None,
            "rows": [
                _make_parsed_row("2026-03-14T17:23:01Z"),
                _make_parsed_row("2026-03-14T17:24:59Z", time_to_close=1592),
            ],
        }

    def test_output_is_valid_json_array(self):
        episodes = [self._make_episode("UP",  session_id="2026-03-14T17:20:00Z"),
                    self._make_episode("DOWN", session_id="2026-03-14T17:25:00Z")]
        arr = json.loads(combine_tsv.format_output(episodes))
        self.assertIsInstance(arr, list)
        self.assertEqual(len(arr), 2)

    def test_session_id_present_in_episode(self):
        episodes = [self._make_episode(session_id="2026-03-14T17:20:00Z")]
        arr = json.loads(combine_tsv.format_output(episodes))
        self.assertEqual(arr[0]["session_id"], "2026-03-14T17:20:00Z")

    def test_outcome_and_metadata_in_output(self):
        episodes = [self._make_episode("DOWN", hour=9, day=0,
                                       session_id="2026-03-14T17:20:00Z")]
        arr = json.loads(combine_tsv.format_output(episodes))
        ep = arr[0]
        self.assertEqual(ep["outcome"], "DOWN")
        self.assertEqual(ep["hour"], 9)
        self.assertEqual(ep["day"], 0)
        self.assertAlmostEqual(ep["start_price"], 70679.78)

    def test_rows_present_in_output(self):
        episodes = [self._make_episode()]
        arr = json.loads(combine_tsv.format_output(episodes))
        self.assertEqual(len(arr[0]["rows"]), 2)
        self.assertEqual(arr[0]["rows"][0]["timestamp"], "2026-03-14T17:23:01Z")

    def test_price_to_beat_excluded_from_rows(self):
        episodes = [self._make_episode()]
        arr = json.loads(combine_tsv.format_output(episodes))
        for row in arr[0]["rows"]:
            self.assertNotIn("price_to_beat", row)


import tempfile


class TestCollectFiles(unittest.TestCase):

    def test_returns_sorted_list(self):
        with tempfile.TemporaryDirectory() as d:
            # Create two fake TSV files
            for name in ["btc_polymarket_20260315_120000.tsv",
                         "btc_polymarket_20260314_090000.tsv"]:
                open(os.path.join(d, name), "w").close()
            result = combine_tsv.collect_files(d)
        basenames = [os.path.basename(f) for f in result]
        self.assertEqual(basenames, sorted(basenames))

    def test_raises_on_no_files(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as ctx:
                combine_tsv.collect_files(d)
            self.assertNotEqual(ctx.exception.code, 0)


class TestReadFileRows(unittest.TestCase):

    def _write_tsv(self, path, lines):
        header = "Timestamp\tUp Bid\tUp Ask\tDown Bid\tDown Ask\tPrice to Beat\tCurrent Price\tDifference %\tDifference $\tTime to Close (ms)\n"
        with open(path, "w") as f:
            f.write(header)
            for line in lines:
                f.write(line + "\n")

    def test_skips_header_line(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as tmp:
            path = tmp.name
        try:
            self._write_tsv(path, [
                "2026-03-14T17:23:01Z\t55.00\t56.00\t44.00\t45.00\t70679.78\t70685.94\t0.008715\t6.16\t119733",
                "2026-03-14T17:23:03Z\t55.00\t56.00\t44.00\t45.00\t70679.78\t70685.94\t0.008715\t6.16\t117661",
            ])
            rows, first_ts, last_ts = combine_tsv.read_file_rows(path)
            self.assertEqual(len(rows), 2)           # header not included
            self.assertEqual(first_ts, "2026-03-14T17:23:01Z")
            self.assertEqual(last_ts, "2026-03-14T17:23:03Z")
        finally:
            os.unlink(path)

    def test_returns_parsed_dicts(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as tmp:
            path = tmp.name
        try:
            self._write_tsv(path, [
                "2026-03-14T17:23:01Z\t55.00\t56.00\t44.00\t45.00\t70679.78\t70685.94\t0.008715\t6.16\t119733",
            ])
            rows, _, _ = combine_tsv.read_file_rows(path)
            self.assertAlmostEqual(rows[0]["up_bid"], 55.0)
            self.assertEqual(rows[0]["time_to_close"], 119733)
        finally:
            os.unlink(path)


class TestFormatDuration(unittest.TestCase):

    def test_multi_hour_duration(self):
        result = combine_tsv.format_duration(
            "2026-03-14T17:00:00Z", "2026-03-14T23:36:00Z"
        )
        self.assertEqual(result, "6h 36m")

    def test_sub_hour_duration(self):
        result = combine_tsv.format_duration(
            "2026-03-14T17:00:00Z", "2026-03-14T17:42:00Z"
        )
        self.assertEqual(result, "0h 42m")

    def test_exactly_one_hour(self):
        result = combine_tsv.format_duration(
            "2026-03-14T17:00:00Z", "2026-03-14T18:00:00Z"
        )
        self.assertEqual(result, "1h 00m")

    def test_midnight_crossing(self):
        result = combine_tsv.format_duration(
            "2026-03-14T23:00:00Z", "2026-03-15T01:30:00Z"
        )
        self.assertEqual(result, "2h 30m")


import glob as _glob

def _find_latest_combined():
    files = sorted(_glob.glob("data/btc_polymarket_combined_*.json"))
    return files[-1] if files else None

COMBINED_FILE = _find_latest_combined()
TSV_DIR = "tsv"


@unittest.skipUnless(COMBINED_FILE is not None,
                     "combined file not present — run combine_tsv.py first")
class TestSpotChecks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(COMBINED_FILE) as f:
            cls.episodes = json.load(f)

    def test_spot_episode_1_first_window(self):
        """
        First episode is a partial-start window (17:20-17:25 UTC on 2026-03-14).
        Expected values are derived directly from the source TSV.

        Source TSV: btc_polymarket_20260314_132259.tsv
          First row in window: 2026-03-14T17:22:59.879Z  (price_to_beat=70679.78)
          Last row in window:  2026-03-14T17:24:59.792Z  (current_price=70694.50)
        """
        # Read expected values from the source TSV
        source_path = os.path.join(TSV_DIR, "btc_polymarket_20260314_132259.tsv")
        rows, _, _ = combine_tsv.read_file_rows(source_path)
        # Filter to the first window (17:20-17:25 key)
        first_key = combine_tsv.get_window_key(rows[0]["timestamp"])
        window_rows = [r for r in rows if combine_tsv.get_window_key(r["timestamp"]) == first_key]
        expected_start_price = window_rows[0]["price_to_beat"]
        expected_end_price   = window_rows[-1]["current_price"]
        expected_outcome     = "UP" if expected_end_price >= expected_start_price else "DOWN"
        expected_hour        = 17
        expected_day         = 5   # Saturday: datetime(2026,3,14).weekday() == 5

        ep = self.episodes[0]
        self.assertEqual(ep["outcome"], expected_outcome)
        self.assertEqual(ep["hour"], expected_hour)
        self.assertEqual(ep["day"], expected_day)
        self.assertAlmostEqual(ep["start_price"], expected_start_price, places=2)
        self.assertAlmostEqual(ep["end_price"], expected_end_price, places=2)

    def test_spot_episode_2_first_episode_from_file3(self):
        """
        First complete episode from btc_polymarket_20260315_210627.tsv.
        That file's data starts at 2026-03-16T01:06:xx UTC (Monday = day 0).
        Expected values derived from source TSV.
        """
        source_path = os.path.join(TSV_DIR, "btc_polymarket_20260315_210627.tsv")
        rows, _, _ = combine_tsv.read_file_rows(source_path)
        first_key = combine_tsv.get_window_key(rows[0]["timestamp"])
        window_rows = [r for r in rows if combine_tsv.get_window_key(r["timestamp"]) == first_key]

        # This window must be complete (has a near-close row) to appear in output
        has_close = any(r["time_to_close"] is not None and r["time_to_close"] < 15000
                        for r in window_rows)
        if not has_close:
            self.skipTest("First window of file 3 was incomplete — not in combined output")

        expected_start_price = window_rows[0]["price_to_beat"]
        expected_end_price   = window_rows[-1]["current_price"]
        expected_outcome     = "UP" if expected_end_price >= expected_start_price else "DOWN"

        # Find matching episode: first row must fall within file 3's timestamp range
        _, first_ts, last_ts = combine_tsv.read_file_rows(source_path)
        ep = next((e for e in self.episodes
                   if first_ts <= e["rows"][0]["timestamp"] <= last_ts), None)
        self.assertIsNotNone(ep, "No episode found from file 3")

        self.assertEqual(ep["hour"], 1)    # 01:xx UTC
        self.assertEqual(ep["day"], 0)     # Monday: datetime(2026,3,16).weekday() == 0
        self.assertEqual(ep["outcome"], expected_outcome)
        self.assertAlmostEqual(ep["start_price"], expected_start_price, places=2)
        self.assertAlmostEqual(ep["end_price"], expected_end_price, places=2)

    def test_spot_episode_3_last_episode(self):
        """
        Last episode in combined output comes from btc_polymarket_20260316_093828.tsv.
        Verifies end_price matches the last row's current_price in that episode,
        and that the episode's last row exists verbatim in the source TSV.
        """
        source_path = os.path.join(TSV_DIR, "btc_polymarket_20260316_093828.tsv")
        rows, _, _ = combine_tsv.read_file_rows(source_path)

        ep = self.episodes[-1]
        # end_price must be consistent within the episode itself
        self.assertAlmostEqual(ep["end_price"], ep["rows"][-1]["current_price"], places=4)

        # The episode's last row must exist verbatim in the source file
        last_ts = ep["rows"][-1]["timestamp"]
        source_row = next((r for r in rows if r["timestamp"] == last_ts), None)
        self.assertIsNotNone(source_row, f"Episode last row {last_ts} not found in source TSV")
        self.assertAlmostEqual(ep["end_price"], source_row["current_price"], places=4)
