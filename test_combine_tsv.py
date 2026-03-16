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


class TestWriteOutput(unittest.TestCase):

    def _make_episode(self, outcome="UP", hour=17, day=5):
        return {
            "outcome": outcome,
            "hour": hour,
            "day": day,
            "start_price": 70679.78,
            "end_price": 70694.50,
            "rows": [
                _make_parsed_row("2026-03-14T17:23:01Z"),
                _make_parsed_row("2026-03-14T17:24:59Z", time_to_close=1592),
            ],
        }

    def test_output_is_parseable_json_per_block(self):
        episodes = [self._make_episode("UP"), self._make_episode("DOWN")]
        text = combine_tsv.format_output(episodes)
        blocks = text.strip().split("\n\n")
        self.assertEqual(len(blocks), 2)
        parsed = json.loads(blocks[0])
        self.assertEqual(parsed["outcome"], "UP")

    def test_each_row_on_its_own_line(self):
        episodes = [self._make_episode()]
        text = combine_tsv.format_output(episodes)
        block = text.strip().split("\n\n")[0]
        lines = block.split("\n")
        # First line: opening brace with metadata and rows:[
        # Middle lines: one row dict each
        # Last line: closing ]}
        self.assertIn('"outcome"', lines[0])
        self.assertIn('"timestamp"', lines[1])  # first row
        self.assertIn('"timestamp"', lines[2])  # second row

    def test_no_blank_lines_within_episode(self):
        episodes = [self._make_episode()]
        text = combine_tsv.format_output(episodes)
        block = text.strip().split("\n\n")[0]
        self.assertNotIn("\n\n", block)

    def test_episodes_separated_by_exactly_one_blank_line(self):
        episodes = [self._make_episode(), self._make_episode(), self._make_episode()]
        text = combine_tsv.format_output(episodes)
        # Should be exactly 2 blank-line separators for 3 episodes
        self.assertEqual(text.count("\n\n"), 2)

    def test_outcome_and_metadata_in_output(self):
        episodes = [self._make_episode("DOWN", hour=9, day=0)]
        text = combine_tsv.format_output(episodes)
        parsed = json.loads(text.strip().split("\n\n")[0])
        self.assertEqual(parsed["outcome"], "DOWN")
        self.assertEqual(parsed["hour"], 9)
        self.assertEqual(parsed["day"], 0)
        self.assertAlmostEqual(parsed["start_price"], 70679.78)
