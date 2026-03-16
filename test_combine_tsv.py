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
