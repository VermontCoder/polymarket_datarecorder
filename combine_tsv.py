"""
combine_tsv.py
--------------
Merges all Polymarket BTC 5-minute TSV snapshot files into a single
JSON episode file for reinforcement learning training.

Usage:
    python combine_tsv.py
"""

import glob
import json
import os
import time
import sys
from datetime import datetime, timezone

TSV_DIR     = "tsv"
TSV_PATTERN = "btc_polymarket_*.tsv"
OUTPUT_FILE = "btc_polymarket_combined.json"
CLOSE_THRESHOLD_MS = 15_000  # segment must have a row within 15s of close
