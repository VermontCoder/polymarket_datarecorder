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

_COLUMNS = [
    "timestamp", "up_bid", "up_ask", "down_bid", "down_ask",
    "price_to_beat", "current_price", "diff_pct", "diff_usd", "time_to_close",
]


def parse_row(line: str) -> dict:
    """Parse a single TSV data line into a dict. Empty fields become None."""
    parts = line.rstrip("\n").split("\t")
    row = {}
    for key, raw in zip(_COLUMNS, parts):
        if raw == "":
            row[key] = None
        elif key == "timestamp":
            row[key] = raw
        elif key == "time_to_close":
            row[key] = int(float(raw))
        else:
            row[key] = float(raw)
    return row


def get_window_key(timestamp: str) -> tuple:
    """Return the 5-minute window key (y, m, d, h, floored_minute) for a UTC timestamp."""
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return (dt.year, dt.month, dt.day, dt.hour, (dt.minute // 5) * 5)


def segment_rows(rows: list[dict]) -> list[list[dict]]:
    """Group a flat list of parsed rows into 5-minute segments by clock boundary."""
    if not rows:
        return []
    segments = []
    current = [rows[0]]
    for row in rows[1:]:
        if get_window_key(row["timestamp"]) != get_window_key(current[-1]["timestamp"]):
            segments.append(current)
            current = [row]
        else:
            current.append(row)
    segments.append(current)
    return segments


def filter_segments(segments: list[list[dict]]) -> tuple[list[list[dict]], int]:
    """
    Discard segments with no near-close row.
    Returns (kept_segments, dropped_count).
    """
    kept = []
    dropped = 0
    for seg in segments:
        if any(r["time_to_close"] is not None and r["time_to_close"] < CLOSE_THRESHOLD_MS
               for r in seg):
            kept.append(seg)
        else:
            dropped += 1
    return kept, dropped


def annotate_segment(rows: list[dict]) -> dict:
    """Compute episode metadata for a validated segment."""
    first, last = rows[0], rows[-1]
    dt = datetime.fromisoformat(first["timestamp"].replace("Z", "+00:00"))
    outcome = "UP" if last["current_price"] >= last["price_to_beat"] else "DOWN"
    return {
        "outcome":     outcome,
        "hour":        dt.hour,
        "day":         dt.weekday(),   # 0=Monday, 6=Sunday
        "start_price": last["price_to_beat"],
        "end_price":   last["current_price"],
        "rows":        rows,
    }


def format_episode(episode: dict) -> str:
    """
    Serialize one episode to a string.
    Opening line: JSON object with metadata and 'rows':[
    Middle lines: one row JSON object per line (comma-separated)
    Closing line: ]}
    No blank lines within the block.
    """
    rows = episode["rows"]
    meta = {k: v for k, v in episode.items() if k != "rows"}
    # Build opening line: {"outcome":"UP","hour":17,...,"rows":[
    opening = json.dumps(meta, separators=(",", ":"))[:-1] + ',"rows":['
    row_lines = []
    for i, row in enumerate(rows):
        suffix = "," if i < len(rows) - 1 else ""
        row_lines.append(json.dumps(row, separators=(",", ":")) + suffix)
    closing = "]}"
    return "\n".join([opening] + row_lines + [closing])


def format_output(episodes: list[dict]) -> str:
    """Serialize all episodes separated by a single blank line."""
    return "\n\n".join(format_episode(ep) for ep in episodes)


def collect_files(tsv_dir: str = TSV_DIR) -> list[str]:
    """Return sorted list of TSV file paths. Exits with error if none found."""
    pattern = f"{tsv_dir}/{TSV_PATTERN}"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Error: no files matching '{pattern}' found.", file=sys.stderr)
        sys.exit(1)
    return files


def read_file_rows(filepath: str) -> tuple[list[dict], str, str]:
    """
    Parse all data rows from a TSV file.
    Returns (rows, first_timestamp, last_timestamp).
    Skips header lines (lines starting with 'Timestamp').
    """
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Timestamp"):
                continue
            rows.append(parse_row(line))
    first_ts = rows[0]["timestamp"] if rows else ""
    last_ts  = rows[-1]["timestamp"] if rows else ""
    return rows, first_ts, last_ts


def format_duration(first_ts: str, last_ts: str) -> str:
    """Return elapsed time between two ISO 8601 UTC timestamps as 'Xh YYm'."""
    dt1 = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
    dt2 = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    total_minutes = int((dt2 - dt1).total_seconds() // 60)
    return f"{total_minutes // 60}h {total_minutes % 60:02d}m"


def print_stats(files: list[str], file_meta: list[dict],
                episodes_written: int, dropped: int, elapsed: float) -> None:
    """
    Print processing summary to stdout.

    file_meta is a list of dicts, one per file:
        {"name": str, "first_ts": str, "last_ts": str, "episode_count": int}
    """
    bar = "─" * 65
    print(bar)
    print(f"Combined: {OUTPUT_FILE}")
    print(bar)
    print(f"Episodes written:  {episodes_written:>6}")
    print(f"Dropped (no close):{dropped:>6}")
    print(f"Processing time:   {elapsed:.1f}s")
    print()
    print("Date ranges by source file:")
    for m in file_meta:
        duration = format_duration(m["first_ts"], m["last_ts"])
        name = os.path.basename(m["name"])
        print(f"  {name:<42} {m['first_ts']} → {m['last_ts']}   {duration:>7}   {m['episode_count']:>3} episodes")
    print(bar)
