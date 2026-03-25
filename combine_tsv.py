"""
combine_tsv.py
--------------
Merges all Polymarket TSV snapshot files into per-market JSON episode files
for reinforcement learning training.

Supported markets:
  - BTC 15-minute  (tsv/btc-15/ → data/btc_15_combined_*.json)
  - ETH 15-minute  (tsv/eth-15/ → data/eth_15_combined_*.json)
  - ETH 5-minute   (tsv/eth-5/  → data/eth_5_combined_*.json)

Usage:
    python combine_tsv.py
"""

import glob
import json
import os
import time
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

TSV_DIR    = "tsv"
OUTPUT_DIR = "data"
CLOSE_THRESHOLD_MS       = 15_000  # segment must have a row within 15s of close
STRAY_CLOSE_THRESHOLD_MS =  5_000  # first-row time_to_close below this → belongs to prev segment

_COLUMNS = [
    "timestamp", "up_bid", "up_ask", "down_bid", "down_ask",
    "price_to_beat", "current_price", "diff_pct", "diff_usd", "time_to_close",
]


@dataclass
class MarketConfig:
    name:          str   # display label, e.g. "BTC-15m"
    tsv_subdir:    str   # subdirectory under TSV_DIR, e.g. "btc-15"
    window_min:    int   # market window in minutes: 5 or 15
    output_prefix: str   # output filename prefix, e.g. "btc_15"


MARKETS = [
    MarketConfig(name="BTC-15m", tsv_subdir="btc-15", window_min=15, output_prefix="btc_15"),
    MarketConfig(name="ETH-15m", tsv_subdir="eth-15", window_min=15, output_prefix="eth_15"),
    MarketConfig(name="ETH-5m",  tsv_subdir="eth-5",  window_min=5,  output_prefix="eth_5"),
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


def get_window_key(timestamp: str, window_min: int = 5) -> tuple:
    """Return the window key (y, m, d, h, floored_minute) for a UTC timestamp."""
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return (dt.year, dt.month, dt.day, dt.hour, (dt.minute // window_min) * window_min)


def segment_rows(rows: list[dict], window_min: int = 5) -> list[list[dict]]:
    """Group a flat list of parsed rows into segments by clock boundary."""
    if not rows:
        return []
    segments = []
    current = [rows[0]]
    for row in rows[1:]:
        if get_window_key(row["timestamp"], window_min) != get_window_key(current[-1]["timestamp"], window_min):
            segments.append(current)
            current = [row]
        else:
            current.append(row)
    segments.append(current)
    return segments


def reassign_stray_close_rows(segments: list[list[dict]]) -> list[list[dict]]:
    """
    Fix misassigned closing rows caused by clock-boundary misalignment.

    If the first row of a segment has a time_to_close below STRAY_CLOSE_THRESHOLD_MS
    it cannot be the opening row of a new market window; it belongs to the previous
    segment's close. Move it there. If the segment becomes empty after the move,
    remove it entirely.
    """
    if not segments:
        return segments
    result = [list(seg) for seg in segments]
    i = 0
    while i < len(result):
        first = result[i][0]
        if first["time_to_close"] is not None and first["time_to_close"] < STRAY_CLOSE_THRESHOLD_MS:
            if i == 0:
                result[i].pop(0)          # no previous segment — discard
            else:
                result[i - 1].append(result[i].pop(0))
            if not result[i]:
                result.pop(i)
                continue
        i += 1
    return result


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


def annotate_segment(rows: list[dict], window_min: int = 5) -> dict:
    """Compute episode metadata for a validated segment."""
    first, last = rows[0], rows[-1]
    dt = datetime.fromisoformat(first["timestamp"].replace("Z", "+00:00"))
    key = get_window_key(first["timestamp"], window_min)
    session_id = f"{key[0]:04d}-{key[1]:02d}-{key[2]:02d}T{key[3]:02d}:{key[4]:02d}:00Z"
    outcome = "UP" if last["current_price"] >= last["price_to_beat"] else "DOWN"
    return {
        "session_id":  session_id,
        "outcome":     outcome,
        "hour":        dt.hour,
        "day":         dt.weekday(),   # 0=Monday, 6=Sunday
        "start_price": last["price_to_beat"],
        "end_price":   last["current_price"],
        "rows":        rows,
    }


def annotate_cross_episode(episodes: list[dict], window_min: int = 5) -> list[dict]:
    """
    Add three cross-episode fields to each episode (mutates in place):
      diff_pct_prev_session  : diff_pct from the last row of the previous episode
                               (None for the first episode)
      diff_pct_hour          : (start_price - hour_ago_start_price) / hour_ago_start_price
                               where hour_ago is the session whose session_id is exactly
                               1 hour before this one (None if that session doesn't exist)
      avg_pct_variance_hour  : rolling average of |diff_pct| (last row) across the prior
                               slots within the last 60 minutes, stepping by window_min
                               (None if none of the prior slots exist)
    """
    by_session = {ep["session_id"]: ep for ep in episodes}

    for i, ep in enumerate(episodes):
        ep["diff_pct_prev_session"] = (
            None if i == 0 else episodes[i - 1]["rows"][-1]["diff_pct"]
        )
        dt = datetime.fromisoformat(ep["session_id"].replace("Z", "+00:00"))
        prev_hour_id = (dt - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:00Z")
        prev_hour_ep = by_session.get(prev_hour_id)
        ep["diff_pct_hour"] = (
            (ep["start_price"] - prev_hour_ep["start_price"]) / prev_hour_ep["start_price"] * 100
            if prev_hour_ep else None
        )

        prior_slots = [
            (dt - timedelta(minutes=m)).strftime("%Y-%m-%dT%H:%M:00Z")
            for m in range(window_min, 60 + window_min, window_min)
        ]
        values = [
            abs(by_session[sid]["rows"][-1]["diff_pct"])
            for sid in prior_slots
            if sid in by_session and by_session[sid]["rows"][-1]["diff_pct"] is not None
        ]
        ep["avg_pct_variance_hour"] = sum(values) / len(values) if values else None

    return episodes


def format_output(episodes: list[dict]) -> str:
    """Serialize all episodes as a JSON array."""
    out = []
    for ep in episodes:
        ep_out = {
            "session_id":            ep["session_id"],
            "outcome":               ep["outcome"],
            "hour":                  ep["hour"],
            "day":                   ep["day"],
            "start_price":           ep["start_price"],
            "end_price":             ep["end_price"],
            "diff_pct_prev_session": ep["diff_pct_prev_session"],
            "diff_pct_hour":         ep["diff_pct_hour"],
            "avg_pct_variance_hour": ep["avg_pct_variance_hour"],
            "rows": [
                {k: v for k, v in row.items() if k != "price_to_beat"}
                for row in ep["rows"]
            ],
        }
        out.append(ep_out)
    return json.dumps(out, indent=2)


def collect_files(tsv_dir: str) -> list[str]:
    """Return sorted list of *.tsv file paths in tsv_dir. Returns empty list if none found."""
    return sorted(glob.glob(os.path.join(tsv_dir, "*.tsv")))


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
                episodes_written: int, dropped: int, elapsed: float,
                output_file: str = "") -> None:
    """Print processing summary to stdout."""
    bar = "-" * 65
    print(bar)
    print(f"Combined: {output_file}")
    print(bar)
    print(f"Episodes written:  {episodes_written:>6}")
    print(f"Dropped (no close):{dropped:>6}")
    print(f"Processing time:   {elapsed:.1f}s")
    print()
    print("Date ranges by source file:")
    for m in file_meta:
        duration = format_duration(m["first_ts"], m["last_ts"])
        name = os.path.basename(m["name"])
        print(f"  {name:<42} {m['first_ts']} -> {m['last_ts']}   {duration:>7}   {m['episode_count']:>3} episodes")
    print(bar)


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cfg in MARKETS:
        t0 = time.perf_counter()
        tsv_subdir = os.path.join(TSV_DIR, cfg.tsv_subdir)
        files = collect_files(tsv_subdir)
        if not files:
            print(f"[{cfg.name}] No TSV files found in '{tsv_subdir}' — skipping.",
                  file=sys.stderr)
            continue

        all_rows: list[dict] = []
        file_meta: list[dict] = []
        for filepath in files:
            rows, first_ts, last_ts = read_file_rows(filepath)
            file_meta.append({
                "name": filepath,
                "first_ts": first_ts,
                "last_ts": last_ts,
                "episode_count": 0,
            })
            all_rows.extend(rows)

        segments = segment_rows(all_rows, cfg.window_min)
        segments = reassign_stray_close_rows(segments)
        kept, dropped = filter_segments(segments)
        episodes = annotate_cross_episode(
            [annotate_segment(seg, cfg.window_min) for seg in kept],
            cfg.window_min,
        )

        for ep in episodes:
            ep_ts = ep["rows"][0]["timestamp"]
            for meta in file_meta:
                if meta["first_ts"] <= ep_ts <= meta["last_ts"]:
                    meta["episode_count"] += 1
                    break

        output_file = os.path.join(OUTPUT_DIR, f"{cfg.output_prefix}_combined_{ts}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(format_output(episodes))

        elapsed = time.perf_counter() - t0
        print_stats(files, file_meta, len(episodes), dropped, elapsed, output_file)


if __name__ == "__main__":
    main()
