# TSV Combiner — Design Spec
**Date:** 2026-03-16
**Status:** Approved

---

## Overview

A one-shot Python script (`combine_tsv.py`) that reads all Polymarket BTC 5-minute snapshot TSV files, combines them into a single JSON episode file for use as reinforcement learning training data.

Each 5-minute market period becomes one **episode** — a JSON object containing the outcome (UP/DOWN), metadata, and all observed rows in sequential order. The training agent will be fed rows one at a time within an episode and rewarded/punished based on its buy/sell/hold decisions versus the actual outcome.

---

## Assumptions

- TSV files are **non-overlapping** in time. Each file represents a separate, non-concurrent recording session. If two sessions recorded the same moment, results are undefined.
- Within a single recording session there are **no gaps** in the data — the only discontinuities are at the start (session may begin mid-period) and at the end (session may be cut off before a period closes).

---

## Input

- Source directory: `tsv/`
- File pattern: `btc_polymarket_*.tsv`
- Files are sorted alphabetically — filenames follow the pattern `btc_polymarket_YYYYMMDD_HHMMSS.tsv`, so alphabetical order is guaranteed to be chronological
- Each TSV has a header row:
  ```
  Timestamp  Up Bid  Up Ask  Down Bid  Down Ask  Price to Beat  Current Price  Difference %  Difference $  Time to Close (ms)
  ```
- Polling interval: ~2 seconds; a full 5-minute window yields ~150 rows
- If no files match the glob, the script exits with a clear error message and a non-zero exit code

---

## Processing Pipeline

### 1. Collect
Glob `tsv/btc_polymarket_*.tsv`, sort alphabetically, open each in sequence. Skip duplicate header lines (any line starting with `Timestamp`).

### 2. Parse
Parse each row into a dict with snake_case keys:
```
timestamp, up_bid, up_ask, down_bid, down_ask,
price_to_beat, current_price, diff_pct, diff_usd, time_to_close
```
All numeric fields stored as `float`. `time_to_close` stored as `int` via truncation (`int(float(value))`).

### 3. Segment
Group consecutive rows into segments using the **timestamp-based window key**:

```python
window_key = (year, month, day, hour, (minute // 5) * 5)
```

**Boundary rule:** when `window_key` changes between row N and row N+1, row N is the **last row of the current segment** and row N+1 is the **first row of the next segment**.

> Rationale: Polymarket's 5-minute windows are clock-aligned at :00, :05, :10, :15, :20, :25, :30, :35, :40, :45, :50, :55 past the hour. The window key derived from the row's UTC timestamp is an unambiguous boundary signal regardless of price. `price_to_beat` was considered as an alternative but is unreliable — if BTC opens two consecutive windows at the same price, `price_to_beat` would not change and two distinct episodes would be incorrectly merged.

**Consequence for `end_price`:** The closing row of a segment is the last row whose timestamp falls within that window key. Its `current_price` is the final observed price and is used as `end_price`.

### 4. Filter (completeness)
Discard any segment where no row has `time_to_close < 15000` (15 seconds remaining).

- **Kept:** segment has a near-close row — outcome is knowable from the final observed price
- **Dropped:** recording cut off before the window closed — outcome is unknown

Partial-start segments (recording began mid-period) are **kept** as long as they satisfy the close condition. No minimum row count is required — the agent simply starts making decisions from wherever the data begins. This is intentional: partial-start segments are valid training episodes.

### 5. Annotate (per-segment)
For each valid segment:
- `session_id`: ISO 8601 UTC timestamp of the window start (e.g. `"2026-03-14T17:20:00Z"`), derived from the window key of the first row
- `outcome`: `"UP"` if last row's `current_price >= price_to_beat`, else `"DOWN"`. An exact tie resolves as `"UP"` — consistent with how Polymarket resolves the market.
- `start_price`: the segment's `price_to_beat` value
- `end_price`: `current_price` from the last row in the segment (i.e., the last row with the old `price_to_beat`, per the boundary rule above)
- `hour`: integer 0–23, from `datetime.fromisoformat(timestamp).hour` (UTC)
- `day`: integer 0–6 (Monday=0, Sunday=6), from `datetime.fromisoformat(timestamp).weekday()` (UTC)

> Note: for partial-start segments, `hour` and `day` are derived from the first *captured* row, not the true market window open. The hour and day are the same as the true window open since the window is only 5 minutes, so this is correct.

### 6. Annotate (cross-episode)
After all episodes are assembled, two cross-episode fields are computed:
- `diff_pct_prev_session`: the `diff_pct` value from the last row of the immediately preceding episode (`null` for the first episode)
- `diff_pct_hour`: `(start_price - hour_ago_start_price) / hour_ago_start_price * 100`, where `hour_ago` is the episode whose `session_id` is exactly 1 hour before the current one (`null` if that episode does not exist in the data)

### 7. Write
Emit all episodes as a JSON array to a timestamped file in the `data/` directory:
```
data/btc_polymarket_combined_YYYYMMDD_HHMMSS.json
```
The timestamp in the filename is the UTC wall-clock time at the moment the script runs. The `data/` directory is gitignored.

---

## Output Format

A single JSON array of episode objects, serialized with `json.dumps(indent=2)`. Parsed with `json.load()`.

Field order within each episode object:

```json
[
  {
    "session_id": "2026-03-14T17:20:00Z",
    "outcome": "UP",
    "hour": 17,
    "day": 5,
    "start_price": 70679.78,
    "end_price": 70694.50,
    "diff_pct_prev_session": null,
    "diff_pct_hour": null,
    "rows": [
      {"timestamp":"2026-03-14T17:23:01Z","up_bid":55.0,"up_ask":56.0,"down_bid":44.0,"down_ask":45.0,"current_price":70685.94,"diff_pct":0.008715,"diff_usd":6.16,"time_to_close":119733},
      {"timestamp":"2026-03-14T17:24:59Z","up_bid":55.0,"up_ask":56.0,"down_bid":44.0,"down_ask":45.0,"current_price":70694.50,"diff_pct":0.020826,"diff_usd":14.72,"time_to_close":1592}
    ]
  },
  {
    "session_id": "2026-03-14T17:25:00Z",
    "outcome": "DOWN",
    ...
  }
]
```

Notes:
- `price_to_beat` is **omitted from each row** — it is redundant with the episode-level `start_price`
- `diff_pct_prev_session` and `diff_pct_hour` use the same scale as the row-level `diff_pct` field (i.e. `* 100`, not a raw ratio)

### Parsing convention for training code
```python
import json

with open("data/btc_polymarket_combined_YYYYMMDD_HHMMSS.json") as f:
    episodes = json.load(f)
```

---

## Console Stats Output

Printed to stdout on completion:

```
-----------------------------------------------------------------
Combined: data\btc_polymarket_combined_20260316_230233.json
-----------------------------------------------------------------
Episodes written:     620
Dropped (no close):     3
Processing time:      0.6s

Date ranges by source file:
  btc_polymarket_20260314_132259.tsv   2026-03-14T17:22:59Z -> 2026-03-15T01:00:17Z   7h 37m    92 episodes
  btc_polymarket_20260314_210528.tsv   2026-03-15T01:05:29Z -> 2026-03-15T23:52:37Z  22h 47m   273 episodes
  btc_polymarket_20260315_210627.tsv   2026-03-16T01:06:27Z -> 2026-03-16T12:50:13Z  11h 43m   141 episodes
  btc_polymarket_20260316_093828.tsv   2026-03-16T13:38:29Z -> 2026-03-16T23:02:31Z   9h 24m   114 episodes
-----------------------------------------------------------------
```

Fields:
- **Episodes written**: count of valid segments written to the output file
- **Dropped (no close)**: count of segments discarded because no row had `time_to_close < 15000`
- **Processing time**: wall-clock seconds from script start to file write complete
- **Date ranges by source file**: one line per TSV file showing:
  - First row timestamp → last row timestamp (ISO 8601 UTC)
  - Duration: elapsed time between first and last row, formatted as `Xh YYm`
  - Episode count: number of valid episodes sourced from that file

---

## Tests (`test_combine_tsv.py`)

Using Python's built-in `unittest`. The script imports processing functions directly from `combine_tsv.py`.

### Unit tests

| Test class | Description |
|------------|-------------|
| `TestParseRow` | TSV line parsing, empty fields → `None`, `time_to_close` truncated to int |
| `TestGetWindowKey` | 5-minute window key derivation, boundary and rollover cases |
| `TestSegmentRows` | Clock-boundary segmentation, boundary row ownership, empty input |
| `TestFilterSegments` | Completeness filter at threshold (14999 kept, 15000 dropped) |
| `TestAnnotateSegment` | `outcome`, `start_price`, `end_price`, `hour`, `day`, `session_id`, row preservation |
| `TestAnnotateCrossEpisode` | `diff_pct_prev_session` (None for first episode, value from last row of prior), `diff_pct_hour` (None when no exact hour-prior session, computed value when present) |
| `TestWriteOutput` | Valid JSON array, session_id as attribute, `price_to_beat` excluded from rows, metadata and rows present |
| `TestCollectFiles` | Sorted file list, non-zero exit on empty directory |
| `TestReadFileRows` | Header skipping, parsed dicts returned |
| `TestFormatDuration` | Duration formatting across hours and midnight |

### Integration / spot-check tests

Load the most recent combined file from `data/` and verify known episodes against the source TSVs:

| Test | Source file | Known boundary | Expected fields |
|------|-------------|----------------|-----------------|
| `test_spot_episode_1` | `btc_polymarket_20260314_132259.tsv` | First window (~17:20 UTC) | `outcome`, `start_price`, `end_price`, `hour=17`, `day=5` |
| `test_spot_episode_2` | `btc_polymarket_20260315_210627.tsv` | First complete segment in that file | `outcome`, `hour=1`, `day=0` (Monday) |
| `test_spot_episode_3` | `btc_polymarket_20260316_093828.tsv` | Last complete segment | `end_price` consistent within episode and with source TSV |

Spot-check tests are skipped automatically if no combined file exists in `data/` (they require `python combine_tsv.py` to have been run first).

---

## Files Created / Modified

| File | Action |
|------|--------|
| `combine_tsv.py` | New — main script |
| `test_combine_tsv.py` | New — unit + integration tests |
| `data/btc_polymarket_combined_*.json` | Generated — gitignored via `data/` |

---

## Out of Scope

- Deduplication of rows across TSV files (source files are assumed non-overlapping)
- Timezone conversion (all timestamps remain UTC)
- Streaming/incremental output (full reprocess each run)
