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

### 5. Annotate
For each valid segment:
- `outcome`: `"UP"` if last row's `current_price >= price_to_beat`, else `"DOWN"`. An exact tie resolves as `"UP"` — consistent with how Polymarket resolves the market.
- `start_price`: the segment's `price_to_beat` value
- `end_price`: `current_price` from the last row in the segment (i.e., the last row with the old `price_to_beat`, per the boundary rule above)
- `hour`: integer 0–23, from `datetime.fromisoformat(timestamp).hour` (UTC)
- `day`: integer 0–6 (Monday=0, Sunday=6), from `datetime.fromisoformat(timestamp).weekday()` (UTC)

> Note: for partial-start segments, `hour` and `day` are derived from the first *captured* row, not the true market window open. The hour and day are the same as the true window open since the window is only 5 minutes, so this is correct.

### 6. Write
Emit episodes to `btc_polymarket_combined.json` in the project root.

---

## Output Format

One JSON object per episode. Episodes separated by **exactly one blank line** (`\n\n` between episodes). Each row in the `rows` array appears on its own line. **No blank lines appear within a single episode block.**

```json
{"outcome":"UP","hour":17,"day":4,"start_price":70679.78,"end_price":70720.10,"rows":[
{"timestamp":"2026-03-14T17:23:01Z","up_bid":55.0,"up_ask":56.0,"down_bid":44.0,"down_ask":45.0,"price_to_beat":70679.78,"current_price":70685.94,"diff_pct":0.008715,"diff_usd":6.16,"time_to_close":119733},
{"timestamp":"2026-03-14T17:23:03Z","up_bid":45.0,"up_ask":51.0,"down_bid":49.0,"down_ask":55.0,"price_to_beat":70679.78,"current_price":70685.94,"diff_pct":0.008715,"diff_usd":6.16,"time_to_close":117661}
]}

{"outcome":"DOWN","hour":17,"day":4,"start_price":70694.51,"end_price":70650.22,"rows":[
...
]}
```

### Parsing convention for training code
```python
with open("btc_polymarket_combined.json") as f:
    content = f.read()

episodes = [json.loads(block) for block in content.strip().split("\n\n")]
```

This works because no blank lines appear within any episode block — the only `\n\n` sequences in the file are the episode separators.

---

## Console Stats Output

Printed to stdout on completion:

```
─────────────────────────────────────────────────────────────────
Combined: btc_polymarket_combined.json
─────────────────────────────────────────────────────────────────
Episodes written:     142
Dropped (no close):    12
Processing time:      1.3s

Date ranges by source file:
  btc_polymarket_20260314_132259.tsv   2026-03-14T17:22:59Z → 2026-03-14T23:59:51Z
  btc_polymarket_20260314_210528.tsv   2026-03-14T21:05:28Z → 2026-03-15T03:12:44Z
  btc_polymarket_20260315_210627.tsv   2026-03-15T21:06:27Z → 2026-03-16T04:00:03Z
  btc_polymarket_20260316_093828.tsv   2026-03-16T09:38:29Z → 2026-03-16T13:40:01Z
─────────────────────────────────────────────────────────────────
```

Fields:
- **Episodes written**: count of valid segments written to the output file
- **Dropped (no close)**: count of segments discarded because no row had `time_to_close < 15000`
- **Processing time**: wall-clock seconds from script start to file write complete
- **Date ranges by source file**: for each TSV file, the `timestamp` of its first row → `timestamp` of its last row, as ISO 8601 UTC strings. One line per file, showing coverage and gaps between sessions at a glance.

---

## Tests (`test_combine_tsv.py`)

Using Python's built-in `unittest`. The script imports processing functions directly from `combine_tsv.py`.

### Unit tests

| Test | Description |
|------|-------------|
| `test_segment_boundary` | Two rows whose timestamps fall in different 5-minute windows → split into 2 segments |
| `test_no_boundary_same_window` | Two rows whose timestamps fall in the same 5-minute window → one segment |
| `test_boundary_row_belongs_to_new_segment` | First row after a clock boundary starts the new segment; last row before belongs to the old segment |
| `test_same_price_to_beat_across_boundary` | Two consecutive windows with identical `price_to_beat` → still split correctly by timestamp |
| `test_truncation_filter_drop` | Segment with no row having `time_to_close < 15000` → dropped |
| `test_truncation_filter_keep` | Segment with one row having `time_to_close = 8000` → kept |
| `test_outcome_up` | Last row `current_price > price_to_beat` → `"UP"` |
| `test_outcome_up_tie` | Last row `current_price == price_to_beat` → `"UP"` |
| `test_outcome_down` | Last row `current_price < price_to_beat` → `"DOWN"` |
| `test_end_price_is_last_row_in_window` | `end_price` equals `current_price` of the last row whose timestamp falls within the window |
| `test_hour_annotation` | Timestamp `2026-03-14T17:23:01Z` → `hour=17` |
| `test_day_annotation` | Timestamp `2026-03-14T17:23:01Z` → `day=5` (Saturday) |
| `test_no_files_error` | Empty `tsv/` directory → script exits with non-zero code and clear error message |
| `test_output_no_blank_lines_within_episode` | No `\n\n` appears within any single episode block in the output |

### Integration / spot-check tests

Load the actual combined output file and verify known episodes from the source TSV files:

| Test | Source file | Known boundary | Expected fields |
|------|-------------|----------------|-----------------|
| `test_spot_episode_1` | `btc_polymarket_20260314_132259.tsv` | First complete segment starting ~17:25 UTC | `outcome`, `start_price`, `end_price`, `hour=17`, `day=5` |
| `test_spot_episode_2` | `btc_polymarket_20260315_210627.tsv` | First complete segment in that file | `outcome`, `hour`, `day=6` (Sunday) |
| `test_spot_episode_3` | `btc_polymarket_20260316_093828.tsv` | Last complete segment | `outcome`, correct `end_price` matches last TSV row's `current_price` |

Spot-check tests are skipped automatically if `btc_polymarket_combined.json` does not exist (they require the combiner to have been run first).

---

## Files Created / Modified

| File | Action |
|------|--------|
| `combine_tsv.py` | New — main script |
| `test_combine_tsv.py` | New — unit + integration tests |
| `btc_polymarket_combined.json` | Generated — not committed to git |

---

## Out of Scope

- Deduplication of rows across TSV files (source files are assumed non-overlapping)
- Timezone conversion (all timestamps remain UTC)
- Streaming/incremental output (full reprocess each run)
