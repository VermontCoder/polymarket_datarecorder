# TSV Combiner — Design Spec
**Date:** 2026-03-16
**Status:** Approved

---

## Overview

A one-shot Python script (`combine_tsv.py`) that reads all Polymarket BTC 5-minute snapshot TSV files, combines them into a single JSON episode file for use as reinforcement learning training data.

Each 5-minute market period becomes one **episode** — a JSON object containing the outcome (UP/DOWN), metadata, and all observed rows in sequential order. The training agent will be fed rows one at a time within an episode and rewarded/punished based on its buy/sell/hold decisions versus the actual outcome.

---

## Input

- Source directory: `tsv/`
- File pattern: `btc_polymarket_*.tsv`
- Files are sorted alphabetically (filenames encode datetime, so alphabetical = chronological)
- Each TSV has a header row:
  ```
  Timestamp  Up Bid  Up Ask  Down Bid  Down Ask  Price to Beat  Current Price  Difference %  Difference $  Time to Close (ms)
  ```
- Polling interval: ~2 seconds; a full 5-minute window yields ~150 rows

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
All numeric fields stored as `float`. `time_to_close` stored as `int`.

### 3. Segment
Group consecutive rows into segments by detecting changes in `price_to_beat`. When `price_to_beat` changes between two adjacent rows, the prior segment ends and a new one begins.

> Rationale: `price_to_beat` is fetched fresh from the Binance opening price at the start of each 5-minute window and is constant within a window — it is the most reliable boundary signal.

### 4. Filter (completeness)
Discard any segment where no row has `time_to_close < 15000` (15 seconds).

- **Kept:** segment has a row confirming the near-close price — outcome is knowable
- **Dropped:** recording cut off before the window closed — outcome is unknown

Partial-start segments (recording began mid-period) are **kept** as long as they satisfy the close condition. The agent simply starts making decisions from wherever the data begins.

### 5. Annotate
For each valid segment:
- `outcome`: `"UP"` if last row's `current_price > price_to_beat`, else `"DOWN"`
- `start_price`: the segment's `price_to_beat` value
- `end_price`: `current_price` from the last row
- `hour`: 0–23, derived from the first row's UTC timestamp
- `day`: 0–6 (Monday=0, Sunday=6), derived from the first row's UTC timestamp

### 6. Write
Emit episodes to `btc_polymarket_combined.json` in the project root.

---

## Output Format

One JSON object per episode. Episodes separated by a blank line. Each row in the `rows` array appears on its own line for human readability. The file is valid JSON when each episode block is parsed individually.

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

---

## Console Stats Output

Printed to stdout on completion:

```
─────────────────────────────────────
Combined: btc_polymarket_combined.json
─────────────────────────────────────
Episodes written:     142
Truncated (dropped):   12
Date range:           2026-03-14T17:22:59Z → 2026-03-16T13:40:01Z
Processing time:      1.3s
─────────────────────────────────────
```

Fields:
- **Episodes written**: count of valid episodes in the output file
- **Truncated (dropped)**: count of segments discarded due to missing close confirmation
- **Date range**: first timestamp of first episode → last timestamp of last episode
- **Processing time**: wall-clock seconds from script start to file write complete

---

## Tests (`test_combine_tsv.py`)

Using Python's built-in `unittest`. The script imports processing functions directly from `combine_tsv.py`.

### Unit tests

| Test | Description |
|------|-------------|
| `test_segment_boundary` | Two rows with different `price_to_beat` → split into 2 segments |
| `test_no_boundary_same_price` | Two rows with same `price_to_beat` → one segment |
| `test_truncation_filter_drop` | Segment with no row having `time_to_close < 15000` → dropped |
| `test_truncation_filter_keep` | Segment with one row having `time_to_close = 8000` → kept |
| `test_outcome_up` | Last row `current_price > price_to_beat` → `"UP"` |
| `test_outcome_down` | Last row `current_price < price_to_beat` → `"DOWN"` |
| `test_hour_annotation` | Timestamp `2026-03-14T17:23:01Z` → `hour=17` |
| `test_day_annotation` | Timestamp `2026-03-14T17:23:01Z` → `day=5` (Saturday) |

### Integration / spot-check tests

Load the actual combined output file and verify known episodes from the source TSV files:

| Test | Source file | Known boundary | Expected fields |
|------|-------------|----------------|-----------------|
| `test_spot_episode_1` | `btc_polymarket_20260314_132259.tsv` | First complete segment starting ~17:25 UTC | `outcome`, `start_price`, `end_price`, `hour=17`, `day=5` |
| `test_spot_episode_2` | `btc_polymarket_20260315_210627.tsv` | First complete segment in that file | `outcome`, `hour`, `day=6` (Sunday) |
| `test_spot_episode_3` | `btc_polymarket_20260316_093828.tsv` | Last complete segment | `outcome`, correct `end_price` matches last TSV row |

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

- Deduplication of rows that appear in multiple TSV files (not expected based on file naming)
- Timezone conversion (all timestamps remain UTC)
- Streaming/incremental output (full reprocess each run)
