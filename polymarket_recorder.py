"""
polymarket_recorder.py
--------------------------
Records Polymarket order book data for multiple markets to tab-delimited files,
polling every 2 seconds until stopped with Ctrl+C.

Markets tracked:
  - BTC 15-minute  (tsv/btc-15/)
  - ETH 15-minute  (tsv/eth-15/)
  - ETH 5-minute   (tsv/eth-5/)

Output columns (per market):
    Timestamp, Up Bid, Up Ask, Down Bid, Down Ask, Price to Beat,
    Current Price, Difference %, Difference $, Time to Close (ms)

Usage:
    python3 polymarket_recorder.py
"""

import urllib.request
import json
import time
import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import rich.box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


INTERVAL_SEC = 2
TSV_DIR      = "tsv"
DISPLAY_ROWS = 10

COLUMNS = [
    "Timestamp",
    "Up Bid", "Up Ask",
    "Down Bid", "Down Ask",
    "Price to Beat", "Current Price",
    "Difference %", "Difference $",
    "Time to Close (ms)",
]

_stop = threading.Event()


@dataclass
class MarketConfig:
    name:           str   # display label, e.g. "BTC-15m"
    slug_prefix:    str   # Polymarket slug base, e.g. "btc-updown-15m"
    window_sec:     int   # epoch-alignment window: 900 for 15m, 300 for 5m
    binance_symbol: str   # e.g. "BTCUSDT" or "ETHUSDT"
    asset_label:    str   # e.g. "BTC" or "ETH"
    output_dir:     str   # subdirectory under TSV_DIR


MARKETS = [
    MarketConfig(
        name="BTC-15m",
        slug_prefix="btc-updown-15m",
        window_sec=900,
        binance_symbol="BTCUSDT",
        asset_label="BTC",
        output_dir="btc-15",
    ),
    MarketConfig(
        name="ETH-15m",
        slug_prefix="eth-updown-15m",
        window_sec=900,
        binance_symbol="ETHUSDT",
        asset_label="ETH",
        output_dir="eth-15",
    ),
    MarketConfig(
        name="ETH-5m",
        slug_prefix="eth-updown-5m",
        window_sec=300,
        binance_symbol="ETHUSDT",
        asset_label="ETH",
        output_dir="eth-5",
    ),
]


# ── Fetching ──────────────────────────────────────────────────────────────────

def fetch_url(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def fetch_open_price(event_start_time: str, binance_symbol: str) -> float | None:
    dt = datetime.fromisoformat(event_start_time.replace("Z", "+00:00"))
    start_ms = int(dt.timestamp() * 1000)
    url = (f"https://api.binance.com/api/v3/klines"
           f"?symbol={binance_symbol}&interval=1m&startTime={start_ms}&limit=1")
    data = fetch_url(url)
    return float(data[0][1]) if data else None


def fetch_current_price(binance_symbol: str) -> float | None:
    data = fetch_url(f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}")
    return float(data["price"]) if data else None


def fetch_book(outcome, token_id):
    url = f"https://clob.polymarket.com/book?token_id={token_id}"
    try:
        return outcome, fetch_url(url)
    except Exception as e:
        return outcome, {"error": str(e)}


def collect_snapshot(cfg: MarketConfig) -> dict | None:
    now = int(time.time())
    window_ts = now - (now % cfg.window_sec)
    slug = f"{cfg.slug_prefix}-{window_ts}"

    markets = fetch_url(f"https://gamma-api.polymarket.com/markets?slug={slug}")
    if not markets:
        return None

    market           = markets[0]
    outcomes         = json.loads(market["outcomes"])   # ["Up", "Down"]
    token_ids        = json.loads(market["clobTokenIds"])
    event_start_time = market.get("eventStartTime")

    with ThreadPoolExecutor(max_workers=5) as executor:
        f_current = executor.submit(fetch_current_price, cfg.binance_symbol)
        f_open    = executor.submit(fetch_open_price, event_start_time, cfg.binance_symbol) if event_start_time else None
        book_futures = {executor.submit(fetch_book, o, t): o
                        for o, t in zip(outcomes, token_ids)}

        current_price = f_current.result()
        price_to_beat = f_open.result() if f_open else None

        clob_results = {}
        book_timestamp = None
        for future in as_completed(book_futures):
            outcome, data = future.result()
            clob_results[outcome] = data
            if book_timestamp is None and "timestamp" in data:
                book_timestamp = int(data["timestamp"])

    books = {}
    for outcome in outcomes:
        d = clob_results.get(outcome, {})
        bids = d.get("bids", [])
        asks = d.get("asks", [])
        books[outcome] = {
            "bid": float(bids[-1]["price"]) * 100 if bids else None,
            "ask": float(asks[-1]["price"]) * 100 if asks else None,
        }

    close_ts_ms   = (window_ts + cfg.window_sec) * 1000
    time_to_close = max(close_ts_ms - book_timestamp, 0) if book_timestamp else None

    diff_pct = None
    diff_usd = None
    if current_price is not None and price_to_beat is not None:
        diff_usd = current_price - price_to_beat
        diff_pct = diff_usd / price_to_beat * 100

    return {
        "timestamp":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "up_bid":        books.get("Up",   {}).get("bid"),
        "up_ask":        books.get("Up",   {}).get("ask"),
        "down_bid":      books.get("Down", {}).get("bid"),
        "down_ask":      books.get("Down", {}).get("ask"),
        "price_to_beat": price_to_beat,
        "current_price": current_price,
        "diff_pct":      diff_pct,
        "diff_usd":      diff_usd,
        "time_to_close": time_to_close,
    }


# ── Display ───────────────────────────────────────────────────────────────────

def _make_market_panel(cfg: MarketConfig, rows: list[dict], output_file: str) -> Panel:
    table = Table(
        box=rich.box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Time (UTC)",  width=12)
    table.add_column("Price",       justify="right", width=11)
    table.add_column("Up B/A",      justify="center", width=9)
    table.add_column("Dn B/A",      justify="center", width=9)
    table.add_column("Diff %",      justify="right",  width=10)
    table.add_column("TTC",         justify="right",  width=6)
    table.add_column("ms",          justify="right",  width=5, style="dim")

    for i, snap in enumerate(reversed(rows)):
        row_style = "on grey15" if i == 0 else ""

        if snap.get("is_error"):
            table.add_row(
                snap["timestamp"][11:23],
                Text(snap.get("message", "error"), style="red"),
                "", "", "", "", "",
                style=row_style,
            )
            continue

        ts    = snap["timestamp"][11:23]   # HH:MM:SS.mmm
        price = f"{snap['current_price']:.2f}" if snap.get("current_price") is not None else "—"

        ub = snap.get("up_bid");   ua = snap.get("up_ask")
        db = snap.get("down_bid"); da = snap.get("down_ask")
        up_ba = f"{ub:.0f}/{ua:.0f}" if ub is not None and ua is not None else "—/—"
        dn_ba = f"{db:.0f}/{da:.0f}" if db is not None and da is not None else "—/—"

        diff = snap.get("diff_pct")
        diff_text = (Text(f"{diff:+.4f}%", style="green" if diff >= 0 else "red")
                     if diff is not None else Text("—", style="dim"))

        ttc     = int(snap.get("time_to_close") or 0) // 1000
        elapsed = str(snap.get("elapsed_ms", "—"))

        table.add_row(ts, price, up_ba, dn_ba, diff_text, f"{ttc}s", elapsed,
                      style=row_style)

    if not rows:
        table.add_row(Text("waiting…", style="dim"), "", "", "", "", "", "")

    subtitle = Text(os.path.basename(output_file), style="dim")
    return Panel(table, title=f"[bold]{cfg.name}[/bold]", subtitle=subtitle,
                 border_style="cyan")


def _make_display(market_buffers: dict[str, deque],
                  market_files:   dict[str, str]) -> Group:
    return Group(*[
        _make_market_panel(cfg, list(market_buffers[cfg.name]), market_files[cfg.name])
        for cfg in MARKETS
    ])


# ── Poll loop ─────────────────────────────────────────────────────────────────

def fmt(val, decimals=2) -> str:
    if val is None:
        return ""
    return f"{val:.{decimals}f}"


def run_market(cfg: MarketConfig, output_file: str, buffer: deque):
    """Poll loop for one market — runs until _stop is set."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    write_header = not os.path.exists(output_file)
    with open(output_file, "a", newline="") as f:
        if write_header:
            f.write("\t".join(COLUMNS) + "\n")

        while not _stop.is_set():
            t0 = time.perf_counter()
            now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            try:
                snap = collect_snapshot(cfg)
                if snap is None:
                    buffer.append({"is_error": True, "timestamp": now_ts,
                                   "message": "no market found"})
                else:
                    snap["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
                    row = "\t".join([
                        snap["timestamp"],
                        fmt(snap["up_bid"]),
                        fmt(snap["up_ask"]),
                        fmt(snap["down_bid"]),
                        fmt(snap["down_ask"]),
                        fmt(snap["price_to_beat"], 2),
                        fmt(snap["current_price"], 2),
                        fmt(snap["diff_pct"], 6),
                        fmt(snap["diff_usd"], 2),
                        fmt(snap["time_to_close"], 0),
                    ])
                    f.write(row + "\n")
                    f.flush()
                    buffer.append(snap)
            except Exception as e:
                buffer.append({"is_error": True, "timestamp": now_ts,
                               "message": str(e)})

            elapsed = time.perf_counter() - t0
            _stop.wait(timeout=max(0.0, INTERVAL_SEC - elapsed))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    market_files:   dict[str, str]   = {}
    market_buffers: dict[str, deque] = {}
    threads: list[threading.Thread]  = []

    for cfg in MARKETS:
        out = os.path.join(TSV_DIR, cfg.output_dir, f"{cfg.slug_prefix}_{ts}.tsv")
        market_files[cfg.name]   = out
        market_buffers[cfg.name] = deque(maxlen=DISPLAY_ROWS)
        t = threading.Thread(target=run_market,
                             args=(cfg, out, market_buffers[cfg.name]),
                             daemon=True, name=cfg.name)
        threads.append(t)

    for t in threads:
        t.start()

    with Live(_make_display(market_buffers, market_files),
              refresh_per_second=4, auto_refresh=False) as live:
        try:
            while True:
                live.update(_make_display(market_buffers, market_files))
                live.refresh()
                time.sleep(0.5)
        except KeyboardInterrupt:
            _stop.set()

    for t in threads:
        t.join(timeout=5)
    print("Stopped.")


if __name__ == "__main__":
    main()
