"""Download one static snapshot of a real QQQ option chain. Run manually, once.

CLAUDE.md section 4 is unambiguous about what this is and is not:

    Secondary source: exactly one static snapshot of a real QQQ option chain, downloaded
    once by a manual script, committed to data/real/. It is a fixture, not a pipeline. Do
    not build a scheduler around it. Do not re-fetch it automatically.

So this script is not imported anywhere, is not wired to a systemd timer, is not called by
any management command, and refuses to overwrite an existing snapshot unless told to. The
committed parquet is the artefact; this script is the provenance record for it.

Usage::

    python scripts/fetch_snapshot_once.py
    python scripts/fetch_snapshot_once.py --ticker SPY --out data/real/

The snapshot feeds exactly one thing: figure 6, the fitted smile against real market data
with residuals in volatility basis points. Every other result in this project comes from
synthetic data with known ground truth, because recovery cannot be measured without it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

import polars as pl


def fetch(ticker: str) -> pl.DataFrame:
    """Pull the full option chain for every listed expiry.

    Uses ``yfinance``, which is a dev-only dependency for exactly this reason: nothing in
    the runtime path may depend on a market data source.
    """
    try:
        import yfinance
    except ImportError:
        print("yfinance is not installed. It is in requirements-dev.txt:")
        print("    pip install -r requirements-dev.txt")
        raise SystemExit(2) from None

    security = yfinance.Ticker(ticker)
    expiries = security.options
    if not expiries:
        print(f"No option expiries returned for {ticker}. The source may be unavailable.")
        raise SystemExit(1)

    history = security.history(period="1d")
    if history.empty:
        print(f"No spot price returned for {ticker}.")
        raise SystemExit(1)
    spot = float(history["Close"].iloc[-1])
    as_of = dt.date.today()

    frames: list[pl.DataFrame] = []
    for expiry in expiries:
        chain = security.option_chain(expiry)
        for option_type, table in (("call", chain.calls), ("put", chain.puts)):
            if table.empty:
                continue
            frame = pl.from_pandas(
                table[
                    [
                        "strike",
                        "bid",
                        "ask",
                        "lastPrice",
                        "volume",
                        "openInterest",
                        "impliedVolatility",
                    ]
                ]
            ).with_columns(
                pl.lit(expiry).alias("expiry"),
                pl.lit(option_type).alias("option_type"),
                pl.lit(spot).alias("spot"),
                pl.lit(as_of.isoformat()).alias("as_of"),
                pl.lit(ticker).alias("ticker"),
            )
            frames.append(frame)

    if not frames:
        print(f"Every expiry for {ticker} came back empty.")
        raise SystemExit(1)

    combined = pl.concat(frames, how="vertical_relaxed")
    # Years to expiry, using the same ACT/365 convention as the rest of the project.
    return combined.with_columns(
        maturity=(pl.col("expiry").str.to_date() - pl.lit(as_of)).dt.total_days().cast(pl.Float64)
        / 365.0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--out", default="data/real", type=pathlib.Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing snapshot. Think before using this: the committed "
        "snapshot is what makes figure 6 reproducible, and replacing it silently "
        "changes a published number.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.out.glob(f"{args.ticker.lower()}_snapshot_*.parquet"))
    if existing and not args.force:
        print(f"A snapshot already exists: {existing[-1]}")
        print("This is a fixture, not a pipeline (CLAUDE.md section 4). Nothing to do.")
        print("Pass --force only if you intend to change a published figure.")
        return 0

    frame = fetch(args.ticker)
    path = args.out / f"{args.ticker.lower()}_snapshot_{dt.date.today().isoformat()}.parquet"
    frame.write_parquet(path)

    print(f"Wrote {path}")
    print(f"  rows      : {frame.height}")
    print(f"  expiries  : {frame['expiry'].n_unique()}")
    print(f"  spot      : {frame['spot'][0]:.2f}")
    print(f"  maturities: {frame['maturity'].min():.4f} to {frame['maturity'].max():.4f} years")
    print("\nCommit this file. Do not schedule this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
