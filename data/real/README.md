# Real data

One file lives here, and one file only.

`qqq_snapshot_<date>.parquet` is a single static snapshot of the listed QQQ option chain,
downloaded once by `scripts/fetch_snapshot_once.py` and committed. `CLAUDE.md` section 4:

> It is a fixture, not a pipeline. Do not build a scheduler around it. Do not re-fetch it
> automatically.

## What is in the committed snapshot

| | |
|---|---|
| Ticker | QQQ |
| As of | 2026-08-19 |
| Spot | 716.08 |
| Rows | 9,315 |
| Expiries | 31 |
| Maturities | 0 to 2.33 years |
| Two-sided quotes | 8,702 |
| Zero-bid quotes | 604 |
| Crossed quotes | 0 |

The zero-bid count is worth noting: 6.5% of a real listed chain has no bid at all, which
is the same defect the synthetic generator injects deliberately. The crossed-quote count
being zero is also worth noting — a consolidated end-of-day snapshot has already had those
cleaned out of it, which is precisely why the synthetic generator has to inject them if
the quality layer is going to be exercised at all.

## What it is used for

Exactly one thing: **figure 6**, the fitted smile against real market data with residuals
in volatility basis points.

Every other result in this project comes from synthetic data, because with real data there
is no ground truth and parameter *recovery* — the thing this project measures — cannot be
evaluated at all. Only fit residual can, and in an ill-conditioned problem a small residual
is not evidence of a good parameter estimate. That is the whole argument.
