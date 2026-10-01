# Changelog

## v7 (this repo)
- Forked from `trader_v6` after verifying: 19/19 unit tests pass, `scripts/preflight.py`
  passes (builds the offline cache, runs one full synthetic epoch, writes all 8 charts).
- No architecture changes yet relative to v6 — this is the clean, verified baseline that
  training proper starts from. Dataset (`data/klines/`, 2017-08-17 -> 2026-09-30,
  BTCUSDT/ETHUSDT/LTCUSDT 1-minute bars) carried over unchanged.
