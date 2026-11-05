# Limit order book alpha from crypto L2/L3 data

> Record real order books, rebuild them correctly, and find short-horizon signals that survive fees and latency.

**Status:** capture (M2). Research results arrive in the memo at M7.

| | |
|---|---|
| Track | Quant Researcher |
| Estimated effort | 6–8 weeks |
| Stack | Python, plus a Rust or C++ recorder (optional) |
| Data cost | Free (self-recorded) |
| Strongest for | HFT and market-making firms |

## Requirements

- [x] A websocket recorder for L2 depth updates and trades on 3+ instruments, with sequence-gap detection and snapshot resync. (FR-01 to FR-11, AC-W-06)
- [ ] Order book reconstruction verified against exchange checksums or periodic REST snapshots.
- [ ] Features: order flow imbalance (Cont, Kukanov and Stoikov 2014), multi-level OFI, queue imbalance, microprice (Stoikov 2018), trade-sign imbalance, spread, short-horizon volatility.
- [ ] Labels: mid-price change in ticks at horizons from 100 ms to 30 s, in both event time and clock time.
- [ ] Models: a linear or ridge baseline, then gradient boosting; walk-forward splits by day, never shuffled.
- [ ] An execution check that turns predictions into taker and maker trades with fees, latency and spread.

## Data

Raw and licensed data live in `data/`, which is git-ignored. Only download scripts and small synthetic fixtures are committed.

| Dataset | Cost | What it gives you |
|---|---|---|
| [Binance, Bybit, OKX public websockets](https://developers.binance.com) | Free | Record your own L2 and trades. OKX sends CRC32 book checksums you can verify against. |
| [Binance public data](https://data.binance.vision) | Free | Historical trades, aggTrades and klines; futures book-depth files for some symbols and periods. |
| [Tardis.dev](https://tardis.dev) | Paid | Tick-level L2/L3 history for most crypto venues. Free sample on the first day of each month. |
| [Coinbase Exchange feed](https://docs.cdp.coinbase.com) | Free | The full channel gives order-level (L3) data. |
| [LOBSTER](https://lobsterdata.com) | Free samples | Reconstructed Nasdaq order books. Sample files are free; full access is academic. |

## Success criteria

### Works (correct and reproducible)

- [ ] At least two weeks of continuous capture, with a published gap log and zero crossed books after reconstruction.
- [ ] The rebuilt book matches checksums or snapshots at 100% of checkpoints.
- [ ] OFI explains a meaningful share of short-horizon mid-price changes, matching the literature in sign and rough size.

### Impresses hirers

- [ ] An alpha decay curve: out-of-sample R² against horizon and against added latency, giving the signal’s half-life in milliseconds.
- [ ] A break-even analysis of edge in bps against taker fee plus half-spread, showing honestly where it pays (often only as a maker).
- [ ] Stability across days, assets and venues, with confidence intervals.
- [ ] Feature importance explained in plain microstructure terms.

## Questions this project must answer

- Why event time instead of clock time?
- Your signal works at 1 s. What latency do you need to capture it?
- How does adverse selection change the result if you post passively?
