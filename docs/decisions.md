# Decisions

Owner decisions and agent ambiguity choices for `crypto-lob-alpha`. Owner entries are copied from the portfolio `DECISIONS.md` (recorded 2026-10-31) and spec section 3. `DECISIONS.md` overrides the spec, and the spec overrides `README.md`.

## Owner decisions

| ID | Decision | Status |
|---|---|---|
| D-01 | Venue set | `binance-usdm` (primary), `coinbase`, `kraken`. OKX is not allowed. No VPN or proxy. Bybit only after P-16. |
| D-02 | Always-on host | Owner's Windows 11 Pro PC under WSL2. Gaps from reboots, sleep or outages are logged, never hidden. |
| D-03 | Service manager | systemd in WSL2, as a system-level unit because lingering is off for user `ze`. Docker files are still provided. |
| D-04 | Tardis paid history | **Not approved.** Free first-of-month samples only. |
| D-05 | VPS budget | USD 0, not needed. |
| D-06 | Raw archive retention | Rolling 7 days. |
| D-07 | Coinbase full channel (L3) | Off. FR-04 is skipped and noted in the memo. |
| D-08 | Instruments | BTC, ETH, SOL on every enabled venue. |
| D-09 | Fee tier | Base retail tier from `config/instruments.yaml`, plus sensitivity rows (decided at M6). |
| D-10 | Home latency | 2 × median feed latency per venue (decided at M6). |
| D-11 | Recorder language | Python only. |
| D-12 | Websocket client | `websockets`; `picows` only if NFR-01 fails. |
| D-13 | Licence | MIT. |
| D-14 | Stream-down alerts | Off. Status through `lob record status` and logs. |

## Portfolio decisions that apply here

- **Merging:** feature branches and PRs; PRs merge automatically once CI is green. Branch protection and auto-merge turned on once CI exists.
- **Identity:** `gh` account `avmhdt`, with admin on the repo. Country: Cayman Islands.
- **Downloads:** Binance public data and Tardis free samples are approved. LOBSTER stays manual: the owner must accept its terms and download it.
- **Money:** this project spends USD 0.
- **Storage:** one shared crypto capture on `D:`. Free space is checked at each milestone. The disk holding the WSL distro is a spinning HDD shared by all projects.
- **Benchmarks:** published benchmark runs in other projects happen when no multi-week capture is running on this PC.

## Prerequisites resolved

| ID | Item | Status |
|---|---|---|
| P-01 | WSL2 Ubuntu 22.04.5 with systemd | Verified 2026-11-01. |
| P-02 | Venue permission | Resolved in D-01. |
| P-03 | Always-on PC | Sleep off on AC, Windows Update paused to 2026-12-06, "Start WSL at boot" task passed a manual run but is not yet tested across a real reboot. |
| P-04 | Disk | D: about 903 GB free on 2026-11-01, shared by all projects. |
| P-07 | GitHub settings | Actions enabled; branch protection on `main` requires `ci` once it exists. |
| P-08 | `uv` | 0.12.23 in WSL. |
| P-09 | Time sync | **Not met.** The WSL clock follows the Windows clock, which measured about 165 ms fast against four public NTP servers and Kraken on 2026-11-01. The spec's `chronyc tracking` check misleads here: see [ops.md](ops.md). The fix is on the Windows side and is the owner's call. |
| P-11 | Coinbase key | Not used (D-07). |
| P-12 | Tardis key | Not used (D-04). |
| P-14 | LOBSTER samples | Manual owner download. |
| P-15 | Alert webhook | Not used (D-14). |
| P-16 | Bybit | Not answered, so no Bybit connection is opened. |

## Capture log

| Time (UTC) | Event |
|---|---|
| 2026-11-01 22:11:59 | First capture start. Run `20261101T221159`, version `0.1.0+caf3df3df091`, system-level service enabled. All 21 streams LIVE within a minute. |
| 2026-11-01 22:11:33 onward | **Clock disturbed.** An agent-recommended chrony change left two chronyd instances steering one clock. `ts_recv` wandered between about +9 ms and -289 ms against NTP, with backward steps. This interval runs until the change is reverted, and its data must not be used for latency or sub-second timing work. |
| No earlier than 2026-11-02 20:15 | Planned owner reboot. It applies `instanceIdleTimeout=-1` and tests the boot task. |

The 14-day consecutive capture for AC-W-01 is counted from the first clean start after that reboot. Anything recorded before it is a bonus.

## Agent choices where the spec was ambiguous or wrong

Dates are when the choice was made. Each follows spec rule 9: the simplest option consistent with the spec.

| Date | Topic | Choice | Why |
|---|---|---|---|
| 2026-11-01 | Coinbase trade gap detection | Use `trade_id` jumps, not `sequence` jumps. | The spec says "gap = jump > 1" on `sequence`, but `sequence` counts every event in the product's full feed, so consecutive matches jump by more than 1 in normal operation (recorded: 138632483145 then 138632483148). `trade_id` was contiguous in the same sample, and Coinbase's docs point at `last_trade_id` for missed trades. |
| 2026-11-01 | Clock probe on Kraken | Use the WS v2 `ping`/`pong` `time_in` field. | `/0/public/Time` has 1 s resolution, which would trip DQ-15's 50 ms limit on every probe. `time_in` has microseconds. `req_id` carries the send time so one pong gives the RTT. |
| 2026-11-01 | Clock probe on Binance | Use `/fapi/v1/time`. | `exchangeInfo.serverTime` came back about 10.5 hours old on 2026-11-01; it is cached. |
| 2026-11-01 | `is_crossed` | True for best bid >= best ask, as DQ-08 defines it; `is_locked` is the separate equality test. | AC-W-02 and DQ-08 count both as crossed and report locked separately. |
| 2026-11-01 | Kraken number text | Book frames are parsed with `json.loads(parse_float=str)`. | `px_str` and `qty_str` must be the wire text for the CRC. `orjson` would give floats and lose trailing zeros. |
| 2026-11-01 | Kraken local book | Truncated to the subscribed depth (1000) after every message; the offline replay must do the same. | Kraken's guide says to truncate, and that no `qty: 0` is sent for levels that fall out of scope. |
| 2026-11-01 | Kraken trade ids | Not treated as contiguous until recorded data shows they are. | The docs promise only "unique per book". |
| 2026-11-01 | `batch_id`, and `seq` on venues without one | `max(ts_recv, last + 1)`. | Strictly increasing as the schema requires, monotone across restarts, and needs no counter state, so `recover` reproduces it from raw frames alone. |
| 2026-11-01 | `ts_recv` of bridged diffs | `max(own receipt, snapshot receipt)`. | A Binance diff buffered before the REST snapshot arrived would otherwise sort before the snapshot under the replay rule (`ts_recv`, then `batch_id`). It also was not usable until the snapshot arrived, so this stays point-in-time correct. Original times remain in the raw archive. |
| 2026-11-01 | Crossed book while LIVE | Counted and logged, not a resync. | `crossed` is not in the spec's gap reason list and FR-07 lists the resync triggers. Offline validation (DQ-08) is the gate. |
| 2026-11-01 | `checkpoint_levels` size | Top 1000 levels per side per checkpoint. | Coinbase `/book?level=2` returns the whole book (about 1 MB), which would add 13 M rows a day. FR-15 compares the top 100. The raw archive keeps full bodies for 7 days. |
| 2026-11-01 | Silence watchdog scope | 30 s silence is per connection and per book stream, never per trades stream. | A quiet symbol can go 30 s without a trade while its connection is healthy. |
| 2026-11-01 | Make-before-break scope | Binance only. | Binance is the only venue with a documented connection lifetime (24 h). |
| 2026-11-01 | Bybit adapter | Not written. | FR-05 is a Could, and P-16 is unanswered. |
| 2026-11-01 | Raw archive contents | Frames per connection, plus an hourly `control` file holding REST bodies, connection events, timers and clock probes. | The spec asks for `ts_recv\tpayload` frames, but row-identical recovery needs every input of the state machine. Recovery replays them through the same `VenuePipeline`. |
| 2026-11-01 | Hourly state record | At the first input of each hour, per-stream counters go to the control file, and for Kraken the book too. | One hour can then be replayed alone, even after older raw days are pruned. Only Kraken's rows depend on book contents, through the checksum. |
| 2026-11-01 | Interrupted hours | The recorder runs `lob record recover --interrupted` at low priority two minutes after it starts. | A crash leaves the hour only in the raw archive, which is pruned after 7 days. An unattended capture should not depend on someone noticing. |
| 2026-11-01 | Runtime location | The service runs from a built venv in `~/lob-recorder`, not from the repo. | Merged code must not change a running recorder (project rule 6), and the recorder should not depend on `/mnt/d`. |
| 2026-11-01 | `run_id` and restart parts | Run id is the UTC start second. Parts within an hour use `-rN` in the order runs wrote them, and the part after a killed run skips its leftover `.tmp`. | `recover` derives the same names, so it fills exactly the file a killed run failed to close. |
| 2026-11-01 | Bridging across a top of hour | Recovering the later hour can differ for the few diffs buffered across the boundary. | The state record does not carry the sync buffer. It needs a resync in the same second as an hour boundary and a crash in that hour. Those rows stay in the raw archive. |
| 2026-11-01 | `.env` comment values | A value starting with `#` is treated as empty. | `python-dotenv` keeps `KEY=   # note` as the value `# note`, and `.env.example` is written that way. |
| 2026-11-01 | Planned capture length | `health.planned_capture_days: 28`, added to `config/recorder.yaml` defaults. | `lob record status --disk` needs an end date to project to, and R-11 targets at least 28 days. |
