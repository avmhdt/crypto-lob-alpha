# Recorder runbook

The recorder is a system-level systemd unit in WSL, because lingering is off for user `ze`. It runs from `~/lob-recorder` on ext4, so nothing on `/mnt/d` can stall or stop it, and merging code does not change it. Data lives in `~/data/crypto-lob-alpha`.

## Install

From the repo in WSL, as `ze`:

```bash
bash deploy/install.sh
```

Then, once, with sudo:

```bash
sudo install -m 644 deploy/lob-recorder.service /etc/systemd/system/lob-recorder.service
```

```bash
sudo systemctl daemon-reload
```

```bash
sudo systemctl enable --now lob-recorder
```

## Clock

Do not set `SYNC_IN_CONTAINER="yes"` in `/etc/default/chrony`. All WSL2 distros share one kernel clock, and a second chronyd outside this distro, most likely Docker Desktop's, already steers it to the Windows clock through the `PHC0` refclock. Letting Ubuntu's chrony steer as well makes the two fight: on 2026-11-01 the clock swung between +9 ms and -289 ms against NTP within seconds.

`chronyc tracking` cannot be trusted alone here. Both instances answer on `127.0.0.1:323`, so replies alternate, and the `PHC0` one compares itself only to Windows. Check the real offset in `lob record status`, which measures against each venue. `ts_recv` follows the Windows clock, and a large offset never stops capture: DQ-15 logs a `clock` interval.

## Check

```bash
~/lob-recorder/venv/bin/lob record status
```

Run commands from `~/lob-recorder`, which holds `.env` and `config/`. Add `--disk` for the measured daily rate and the projection to the planned end date. After the first 24 hours, compare that rate with the budget of 6 GB/day canonical and 6 GB/day raw.

Service state and logs:

```bash
systemctl status lob-recorder
```

```bash
journalctl -u lob-recorder --since "1 hour ago"
```

## Stop and start

`sudo systemctl stop lob-recorder` flushes and closes every file. Any stop or restart appears in the gap log as `process_restart`, and a WSL or Windows restart appears as `host_down`.

A restart of WSL is safe. The unit is enabled, so it starts again with the distro, and the "Start WSL at boot" task starts the distro after a Windows reboot. `.wslconfig` needs both `vmIdleTimeout=-1` and `instanceIdleTimeout=-1`, otherwise WSL can stop the distro when no `wsl.exe` client is attached.

## Deploy a code change

The running service only changes when you do this. Each restart costs a few seconds of capture, which is logged.

```bash
bash deploy/install.sh
```

```bash
sudo systemctl restart lob-recorder
```

## Recover

After a crash, each interrupted hour exists only in the raw archive, as a leftover `*.parquet.tmp`. The recorder rebuilds those at low priority two minutes after it starts. To do it by hand from `~/lob-recorder`:

```bash
venv/bin/lob record recover --from-raw --interrupted
```

For one venue and hour:

```bash
venv/bin/lob record recover --from-raw --venue kraken --date 2026-11-02 --hour 14
```

Closed files are left alone unless you add `--replace`. Recovery needs the raw archive, which is pruned after 7 days, so run it within that window. Do it before relying on a parser fix too: `--replace` re-parses raw frames with the current code.

## Publish the gap log

```bash
uv run lob record gaplog --start 2026-11-02 --end 2026-11-16
```

This writes `reports/gap_log.csv` and `reports/tables/coverage.csv`. Rows with reason `clock` are kept in the log but do not count as downtime, because data still arrived.

## Disk

- Retention: canonical data, gaps, checkpoints and clock are kept. The raw archive is pruned after `LOB_RAW_ARCHIVE_DAYS`.
- Below `LOB_MIN_FREE_GB` free, the raw archive stops.
- Below 5 GB free, all capture stops with a `disk_guard` gap and resumes when space returns.
- At 80% used the recorder logs a warning every minute.

Free space only by the retention rules. D: is a spinning disk shared by all projects.

## Reasons in the gap log

| Reason | Meaning |
|---|---|
| `disconnect` | Connection closed or failed |
| `stale` | 30 s of silence |
| `seq_gap` | Binance `pu` chain broke, or a trade id jumped |
| `checksum_fail` | Kraken CRC32 mismatch |
| `stale_snapshot` | REST snapshot older than the buffered diffs |
| `snapshot_error` | REST failure, or a message with an unknown shape |
| `process_restart` | The recorder restarted; the interval starts up to 10 s early, never late |
| `host_down` | WSL or Windows restarted |
| `disk_guard` | Capture stopped to protect the disk |
| `clock` | Clock offset above 50 ms on 5 probes in a row; capture continued |

## Windows

- Windows Update is paused until 2026-12-06. Extend the pause before then if capture is still running.
- Sleep is off on AC.
