# Reference hardware

Recorded 2026-11-01. Every hardware-dependent NFR number in this repo refers to this machine.

| Item | Value |
|---|---|
| CPU | Intel Core i9-14900K, 32 threads visible to WSL |
| RAM | 62 GB available to WSL2 |
| OS | Windows 11 Pro 10.0.26200, WSL 3.0.3.0, kernel 6.18.54.1 |
| Distro | Ubuntu 22.04.5 LTS, systemd running |
| Data disk | `D:\WSL\Ubuntu\ext4.vhdx` on an internal 2 TB SATA HDD, about 903 GB free on D: |
| Python | 3.12 through `uv` 0.12.23 |
| Clock | See below |
| Network | Median RTT on 2026-11-01: Binance REST 460 ms, Coinbase REST 228 ms, Kraken websocket 197 ms |

## Clock

The figure that matters is `System time` in `chronyc tracking`, which says how far the system clock is from NTP. `Last offset` and `RMS offset` describe chrony's own estimate and look healthy even when chrony cannot steer the clock.

Until chrony controls the clock, `ts_recv` follows the Windows clock. On 2026-11-01 that was 165 to 170 ms fast and chrony estimated 6.7 ppm of drift. Median feed latency measured then was Binance 298 ms (including 100 ms batching), Kraken 253 ms and Coinbase 242 ms, and each figure includes that error, so they overstate true latency.
