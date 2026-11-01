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
| Clock | chrony, stratum 1, offset under 0.1 ms to the hypervisor clock |
| Network | Median feed latency from a 25 s smoke on 2026-11-01: Binance 298 ms (includes 100 ms batching), Kraken 253 ms, Coinbase 242 ms |
