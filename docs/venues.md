# Venues

Endpoints, channels and sync rules as verified on **2026-11-01**, against the official docs and against live frames from this PC. Re-check this file whenever a stream fails to reach LIVE. Choices that differ from the spec are explained in [decisions.md](decisions.md).

## binance-usdm (primary)

| Item | Value |
|---|---|
| Depth | `wss://fstream.binance.com/public/stream?streams=<s>@depth@100ms/...` |
| bbo | `wss://fstream.binance.com/public/stream?streams=<s>@bookTicker/...`, a separate connection from depth |
| Trades | `wss://fstream.binance.com/market/stream?streams=<s>@aggTrade/...` |
| REST | `https://fapi.binance.com`: `/fapi/v1/depth?limit=1000` (weight 20), `/fapi/v1/time`, `/fapi/v1/exchangeInfo` |
| Limits | 24 h connection lifetime; server ping every 3 min, pong within 10 min; 10 incoming messages/s; 1024 streams per connection; 2400 request weight/min (from `exchangeInfo`) |

Sync rule: buffer diffs from subscribe, fetch a REST snapshot, drop events with `u < lastUpdateId`, the first applied event has `U <= lastUpdateId <= u`, then each `pu` must equal the previous `u`. Quantities are absolute, zero removes a level, and removing an unknown level is normal.

Field meanings: depth `T` is transaction time and `E` event time. `bookTicker` `T`/`u`. `aggTrade` `a` is the aggregate id, `T` the trade time; `m=true` means the buyer is the maker, so the seller aggressed (`side = -1`).

Notes from live frames: the legacy `/ws` and `/stream` paths are retired, and this project uses only the routed paths. `aggTrade` frames also carry `nq`, and frames carry `st`; neither is used. The REST depth endpoint excludes RPI orders. `exchangeInfo.serverTime` is cached and stale, so it is never used for the clock.

Docs:
- <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>
- <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Important-WebSocket-Change-Notice>
- <https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Connect>
- <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Order-Book>

## kraken

| Item | Value |
|---|---|
| Websocket | `wss://ws.kraken.com/v2`, channels `book` (depth 10, 25, 100, 500 or 1000) and `trade` |
| REST | `https://api.kraken.com`: `/0/public/AssetPairs` |
| Clock | WS `{"method":"ping","req_id":<send ns>}`, pong carries `time_in` and `time_out` in microseconds |

Sync rule: subscribe, receive a `snapshot` on the websocket, verify its CRC32, then every `update` has a CRC32 over the top 10 levels regardless of subscribed depth. Process all updates in a message, truncate to the subscribed depth, then compute the checksum. `qty: 0` removes a level; no removal is sent for levels that fall out of scope.

Checksum: asks low to high, then bids high to low, top 10 of each. For each level take the price text then the quantity text with the `.` and leading zeros removed. CRC32 of the concatenation, unsigned. Doc worked example: 3310070434, committed in `tests/unit/test_book.py`. Numbers on the wire keep trailing zeros (`0.00006120`), and they must be read as text.

Field meanings: `trade.side` is the taker side. `trade_id` is "unique per book". `AssetPairs.wsname` says `XBT/USD` while WS v2 publishes `BTC/USD`.

Docs:
- <https://docs.kraken.com/api/docs/websocket-v2/book>
- <https://docs.kraken.com/api/docs/guides/spot-ws-book-v2>
- <https://docs.kraken.com/api/docs/websocket-v2/trade>

## coinbase

| Item | Value |
|---|---|
| Websocket | `wss://ws-feed.exchange.coinbase.com`, channels `level2_batch` (the server reports it as `level2_50`), `matches`, `heartbeat`. No auth for these. `level2` and `full` need auth. |
| REST | `https://api.exchange.coinbase.com`: `/products/{id}/book?level=2` (whole aggregated book, about 1 MB for BTC-USD, rows are `[price, size, num_orders]`), `/products/{id}`, `/time` |
| Limits | 10 inbound messages/s with bursts to 1000; 10 subscriptions per product per channel per account |

Sync rule: the snapshot arrives on the websocket after subscribe and replaces the book. `l2update.changes` are `[side, price, size]`, `size` is absolute, `"0"`-valued sizes remove the level. Batches arrive every 50 ms. There is no sequence on L2, so integrity rests on the heartbeat, the message-age watchdog and REST checkpoints every 600 s.

Field meanings: `match.side` is the **maker** side, so `side=sell` is buyer-initiated (`+1`). `sequence` counts the product's whole feed and is not contiguous across matches. `trade_id` is contiguous. The first message on `matches` is `last_match`, which is not written as a trade. The WS snapshot has no `time` field, so its `ts_exchange` falls back to `ts_recv`.

Docs:
- <https://docs.cdp.coinbase.com/exchange/websocket-feed/channels>
- <https://docs.cdp.coinbase.com/exchange/websocket-feed/rate-limits>
- <https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-book>

## Fees entered in `config/instruments.yaml`

Read 2026-11-01. Base tier for each venue.

| Venue | Maker bps | Taker bps | Source |
|---|---|---|---|
| binance-usdm | 2.0 | 5.0 | <https://www.binance.com/en/fee/futureFee> (USDT, Regular User) |
| coinbase | 40.0 | 60.0 | <https://help.coinbase.com/en/exchange/trading-and-funding/exchange-fees> ($0-$10K) |
| kraken | 40.0 | 80.0 | <https://www.kraken.com/features/fee-schedule> (Kraken Pro, lowest tier) |
