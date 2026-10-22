# /netimpact — 20-ticker near-term build

The Discord command and local chart command use the same service and renderer. Existing heatmap commands remain available. Tradier collection runs in one separate process, so Discord requests never create competing streams.

## Modes

- `/netimpact` reads observed executions for the latest applicable session.
- `/netimpact window:15m` and `window:60m` use execution timestamps within the selected window. After close, windows end at the session close.
- `/netimpact mode:close_estimate` creates an explicitly labeled after-close **activity estimate** from daily call/put volume times final observed quote midpoints. It is not a reconstruction of buys/sells, traded premium, or Unusual Whales' net impact. It supports `window:session` only, on the current trading date after regular close.

Missing data is unavailable, never fabricated zero. Live mode never silently falls back to an estimate. Unknown trade conditions are excluded. Ambiguous cancellations or corrections make the affected ticker unavailable until reconciliation rules are verified.

## Default coverage

SPY, QQQ, IWM, TSLA, NVDA, AMD, AAPL, MSFT, META, AMZN, GOOGL, AVGO, ORCL, PLTR, MU, NFLX, COIN, MSTR, INTC, CRM.

For each ticker: the nearest available expiry 0–7 calendar days away, 11 listed strikes centered on the selected underlying price, calls and puts. Up to 440 standard option contracts. Missing paired strikes or unsupported adjusted contracts reduce coverage and are recorded. This is a defined watchlist slice, not whole-market or full-chain flow.

Live selection uses the observed underlying price when the collector establishes its session cohort; that cohort is frozen and persisted for the day. Close previews use date-matched underlying daily closes where available, with provenance recorded. After-close expired contracts may no longer be returned by Tradier, so preview selection can differ from a live cohort chosen at the open.

## Local setup

Use Python 3.12–3.14 and install `requirements.txt`. Copy `env.template` to `.env` for a new installation, or retain your existing Discord configuration and add the net-impact fields. Never share `.env` or include it in archives.

Set either `TRADIER_ACCESS_TOKEN` in the private environment or `TRADIER_ENV_FILE` to a separate private file containing `TRADIER_ACCESS_TOKEN=...`. `NETIMPACT_TICKERS` is configurable up to 20 unique names. `NETIMPACT_DB_PATH` must be the same for the collector and bot.

Run the collector in its own terminal/process:

```console
python -m tpr_terminal.netimpact_cli collect
```

Run the Discord bot separately using the project's normal launch command. The collector must remain awake and connected during trading hours; one collector per Tradier account. An OS-held lock prevents competing collectors for the same database. All bot users read the shared ledger.

The collector checks Tradier's clock, observes holidays/early closes, selects contracts, opens one streaming session and reconnects with bounded backoff. Startup and reconnect gaps are recorded. V1 uses the common equity session, normally 09:30–16:00 New York time, excluding ETF options executions after that cutoff. It does not claim to backfill missing trade-side history.

The initial cohort requires multiple requests after the open, so initial startup coverage is explicitly partial. A later refinement can preselect contracts before the open; it must still confirm the final cohort and subscription before claiming a full session.

## Local chart command

```console
python -m tpr_terminal.netimpact_cli chart --mode close_estimate --output close.png --report close.json
python -m tpr_terminal.netimpact_cli chart --window session --output flow.png
python -m tpr_terminal.netimpact_cli status
```

The preview JSON includes source contract snapshots and quote timestamps for audit/re-rendering. Only option bid/ask marks within 15 minutes of regular session close qualify; missing or stale required marks make the ticker unavailable. Marks are asynchronous and may include ETF quotes up to 15 minutes after the common cutoff. Zero-volume contracts contribute zero to this activity estimate.

## Calculation and integrity

Execution premium = price × contracts × metadata multiplier. Calls at/near ask add; calls at/near bid subtract. Puts at/near ask subtract; puts at/near bid add. Midpoint, invalid/locked/crossed quotes, out-of-session events, missing identity fields and unrecognized flags are excluded. These are execution-side inferences, not proof of opening trades or trader intent.

Raw trade records, excluded reasons, source and receipt timestamps, duplicates, gaps and cohort metadata are kept in a separate SQLite ledger. Duplicate identities do not add twice. Conflicting payloads and unresolved amendments suppress the affected ticker rather than quietly preserving a questionable bar. There is no per-print $2m cap; UW's exact Top Net Impact filtering is not verified.

Quote traffic refreshes stream health without adding premium. The feed and data timestamps stay separate. Collection beginning late or encountering gaps is labeled partial; a stale feed remains labeled stale. No confidence score is generated.

## Deployment and remaining validation

The supplied Compose configuration runs the bot and collector with a shared persistent `data/` directory. Supply the Tradier token as an environment value inside Docker, or mount a private token file and set its container path. Do not put credentials in an image. The Docker ignore file excludes secrets and collected data.

An account connection is not authorization to redistribute Tradier data to members. Confirm the appropriate provider permission before a shared-server rollout; the present artifacts support private development and validation.

Market-hours validation remains necessary for real option timesale fields, condition codes, sequence stability, latency, subscription completeness and the 440-contract load. Unit tests and after-close snapshots cannot establish those properties. No live Discord deployment or unattended service is started by building this package.
