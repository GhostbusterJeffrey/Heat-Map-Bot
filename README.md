> This sanitized cloud build includes `/netimpact`. Read [CLOUD_UPDATE.md](CLOUD_UPDATE.md) before replacing server files. Keep your existing private environment configuration and persistent data.

# TPR Terminal V4 Alpha 4.2 Dev.1 — Canonical Contract State Foundation

Alpha 4.2 development has started without promoting candidate mathematics to
production. Dev.1 gives `/heatmap`, `/greek`, and `/greeks` one authoritative,
snapshot-bound contract ledger. Every Greek report now uses the exact fitted IV,
rate, carry, expiration time, and Gamma already calculated for its heatmap
snapshot. Snapshot and contract-state IDs make that relationship auditable.

Alpha 4.0 branches from the working Alpha 3 Renderer Patch and adds the first production foundation for contract-level Greek analytics and historical public-options research.

Alpha 4.1 adds an incremental accuracy pipeline without replacing the working
Alpha 4.0 result. The default `legacy` mode is the locked BullFlow-comparison
baseline. `shadow` mode displays that same result while calculating candidate
IV and GEX audit columns. `candidate` explicitly opts into the stricter quote
sanitation and volatility-surface blend.

```env
QUANT_ENGINE_MODE=legacy
```

Use `shadow` while gathering references. Promote `candidate` only after the
regression report demonstrates an improvement on timestamp-aligned comparisons.

## What is live in Alpha 4.2 Dev.1

- Existing SVI/SSVI gamma-exposure heatmap pipeline
- One canonical contract ledger shared by `/heatmap`, `/greek`, and `/greeks`
- Deterministic snapshot-bound contract-state IDs
- Direct contract-Gamma-to-heatmap consistency regression
- Structural walls, gamma flips, pin diagnostics, expected move, and confidence scoring
- Global Discord commands with Printing Room Elite-only access and open access elsewhere
- `/greek` for one selected first-, second-, or third-order Greek
- `/greeks` for the full contract Greek stack
- Exact ticker, expiry, strike, and call/put contract resolution
- Expiry and strike autocomplete
- Bid/mid/ask-consistent implied-volatility intervals when valid quotes permit
- Per-share, per-contract, custom-position, and market-open-interest scopes
- Standardized units for every Greek
- Dollar gamma for a 1% underlying move
- Analytic delta/gamma cross-checks against adaptive finite differences
- CRR American-price diagnostics for equity and ETF options
- Automatic historical option-chain snapshots in SQLite
- `/oichange` snapshot-to-snapshot listed open-interest comparisons
- Absolute logarithmic color intensity from $0 through ±$500K
- Fixed high-exposure semantic tiers retained across tickers
- Canonical contract-state audit fields shared with the heatmap pipeline
- No-arbitrage quote sanitation and diagnostic Brent IV solving
- Per-contract legacy/candidate IV and GEX comparison columns
- Core Delta/Gamma/Theta/Vega/Rho adaptive finite-difference validation
- Higher-Greek bump-stability flags
- Ranked global node outlines: blue `P1/P2/P3` and purple `N1/N2/N3`, with
  the absolute positive maximum and negative minimum receiving the strongest treatment
- BullFlow/reference CSV regression harness

## Main commands

```text
/heatmap ticker:SPY preset:swing
```

```text
/greek
  ticker:SPY
  expiry:2026-07-17
  strike:750
  option_type:call
  greek:gamma
  scope:per_contract
```

```text
/greeks
  ticker:NVDA
  expiry:2026-08-21
  strike:200
  option_type:put
  scope:custom_position
  quantity:10
  position:long
```


```text
/oichange
  ticker:SPY
  expiry:2026-07-17
```

This compares the latest two saved snapshots that contain the requested expiration. It reports listed OI changes, not opening/closing trade flow or dealer ownership.

Supported Greek choices:

```text
delta, gamma, theta, vega, rho,
vanna, charm, vomma,
speed, color, zomma, ultima
```

## Greek scopes

- `per_share`: sensitivity for one option share
- `per_contract`: sensitivity for one standard contract
- `custom_position`: requested contract quantity with long/short sign
- `market_open_interest`: listed-OI aggregate using the public call-positive/put-negative sign proxy

The market-OI scope is not observed dealer inventory.

## Historical snapshot database

Every fetched option chain can be stored locally at:

```text
data/tpr_options_history.sqlite3
```

Configuration:

```env
SNAPSHOT_ENABLED=true
SNAPSHOT_DB_PATH=data/tpr_options_history.sqlite3
SNAPSHOT_RETENTION_DAYS=730
```

The database stores compressed raw call and put chain snapshots by ticker, timestamp, and expiration. This begins accumulating the history required for later OI-change, wall-persistence, and probabilistic positioning research.

## Low-exposure rendering

Values from $0 through ±$500K use a logarithmic absolute intensity curve. This improves separation on lower-GEX tickers without normalizing each chart independently or making small exposures look equivalent to SPY-scale exposures.

Existing high-end meanings remain fixed:

```text
Positive ≥ $1B: gold
Positive ≥ $10B: bright blue
Negative ≥ $1B magnitude: black with readable red treatment
```

## Installation

Alpha 4 should run from its own folder while Alpha 3 remains available as rollback.

```bash
chmod +x install_mac.command run_mac.command
./install_mac.command
./run_mac.command
```

The installer creates `.venv-tpr`, installs dependencies, and opens the secure configuration wizard when `.env` is absent.

Existing branding images can be copied into `assets/` using these stems:

```text
main.jpg
developer.jpg
trader.jpg
sibei.jpg
jeff.jpg
```

JPG, JPEG, PNG, and WebP are supported. Missing images use monogram fallbacks.

## Validation

The packaged automated suite covers:

- Existing Alpha 3 analytics and renderer behavior
- Complete twelve-Greek stack availability
- Contract resolution and quote-IV intervals
- Position-scope scaling
- Numerical delta/gamma cross-validation
- Snapshot database round trips
- Absolute low-exposure intensity behavior
- Discord command registration

Run:

```bash
source .venv-tpr/bin/activate
python -m pytest -q
```

## Accuracy boundary

Alpha 4.0 improves mathematical consistency and begins collecting historical public data. It does not reveal actual dealer books, OTC inventory, opening/closing trade direction, or cross-product hedges.

For American equity and ETF options, Alpha 4.0 reports BSM-derived Greek point estimates while using a CRR tree as an exercise-premium diagnostic. A fully integrated American/discrete-dividend Greek engine remains an Alpha 4.2 milestone.

The strongest defensible description is:

> Institutional-style quantitative analytics built from public market data.
