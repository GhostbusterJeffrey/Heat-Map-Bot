# TPR Terminal V4 Alpha 4.2 Dev.1 Validation

## Automated result

```text
56 passed, 2 subtests passed
```

The suite validates the inherited Alpha 3 analytics pipeline plus the Alpha 4 foundation.

The Alpha 4.1 consistency regression additionally proves that overlapping
expiration/strike cells are bit-for-bit identical when different preset views
are derived from the same canonical snapshot and volatility surface.

The Alpha 4.2 Dev.1 regression additionally proves that:

- Every calculated contract is materialized once in a snapshot-bound ledger.
- Contract-state IDs are deterministic for identical canonical snapshots.
- Expiry-limited views retain the original canonical ledger.
- `/greek` and `/greeks` Gamma comes from the exact corresponding heatmap contract state.
- Market-OI dollar Gamma matches heatmap GEX when reported OI is the selected liquidity source.
- Locked legacy synthetic contract and matrix GEX values remain bit-for-bit unchanged.

## Alpha 4 checks

- All twelve requested Greeks return finite values on deterministic contracts.
- Contract resolution requires an exact listed strike and expiration.
- Bid/mid/ask quote inputs produce IV and Greek intervals when arbitrage-valid.
- Custom-position and market-OI scope scaling is explicit.
- Analytic delta and gamma agree with adaptive finite differences within tolerance.
- SQLite snapshot runs and compressed call/put chains round-trip correctly.
- Latest-snapshot OI comparisons correctly identify call, put, and contract-level changes.
- The low-exposure intensity curve is monotonic from $0 through $500K.
- The renderer uses a fixed $5K display floor rather than chart-relative normalization.
- Discord registers one canonical global command set including `/greek` and `/greeks`.

## Synthetic full-pipeline preview

`preview_v4_alpha4.png` was generated through the surface calibration, Greek repricing, wall/flip/pin analytics, confidence system, and Alpha 4 renderer using deterministic SPY-like data.

Synthetic preview result:

```text
Surface mode: SSVI
Surface score: 100.0
Confidence score: 98.6
```

## Live-data limitation of this build environment

Live Yahoo retrieval and a rendered heatmap were validated on the target Windows
machine before Alpha 4.2 development began. Dev.1 is additionally validated
through deterministic provider fixtures. Run `python doctor.py`, then compare a
live `/heatmap` and `/greeks` request that share a snapshot ID.

## Boundary

Passing tests establish software consistency against the implemented models. They do not establish knowledge of private dealer inventory or guarantee that public Yahoo quotes are complete, current, or redistribution-licensed.
