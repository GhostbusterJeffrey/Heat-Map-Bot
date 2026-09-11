# TPR Terminal V4 Alpha 4 Build Specification

## Status

Architecture approved. Implementation must branch from the user's current **V4 Alpha 3 Renderer Patch** source tree, not from an older snapshot.

## Product objective

Build the strongest defensible public-data options analytics engine possible while clearly separating:

1. observed market data,
2. model-derived contract Greeks,
3. aggregate exposure calculations,
4. inferred dealer-position proxies.

No output may describe inferred dealer inventory as observed.

---

## Non-negotiable rules

- Preserve Alpha 3 as a rollback build.
- Never include `.env`, Discord tokens, or credentials in build archives.
- Every exposure value must state its units and shock convention.
- Quality weighting must never silently redefine raw exposure.
- Raw, quality-filtered, and inferred-position outputs must be separately labeled.
- More complex math is accepted only when it improves validation, stability, or known model bias.
- Every generated result must carry a timestamp, data provider, model version, and input-quality report.
- The public-data ceiling must be disclosed: exact dealer books are not observable.

---

## Alpha 4 command design

### Primary contract commands

```text
/greek
ticker:
expiry:
strike:
option_type:
greek:
quantity:
position:
scope:
```

`greek` choices:

```text
delta
gamma
theta
vega
rho
vanna
charm
vomma
speed
color
zomma
ultima
```

### Full contract report

```text
/greeks
ticker:
expiry:
strike:
option_type:
quantity:
position:
scope:
```

### Supported scopes

```text
per_share
per_contract
custom_position
market_open_interest
```

Individual aliases such as `/gamma` may be added later, but `/greek` and `/greeks` are the canonical interfaces to avoid duplicated command logic.

### Contract resolution rules

- Expiry autocomplete must use actual listed expirations.
- Strike autocomplete must depend on ticker and expiry.
- Invalid strikes must not be silently replaced.
- Adjusted/nonstandard contracts must be detected and clearly labeled.
- Option type must be explicit.

---

## Standard Greek conventions

- Delta: option-value sensitivity to a $1 spot move.
- Gamma: delta sensitivity to a $1 spot move.
- Vega: option-value sensitivity to a 1 volatility-point move.
- Theta: option-value sensitivity per stated calendar/trading interval.
- Rho: option-value sensitivity to a 1 percentage-point rate move.
- Vanna: delta sensitivity to a 1 volatility-point move.
- Charm: delta sensitivity over an explicitly stated time interval.
- Vomma: vega sensitivity to a 1 volatility-point move.
- Speed: gamma sensitivity to a $1 spot move.
- Color: gamma sensitivity over an explicitly stated time interval.
- Zomma: gamma sensitivity to a 1 volatility-point move.
- Ultima: vomma sensitivity to a 1 volatility-point move.

All Discord outputs must display the convention used.

---

## Quantitative architecture

### 1. Market-data adapter layer

Create a provider-neutral interface so Yahoo/yfinance can later be replaced or supplemented by licensed data.

Required normalized fields:

```text
ticker
contract_symbol
option_type
strike
expiration
bid
ask
last
volume
open_interest
vendor_iv
quote_timestamp
multiplier
deliverable
exercise_style
settlement_style
```

### 2. Snapshot persistence

Begin storing option-chain snapshots immediately.

Recommended first implementation:

```text
SQLite metadata + compressed Parquet chain snapshots
```

Store:

```text
snapshot_id
ticker
captured_at_utc
provider
spot
rates
dividend assumptions
raw chain
cleaned chain
model version
```

This is required because Yahoo does not provide a reliable historical open-interest database. Probabilistic positioning cannot be responsibly developed without accumulated history.

### 3. Quote cleaning and IV intervals

For each contract:

- validate bid/ask,
- enforce model-independent bounds,
- derive midpoint when valid,
- solve bid-consistent IV,
- solve midpoint IV,
- solve ask-consistent IV,
- calculate spread and liquidity quality,
- record every fallback source.

Return a point estimate and quote-consistent interval.

### 4. Instrument-aware pricing

Model selection:

- European index options: forward Black-Scholes/Black-76 representation.
- American equity/ETF options: production approximation plus binomial/trinomial validation.
- Discrete dividends: explicit schedule when available.
- Exact time to settlement: exchange calendar, early close, AM/PM settlement.

### 5. Volatility surface

- Weighted SVI slices.
- Global SSVI where supported.
- Robust calibration loss.
- Multiple optimizer starts.
- Spread/liquidity weighting.
- Butterfly and calendar diagnostics.
- Weak-slice fallback hierarchy.
- Surface residual and coverage reporting.

### 6. Greek engine

One shared engine must power heatmaps and all Discord Greek commands.

Calculation hierarchy:

1. analytic formula where stable,
2. automatic differentiation,
3. adaptive finite-difference verification.

A result is downgraded or rejected when methods disagree beyond tolerance.

### 7. Scenario engine

Canonical cube:

```text
spot shock × volatility shock × elapsed time
```

Initial grid:

```text
Spot: -3%, -2%, -1%, current, +1%, +2%, +3%
IV:   -3, -1, current, +1, +3 volatility points
Time: now, close, next open, next close
```

Outputs:

- wall persistence,
- gamma-flip persistence,
- wall migration,
- scenario hedge change,
- vanna pressure,
- charm pressure,
- regime-transition probability.

### 8. Positioning layers

Maintain three separate views:

```text
Unsigned gross exposure
Classic public-data call/put sign proxy
Probabilistic dealer-side estimate
```

The probabilistic layer must not be enabled until enough historical snapshots exist for validation.

---

## Low-exposure renderer specification

Preserve fixed high-exposure semantic tiers.

For magnitudes from $0 to $500,000, apply a logarithmic micro-exposure intensity curve:

```python
magnitude = min(abs(value), 500_000.0)
normalized = log1p(magnitude / 5_000.0) / log1p(500_000.0 / 5_000.0)
intensity = clip(0.08 + 0.92 * normalized, 0.08, 1.0)
```

Positive cells transition from near-black green to neon green.

Negative cells transition from near-black burgundy to deep red.

Rules:

- Preserve absolute cross-ticker meaning.
- Do not normalize every chart independently.
- Keep text readable at low intensity.
- Values below the display floor may render as an em dash.
- Quality dimming and magnitude intensity must be distinct calculations.

---

## Build phases

### Alpha 4.0 — Foundation

- Provider-neutral data models.
- Snapshot persistence.
- Shared contract resolver.
- `/greek` and `/greeks`.
- Standardized units.
- Bid/mid/ask intervals.
- Audit metadata.
- Regression tests against Alpha 3.

### Alpha 4.1 — Renderer and diagnostics

- $0–$500K logarithmic micro-exposure intensity.
- Updated legend.
- Improved low-GEX ticker readability.
- Component confidence display.
- Raw versus filtered exposure labeling.

### Alpha 4.2 — Model rigor

- Production American-option model.
- Discrete-dividend schedules.
- Adaptive finite-difference validation.
- Analytic/autodiff/finite-difference agreement tests.
- Exact settlement metadata.

### Alpha 4.3 — Scenario intelligence

- Full spot/IV/time cube.
- Persistent walls.
- Persistent gamma flips.
- Wall migration.
- Regime-transition classification.

### Alpha 4.4 — Historical positioning research

- OI-change database.
- Position persistence.
- Flow-derived features where available.
- Probabilistic sign estimates.
- Classic-versus-probabilistic comparison.
- Calibration and uncertainty intervals.

---

## Acceptance tests

### Numerical

- Black-Scholes benchmark recovery.
- Put-call parity benchmark recovery.
- Analytic/autodiff/finite-difference Greek agreement.
- Greek symmetry and put-call relationships.
- Stable results across adaptive bump sizes.
- American option value not below equivalent European exercise value where applicable.
- Surface diagnostics catch synthetic butterfly/calendar violations.
- Scenario cube reproduces current-state values at zero shock.

### Data quality

- Wide or crossed markets are rejected or downgraded.
- Missing IV paths are labeled.
- Adjusted contracts are not treated as standard 100-share contracts.
- Snapshot writes are atomic.
- No credential can enter logs, fixtures, or archives.

### Discord

- Expiry and strike autocomplete.
- Clear invalid-contract errors.
- Units included in every Greek output.
- Global commands function outside The Printing Room.
- Printing Room access policy remains Elite-only.
- No duplicate commands.

### Renderer

- Low-GEX cells show meaningful separation.
- SPY-scale extreme tiers remain unchanged.
- Positive and negative signs remain visually unmistakable.
- Text remains readable at minimum and maximum intensity.
- Synthetic and saved live fixtures render deterministically.

---

## Deployment rule

Alpha 4 must run as a separate folder/build until:

- all automated tests pass,
- saved live fixtures pass,
- SPY, QQQ, NVDA, TSLA, and AAPL comparisons are reviewed,
- no unexplained Alpha 3 regression remains,
- the existing Alpha 3 folder remains available for rollback.
