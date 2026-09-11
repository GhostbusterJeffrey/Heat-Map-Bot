# Alpha 4.1 Quant Engine audit

## Baseline findings

The production heatmap was already mathematically coherent and its GEX scaling
was correct:

```text
signed contracts × contract multiplier × gamma/share × spot² × 1%
```

Exact seconds to the configured market close, tenor-specific rates, a robust
put-call-parity forward, and an SVI/SSVI surface were also already present.
Those components were preserved.

The main gaps were architectural rather than a missing Black–Scholes formula:

- Contract commands and the heatmap did not share an explicit canonical state.
- Heatmap quote selection accepted weak last-price fallbacks before enforcing
  contract-specific no-arbitrage bounds.
- The IV solver returned only a number, with no convergence or residual audit.
- A successful SVI/SSVI fit replaced every usable raw IV rather than blending
  market and surface estimates according to input quality.
- Adaptive finite-difference validation covered Delta and Gamma only.
- Higher-order Greeks returned a value without checking bump-size stability.
- There was no formal reference-map regression harness.

## Implemented in this preview

1. `legacy`, `shadow`, and `candidate` rollout modes.
2. A scalar canonical `ContractState` plus matching heatmap audit columns.
3. No-arbitrage quote sanitation with explicit mark source and warnings.
4. Diagnostic Brent IV solving with residual, iteration count, and status.
5. Quality-aware market/surface IV blending in candidate mode.
6. Canonical GEX scaling in one tested function.
7. Adaptive price-difference validation for Delta, Gamma, Theta, Vega, and Rho.
8. Multi-bump stability checks for Vanna, Charm, Vomma, Speed, Color, Zomma,
   and Ultima.
9. Legacy/candidate GEX audit columns on every contract in shadow mode.
10. Heatmap aggregation and scaling regression tests.
11. Reference-map metrics for signs, weighted errors, ranks, top-node overlap,
    and positive/negative maximum agreement.
12. Ranked blue `P1/P2/P3` and purple `N1/N2/N3` global-node highlights
    without changing cell values, opacity, or ranking inputs.

## Safety rule

`QUANT_ENGINE_MODE=legacy` remains the default. Shadow mode was tested to match
the legacy heatmap exactly on a deterministic option chain. Candidate mode must
not become the default until real, timestamp-aligned reference samples show a
repeatable improvement rather than a cosmetic or theoretical change.

## Deliberately deferred

- Automatic differentiation as a third independent derivative engine
- Production American-option Greeks with discrete dividend schedules
- Full no-static-arbitrage SSVI constraints for every sparse-chain edge case
- Licensed contract metadata for adjusted deliverables and settlement styles
- Dealer-side inference beyond the public call-positive/put-negative proxy

These are separate validation projects. They should not be rushed into the
benchmark formula that is already matching the reference maps closely.
