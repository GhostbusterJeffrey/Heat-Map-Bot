# Changelog

## 4.2.0-alpha.1-dev.1 — Canonical Contract State Foundation

- Added one immutable, snapshot-bound contract ledger for every canonical ticker analysis.
- Added deterministic contract-state IDs tied to the analysis snapshot, expiration, option side, strike, and listed contract symbol.
- Preserved raw bid, ask, last, vendor IV, and contract symbol fields through the heatmap calculation so contract commands no longer need an independent quote fetch.
- Routed `/greek` and `/greeks` through the same canonical snapshot, fitted IV, rate, carry, exact expiration time, and per-contract Greek state used by `/heatmap`.
- Added snapshot and contract-state IDs to Greek-command footers for direct auditability.
- Limited Greek expiration autocomplete to the configured canonical full-chain window.
- Added direct contract-Gamma and contract-GEX consistency tests between Greek reports and heatmap records.
- Added a locked synthetic legacy-GEX regression proving the new ledger does not change the existing production calculation.
- Kept `QUANT_ENGINE_MODE=legacy` as the default and retained shadow/candidate isolation.
- Preserved public-data sign-convention disclosures; no field is described as observed dealer inventory.

## 4.1.0-alpha.4 — Canonical Snapshot Consistency Patch

- Each ticker is now analyzed from one canonical full-chain quote snapshot and one volatility-surface calibration.
- Intraday, swing, macro, and custom expiry counts are derived views of that canonical result rather than independent recalculations.
- Overlapping expiration/strike cells are now bit-for-bit identical across presets that share the same snapshot ID.
- Added a five-minute minimum coherence window for preset-to-preset comparisons.
- Added per-ticker analysis locks so simultaneous member requests cannot create competing canonical snapshots.
- Render caching now keys on the analysis snapshot ID instead of the potentially unchanged underlying quote timestamp.
- Added snapshot IDs to the heatmap intelligence panel and Discord footer for auditable comparisons.
- Recomputed view-level totals, walls, flips, pins, liquidity, confidence, and Greek exposure from the selected expirations without refitting contract IV or Greeks.
- Preserved all existing GEX formulas, Greek calculations, surface math, colors, branding, and ranked-node styling.

## 4.1.0-alpha.3 — Jeff Branding Patch

- Added JEFF THE SCRIPTER as the fourth adaptive header member.
- Added automatic `assets/jeff.jpg`, `.jpeg`, `.png`, or `.webp` discovery.
- Added a gold `JT` monogram fallback when Jeff's image is not installed yet.
- Updated diagnostics, environment documentation, quick start, and branding regression coverage.

## 4.1.0-alpha.2 — Color Intelligence Patch

- Restored the original dark, structural renderer and removed the experimental header gradient.
- Positive exposure now progresses continuously from dark green through light green and near-white below +$1B.
- Positive exposure turns gold at +$1B and bright blue at +$10B.
- Negative exposure now progresses from dark burgundy through increasingly vivid red, then darkens into black at -$1B.
- Restored the original logarithmic $0–$500K micro-opacity behavior and input-quality dimming.
- Preserved the global blue P1/P2/P3 and purple N1/N2/N3 ranked-node outlines.
- Preserved all GEX values, rankings, surfaces, Greek calculations, and analytics behavior.
- Added deterministic regression checks for every requested color transition.

## 4.0.0-alpha.4 — Historical Greek Foundation

- Added automatic SQLite option-chain snapshot capture.
- Added `/greek` and `/greeks` contract analytics commands.
- Added delta, gamma, theta, vega, rho, vanna, charm, vomma, speed, color, zomma, and ultima.
- Added bid/mid/ask-consistent IV and Greek ranges where valid quotes permit.
- Added per-share, per-contract, custom-position, and market-OI scopes.
- Added adaptive finite-difference delta/gamma validation and CRR American price diagnostics.
- Added logarithmic absolute micro-exposure intensity from $0 through ±$500K.
- Preserved fixed institutional-style high-exposure color tiers.
- Added snapshot statistics to `/health`.
- Added `/oichange` for the latest comparable listed-OI snapshots.
- Expanded the automated suite to 36 passing tests.

## 4.0.0-alpha.3.1 — Cyberpunk Renderer Patch

- Replaced percentile-only extreme colors with fixed financial thresholds.
- Positive cells turn gold at +$250M, electric blue at +$1B, and brilliant white at +$10B.
- Negative cells progress from pink to blood red and enter a near-black danger tier at -$500M.
- Extreme negative cells use neon-red text and borders for readability on black.
- Capped the significance floor so meaningful six-figure cells remain visible.
- Added a dedicated adaptive header region with width-aware title and subtitle sizing.
- Lowered the header away from the save boundary and increased output padding.
- Preserved adaptive Viola Trades, JP The Developer, and Sibei Shiok branding.
- Replaced angled numeric expirations with centered horizontal labels such as `Jul 17 '26`.
- Added adaptive expiration-label sizing and nearest-expiration emphasis.
- Added renderer-patch regression tests and an updated synthetic preview.

## 4.0.0-alpha.3 — Surface & Scenario Build

- Added SVI slice calibration and SSVI term-structure calibration.
- Repriced the contract Greek stack from fitted surface IV where available.
- Added vanna, charm, vomma, speed, color, and zomma exposure diagnostics.
- Added sticky-strike and sticky-delta gamma-flip comparison.
- Added gamma-flip model spread and surface-aware shock stability.
- Corrected structural-wall breadth to ignore insignificant expiry contributions.
- Added wall shock stability under spot, volatility, and time perturbations.
- Added explicit expected-move horizon.
- Renamed confidence to model/input confidence in the renderer.
- Added cautious weak/moderate/high pin wording.
- Widened and rescaled the intelligence panel for Discord previews.
- Added adaptive three-person branding and `assets/sibei.*` support.
- Added monogram and corrupt-image fallbacks.
- Added American CRR binomial validation utilities.
- Added Alpha.3 surface, Greek, breadth, and branding tests.
