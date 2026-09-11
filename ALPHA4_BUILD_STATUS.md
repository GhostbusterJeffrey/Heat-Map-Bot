# Alpha 4 Build Status

## Implemented in 4.0

- Historical SQLite option-chain snapshots
- Basic `/oichange` snapshot comparison
- Provider wrapper for automatic capture
- `/greek` and `/greeks`
- Twelve-Greek contract stack
- Bid/mid/ask IV intervals
- Position and OI scaling
- Numerical delta/gamma verification
- American price diagnostics
- $0–$500K logarithmic renderer intensity
- Updated health diagnostics and automated tests

## Deliberately deferred

- Full production American/discrete-dividend Greeks
- Automatic differentiation library integration
- Full spot × IV × time scenario cube
- Multi-day OI-change features and persistence models
- Probabilistic dealer-side inference
- Parquet archival compaction

These items require validation and accumulated snapshots rather than being safely rushed into the first Alpha 4 build.
