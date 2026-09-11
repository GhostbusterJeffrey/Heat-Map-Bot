# Alpha 4.2 Quant Precision Development

## Current milestone

Version: `4.2.0-alpha.1-dev.1`

Release name: Canonical Contract State Foundation

This milestone changes the calculation architecture without replacing the
locked legacy production mathematics. `QUANT_ENGINE_MODE=legacy` remains the
default BullFlow-comparison mode.

## What Dev.1 establishes

- One canonical full-chain analysis per ticker and coherence window.
- One immutable ledger entry for every calculated listed contract.
- A deterministic contract-state ID beneath the existing snapshot ID.
- Shared quote, spot, time, rate, carry, IV, Greek, liquidity, and GEX state.
- `/greek` and `/greeks` resolve from the same state used by `/heatmap`.
- Expiry-limited heatmap views retain the full canonical ledger without
  refitting or repricing shared contracts.
- Bid/ask Greek intervals may be recalculated for display, but the reported
  midpoint Gamma is the canonical heatmap Gamma.

## Preserved boundaries

- The public net-GEX view remains a call-positive/put-negative convention proxy.
- Reported open interest does not identify dealer ownership.
- Volume fallback is provisional activity, not position inventory.
- Legacy, shadow, and candidate quant-engine modes remain separate.
- Candidate mathematics is not a production default.
- Heatmap colors, micro-opacity, P1/P2/P3, N1/N2/N3, branding, commands, access
  controls, and disclosures remain intact.

## Promotion gates for later Alpha 4.2 mathematics

1. Deterministic unit and regression tests.
2. Same-snapshot contract and heatmap consistency.
3. Legacy-versus-candidate shadow exports.
4. Timestamp-aligned BullFlow observations where available.
5. Out-of-sample quote validation.
6. Explicit review before changing the production default.

## Windows validation command

From this project directory:

```powershell
$testTemp = Join-Path $PWD ".test-temp"
New-Item -ItemType Directory -Force -Path $testTemp | Out-Null
$env:TEMP = $testTemp
$env:TMP = $testTemp
$env:MPLCONFIGDIR = Join-Path $PWD ".matplotlib-cache"
.\.venv-tpr\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp "$testTemp\pytest-run"
```

## Startup

```powershell
.\.venv-tpr\Scripts\python.exe doctor.py
.\.venv-tpr\Scripts\python.exe main.py
```

Do not run two bot processes with the same Discord token simultaneously.
