# BullFlow regression harness

Alpha 4.1 keeps the Alpha 4.0 GEX result as the locked legacy baseline. Reference
comparisons use ordinary CSV files with these columns:

```text
expiry,strike,gex
07/24/2026,340,30400000
07/24/2026,335,28900000
07/24/2026,320,-2400000
```

Capture the ticker, source, spot, and timestamp beside every reference file. A
comparison is meaningful only when the option-chain and spot timestamps are as
close as possible. Do not commit licensed or proprietary reference data unless
its terms allow it.

Run a comparison with:

```bash
python -m tpr_terminal.regression reference.csv candidate.csv
```

For an Alpha 4.1 shadow export containing both engines, select a column with:

```bash
python -m tpr_terminal.regression \
  reference.csv shadow-export.csv \
  --candidate-value candidate_gex
```

The report includes sign accuracy, weighted absolute error, a small-cell-robust
weighted percentage error, strike-rank correlation, top-five overlap, and exact
positive/negative maximum-node agreement. Large nodes matter more than tiny
cells, but no single cell is allowed to determine the full score.
