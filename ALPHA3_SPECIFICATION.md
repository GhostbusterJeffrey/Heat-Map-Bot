# TPR Terminal V4 Alpha.3 Implementation Boundary

## Implemented

- Quote cleaning and market-price IV solving inherited from Alpha.2
- Parity-implied forward/carry estimates
- Weighted raw-SVI calibration per expiration
- Global SSVI calibration with sufficient-condition penalties
- Fallback to valid SVI slices or raw IV when calibration is weak
- Surface-derived Greek stack
- Sticky-strike and sticky-delta gamma-flip calculations
- Surface-aware pin delta scenarios
- Meaningful expiration-breadth thresholds
- Wall shock stability
- Expected-move horizon
- Adaptive three-member branding

## Deliberately not claimed

- Observed dealer inventory
- Formal proof that every fitted surface is fully arbitrage-free
- Live OPRA-grade quotes or trade classification
- Full production American/discrete-dividend repricing for every contract
- Buy/sell and open/close participant inference

Alpha.3's SSVI sufficient-condition checks and numerical diagnostics are strong engineering controls, but they are not a substitute for a licensed feed and an independently audited institutional risk library.
