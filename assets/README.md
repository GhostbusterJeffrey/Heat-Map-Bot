# Branding assets

Place permanent image files in this folder. The loader accepts `.jpg`, `.jpeg`, `.png`, or `.webp` and discovers each asset by stem.

- `main.*` — The Printing Room logo
- `developer.*` — JP The Developer
- `trader.*` — Viola Trades
- `sibei.*` — Sibei Shiok

Recommended minimum: 500×500 pixels. Wide images are center-cropped to a square without stretching. Missing or unreadable member images automatically use a colored monogram, so rendering never fails.

The displayed order is:

1. Viola Trades
2. JP The Developer
3. Sibei Shiok

`calendar_overrides.json` supports exceptional exchange dates:

```json
{
  "closed": ["2027-01-02"],
  "closes": {"2027-07-02": "13:00"}
}
```
