# V4 Alpha.3.1 Cyberpunk Renderer Patch

This patch changes rendering semantics only. It does not alter SVI/SSVI calibration, Greek calculations, structural walls, gamma flips, pin scoring, expected moves, or confidence formulas.

## Fixed positive tiers

| Exposure | Meaning | Appearance |
|---:|---|---|
| below $250M | ordinary to strong positive | dark green through neon lime |
| $250M–$1B | exceptional positive | gold |
| $1B–$10B | billion-dollar positive | electric neon blue |
| $10B+ | God level | brilliant white with cyan border |

## Fixed negative tiers

| Exposure magnitude | Meaning | Appearance |
|---:|---|---|
| below $300M | ordinary to major negative | pink through blood red |
| $300M–$500M | approaching danger | deep dark red |
| $500M–$1B | Oh-shit tier | near black with neon-red text |
| $1B+ | extreme negative | pure black with neon-red text and border |

## Header and expiration labels

- Title and subtitle are rendered inside a dedicated header band.
- Font sizes adjust to image width and available space before the contributor block.
- Branding scales automatically for Viola Trades, JP The Developer, and Sibei Shiok.
- Expirations are centered horizontally using `Mon D 'YY`, for example `Jul 17 '26`.
- The nearest expiration is slightly brighter than later expirations.

## Upgrade

Replace the existing Alpha.3 folder with this full build, or replace `tpr_terminal/renderer.py` and restart the bot. Keep your existing `.env` and assets private when moving to the new folder.
