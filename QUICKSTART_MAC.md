# TPR Terminal V4 Alpha 4.1 — Mac Quick Start

1. Stop any older V3/V4 process using the same Discord token.
2. Extract this ZIP into a fresh folder, such as:

```text
Heat map bot V4 Alpha 4
```

3. Copy these optional branding assets into `assets/`:

```text
main.jpg
developer.jpg
trader.jpg
sibei.jpg
jeff.jpg
```

4. Open Terminal and run:

```bash
cd "/Users/cesarpascual/Documents/Heat map bot V4 Alpha 4"
chmod +x install_mac.command run_mac.command
./install_mac.command
```

5. Paste a current Discord token into the hidden token prompt and press Return. No characters will appear while pasting.
6. Launch:

```bash
./run_mac.command
```

7. Test inside the configured server:

```text
/heatmap ticker:SPY preset:swing
```

## Manual launch

```bash
source .venv-tpr/bin/activate
python doctor.py
python main.py
```

## Important

Never paste a Discord token into chat, screenshots, GitHub, or public files. Reset any token that has ever been exposed.
