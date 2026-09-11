# Upgrade from the Alpha 3 Renderer Patch

Keep the Alpha 3 folder untouched as rollback. Alpha 4 should live in its own folder.

Assuming both folders are in Documents:

```bash
cd "/Users/cesarpascual/Documents"
cp "Heat map bot V4 Alpha 3 Renderer Patch/.env" "Heat map bot V4 Alpha 4/.env"
mkdir -p "Heat map bot V4 Alpha 4/assets"
cp -R "Heat map bot V4 Alpha 3 Renderer Patch/assets/." "Heat map bot V4 Alpha 4/assets/"
cd "Heat map bot V4 Alpha 4"
chmod +x install_mac.command run_mac.command
./install_mac.command
```

The installer creates a separate `.venv-tpr` for Alpha 4. Do not run Alpha 3 and Alpha 4 at the same time with the same Discord token.

The first heatmap or Greek request creates:

```text
data/tpr_options_history.sqlite3
```

That file belongs to Alpha 4 and should not be copied back into Alpha 3.
