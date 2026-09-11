# Upgrade from Alpha.2

1. Stop Alpha.2.
2. Keep the entire Alpha.2 folder as a rollback copy.
3. Extract Alpha.3 into a new folder.
4. Copy only your branding images from the old `assets/` folder.
5. Do not copy an old `.venv-tpr` environment.
6. You may copy `.env` only when its Discord token has never been exposed. Otherwise reset the token and run the installer.
7. Run `./install_mac.command` and then `./run_mac.command`.
8. Compare SPY, QQQ, TSLA, NVDA, and AAPL against Alpha.2 before replacing the live production build.
