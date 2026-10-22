# Cloud update: Alpha 4.2 Dev1 + /netimpact

This package starts from your sanitized Alpha 4.2 Dev1 build, preserves its pinned dependency versions, and adds `/netimpact` with the 20-ticker near-term setup. The existing bot entrypoint is still `python main.py`.

## Keep your server settings and data

1. Back up the running bot's private environment settings and persistent data before replacing application files.
2. Keep your server's real Discord token, role IDs, guild ID, and other settings. The supplied `.env.example` and `env.template` are blank configuration templates; do not use them to overwrite a working server `.env`.
3. Add `TRADIER_ACCESS_TOKEN` using your hosting provider's secret settings or the server's private `.env`. Leave `TRADIER_ENV_FILE` blank unless you deliberately mount a separate secret file at a path accessible inside the container.
4. Persist the entire `data` directory, including the SQLite database, its journal files, and collector state. If the old bot stores data only inside a container, copy that data out before replacing the container or mounting a new directory over it.

New settings and defaults are listed in the environment templates. Both processes must use the same `NETIMPACT_DB_PATH` (default `data/netimpact.sqlite3`) and ticker configuration.

## Run the bot and the collector

The bot answers commands:

```sh
python main.py
```

The separate collector records live option executions:

```sh
python -m tpr_terminal.netimpact_cli collect
```

Run one collector replica. The two processes must have access to the same persistent data directory on a host/storage setup that supports SQLite locking. Starting only the bot enables close estimates but does not accumulate live flow.

The included `docker-compose.yml` already defines both services and mounts the same host `./data` directory into both containers. Keep the private `.env` beside the Compose file on the server. After preserving existing settings and data, rebuild and start with your normal deployment process; for Docker Compose:

```sh
docker compose up -d --build
```

For hosts without Compose, configure the two start commands as separate processes sharing the same persistent directory. Do not point each process at a separate local disk.

## Use the command

- `/netimpact`: recorded signed executions for the current or latest recorded session.
- `/netimpact window:15m` or `/netimpact window:60m`: a shorter window of recorded executions.
- `/netimpact mode:close_estimate`: an after-close options activity estimate for the current trading date. It uses final chain volume and eligible closing quote marks; it is explicitly labeled as an estimate and does not reconstruct actual trade direction.

The default universe is 20 tickers, the nearest expiration within seven calendar days, and up to 11 paired strikes per ticker (440 call/put contracts total). Missing, stale, partial, and unclassifiable data are labeled rather than presented as complete flow. Collection cannot recover executions from before the collector started.

## Validation and package contents

The development build passed its local tests, a real 20-ticker close-estimate check, and a local invocation of the Discord command callback. See `NETIMPACT_VALIDATION.json` and `NETIMPACT.md` for details. Live execution classification and full subscription load still need market-hours validation. This package has not been deployed or built as a Linux image here.

Credentials, deployment IDs, local secret-file paths, databases, logs, virtual environments, and caches are excluded. Install the supplied pinned `requirements.txt` using the existing server deployment process. This update adds `/netimpact`; it does not change the existing heatmap confidence display or king-node colors.
