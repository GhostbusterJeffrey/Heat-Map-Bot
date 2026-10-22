from __future__ import annotations

import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
TEMPLATE_PATH = ROOT / "env.template"


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def main() -> None:
    print("\nTPR TERMINAL V4 ALPHA 4.2 DEV.1 CONFIGURATION\n")
    print("Use a freshly regenerated Discord token. It is written only to your local .env file.")
    print("The token prompt is hidden: paste it, press Return, and no characters will appear.\n")
    token = getpass.getpass("Discord bot token: ").strip()
    if not token or token == "replace_me":
        raise SystemExit("A fresh Discord token is required.")
    guild = ask("Discord server ID")
    role_id = ask("Elite role ID")
    role_name = ask("Elite role name", "Tier 2 Elite Member")
    owner_id = ask("Owner Discord user ID")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    values = {
        "DISCORD_TOKEN": token,
        "DEV_GUILD_ID": guild,
        "ELITE_ROLE_ID": role_id,
        "ELITE_ROLE_NAME": role_name,
        "OWNER_USER_ID": owner_id,
    }
    lines = []
    for line in template.splitlines():
        key, separator, _ = line.partition("=")
        if separator and key.strip() in values:
            line = f"{key.strip()}={values[key.strip()]}"
        lines.append(line)
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        ENV_PATH.chmod(0o600)
    except OSError:
        pass
    print(f"\nSaved configuration to {ENV_PATH}")
    print("Run `python doctor.py`, then `python main.py`.\n")


if __name__ == "__main__":
    main()
