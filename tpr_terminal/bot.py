from __future__ import annotations

import asyncio
import io
import logging
import platform
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Literal

import discord
from discord import app_commands
from discord.ext import commands

from . import __version__
from .cache import TTLCache
from .models import AnalysisResult, ContractGreekReport, RenderedHeatmap
from .netimpact_service import NetImpactService
from .service import TerminalService
from .settings import Settings
from .snapshot_store import OIComparison

logger = logging.getLogger(__name__)
MODULE_SETTINGS = Settings.from_env()
TICKER_PATTERN = re.compile(r"^[A-Z0-9.^=\-]{1,15}$")
PresetChoice = Literal["intraday", "swing", "macro"]
OptionTypeChoice = Literal["call", "put"]
GreekChoice = Literal[
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "vanna",
    "charm",
    "vomma",
    "speed",
    "color",
    "zomma",
    "ultima",
]
ScopeChoice = Literal["per_share", "per_contract", "custom_position", "market_open_interest"]
PositionChoice = Literal["long", "short"]


class TPRTerminalBot(commands.Bot):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            command_prefix=commands.when_mentioned_or("!"),
            intents=discord.Intents.default(),
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.settings = settings
        self.started_at = time.monotonic()
        self.service = TerminalService(settings)
        self.netimpact_service: NetImpactService | None = None
        self.analysis_executor = ThreadPoolExecutor(
            max_workers=settings.analysis_workers,
            thread_name_prefix="tpr-analysis",
        )
        self.render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tpr-render")
        self.analysis_semaphore = asyncio.Semaphore(settings.max_concurrent_analyses)
        self.render_semaphore = asyncio.Semaphore(1)
        self.analysis_cache: TTLCache[tuple[object, ...], AnalysisResult] = TTLCache(
            settings.result_cache_ttl_seconds, max_entries=128
        )
        # Keep one full-chain calculation coherent long enough for members to
        # compare intraday/swing/macro views without silently changing snapshots.
        self.canonical_analysis_cache: TTLCache[tuple[object, ...], AnalysisResult] = TTLCache(
            max(settings.result_cache_ttl_seconds, 300), max_entries=64
        )
        self._canonical_analysis_locks: dict[str, asyncio.Lock] = {}
        self.render_cache: TTLCache[tuple[object, ...], RenderedHeatmap] = TTLCache(
            settings.result_cache_ttl_seconds, max_entries=128
        )

    async def setup_hook(self) -> None:
        await self.add_cog(TerminalCog(self))

        # DEV_GUILD_ID doubles as the Printing Room's restricted guild ID.
        # Remove any older guild-only copies first so Discord shows one canonical
        # global command set rather than duplicate global + guild commands.
        if self.settings.dev_guild_id:
            printing_room = discord.Object(id=self.settings.dev_guild_id)
            self.tree.clear_commands(guild=printing_room)
            await self.tree.sync(guild=printing_room)
            logger.info(
                "Cleared guild-specific commands from Printing Room guild %s",
                printing_room.id,
            )

        # Global commands are available in every server where the bot is installed.
        synced = await self.tree.sync()
        logger.info("Synced %d global commands for multi-server use", len(synced))

    async def on_ready(self) -> None:
        logger.info(
            "TPR Terminal v%s online as %s (%s)",
            __version__,
            self.user,
            getattr(self.user, "id", "unknown"),
        )

    async def get_analysis(self, ticker: str, expiries: int) -> tuple[AnalysisResult, str]:
        canonical_key = (
            "canonical",
            ticker,
            self.settings.max_expiries,
            "auto",
            "net_proxy",
        )
        canonical = self.canonical_analysis_cache.get(canonical_key)
        cache_status = "HIT"
        if canonical is None:
            lock = self._canonical_analysis_locks.setdefault(ticker, asyncio.Lock())
            async with lock:
                # A concurrent request may have completed while this request waited.
                canonical = self.canonical_analysis_cache.get(canonical_key)
                if canonical is None:
                    loop = asyncio.get_running_loop()
                    async with self.analysis_semaphore:
                        canonical = await loop.run_in_executor(
                            self.analysis_executor,
                            lambda: self.service.analyze(
                                ticker,
                                expiries=self.settings.max_expiries,
                                liquidity_mode="auto",
                                exposure_mode="net_proxy",
                            ),
                        )
                    self.canonical_analysis_cache.set(canonical_key, canonical)
                    cache_status = "MISS"

        view_key = (
            "view",
            ticker,
            canonical.snapshot_id,
            int(expiries),
            "auto",
            "net_proxy",
        )
        cached_view = self.analysis_cache.get(view_key)
        if cached_view is not None:
            return cached_view, "HIT"
        result = self.service.analysis_view(canonical, expiries=int(expiries))
        self.analysis_cache.set(view_key, result)
        return result, cache_status

    async def get_render(
        self,
        result: AnalysisResult,
        expiries: int,
        strikes: int,
    ) -> tuple[RenderedHeatmap, str]:
        key = (result.ticker, expiries, strikes, result.snapshot_id)
        cached = self.render_cache.get(key)
        if cached is not None:
            return cached, "HIT"
        loop = asyncio.get_running_loop()
        async with self.render_semaphore:
            rendered = await loop.run_in_executor(
                self.render_executor,
                lambda: self.service.render(result, strikes),
            )
        self.render_cache.set(key, rendered)
        return rendered, "MISS"

    async def get_contract_greeks(
        self,
        ticker: str,
        *,
        expiration: date,
        strike: float,
        option_side: OptionTypeChoice,
        quantity: int,
        position: PositionChoice,
        scope: ScopeChoice,
    ) -> ContractGreekReport:
        canonical, _ = await self.get_analysis(
            ticker,
            self.settings.max_expiries,
        )
        loop = asyncio.get_running_loop()
        async with self.analysis_semaphore:
            return await loop.run_in_executor(
                self.analysis_executor,
                lambda: self.service.contract_greeks(
                    ticker,
                    expiration=expiration,
                    strike=strike,
                    option_side=option_side,
                    quantity=quantity,
                    position=position,
                    scope=scope,
                    canonical=canonical,
                ),
            )

    async def list_expirations(self, ticker: str) -> list[date]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.analysis_executor,
            lambda: self.service.list_expirations(
                ticker,
                limit=self.settings.max_expiries,
            ),
        )

    async def list_strikes(
        self,
        ticker: str,
        *,
        expiration: date,
        option_side: OptionTypeChoice,
    ) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.analysis_executor,
            lambda: self.service.list_strikes(
                ticker,
                expiration=expiration,
                option_side=option_side,
                limit=500,
            ),
        )

    async def get_oi_comparison(
        self,
        ticker: str,
        *,
        expiration: date | None = None,
    ) -> OIComparison:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.analysis_executor,
            lambda: self.service.compare_open_interest(ticker, expiration=expiration),
        )

    def has_terminal_access(self, interaction: discord.Interaction) -> bool:
        # The configured owner always bypasses role restrictions.
        if self.settings.owner_user_id and interaction.user.id == self.settings.owner_user_id:
            return True

        guild_id = interaction.guild_id
        if guild_id is None:
            return False

        # DEV_GUILD_ID is the Printing Room server. Every other server is open
        # access, regardless of the member's role there.
        printing_room_id = self.settings.dev_guild_id
        if printing_room_id is None or guild_id != printing_room_id:
            return True

        # Inside The Printing Room, enforce the configured Elite role.
        if not isinstance(interaction.user, discord.Member):
            return False
        if self.settings.elite_role_id:
            return any(role.id == self.settings.elite_role_id for role in interaction.user.roles)
        return any(role.name == self.settings.elite_role_name for role in interaction.user.roles)

    async def close(self) -> None:
        self.analysis_executor.shutdown(wait=False, cancel_futures=True)
        self.render_executor.shutdown(wait=False, cancel_futures=True)
        await super().close()


class TerminalCog(commands.Cog):
    def __init__(self, bot: TPRTerminalBot) -> None:
        self.bot = bot

    async def _authorize(self, interaction: discord.Interaction) -> bool:
        if not self.bot.has_terminal_access(interaction):
            role = (
                f"role ID `{self.bot.settings.elite_role_id}`"
                if self.bot.settings.elite_role_id
                else f"**{self.bot.settings.elite_role_name}**"
            )
            await interaction.response.send_message(
                f"Inside The Printing Room, this command is exclusive to {role}.",
                ephemeral=True,
            )
            return False
        return True

    @app_commands.command(name="netimpact", description="Rank tracked near-term options by recorded net premium.")
    @app_commands.guild_only()
    @app_commands.checks.cooldown(
        1, MODULE_SETTINGS.cooldown_seconds,
        key=lambda interaction: (interaction.guild_id, interaction.user.id),
    )
    @app_commands.describe(
        window="Recorded session, last 15 minutes, or last 60 minutes",
        mode="Flow uses recorded executions; close_estimate is call/put activity with unknown trade direction",
    )
    async def netimpact(
        self, interaction: discord.Interaction,
        window: Literal["session", "15m", "60m"] = "session",
        mode: Literal["flow", "close_estimate"] = "flow",
    ) -> None:
        if not await self._authorize(interaction):
            return
        await interaction.response.defer(thinking=True)
        try:
            if self.bot.netimpact_service is None:
                self.bot.netimpact_service = NetImpactService()
            service = self.bot.netimpact_service
            loop = asyncio.get_running_loop()
            async with self.bot.analysis_semaphore:
                report = await loop.run_in_executor(
                    self.bot.analysis_executor,
                    lambda: service.report(window, mode),
                )
            async with self.bot.render_semaphore:
                png = await loop.run_in_executor(
                    self.bot.render_executor,
                    lambda: service.render(report),
                )
            if len(png) > int(self.bot.settings.max_upload_mb * 1024 * 1024):
                raise ValueError("Net-impact chart exceeds the configured upload limit.")
            filename = f"tpr_netimpact_{report['session_date']}_{mode}.png"
            estimate = mode == "close_estimate"
            embed = discord.Embed(
                title="TPR • Close Activity Estimate" if estimate else "TPR • Net Impact",
                description=(
                    "Daily call/put volume valued at final observed marks. **Trade direction is unknown; this is not signed execution flow.**"
                    if estimate else
                    f"{report.get('scope', '')}\nStatus: **{report.get('status', 'unknown')}**"
                ),
                color=0xD8B45A if estimate else 0x2DB98E,
            )
            embed.set_image(url=f"attachment://{filename}")
            embed.set_footer(text=f"{report['session_date']} • {window} • Tradier • Recorded coverage only")
            await interaction.followup.send(embed=embed, file=discord.File(io.BytesIO(png), filename=filename))
        except Exception as exc:
            # Never route provider/network exceptions through the generic traceback logger.
            logger.warning("Net-impact request failed (%s)", type(exc).__name__)
            await interaction.followup.send(
                "Net impact is unavailable. Check the local Tradier setup and collector. "
                "Close estimates require the current session to have closed and window:session.",
                ephemeral=True,
            )

    @app_commands.command(
        name="heatmap",
        description="Generate the canonical TPR V4 Alpha 4.2 development gamma matrix.",
    )
    @app_commands.guild_only()
    @app_commands.checks.cooldown(
        1,
        MODULE_SETTINGS.cooldown_seconds,
        key=lambda interaction: (interaction.guild_id, interaction.user.id),
    )
    @app_commands.describe(
        ticker="Underlying symbol, such as SPY, AAPL, TSLA, or BRK-B",
        preset="Intraday, swing, or macro layout",
        expiries="Optional override for number of future expirations",
        strikes="Optional override for strikes centered around spot",
    )
    async def heatmap(
        self,
        interaction: discord.Interaction,
        ticker: str,
        preset: PresetChoice = "swing",
        expiries: app_commands.Range[int, 1, 24] | None = None,
        strikes: app_commands.Range[int, 11, 101] | None = None,
    ) -> None:
        if not await self._authorize(interaction):
            return
        symbol = self._ticker(ticker)
        if symbol is None:
            await interaction.response.send_message("Invalid ticker format.", ephemeral=True)
            return
        expiry_count, strike_count = self._resolve_preset(preset, expiries, strikes)
        if expiry_count > self.bot.settings.max_expiries or strike_count > self.bot.settings.max_strikes:
            await interaction.response.send_message(
                f"Configured limits: {self.bot.settings.max_expiries} expiries and "
                f"{self.bot.settings.max_strikes} strikes.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)
        request_id = str(interaction.id)
        try:
            result, analysis_cache = await self.bot.get_analysis(symbol, expiry_count)
            rendered, render_cache = await self.bot.get_render(result, expiry_count, strike_count)
            size_limit = int(self.bot.settings.max_upload_mb * 1024 * 1024)
            if len(rendered.png_bytes) > size_limit:
                raise ValueError("Chart exceeds the configured upload limit; request fewer cells.")
            filename = f"{symbol.lower().replace('^', '')}_tpr_v4_gex.png"
            file = discord.File(io.BytesIO(rendered.png_bytes), filename=filename)
            embed = self._heatmap_embed(
                rendered,
                filename,
                f"A:{analysis_cache} R:{render_cache}",
                preset,
            )
            await interaction.followup.send(embed=embed, file=file)
        except Exception as exc:
            await self._failure(interaction, request_id, symbol, exc)

    @app_commands.command(
        name="levels",
        description="Show TPR gamma walls, dynamic gamma flip, and confidence.",
    )
    @app_commands.guild_only()
    @app_commands.checks.cooldown(1, MODULE_SETTINGS.cooldown_seconds)
    async def levels(
        self,
        interaction: discord.Interaction,
        ticker: str,
        expiries: app_commands.Range[int, 1, 16] = MODULE_SETTINGS.default_expiries,
    ) -> None:
        if not await self._authorize(interaction):
            return
        symbol = self._ticker(ticker)
        if symbol is None:
            await interaction.response.send_message("Invalid ticker format.", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            result, cache = await self.bot.get_analysis(symbol, int(expiries))
            await interaction.followup.send(embed=self._levels_embed(result, cache))
        except Exception as exc:
            await self._failure(interaction, str(interaction.id), symbol, exc)

    @app_commands.command(
        name="expectedmove",
        description="Estimate an expiration's cleaned ATM options-implied move.",
    )
    @app_commands.guild_only()
    @app_commands.checks.cooldown(1, MODULE_SETTINGS.cooldown_seconds)
    @app_commands.describe(expiry_number="1 = nearest expiration, 2 = next expiration, etc.")
    async def expectedmove(
        self,
        interaction: discord.Interaction,
        ticker: str,
        expiry_number: app_commands.Range[int, 1, 12] = 1,
    ) -> None:
        if not await self._authorize(interaction):
            return
        symbol = self._ticker(ticker)
        if symbol is None:
            await interaction.response.send_message("Invalid ticker format.", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            result, cache = await self.bot.get_analysis(symbol, int(expiry_number))
            if len(result.expected_moves) < int(expiry_number):
                raise ValueError("That expiration does not have enough usable quote data.")
            move = result.expected_moves[int(expiry_number) - 1]
            embed = discord.Embed(
                title=f"TPR V4 // {result.ticker} Expected Move",
                color=0xB000FF,
                description=(
                    f"Expiration: **{move.expiration:%A, %B %d, %Y}**\n"
                    f"Spot: **${result.spot:,.2f}**\n"
                    f"Move: **±${move.dollars:,.2f} ({move.percent:.2%})**\n"
                    f"Range: **${move.lower:,.2f} — ${move.upper:,.2f}**\n"
                    f"Method: `{move.method}`\n"
                    f"Confidence: **{result.confidence.display}**"
                ),
            )
            embed.set_footer(text=f"Cache: {cache} | Market-implied estimate, not a guarantee.")
            await interaction.followup.send(embed=embed)
        except Exception as exc:
            await self._failure(interaction, str(interaction.id), symbol, exc)

    @app_commands.command(
        name="summary",
        description="Generate a compact V4 quantitative positioning brief.",
    )
    @app_commands.guild_only()
    @app_commands.checks.cooldown(1, MODULE_SETTINGS.cooldown_seconds)
    async def summary(
        self,
        interaction: discord.Interaction,
        ticker: str,
        expiries: app_commands.Range[int, 1, 16] = MODULE_SETTINGS.default_expiries,
    ) -> None:
        if not await self._authorize(interaction):
            return
        symbol = self._ticker(ticker)
        if symbol is None:
            await interaction.response.send_message("Invalid ticker format.", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            result, cache = await self.bot.get_analysis(symbol, int(expiries))
            await interaction.followup.send(embed=self._summary_embed(result, cache))
        except Exception as exc:
            await self._failure(interaction, str(interaction.id), symbol, exc)

    @app_commands.command(
        name="greek",
        description="Calculate one quote-aware Greek for a listed option contract.",
    )
    @app_commands.guild_only()
    @app_commands.checks.cooldown(1, MODULE_SETTINGS.cooldown_seconds)
    @app_commands.describe(
        ticker="Underlying symbol",
        expiry="Listed expiration in YYYY-MM-DD format",
        strike="Exact listed strike",
        option_type="Call or put",
        greek="Greek to calculate",
        scope="Per share, per contract, custom position, or market OI proxy",
        quantity="Contract count used only for custom_position",
        position="Long or short, used only for custom_position",
    )
    async def greek(
        self,
        interaction: discord.Interaction,
        ticker: str,
        expiry: str,
        strike: float,
        option_type: OptionTypeChoice,
        greek: GreekChoice = "gamma",
        scope: ScopeChoice = "per_contract",
        quantity: app_commands.Range[int, 1, 10000] = 1,
        position: PositionChoice = "long",
    ) -> None:
        if not await self._authorize(interaction):
            return
        symbol = self._ticker(ticker)
        expiration = self._expiry(expiry)
        if symbol is None or expiration is None:
            await interaction.response.send_message(
                "Use a valid ticker and an expiry formatted as `YYYY-MM-DD`.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(thinking=True)
        try:
            report = await self.bot.get_contract_greeks(
                symbol,
                expiration=expiration,
                strike=float(strike),
                option_side=option_type,
                quantity=int(quantity),
                position=position,
                scope=scope,
            )
            await interaction.followup.send(embed=self._greek_embed(report, greek))
        except Exception as exc:
            await self._contract_failure(interaction, symbol, exc)

    @app_commands.command(
        name="greeks",
        description="Calculate the full Alpha 4 Greek stack for a listed option contract.",
    )
    @app_commands.guild_only()
    @app_commands.checks.cooldown(1, MODULE_SETTINGS.cooldown_seconds)
    @app_commands.describe(
        ticker="Underlying symbol",
        expiry="Listed expiration in YYYY-MM-DD format",
        strike="Exact listed strike",
        option_type="Call or put",
        scope="Per share, per contract, custom position, or market OI proxy",
        quantity="Contract count used only for custom_position",
        position="Long or short, used only for custom_position",
    )
    async def greeks(
        self,
        interaction: discord.Interaction,
        ticker: str,
        expiry: str,
        strike: float,
        option_type: OptionTypeChoice,
        scope: ScopeChoice = "per_contract",
        quantity: app_commands.Range[int, 1, 10000] = 1,
        position: PositionChoice = "long",
    ) -> None:
        if not await self._authorize(interaction):
            return
        symbol = self._ticker(ticker)
        expiration = self._expiry(expiry)
        if symbol is None or expiration is None:
            await interaction.response.send_message(
                "Use a valid ticker and an expiry formatted as `YYYY-MM-DD`.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(thinking=True)
        try:
            report = await self.bot.get_contract_greeks(
                symbol,
                expiration=expiration,
                strike=float(strike),
                option_side=option_type,
                quantity=int(quantity),
                position=position,
                scope=scope,
            )
            await interaction.followup.send(embed=self._greeks_embed(report))
        except Exception as exc:
            await self._contract_failure(interaction, symbol, exc)

    @app_commands.command(
        name="oichange",
        description="Compare open interest across the two latest saved chain snapshots.",
    )
    @app_commands.guild_only()
    @app_commands.checks.cooldown(1, MODULE_SETTINGS.cooldown_seconds)
    @app_commands.describe(
        ticker="Underlying symbol",
        expiry="Optional listed expiration in YYYY-MM-DD format",
    )
    async def oichange(
        self,
        interaction: discord.Interaction,
        ticker: str,
        expiry: str | None = None,
    ) -> None:
        if not await self._authorize(interaction):
            return
        symbol = self._ticker(ticker)
        expiration = self._expiry(expiry) if expiry else None
        if symbol is None or (expiry and expiration is None):
            await interaction.response.send_message(
                "Use a valid ticker and optional expiry formatted as `YYYY-MM-DD`.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(thinking=True)
        try:
            comparison = await self.bot.get_oi_comparison(symbol, expiration=expiration)
            await interaction.followup.send(embed=self._oi_change_embed(comparison))
        except Exception as exc:
            logger.exception("OI comparison failed | user=%s ticker=%s", interaction.user.id, symbol)
            detail = str(exc) if isinstance(exc, ValueError) else "Internal snapshot-comparison failure."
            await interaction.followup.send(
                f"⚠️ Unable to compare saved OI for **{symbol}**.\n{detail}",
                ephemeral=True,
            )

    @oichange.autocomplete("expiry")
    @greek.autocomplete("expiry")
    @greeks.autocomplete("expiry")
    async def expiry_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        ticker = self._ticker(str(getattr(interaction.namespace, "ticker", "")))
        if ticker is None:
            return []
        try:
            expirations = await asyncio.wait_for(self.bot.list_expirations(ticker), timeout=2.5)
        except Exception:
            return []
        query = current.strip()
        values = [value for value in expirations if not query or query in value.isoformat()]
        return [
            app_commands.Choice(name=value.strftime("%b %d, %Y"), value=value.isoformat())
            for value in values[:25]
        ]

    @greek.autocomplete("strike")
    @greeks.autocomplete("strike")
    async def strike_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[float]]:
        ticker = self._ticker(str(getattr(interaction.namespace, "ticker", "")))
        expiration = self._expiry(str(getattr(interaction.namespace, "expiry", "")))
        option_side = str(getattr(interaction.namespace, "option_type", "call"))
        if ticker is None or expiration is None or option_side not in {"call", "put"}:
            return []
        try:
            strikes = await asyncio.wait_for(
                self.bot.list_strikes(
                    ticker,
                    expiration=expiration,
                    option_side=option_side,
                ),
                timeout=2.5,
            )
        except Exception:
            return []
        query = current.strip()
        if query:
            try:
                target = float(query)
                strikes.sort(key=lambda value: abs(value - target))
            except ValueError:
                strikes = [value for value in strikes if query in f"{value:g}"]
        return [
            app_commands.Choice(name=f"{value:g}", value=float(value))
            for value in strikes[:25]
        ]

    @app_commands.command(name="health", description="Show TPR Terminal system status.")
    @app_commands.guild_only()
    async def health(self, interaction: discord.Interaction) -> None:
        uptime = int(time.monotonic() - self.bot.started_at)
        embed = discord.Embed(title="TPR Terminal Health", color=0x39FF14)
        snapshot_stats = self.bot.service.snapshot_stats()
        snapshot_text = (
            "disabled"
            if snapshot_stats is None
            else (
                f"{snapshot_stats.runs:,} runs / {snapshot_stats.chains:,} chains / "
                f"{snapshot_stats.contracts:,} contracts"
            )
        )
        embed.description = (
            f"Status: **ONLINE**\n"
            f"Version: **{__version__}**\n"
            f"Latency: **{self.bot.latency * 1000:.0f} ms**\n"
            f"Uptime: **{uptime // 3600}h {(uptime % 3600) // 60}m**\n"
            f"Python: **{platform.python_version()}**\n"
            f"Quant engine: **{self.bot.settings.quant_engine_mode.upper()}**\n"
            f"Analysis cache: **{len(self.bot.analysis_cache)} entries** "
            f"({self.bot.analysis_cache.hits} hits / {self.bot.analysis_cache.misses} misses)\n"
            f"Render cache: **{len(self.bot.render_cache)} entries**\n"
            f"Historical snapshots: **{snapshot_text}**"
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def cog_app_command_error(
        self,
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        if isinstance(error, app_commands.CommandOnCooldown):
            message = f"Please retry in **{error.retry_after:.1f}s**."
        elif isinstance(error, app_commands.NoPrivateMessage):
            message = "TPR Terminal commands only work inside the server."
        else:
            logger.error(
                "Application command error: %s",
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
            message = "The command could not be started. Check the bot log."
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)

    async def _contract_failure(
        self,
        interaction: discord.Interaction,
        ticker: str,
        exc: Exception,
    ) -> None:
        logger.exception(
            "Contract Greek request failed | user=%s ticker=%s",
            interaction.user.id,
            ticker,
        )
        detail = str(exc) if isinstance(exc, ValueError) else "Internal contract-analytics failure."
        await interaction.followup.send(
            f"⚠️ Unable to calculate that contract for **{ticker}**.\n{detail}",
            ephemeral=True,
        )

    async def _failure(
        self,
        interaction: discord.Interaction,
        request_id: str,
        ticker: str,
        exc: Exception,
    ) -> None:
        logger.exception(
            "Request failed | id=%s user=%s ticker=%s",
            request_id,
            interaction.user.id,
            ticker,
        )
        reason = "insufficient usable options data" if isinstance(exc, ValueError) else "internal analytics failure"
        await interaction.followup.send(
            content=(
                f"⚠️ Unable to generate a reliable heatmap for **{ticker}**.\n"
                f"Reason: {reason}. Request ID: `{request_id}`"
            ),
            ephemeral=True,
        )

    @staticmethod
    def _ticker(raw: str) -> str | None:
        value = raw.strip().upper().replace("/", "-")
        return value if TICKER_PATTERN.fullmatch(value) else None

    @staticmethod
    def _expiry(raw: str) -> date | None:
        try:
            return datetime.strptime(raw.strip(), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None

    def _resolve_preset(
        self,
        preset: PresetChoice,
        expiries: int | None,
        strikes: int | None,
    ) -> tuple[int, int]:
        defaults = {
            "intraday": (self.bot.settings.intraday_expiries, self.bot.settings.intraday_strikes),
            "swing": (self.bot.settings.swing_expiries, self.bot.settings.swing_strikes),
            "macro": (self.bot.settings.macro_expiries, self.bot.settings.macro_strikes),
        }
        default_expiries, default_strikes = defaults[preset]
        return int(expiries or default_expiries), int(strikes or default_strikes)

    @classmethod
    def _heatmap_embed(
        cls,
        rendered: RenderedHeatmap,
        filename: str,
        cache: str,
        preset: str,
    ) -> discord.Embed:
        result = rendered.result
        embed = discord.Embed(
            title=f"TPR V4 Alpha 4.2 Dev.1 // {result.ticker} Gamma Exposure Matrix",
            description=(
                "Market-derived IV, parity-implied carry, structural walls, dynamic multi-flip "
                "repricing, and scenario-based pin diagnostics. Net exposure remains an "
                "estimated public-data sign-convention proxy."
            ),
            color=0x00F0FF,
        )
        embed.set_image(url=f"attachment://{filename}")
        embed.add_field(
            name="Underlying",
            value=(
                f"Spot: **${result.spot:,.2f}**\n"
                f"Preset: **{preset}**\n"
                f"Displayed strikes: **{rendered.displayed_strikes}**"
            ),
            inline=True,
        )
        flip = result.levels.primary_gamma_flip
        pin = result.levels.primary_pin
        embed.add_field(
            name="Structural positioning",
            value=(
                f"Regime: **{result.levels.regime}**\n"
                f"Primary flip: **{cls._price(None if flip is None else flip.level)}**\n"
                f"Pin estimate: **{cls._price(None if pin is None else pin.strike)}**\n"
                f"Total: **{cls._amount(result.levels.total_net_gex)}**"
            ),
            inline=True,
        )
        embed.add_field(
            name="Model quality",
            value=(
                f"Confidence: **{result.confidence.display}**\n"
                f"Surface: **{result.surface_health.display}**\n"
                f"Quant engine: **{result.quant_engine_mode.upper()}**\n"
                f"Liquidity: **{result.liquidity.mode}**\n"
                f"Fallback exposure: **{result.liquidity.volume_exposure_share:.1%}**"
            ),
            inline=False,
        )
        if result.interpretation:
            embed.add_field(
                name="TPR computed brief",
                value="\n".join(f"• {line}" for line in result.interpretation[:5]),
                inline=False,
            )
        embed.set_footer(
            text=(
                f"Provider: {result.provider_name} | Cache {cache} | "
                f"Snapshot {result.snapshot_id} | V4 Alpha 4.2 Dev.1 | "
                "Analytics only; not investment advice."
            )
        )
        return embed

    @classmethod
    def _levels_embed(cls, result: AnalysisResult, cache: str) -> discord.Embed:
        levels = result.levels
        all_flips = ", ".join(cls._price(value) for value in levels.gamma_flips) or "—"
        embed = discord.Embed(
            title=f"TPR V4 Alpha 4.2 Dev.1 // {result.ticker} Structural Gamma Levels",
            color=0x00F0FF if levels.total_net_gex >= 0 else 0xFF0055,
        )
        embed.description = (
            f"Spot: **${result.spot:,.2f}**\n"
            f"Regime: **{levels.regime}**\n"
            f"Positive wall: **{cls._wall(levels.positive_wall_strike, levels.positive_wall_value)}**\n"
            f"Negative wall: **{cls._wall(levels.negative_wall_strike, levels.negative_wall_value)}**\n"
            f"Primary gamma flip: **{cls._price(levels.gamma_flip)}**\n"
            f"All detected flips: **{all_flips}**\n"
            f"Total net GEX proxy: **{cls._amount(levels.total_net_gex)}**\n"
            f"Confidence: **{result.confidence.display}**"
        )
        embed.add_field(name="Data quality", value=result.quality.compact_summary(), inline=False)
        embed.set_footer(text=f"Cache: {cache} | Net values are a public-data sign-convention proxy.")
        return embed

    @classmethod
    def _summary_embed(cls, result: AnalysisResult, cache: str) -> discord.Embed:
        levels = result.levels
        move = result.expected_moves[0] if result.expected_moves else None
        move_text = (
            f"±${move.dollars:,.2f} ({move.percent:.2%}) through {move.expiration:%m/%d} close"
            if move
            else "Unavailable"
        )
        reason_text = "\n".join(f"• {reason}" for reason in result.confidence.reasons[:3])
        embed = discord.Embed(
            title=f"TPR V4 Alpha 4.2 Dev.1 Intelligence Brief // {result.ticker}",
            color=0xB000FF,
            description=(
                f"**Spot:** ${result.spot:,.2f}\n"
                f"**Gamma regime:** {levels.regime}\n"
                f"**Positive wall:** {cls._wall(levels.positive_wall_strike, levels.positive_wall_value)}\n"
                f"**Negative wall:** {cls._wall(levels.negative_wall_strike, levels.negative_wall_value)}\n"
                f"**Gamma flip:** {cls._price(levels.gamma_flip)}\n"
                f"**Nearest expected move:** {move_text}\n"
                f"**Net GEX proxy:** {cls._amount(levels.total_net_gex)}\n"
                f"**Confidence:** {result.confidence.display}\n\n{reason_text}"
            ),
        )
        embed.add_field(name="Quality", value=result.quality.compact_summary(), inline=False)
        embed.set_footer(text=f"Provider: {result.provider_name} | Cache: {cache}")
        return embed

    @classmethod
    def _oi_change_embed(cls, comparison: OIComparison) -> discord.Embed:
        color = 0x39FF14 if comparison.total_oi_change >= 0 else 0xFF0055
        embed = discord.Embed(
            title=(
                f"TPR Alpha 4 OI Change // {comparison.ticker} "
                f"{comparison.expiration:%b %d, %Y}"
            ),
            color=color,
            description=(
                f"Older snapshot: **{comparison.older_at:%Y-%m-%d %H:%M UTC}**\n"
                f"Newer snapshot: **{comparison.newer_at:%Y-%m-%d %H:%M UTC}**\n"
                f"Contracts compared: **{comparison.contracts_compared:,}**\n\n"
                f"Call OI change: **{comparison.call_oi_change:+,}**\n"
                f"Put OI change: **{comparison.put_oi_change:+,}**\n"
                f"Total OI change: **{comparison.total_oi_change:+,}**"
            ),
        )
        if comparison.top_increases:
            embed.add_field(
                name="Largest OI increases",
                value="\n".join(cls._oi_change_line(item) for item in comparison.top_increases),
                inline=False,
            )
        if comparison.top_decreases:
            embed.add_field(
                name="Largest OI decreases",
                value="\n".join(cls._oi_change_line(item) for item in comparison.top_decreases),
                inline=False,
            )
        embed.set_footer(
            text=(
                "Snapshot-to-snapshot listed OI change. OI is vendor-reported, often updates daily, "
                "and does not identify opening/closing flow or dealer ownership."
            )
        )
        return embed

    @staticmethod
    def _oi_change_line(item) -> str:
        return (
            f"**{item.side[0].upper()} ${item.strike:g}:** "
            f"{item.old_open_interest:,} → {item.new_open_interest:,} "
            f"(**{item.change:+,}**)"
        )

    @classmethod
    def _greek_embed(cls, report: ContractGreekReport, greek: str) -> discord.Embed:
        band = report.values[greek]
        color = 0x39FF14 if band.mid >= 0 else 0xFF0055
        embed = discord.Embed(
            title=(
                f"TPR Alpha 4.2 Dev.1 // {report.ticker} ${report.strike:g} "
                f"{report.option_side.upper()} {greek.upper()}"
            ),
            color=color,
            description=(
                f"Expiration: **{report.expiration:%B %d, %Y}**\n"
                f"Spot: **${report.spot:,.2f}**\n"
                f"Scope: **{report.scope.replace('_', ' ')}**\n\n"
                f"**{cls._format_band(band)}**\n"
                f"_{band.unit}_"
                + (
                    "\n⚠️ This higher-order sensitivity failed the configured bump-stability threshold."
                    if report.quant_engine_mode == "candidate" and not band.stable
                    else ""
                )
            ),
        )
        if greek == "gamma":
            embed.add_field(
                name="Dollar gamma / 1% move",
                value=cls._format_band(report.dollar_gamma_1pct, currency=True),
                inline=False,
            )
        embed.add_field(
            name="Quote and IV",
            value=cls._quote_iv_text(report),
            inline=False,
        )
        embed.add_field(
            name="Numerical validation",
            value=cls._validation_text(report),
            inline=False,
        )
        if report.warnings:
            embed.add_field(
                name="Model disclosure",
                value="\n".join(f"• {warning}" for warning in report.warnings[:3]),
                inline=False,
            )
        state_text = (
            ""
            if report.snapshot_id is None
            else f" | Snapshot {report.snapshot_id} | State {report.contract_state_id}"
        )
        embed.set_footer(
            text=(
                f"{report.provider_name} | {report.model_label} | "
                f"Model-derived analytics, not observed dealer inventory.{state_text}"
            )
        )
        return embed

    @classmethod
    def _greeks_embed(cls, report: ContractGreekReport) -> discord.Embed:
        embed = discord.Embed(
            title=(
                f"TPR Alpha 4.2 Dev.1 Greek Stack // {report.ticker} ${report.strike:g} "
                f"{report.option_side.upper()}"
            ),
            color=0x00F0FF,
            description=(
                f"Expiration: **{report.expiration:%B %d, %Y}**  •  "
                f"Spot: **${report.spot:,.2f}**\n"
                f"Scope: **{report.scope.replace('_', ' ')}**  •  "
                f"Validation: **{report.validation_label.upper()} "
                f"({report.validation_error:.3%})**"
            ),
        )
        groups = (
            ("Core Greeks", ("delta", "gamma", "theta", "vega", "rho")),
            ("Cross Greeks", ("vanna", "charm", "vomma")),
            ("Third-order Greeks", ("speed", "color", "zomma", "ultima")),
        )
        for title, names in groups:
            lines = [
                f"**{name.title()}:** {cls._format_number(report.values[name].mid)}"
                + (
                    " ⚠️"
                    if report.quant_engine_mode == "candidate"
                    and not report.values[name].stable
                    else ""
                )
                for name in names
            ]
            embed.add_field(name=title, value="\n".join(lines), inline=False)
        embed.add_field(
            name="Dollar gamma / 1% move",
            value=cls._format_band(report.dollar_gamma_1pct, currency=True),
            inline=False,
        )
        embed.add_field(name="Quote and IV", value=cls._quote_iv_text(report), inline=False)
        if report.warnings:
            embed.add_field(
                name="Model disclosure",
                value="\n".join(f"• {warning}" for warning in report.warnings[:3]),
                inline=False,
            )
        state_text = (
            ""
            if report.snapshot_id is None
            else f" | Snapshot {report.snapshot_id} | State {report.contract_state_id}"
        )
        embed.set_footer(
            text=(
                f"{report.provider_name} | {report.model_label} | "
                f"Use /greek for a single sensitivity with units and interval.{state_text}"
            )
        )
        return embed

    @classmethod
    def _quote_iv_text(cls, report: ContractGreekReport) -> str:
        bid = "—" if report.bid is None else f"${report.bid:,.2f}"
        ask = "—" if report.ask is None else f"${report.ask:,.2f}"
        last = "—" if report.last_price is None else f"${report.last_price:,.2f}"
        iv_bid = "—" if report.implied_volatility_bid is None else f"{report.implied_volatility_bid:.2%}"
        iv_ask = "—" if report.implied_volatility_ask is None else f"{report.implied_volatility_ask:.2%}"
        text = (
            f"Bid / Ask / Last: **{bid} / {ask} / {last}**\n"
            f"IV bid / mid / ask: **{iv_bid} / {report.implied_volatility_mid:.2%} / {iv_ask}**\n"
            f"IV source: **{report.implied_volatility_source}**  •  "
            f"Quote quality: **{report.quote_quality:.0%}**\n"
            f"OI: **{report.open_interest:,}**  •  Volume: **{report.volume:,}**"
        )
        if report.quant_engine_mode != "legacy":
            mark = "—" if report.selected_mark is None else f"${report.selected_mark:,.3f}"
            forward = "—" if report.forward_price is None else f"${report.forward_price:,.3f}"
            legacy_iv = (
                "—"
                if report.legacy_implied_volatility is None
                else f"{report.legacy_implied_volatility:.2%}"
            )
            candidate_iv = (
                "—"
                if report.candidate_implied_volatility is None
                else f"{report.candidate_implied_volatility:.2%}"
            )
            text += (
                f"\nSelected mark: **{mark} ({report.mark_source})**  •  Forward: **{forward}**"
                f"\nLegacy / candidate IV: **{legacy_iv} / {candidate_iv}**"
            )
        return text

    @staticmethod
    def _validation_text(report: ContractGreekReport) -> str:
        if report.quant_engine_mode == "legacy":
            return (
                f"**{report.validation_label.upper()}** — maximum delta/gamma "
                f"finite-difference disagreement {report.validation_error:.3%}"
            )
        details = "  •  ".join(
            f"{name.title()} {error:.3%}"
            for name, error in report.core_validation_errors.items()
        )
        unstable = (
            "None"
            if not report.unstable_greeks
            else ", ".join(name.title() for name in report.unstable_greeks)
        )
        return (
            f"**{report.validation_label.upper()}** — maximum core-Greek disagreement "
            f"{report.validation_error:.3%}\n{details}\n"
            f"Unstable higher Greeks: **{unstable}**"
        )

    @classmethod
    def _format_band(cls, band, *, currency: bool = False) -> str:
        mid = cls._format_number(band.mid, currency=currency)
        if band.low is None or band.high is None:
            return mid
        low = cls._format_number(band.low, currency=currency)
        high = cls._format_number(band.high, currency=currency)
        return f"{mid}\nQuote-consistent range: {low} — {high}"

    @staticmethod
    def _format_number(value: float, *, currency: bool = False) -> str:
        magnitude = abs(value)
        sign = "+" if value >= 0 else "-"
        if currency:
            if magnitude >= 1e9:
                return f"{sign}${magnitude / 1e9:.3f}B"
            if magnitude >= 1e6:
                return f"{sign}${magnitude / 1e6:.3f}M"
            if magnitude >= 1e3:
                return f"{sign}${magnitude / 1e3:.3f}K"
            if magnitude != 0 and magnitude < 1e-4:
                return f"{sign}${magnitude:.6e}"
            return f"{sign}${magnitude:,.6f}".rstrip("0").rstrip(".")
        if magnitude != 0 and (magnitude < 1e-4 or magnitude >= 1e7):
            return f"{value:+.6e}"
        return f"{value:+,.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _amount(value: float | None) -> str:
        if value is None:
            return "—"
        sign = "+" if value >= 0 else "-"
        magnitude = abs(value)
        if magnitude >= 1e9:
            return f"{sign}${magnitude / 1e9:.2f}B"
        if magnitude >= 1e6:
            return f"{sign}${magnitude / 1e6:.2f}M"
        if magnitude >= 1e3:
            return f"{sign}${magnitude / 1e3:.2f}K"
        return f"{sign}${magnitude:.0f}"

    @staticmethod
    def _price(value: float | None) -> str:
        return "—" if value is None else f"${value:,.2f}"

    @classmethod
    def _wall(cls, strike: float | None, value: float | None) -> str:
        return "—" if strike is None else f"${strike:,.2f} ({cls._amount(value)})"
