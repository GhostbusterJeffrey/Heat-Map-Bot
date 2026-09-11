from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .models import BrandMember

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True, slots=True)
class BrandAsset:
    member: BrandMember
    image: np.ndarray
    path: Path | None
    is_placeholder: bool
    width: int
    height: int
    warning: str | None = None


DEFAULT_MEMBERS = (
    BrandMember(
        key="trader",
        display_name="VIOLA TRADES",
        asset_stem="trader",
        accent="#b000ff",
        priority=1,
    ),
    BrandMember(
        key="developer",
        display_name="JP THE DEVELOPER",
        asset_stem="developer",
        accent="#ff0055",
        priority=2,
    ),
    BrandMember(
        key="sibei",
        display_name="SIBEI SHIOK",
        asset_stem="sibei",
        accent="#00f0ff",
        priority=3,
    ),
    BrandMember(
        key="jeff",
        display_name="JEFF THE SCRIPTER",
        asset_stem="jeff",
        accent="#ffd700",
        priority=4,
    ),
)


def discover_asset(assets_dir: Path, stem: str) -> Path | None:
    for extension in SUPPORTED_EXTENSIONS:
        candidate = assets_dir / f"{stem}{extension}"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    clean = value.lstrip("#")
    if len(clean) != 6:
        return 50, 50, 60
    return tuple(int(clean[index : index + 2], 16) for index in (0, 2, 4))


def _monogram(member: BrandMember, size: int = 512) -> np.ndarray:
    background = _hex_to_rgb(member.accent)
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    initials = "".join(word[0] for word in member.display_name.split() if word)[:2]
    try:
        from matplotlib import font_manager

        font_path = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight="bold")
        )
        font = ImageFont.truetype(font_path, int(size * 0.38))
    except Exception:
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), initials, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text(
        ((size - width) / 2, (size - height) / 2 - size * 0.03),
        initials,
        fill=(5, 5, 10),
        font=font,
    )
    return np.asarray(image)


def _load_square(path: Path, size: int = 512) -> tuple[np.ndarray, int, int, str | None]:
    warning: str | None = None
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source).convert("RGBA")
        width, height = source.size
        if width < 500 or height < 500:
            warning = "Image is below the recommended 500px minimum on one dimension."
        ratio = max(width / max(height, 1), height / max(width, 1))
        if ratio > 4.0:
            warning = "Image has an extreme aspect ratio and will be center-cropped."
        fitted = ImageOps.fit(source, (size, size), method=Image.Resampling.LANCZOS)
        background = Image.new("RGBA", fitted.size, (7, 7, 9, 255))
        background.alpha_composite(fitted)
        return np.asarray(background.convert("RGB")), width, height, warning


class BrandingManager:
    def __init__(
        self,
        assets_dir: Path,
        *,
        members: tuple[BrandMember, ...] = DEFAULT_MEMBERS,
    ) -> None:
        self.assets_dir = Path(assets_dir)
        self.members = tuple(sorted(members, key=lambda member: member.priority))
        self._assets = tuple(self._load_member(member) for member in self.members)
        self.main_logo = self._load_main()

    def _load_main(self) -> np.ndarray | None:
        path = discover_asset(self.assets_dir, "main")
        if path is None:
            return None
        try:
            image, _, _, warning = _load_square(path)
            if warning:
                logger.warning("Branding asset %s: %s", path, warning)
            return image
        except Exception as exc:
            logger.warning("Unable to load main branding asset %s: %s", path, exc)
            return None

    def _load_member(self, member: BrandMember) -> BrandAsset:
        path = discover_asset(self.assets_dir, member.asset_stem)
        if path is None:
            logger.warning(
                "Branding image for %s is missing; using monogram fallback.",
                member.display_name,
            )
            image = _monogram(member)
            return BrandAsset(
                member=member,
                image=image,
                path=None,
                is_placeholder=True,
                width=image.shape[1],
                height=image.shape[0],
                warning="Missing image; monogram fallback used.",
            )
        try:
            image, width, height, warning = _load_square(path)
            if warning:
                logger.warning("Branding asset %s: %s", path, warning)
            return BrandAsset(
                member=member,
                image=image,
                path=path,
                is_placeholder=False,
                width=width,
                height=height,
                warning=warning,
            )
        except Exception as exc:
            logger.warning(
                "Unable to load branding image %s for %s: %s; using monogram.",
                path,
                member.display_name,
                exc,
            )
            image = _monogram(member)
            return BrandAsset(
                member=member,
                image=image,
                path=None,
                is_placeholder=True,
                width=image.shape[1],
                height=image.shape[0],
                warning=str(exc),
            )

    @property
    def assets(self) -> tuple[BrandAsset, ...]:
        return self._assets
