"""Helpers for reading symbol lists from `assets.json`."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_ASSET_CLASS_MAP = {
    "stock": "equity",
    "crypto": "crypto",
    "commodity": "commodity",
    "currency_exchange": "forex",
}

_DEFAULT_ASSETS_PATH = Path(__file__).resolve().parent / "assets.json"


@lru_cache(maxsize=8)
def _load_assets(path: str) -> tuple[dict, ...]:
    """Load and validate the asset registry from disk."""
    json_path = Path(path)
    logger.debug("Loading assets from %s", path)
    raw = json.loads(json_path.read_text(encoding="utf-8"))

    assets = raw.get("assets")
    if not isinstance(assets, list):
        raise ValueError("assets.json must contain an 'assets' list")

    validated_assets: list[dict] = []
    skipped = 0
    for item in assets:
        if not isinstance(item, dict):
            skipped += 1
            continue
        symbol = item.get("symbol")
        asset_class = item.get("asset_class")
        if not isinstance(symbol, str) or not symbol.strip():
            skipped += 1
            continue
        if not isinstance(asset_class, str) or not asset_class.strip():
            skipped += 1
            continue
        validated_assets.append(item)

    logger.info("Loaded %d assets from %s (%d skipped due to invalid entries)",
                len(validated_assets), path, skipped)
    return tuple(validated_assets)


def _symbols_for_asset_class(asset_class: str, path: Path | str = _DEFAULT_ASSETS_PATH) -> list[str]:
    """Return unique symbols for a given asset class, preserving source order."""
    target_class = asset_class.strip().lower()
    seen: set[str] = set()
    symbols: list[str] = []

    for asset in _load_assets(str(path)):
        if asset.get("asset_class", "").strip().lower() != target_class:
            continue
        symbol = asset["symbol"].strip()
        if symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)

    logger.debug("Found %d unique symbols for asset_class=%s", len(symbols), target_class)
    return symbols


def get_stock_symbols(path: Path | str = _DEFAULT_ASSETS_PATH) -> list[str]:
    """Return all stock symbols from `assets.json`."""
    return _symbols_for_asset_class(_ASSET_CLASS_MAP["stock"], path)[:7]


def get_crypto_symbols(path: Path | str = _DEFAULT_ASSETS_PATH) -> list[str]:
    """Return all cryptocurrency symbols from `assets.json`."""
    return _symbols_for_asset_class(_ASSET_CLASS_MAP["crypto"], path)[:1]


def get_commodity_symbols(path: Path | str = _DEFAULT_ASSETS_PATH) -> list[str]:
    """Return all commodity symbols from `assets.json`."""
    return _symbols_for_asset_class(_ASSET_CLASS_MAP["commodity"], path)[:1]


def get_currency_exchange_symbols(path: Path | str = _DEFAULT_ASSETS_PATH) -> list[str]:
    """Return all currency exchange / forex symbols from `assets.json`."""
    return _symbols_for_asset_class(_ASSET_CLASS_MAP["currency_exchange"], path)[:1]


# Optional alias for callers who prefer the shorter market name.
get_forex_symbols = get_currency_exchange_symbols


def get_symbol_metadata(
    symbol: str,
    asset_class: str | None = None,
    path: Path | str = _DEFAULT_ASSETS_PATH,
) -> dict[str, object]:
    """
    Retrieve metadata for a given symbol from assets.json.

    Args:
        symbol: The symbol to look up (e.g., "AAPL", "BTC", "GC=F").
        asset_class: Optional asset class filter (e.g., "equity", "crypto", "commodity", "forex").
                     If provided, will only match symbols in that class.
        path: Path to assets.json file.

    Returns:
        A dictionary with keys: symbol, name, asset_class, exchange.
        Returns a fallback dict with symbol and unknown/empty values if not found.
    """
    target_symbol = symbol.strip().upper()
    target_class = asset_class.strip().lower() if asset_class else None

    for asset in _load_assets(str(path)):
        asset_symbol = asset.get("symbol", "").strip().upper()
        if asset_symbol != target_symbol:
            continue

        if target_class and asset.get("asset_class", "").strip().lower() != target_class:
            continue

        return {
            "symbol": asset.get("symbol", symbol),
            "name": asset.get("name", symbol),
            "asset_class": asset.get("asset_class", "unknown"),
            "exchange": asset.get("exchange", "UNKNOWN"),
        }

    logger.warning("Symbol '%s' not found in assets registry (asset_class filter=%s)", symbol, asset_class)
    return {
        "symbol": symbol,
        "name": symbol,
        "asset_class": asset_class or "unknown",
        "exchange": "UNKNOWN",
    }
