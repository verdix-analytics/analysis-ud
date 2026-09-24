# chart_utilities.py

from __future__ import annotations

from typing import Optional

from .utilities import (
    recency_bias,
    confidence_weight,
    signal_age_days,
    signal_age_decay,
)
from .chart_metadata import CHART_PATTERN_META


def get_chart_meta(pattern_name: str) -> dict:
    """
    Safely fetch metadata for a chart pattern.
    Returns empty dict if pattern is unknown.
    """
    return CHART_PATTERN_META.get(pattern_name, {})


def should_include_pattern(pattern_name: str) -> bool:
    """
    Whether this pattern should contribute to chart context score.
    Neutral patterns are currently marked as False in metadata.
    """
    meta = get_chart_meta(pattern_name)
    return bool(meta.get("include_in_score", False))


def resolve_chart_direction(formation: dict) -> Optional[str]:
    """
    Resolve final direction for a chart formation.

    Priority:
    1. detector output: formation["signal"]["direction"]
    2. metadata fallback: default_direction
    3. None

    Returns:
        "bullish", "bearish", or None
    """
    pattern_name = formation.get("pattern")
    meta = get_chart_meta(pattern_name)

    direction = formation.get("signal", {}).get("direction")
    if direction in {"bullish", "bearish"}:
        return direction

    fallback = meta.get("default_direction")
    if fallback in {"bullish", "bearish"}:
        return fallback

    return None


def expected_pretrend_label(pattern_role: str, direction: Optional[str]) -> Optional[str]:
    """
    Infer the expected pre-trend label for a chart pattern.

    Rules:
    - reversal:
        bullish -> downtrend
        bearish -> uptrend
    - continuation:
        bullish -> uptrend
        bearish -> downtrend
    - neutral:
        None

    Returns:
        "uptrend", "downtrend", or None
    """
    if direction not in {"bullish", "bearish"}:
        return None

    if pattern_role == "reversal":
        return "downtrend" if direction == "bullish" else "uptrend"

    if pattern_role == "continuation":
        return "uptrend" if direction == "bullish" else "downtrend"

    return None


def trend_score_chart(formation: dict) -> float:
    """
    Chart-specific pre-trend compatibility score.

    This function does NOT decide direction.
    It only judges whether the observed pre-trend is consistent
    with the pattern's expected role.

    Score range is roughly [0, 1].

    Heuristic:
    - unknown / missing trend -> 0.5
    - neutral pattern role -> 0.5
    - expected trend match:
        0.6 + 0.4 * strength
    - sideways:
        0.4 + 0.1 * strength
    - mismatch:
        max(0, 0.2 - 0.2 * strength)
    """
    pattern_name = formation.get("pattern")
    meta = get_chart_meta(pattern_name)
    pattern_role = meta.get("pattern_role", "neutral")

    # neutral patterns currently do not contribute to score anyway,
    # but we keep this safe default here.
    if pattern_role == "neutral":
        return 0.5

    direction = resolve_chart_direction(formation)
    if direction is None:
        return 0.5

    trend = formation.get("pre_trend")
    if not trend or trend.get("label") == "unknown":
        return 0.5

    observed_label = trend.get("label")
    strength = min(trend.get("strength", 0), 100) / 100

    expected_label = expected_pretrend_label(pattern_role, direction)

    if expected_label is None:
        return 0.5

    if observed_label == expected_label:
        return 0.6 + 0.4 * strength

    if observed_label == "sideways":
        return 0.4 + 0.1 * strength

    return max(0.0, 0.2 - 0.2 * strength)


def trend_structure_bonus_chart(formation: dict) -> float:
    """
    Chart-specific trend structure bonus.

    For reversal patterns:
        - bullish reversal prefers a clean downtrend
          -> lower_high_ratio, lower_low_ratio
        - bearish reversal prefers a clean uptrend
          -> higher_high_ratio, higher_low_ratio

    For continuation patterns:
        - bullish continuation prefers a clean uptrend
          -> higher_high_ratio, higher_low_ratio
        - bearish continuation prefers a clean downtrend
          -> lower_high_ratio, lower_low_ratio

    Neutral patterns default to 1.0.

    Returns:
        A mild multiplicative bonus, typically around [1.0, 1.07]
    """
    pattern_name = formation.get("pattern")
    meta = get_chart_meta(pattern_name)
    pattern_role = meta.get("pattern_role", "neutral")

    if pattern_role == "neutral":
        return 1.0

    trend = formation.get("pre_trend")
    if not trend:
        return 1.0

    direction = resolve_chart_direction(formation)
    if direction is None:
        return 1.0

    if pattern_role == "reversal":
        if direction == "bullish":
            q1 = trend.get("lower_high_ratio", 0)
            q2 = trend.get("lower_low_ratio", 0)
        else:  # bearish reversal
            q1 = trend.get("higher_high_ratio", 0)
            q2 = trend.get("higher_low_ratio", 0)

    elif pattern_role == "continuation":
        if direction == "bullish":
            q1 = trend.get("higher_high_ratio", 0)
            q2 = trend.get("higher_low_ratio", 0)
        else:  # bearish continuation
            q1 = trend.get("lower_high_ratio", 0)
            q2 = trend.get("lower_low_ratio", 0)

    else:
        return 1.0

    structure_quality = (q1 + q2) / 2
    bonus = 1 + 0.15 * max(0, structure_quality - 0.55)
    return round(bonus, 4)