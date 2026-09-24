from __future__ import annotations
import math
from typing import Any

from .chart_utilities import (
    recency_bias,
    confidence_weight,
    signal_age_decay,
    should_include_pattern,
    resolve_chart_direction,
    trend_score_chart,
    trend_structure_bonus_chart,
    get_chart_meta,
)

TECHNICAL_WEIGHTS = {
    "chart_score": 0.80,
    "mss_score": 0.20,
}


def normalize_mss_score(mss_score: float) -> float:
    """
    Convert market sentiment score to 0-100 scale.

    If MSS is already 0-100, keep it.
    If MSS is -1 to 1, convert it to 0-100.
    If MSS is 0-1, convert it to 0-100.
    """
    if mss_score is None:
        return 50.0

    try:
        mss_score = float(mss_score)
    except Exception:
        return 50.0

    if -1.0 <= mss_score <= 1.0:
        if mss_score < 0:
            return 50 + 50 * mss_score
        return mss_score * 100

    return max(0.0, min(100.0, mss_score))


def extract_market_sentiment_score(formations: list[dict]) -> float:
    """
    Extract Market Sentiment score from the first available technical_analysis field.
    """
    for f in formations:
        technical = f.get("technical_analysis", {})
        if isinstance(technical, dict) and "Market Sentiment" in technical:
            return normalize_mss_score(technical.get("Market Sentiment"))

    return 50.0


def context_score(formations: list[dict]) -> dict[str, Any]:
    if not formations:
        return {
            "score": 50.0,
            "pattern_score": 50.0,
            "mss_score": 50.0,
            "bias": "neutral",
            "agreement": 0.0,
            "bull_score": 0.0,
            "bear_score": 0.0,
            "pattern_count": 0,
            "contributing_patterns": [],
            "skipped_patterns": [],
            "technical_analysis": {},
        }

    bull_score = 0.0
    bear_score = 0.0
    total_weight = 0.0
    contributing_patterns = []
    skipped_patterns = []

    for formation in formations:
        pattern_name = formation.get("pattern")
        meta = get_chart_meta(pattern_name)

        if not should_include_pattern(pattern_name):
            skipped_patterns.append({
                "pattern": pattern_name,
                "reason": "excluded_by_metadata",
            })
            continue

        direction = resolve_chart_direction(formation)

        if direction not in {"bullish", "bearish"}:
            skipped_patterns.append({
                "pattern": pattern_name,
                "reason": "unresolved_direction",
            })
            continue

        window_count = formation.get("window_count", 1)
        confidence = formation.get("confidence", 0.0)
        base_weight = float(meta.get("base_weight", 1.0))

        r_w = recency_bias(window_count)
        c_w = confidence_weight(confidence)
        a_w = signal_age_decay(formation)
        pre_score = trend_score_chart(formation)
        s_bonus = trend_structure_bonus_chart(formation)

        combined_weight = r_w * c_w * a_w * pre_score * s_bonus * base_weight

        if combined_weight <= 0:
            skipped_patterns.append({
                "pattern": pattern_name,
                "reason": "non_positive_weight",
            })
            continue

        total_weight += combined_weight

        if direction == "bullish":
            bull_score += combined_weight
        elif direction == "bearish":
            bear_score += combined_weight

        contributing_patterns.append({
            "pattern": pattern_name,
            "direction": direction,
            "pattern_role": meta.get("pattern_role"),
            "window": window_count,
            "confidence": round(confidence, 4),
            "recency_weight": round(r_w, 4),
            "confidence_weight": round(c_w, 4),
            "age_weight": round(a_w, 4),
            "pre_trend_score": round(pre_score, 4),
            "structure_bonus": round(s_bonus, 4),
            "base_weight": round(base_weight, 4),
            "combined_weight": round(combined_weight, 6),
        })

    if total_weight <= 0:
        return {
            "score": 50.0,
            "pattern_score": 50.0,
            "mss_score": extract_market_sentiment_score(formations),
            "bias": "neutral",
            "agreement": 0.0,
            "bull_score": 0.0,
            "bear_score": 0.0,
            "pattern_count": 0,
            "contributing_patterns": [],
            "skipped_patterns": skipped_patterns,
            "technical_analysis": formations[0].get("technical_analysis", {}) if formations else {},
        }

    norm_bull = bull_score / total_weight
    norm_bear = bear_score / total_weight

    net_diff = norm_bull - norm_bear
    agreement = abs(net_diff)

    if net_diff > 0.05:
        bias = "bullish"
    elif net_diff < -0.05:
        bias = "bearish"
    else:
        bias = "neutral"

    evidence_strength = 1 - math.exp(-total_weight)
    effective_signal = net_diff * evidence_strength

    pattern_score = 50 + 50 * effective_signal
    pattern_score = max(0.0, min(100.0, pattern_score))

    mss_score = extract_market_sentiment_score(formations)

    final_score = (
        TECHNICAL_WEIGHTS["chart_score"] * pattern_score
        + TECHNICAL_WEIGHTS["mss_score"] * mss_score
    )

    final_score = max(0.0, min(100.0, final_score))

    return {
        "score": round(final_score, 3),
        "pattern_score": round(pattern_score, 3),
        "mss_score": round(mss_score, 3),
        "bias": bias,
        "agreement": round(agreement, 3),
        "bull_score": round(norm_bull * 100, 2),
        "bear_score": round(norm_bear * 100, 2),
        "pattern_count": len(contributing_patterns),
        "contributing_patterns": sorted(
            contributing_patterns,
            key=lambda x: x["combined_weight"],
            reverse=True
        ),
        "skipped_patterns": skipped_patterns,
        "technical_analysis": formations[0].get("technical_analysis", {}) if formations else {},
    }