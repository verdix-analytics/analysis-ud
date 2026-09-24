from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import (
    detect_pivots,
    get_last_n_pivots,
    pivots_to_lists,
    safe_ratio,
)
from stock_analysis.confidence_models.chart_patterns.diamond_bottom_model import (
    load_model,
    learned_confidence,
    diamond_bottom_score,
)

import pandas as pd
import numpy as np


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class DiamondBottomChartPattern(BasePattern):
    enabled = True
    category = Category.CHART.value
    expected_pretrend = "downtrend"

    def __init__(self):
        try:
            self._model = load_model()
        except Exception:
            self._model = None

    def _run(self, df: pd.DataFrame) -> dict:
        empty_result = {
            "confirmed": False,
            "confidence": 0,
            "pivots": [],
            "direction": "bullish",
        }

        if len(df) < 18:
            return empty_result

        pivots = detect_pivots(
            df,
            lookback=2,
            min_move_pct=0.005,
        )

        if pivots is None or len(pivots) < 7:
            return empty_result

        last_pivots = get_last_n_pivots(pivots, 7)
        pivot_info = pivots_to_lists(last_pivots)

        types = pivot_info["types"]
        prices = pivot_info["prices"]
        indices = pivot_info["indices"]

        alternating = all(
            types[i] != types[i + 1]
            for i in range(len(types) - 1)
        )

        if not alternating:
            return empty_result

        min_idx = int(np.argmin(prices))

        if min_idx not in {2, 3, 4}:
            return empty_result

        left_prices = prices[: min_idx + 1]
        right_prices = prices[min_idx:]

        left_types = types[: min_idx + 1]
        right_types = types[min_idx:]

        if len(left_prices) < 3 or len(right_prices) < 3:
            return empty_result

        left_highs = [
            p for t, p in zip(left_types, left_prices)
            if t == "H"
        ]
        left_lows = [
            p for t, p in zip(left_types, left_prices)
            if t == "L"
        ]
        right_highs = [
            p for t, p in zip(right_types, right_prices)
            if t == "H"
        ]
        right_lows = [
            p for t, p in zip(right_types, right_prices)
            if t == "L"
        ]

        if (
            len(left_highs) < 2
            or len(left_lows) < 2
            or len(right_highs) < 2
            or len(right_lows) < 2
        ):
            return empty_result

        highs_rising_left = left_highs[-1] > left_highs[0]
        lows_falling_left = left_lows[-1] < left_lows[0]

        if not (highs_rising_left and lows_falling_left):
            return empty_result

        highs_falling_right = right_highs[-1] < right_highs[0]
        lows_rising_right = right_lows[-1] > right_lows[0]

        if not (highs_falling_right and lows_rising_right):
            return empty_result

        left_span = indices[min_idx] - indices[0]
        right_span = indices[-1] - indices[min_idx]

        if left_span <= 0 or right_span <= 0:
            return empty_result

        center_target = (len(prices) - 1) / 2
        center_dist = abs(min_idx - center_target)
        max_center_dist = max(center_target, 1e-9)

        center_proximity_score = max(
            0.0,
            1.0 - center_dist / max_center_dist,
        )

        left_high_rise = safe_ratio(
            left_highs[-1] - left_highs[0],
            max(abs(left_highs[0]), 1e-9),
        )
        left_low_drop = safe_ratio(
            left_lows[0] - left_lows[-1],
            max(abs(left_lows[0]), 1e-9),
        )
        left_expansion_raw = left_high_rise + left_low_drop
        left_expansion_score = min(max(left_expansion_raw / 0.08, 0.0), 1.0)

        right_high_drop = safe_ratio(
            right_highs[0] - right_highs[-1],
            max(abs(right_highs[0]), 1e-9),
        )
        right_low_rise = safe_ratio(
            right_lows[-1] - right_lows[0],
            max(abs(right_lows[0]), 1e-9),
        )
        right_contraction_raw = right_high_drop + right_low_rise
        right_contraction_score = min(max(right_contraction_raw / 0.08, 0.0), 1.0)

        span_symmetry_ratio = safe_ratio(
            min(left_span, right_span),
            max(left_span, right_span),
        )

        right_boundary = max(right_highs) if len(right_highs) > 0 else max(prices)
        right_bottom = min(right_lows) if len(right_lows) > 0 else min(prices)

        right_width = safe_ratio(
            right_boundary - right_bottom,
            max(abs(right_boundary), 1e-9),
        )

        boundary_tightness = max(
            0.0,
            1.0 - min(right_width / 0.12, 1.0),
        )

        pivot_richness_score = min(len(prices) / 9.0, 1.0)

        rule_score = diamond_bottom_score(
            center_proximity_score=center_proximity_score,
            left_expansion_score=left_expansion_score,
            right_contraction_score=right_contraction_score,
            span_symmetry_ratio=span_symmetry_ratio,
            boundary_tightness=boundary_tightness,
            pivot_richness_score=pivot_richness_score,
        )

        if self._model is not None:
            try:
                ml_score = learned_confidence(
                    center_proximity_score=center_proximity_score,
                    left_expansion_score=left_expansion_score,
                    right_contraction_score=right_contraction_score,
                    span_symmetry_ratio=span_symmetry_ratio,
                    boundary_tightness=boundary_tightness,
                    pivot_richness_score=pivot_richness_score,
                    model=self._model,
                )
                overall_score = 0.7 * ml_score + 0.3 * rule_score
            except Exception:
                overall_score = rule_score
        else:
            overall_score = rule_score

        confidence = clamp_0_100(overall_score)

        last_close = float(df.iloc[-1]["Close"])
        confirmed = confidence >= 50 and last_close > float(right_boundary)

        best_pivots = [
            {
                "type": t,
                "price": float(p),
                "time": df.iloc[int(i)]["Datetime"],
                "label": f"{t}{k + 1}",
            }
            for k, (t, p, i) in enumerate(zip(types, prices, indices))
        ]

        best_pivots = sorted(
            best_pivots,
            key=lambda x: pd.to_datetime(x["time"]),
        )

        return {
            "confirmed": bool(confirmed),
            "confidence": round(float(confidence), 2),
            "pivots": best_pivots,
            "direction": "bullish",
        }