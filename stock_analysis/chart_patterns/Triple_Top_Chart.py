from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import (
    detect_pivots,
    get_last_n_pivots,
    pivots_to_lists,
    safe_ratio,
)
from stock_analysis.confidence_models.chart_patterns.triple_top_model import (
    load_model,
    learned_confidence,
    triple_top_score,
)

import pandas as pd


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class TripleTopChartPattern(BasePattern):
    enabled = True
    category = Category.CHART.value
    expected_pretrend = "uptrend"

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
            "direction": "bearish",
        }

        if len(df) < 11:
            return empty_result

        pivots = detect_pivots(df, lookback=2, min_move_pct=0.005)

        if pivots is None or len(pivots) < 5:
            return empty_result

        pivot_info = pivots_to_lists(get_last_n_pivots(pivots, 5))

        types = pivot_info["types"]
        prices = pivot_info["prices"]
        indices = pivot_info["indices"]

        if types != ["H", "L", "H", "L", "H"]:
            return empty_result

        h1, l1, h2, l2, h3 = prices
        i1, i2, i3, i4, i5 = indices

        highs = [h1, h2, h3]
        valleys = [l1, l2]

        avg_high = sum(highs) / 3

        peak_spread_pct = safe_ratio(
            max(highs) - min(highs),
            max(abs(avg_high), 1e-9),
        )

        valley_depths = [
            safe_ratio(avg_high - valley, max(abs(avg_high), 1e-9))
            for valley in valleys
        ]

        valley_depth = min(valley_depths)

        span_1 = i2 - i1
        span_2 = i3 - i2
        span_3 = i4 - i3
        span_4 = i5 - i4
        total_span = i5 - i1

        spans_ok = (
            span_1 >= 1
            and span_2 >= 1
            and span_3 >= 1
            and span_4 >= 1
        )

        if not spans_ok:
            return empty_result

        if peak_spread_pct > 0.08:
            return empty_result

        if valley_depth < 0.005:
            return empty_result

        if total_span < 4:
            return empty_result

        if total_span >= 10:
            span_score = 1.0
        elif total_span >= 8:
            span_score = 0.8
        elif total_span >= 6:
            span_score = 0.6
        elif total_span >= 4:
            span_score = 0.4
        else:
            span_score = 0.0

        left_block = i3 - i1
        right_block = i5 - i3

        symmetry_ratio = safe_ratio(
            min(left_block, right_block),
            max(left_block, right_block),
        )

        valley_balance_ratio = safe_ratio(
            min(l1, l2),
            max(l1, l2),
        )

        rule_score = triple_top_score(
            peak_spread_pct=peak_spread_pct,
            valley_depth=valley_depth,
            span_score=span_score,
            symmetry_ratio=symmetry_ratio,
            valley_balance_ratio=valley_balance_ratio,
            total_span=total_span,
        )

        if self._model is not None:
            try:
                ml_score = learned_confidence(
                    peak_spread_pct=peak_spread_pct,
                    valley_depth=valley_depth,
                    span_score=span_score,
                    symmetry_ratio=symmetry_ratio,
                    valley_balance_ratio=valley_balance_ratio,
                    total_span=total_span,
                    model=self._model,
                )

                overall_score = 0.7 * ml_score + 0.3 * rule_score
            except Exception:
                overall_score = rule_score
        else:
            overall_score = rule_score

        confidence = clamp_0_100(overall_score)

        base_support = min(l1, l2)

        confirmed = (
            confidence >= 50
            and float(df.iloc[-1]["Close"]) < float(base_support)
        )

        best_pivots = [
            {
                "type": "H",
                "price": float(h1),
                "time": df.iloc[int(i1)]["Datetime"],
                "label": "H1",
            },
            {
                "type": "L",
                "price": float(l1),
                "time": df.iloc[int(i2)]["Datetime"],
                "label": "L1",
            },
            {
                "type": "H",
                "price": float(h2),
                "time": df.iloc[int(i3)]["Datetime"],
                "label": "H2",
            },
            {
                "type": "L",
                "price": float(l2),
                "time": df.iloc[int(i4)]["Datetime"],
                "label": "L2",
            },
            {
                "type": "H",
                "price": float(h3),
                "time": df.iloc[int(i5)]["Datetime"],
                "label": "H3",
            },
        ]

        best_pivots = sorted(
            best_pivots,
            key=lambda x: pd.to_datetime(x["time"]),
        )

        return {
            "confirmed": bool(confirmed),
            "confidence": round(float(confidence), 2),
            "pivots": best_pivots,
            "direction": "bearish",
        }
