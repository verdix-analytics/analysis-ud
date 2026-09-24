from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import (
    detect_pivots,
    get_last_n_pivots,
    pivots_to_lists,
    safe_ratio,
    fit_line_params,
    line_value,
)
from stock_analysis.confidence_models.chart_patterns.head_and_shoulders_model import (
    load_model,
    learned_confidence,
    head_and_shoulders_score,
)

import pandas as pd


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class HeadAndShouldersChartPattern(BasePattern):
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

        pivots = detect_pivots(
            df,
            lookback=2,
            min_move_pct=0.005,
        )

        if pivots is None or len(pivots) < 5:
            return empty_result

        last_pivots = get_last_n_pivots(pivots, 5)
        pivot_info = pivots_to_lists(last_pivots)

        types = pivot_info["types"]
        prices = pivot_info["prices"]
        indices = pivot_info["indices"]

        if types != ["H", "L", "H", "L", "H"]:
            return empty_result

        ls, nl1, head, nl2, rs = prices
        i_ls, i_nl1, i_head, i_nl2, i_rs = indices

        head_vs_ls = safe_ratio(head - ls, ls)
        head_vs_rs = safe_ratio(head - rs, rs)
        head_vs_shoulder_min = min(head_vs_ls, head_vs_rs)

        avg_shoulder = (ls + rs) / 2
        shoulder_diff_pct = safe_ratio(abs(ls - rs), avg_shoulder)

        avg_neck = (nl1 + nl2) / 2
        neckline_diff_pct = safe_ratio(abs(nl1 - nl2), avg_neck)

        left_span = i_head - i_ls
        right_span = i_rs - i_head
        total_span = i_rs - i_ls

        symmetry_ratio = safe_ratio(
            min(left_span, right_span),
            max(left_span, right_span),
        )

        if head_vs_shoulder_min < 0.005:
            return empty_result

        if shoulder_diff_pct > 0.08:
            return empty_result

        if left_span < 1 or right_span < 1 or total_span < 4:
            return empty_result

        rule_score = head_and_shoulders_score(
            head_vs_shoulder_min=head_vs_shoulder_min,
            shoulder_diff_pct=shoulder_diff_pct,
            neckline_diff_pct=neckline_diff_pct,
            left_span=left_span,
            right_span=right_span,
            symmetry_ratio=symmetry_ratio,
        )

        if self._model is not None:
            try:
                ml_score = learned_confidence(
                    head_vs_shoulder_min=head_vs_shoulder_min,
                    shoulder_diff_pct=shoulder_diff_pct,
                    neckline_diff_pct=neckline_diff_pct,
                    left_span=left_span,
                    right_span=right_span,
                    symmetry_ratio=symmetry_ratio,
                    model=self._model,
                )

                overall_score = 0.7 * ml_score + 0.3 * rule_score
            except Exception:
                overall_score = rule_score
        else:
            overall_score = rule_score

        confidence = clamp_0_100(overall_score)

        neckline_slope, neckline_intercept = fit_line_params(
            [i_nl1, i_nl2],
            [nl1, nl2],
        )

        neckline_now = line_value(
            neckline_slope,
            neckline_intercept,
            len(df) - 1,
        )

        confirmed = (
            confidence >= 50
            and float(df.iloc[-1]["Close"]) < float(neckline_now)
        )

        best_pivots = [
            {
                "type": "H",
                "price": float(ls),
                "time": df.iloc[int(i_ls)]["Datetime"],
                "label": "LS",
            },
            {
                "type": "L",
                "price": float(nl1),
                "time": df.iloc[int(i_nl1)]["Datetime"],
                "label": "NL1",
            },
            {
                "type": "H",
                "price": float(head),
                "time": df.iloc[int(i_head)]["Datetime"],
                "label": "H",
            },
            {
                "type": "L",
                "price": float(nl2),
                "time": df.iloc[int(i_nl2)]["Datetime"],
                "label": "NL2",
            },
            {
                "type": "H",
                "price": float(rs),
                "time": df.iloc[int(i_rs)]["Datetime"],
                "label": "RS",
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