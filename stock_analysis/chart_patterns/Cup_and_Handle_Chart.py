from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import (
    safe_ratio,
    fit_line_pct_slope,
)
from stock_analysis.confidence_models.chart_patterns.cup_and_handle_model import (
    load_model,
    learned_confidence,
    cup_and_handle_score,
)

import pandas as pd


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class CupAndHandleChartPattern(BasePattern):
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
            "direction": "bullish",
        }

        if len(df) < 20:
            return empty_result

        df = df.reset_index(drop=True).copy()

        best_result = None
        best_confidence = -1

        for cup_end in range(int(len(df) * 0.6), int(len(df) * 0.85)):
            cup_df = df.iloc[:cup_end].copy()
            handle_df = df.iloc[cup_end:].copy()

            if len(cup_df) < 12 or len(handle_df) < 4:
                continue

            left_rim_window = cup_df.iloc[: max(3, len(cup_df) // 3)]
            right_rim_window = cup_df.iloc[-max(3, len(cup_df) // 3):]

            left_rim = left_rim_window["High"].max()
            right_rim = right_rim_window["High"].max()
            cup_low = cup_df["Low"].min()

            avg_rim = (left_rim + right_rim) / 2
            cup_depth = min(left_rim, right_rim) - cup_low

            if avg_rim <= 0 or cup_depth <= 0:
                continue

            rim_gap = safe_ratio(
                abs(left_rim - right_rim),
                max(abs(avg_rim), 1e-9),
            )

            if rim_gap > 0.08:
                continue

            cup_depth_ratio = safe_ratio(
                cup_depth,
                max(abs(avg_rim), 1e-9),
            )

            if cup_depth_ratio < 0.03:
                continue

            handle_low = handle_df["Low"].min()
            handle_pullback = right_rim - handle_low
            handle_ratio = safe_ratio(
                handle_pullback,
                max(cup_depth, 1e-9),
            )

            if handle_ratio > 0.60:
                continue

            handle_len_ratio = safe_ratio(
                len(handle_df),
                max(len(cup_df), 1),
            )

            if handle_len_ratio > 0.70:
                continue

            low_idx = int(cup_df["Low"].idxmin())
            low_pos = safe_ratio(low_idx, max(len(cup_df) - 1, 1))

            if not (0.15 <= low_pos <= 0.85):
                continue

            rim_similarity_score = max(
                0.0,
                1.0 - min(rim_gap / 0.08, 1.0),
            )

            if handle_ratio <= 0.35:
                handle_shallow_score = 1.0
            elif handle_ratio <= 0.50:
                handle_shallow_score = 0.7
            elif handle_ratio <= 0.60:
                handle_shallow_score = 0.4
            else:
                handle_shallow_score = 0.0

            if handle_len_ratio <= 0.40:
                handle_len_score = 1.0
            elif handle_len_ratio <= 0.55:
                handle_len_score = 0.7
            elif handle_len_ratio <= 0.70:
                handle_len_score = 0.4
            else:
                handle_len_score = 0.0

            center_dist = abs(low_pos - 0.5)
            cup_roundness_score = max(
                0.0,
                1.0 - min(center_dist / 0.35, 1.0),
            )

            handle_x = list(range(len(handle_df)))
            handle_y = handle_df["Close"].tolist()
            handle_pct_slope = fit_line_pct_slope(handle_x, handle_y)

            if handle_pct_slope <= 0.0002:
                handle_slope_score = 1.0
            elif handle_pct_slope <= 0.0005:
                handle_slope_score = 0.7
            elif handle_pct_slope <= 0.0010:
                handle_slope_score = 0.4
            else:
                handle_slope_score = 0.0

            rule_score = cup_and_handle_score(
                rim_similarity_score=rim_similarity_score,
                cup_depth_ratio=cup_depth_ratio,
                handle_shallow_score=handle_shallow_score,
                handle_len_score=handle_len_score,
                cup_roundness_score=cup_roundness_score,
                handle_slope_score=handle_slope_score,
            )

            if self._model is not None:
                try:
                    ml_score = learned_confidence(
                        rim_similarity_score=rim_similarity_score,
                        cup_depth_ratio=cup_depth_ratio,
                        handle_shallow_score=handle_shallow_score,
                        handle_len_score=handle_len_score,
                        cup_roundness_score=cup_roundness_score,
                        handle_slope_score=handle_slope_score,
                        model=self._model,
                    )
                    overall_score = 0.7 * ml_score + 0.3 * rule_score
                except Exception:
                    overall_score = rule_score
            else:
                overall_score = rule_score

            confidence = clamp_0_100(overall_score)

            rim_level = max(left_rim, right_rim)

            confirmed = (
                confidence >= 50
                and float(df.iloc[-1]["Close"]) > float(rim_level)
            )

            left_rim_idx = int(left_rim_window["High"].idxmax())
            right_rim_idx = int(right_rim_window["High"].idxmax())
            handle_low_idx = int(handle_df["Low"].idxmin())
            handle_end_idx = int(handle_df.index[-1])

            best_pivots = [
                {
                    "type": "left_rim",
                    "price": float(left_rim),
                    "time": df.iloc[left_rim_idx]["Datetime"],
                    "label": "LeftRim",
                },
                {
                    "type": "cup_low",
                    "price": float(cup_low),
                    "time": df.iloc[int(low_idx)]["Datetime"],
                    "label": "CupLow",
                },
                {
                    "type": "right_rim",
                    "price": float(right_rim),
                    "time": df.iloc[right_rim_idx]["Datetime"],
                    "label": "RightRim",
                },
                {
                    "type": "handle_low",
                    "price": float(handle_low),
                    "time": df.iloc[handle_low_idx]["Datetime"],
                    "label": "HandleLow",
                },
                {
                    "type": "handle_end",
                    "price": float(handle_df.iloc[-1]["Close"]),
                    "time": df.iloc[handle_end_idx]["Datetime"],
                    "label": "HandleEnd",
                },
            ]

            best_pivots = sorted(
                best_pivots,
                key=lambda x: pd.to_datetime(x["time"]),
            )

            if confidence > best_confidence:
                best_confidence = confidence
                best_result = {
                    "confirmed": bool(confirmed),
                    "confidence": round(float(confidence), 2),
                    "pivots": best_pivots,
                    "direction": "bullish",
                }

        if best_result is None:
            return empty_result

        return best_result