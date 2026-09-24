from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import (
    detect_pivots,
    fit_line_params,
    fit_line_pct_slope,
    line_value,
    safe_ratio,
)
from stock_analysis.confidence_models.chart_patterns.pennant_model import (
    load_model,
    learned_confidence,
    pennant_score,
)

import pandas as pd


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class PennantChartPattern(BasePattern):
    enabled = True
    category = Category.CHART.value
    expected_pretrend = None

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
            "direction": "neutral",
        }

        if len(df) < 12:
            return empty_result

        df = df.reset_index(drop=True).copy()
        best_result = None
        best_confidence = -1

        for split_idx in range(4, len(df) - 4):
            pole_df = df.iloc[:split_idx].copy()
            pennant_df = df.iloc[split_idx:].copy()

            if len(pole_df) < 4 or len(pennant_df) < 4:
                continue

            pole_start = pole_df.iloc[0]["Close"]
            pole_end = pole_df.iloc[-1]["Close"]
            pole_move = pole_end - pole_start
            pole_move_pct = safe_ratio(
                abs(pole_move),
                max(abs(pole_start), 1e-9),
            )

            pole_is_bullish = pole_move > 0
            pole_is_bearish = pole_move < 0

            if not (pole_is_bullish or pole_is_bearish):
                continue

            if pole_move_pct < 0.015:
                continue

            pivots = detect_pivots(
                pennant_df,
                lookback=2,
                min_move_pct=0.003,
            )

            if pivots is None or len(pivots) < 4:
                continue

            highs = pivots[pivots["pivot_type"] == "H"].copy()
            lows = pivots[pivots["pivot_type"] == "L"].copy()

            if len(highs) < 2 or len(lows) < 2:
                continue

            high_x = highs["candle_index"].tolist()
            high_y = highs["pivot_price"].tolist()
            low_x = lows["candle_index"].tolist()
            low_y = lows["pivot_price"].tolist()

            high_slope, high_intercept = fit_line_params(high_x, high_y)
            low_slope, low_intercept = fit_line_params(low_x, low_y)

            high_pct_slope = fit_line_pct_slope(high_x, high_y)
            low_pct_slope = fit_line_pct_slope(low_x, low_y)

            if high_pct_slope >= -0.00005 or low_pct_slope <= 0.00005:
                continue

            pennant_range = pennant_df["High"].max() - pennant_df["Low"].min()
            pole_range = pole_df["High"].max() - pole_df["Low"].min()

            range_ratio = safe_ratio(
                pennant_range,
                max(pole_range, 1e-9),
            )

            if range_ratio > 0.70:
                continue

            first_idx = 0
            last_idx = len(pennant_df) - 1

            start_range = line_value(
                high_slope,
                high_intercept,
                first_idx,
            ) - line_value(
                low_slope,
                low_intercept,
                first_idx,
            )

            end_range = line_value(
                high_slope,
                high_intercept,
                last_idx,
            ) - line_value(
                low_slope,
                low_intercept,
                last_idx,
            )

            if start_range <= 0 or end_range <= 0:
                continue

            convergence_ratio = safe_ratio(
                end_range,
                max(start_range, 1e-9),
            )

            if convergence_ratio > 1.00:
                continue

            pole_strength_score = min(max(pole_move_pct / 0.05, 0.0), 1.0)

            if convergence_ratio <= 0.50:
                convergence_score = 1.0
            elif convergence_ratio <= 0.70:
                convergence_score = 0.8
            elif convergence_ratio <= 0.85:
                convergence_score = 0.6
            elif convergence_ratio <= 1.00:
                convergence_score = 0.4
            else:
                convergence_score = 0.0

            if range_ratio <= 0.25:
                compactness_score = 1.0
            elif range_ratio <= 0.40:
                compactness_score = 0.8
            elif range_ratio <= 0.60:
                compactness_score = 0.6
            elif range_ratio <= 0.70:
                compactness_score = 0.4
            else:
                compactness_score = 0.0

            pivot_count = len(highs) + len(lows)
            pivot_richness_score = min(pivot_count / 6.0, 1.0)

            balance_score = safe_ratio(
                min(len(highs), len(lows)),
                max(len(highs), len(lows)),
            )

            left_side = max(high_x[0], low_x[0])
            right_side = min(high_x[-1], low_x[-1])
            span_total = max(last_idx, 1)
            symmetry_width = max(right_side - left_side, 0)

            symmetry_score = min(
                symmetry_width / max(span_total * 0.6, 1e-9),
                1.0,
            )

            rule_score = pennant_score(
                pole_strength_score=pole_strength_score,
                convergence_score=convergence_score,
                compactness_score=compactness_score,
                pivot_richness_score=pivot_richness_score,
                balance_score=balance_score,
                symmetry_score=symmetry_score,
            )

            if self._model is not None:
                try:
                    ml_score = learned_confidence(
                        pole_strength_score=pole_strength_score,
                        convergence_score=convergence_score,
                        compactness_score=compactness_score,
                        pivot_richness_score=pivot_richness_score,
                        balance_score=balance_score,
                        symmetry_score=symmetry_score,
                        model=self._model,
                    )

                    overall_score = 0.7 * ml_score + 0.3 * rule_score
                except Exception:
                    overall_score = rule_score
            else:
                overall_score = rule_score

            confidence = clamp_0_100(overall_score)

            last_pennant_idx = len(pennant_df) - 1

            resistance_now = line_value(
                high_slope,
                high_intercept,
                last_pennant_idx,
            )

            support_now = line_value(
                low_slope,
                low_intercept,
                last_pennant_idx,
            )

            last_close = float(pennant_df.iloc[-1]["Close"])

            confirmed = confidence >= 50 and (
                (pole_is_bullish and last_close > float(resistance_now))
                or (pole_is_bearish and last_close < float(support_now))
            )

            direction = "bullish" if pole_is_bullish else "bearish"

            best_pivots = [
                {
                    "type": "pole_start",
                    "price": float(pole_df.iloc[0]["Close"]),
                    "time": pole_df.iloc[0]["Datetime"],
                    "label": "PoleStart",
                },
                {
                    "type": "pole_end",
                    "price": float(pole_df.iloc[-1]["Close"]),
                    "time": pole_df.iloc[-1]["Datetime"],
                    "label": "PoleEnd",
                },
            ]

            for j, (_, row) in enumerate(highs.iterrows(), start=1):
                pivot_idx = int(split_idx + row["candle_index"])

                best_pivots.append({
                    "type": "H",
                    "price": float(row["pivot_price"]),
                    "time": df.iloc[pivot_idx]["Datetime"],
                    "label": f"H{j}",
                })

            for j, (_, row) in enumerate(lows.iterrows(), start=1):
                pivot_idx = int(split_idx + row["candle_index"])

                best_pivots.append({
                    "type": "L",
                    "price": float(row["pivot_price"]),
                    "time": df.iloc[pivot_idx]["Datetime"],
                    "label": f"L{j}",
                })

            best_pivots = sorted(
                best_pivots,
                key=lambda x: pd.to_datetime(x["time"]),
            )

            deduped = []
            seen = set()

            for p in best_pivots:
                if p["time"] not in seen:
                    deduped.append(p)
                    seen.add(p["time"])

            if confidence > best_confidence:
                best_confidence = confidence
                best_result = {
                    "confirmed": bool(confirmed),
                    "confidence": round(float(confidence), 2),
                    "pivots": deduped,
                    "direction": direction,
                }

        if best_result is None:
            return empty_result

        return best_result
