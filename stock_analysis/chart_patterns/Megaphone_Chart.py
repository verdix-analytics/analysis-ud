from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import (
    detect_pivots,
    get_last_n_pivots,
    pivots_to_lists,
    fit_line_pct_slope,
    safe_ratio,
)
from stock_analysis.confidence_models.chart_patterns.megaphone_model import (
    load_model,
    learned_confidence,
    megaphone_score,
)

import pandas as pd


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class MegaphoneChartPattern(BasePattern):
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

        pivots = detect_pivots(df, lookback=2, min_move_pct=0.005)

        if pivots is None or len(pivots) < 4:
            return empty_result

        last_pivots = get_last_n_pivots(pivots, 6)

        if len(last_pivots) < 4:
            return empty_result

        pivot_info = pivots_to_lists(last_pivots)

        types = pivot_info["types"]
        prices = pivot_info["prices"]
        indices = pivot_info["indices"]

        highs_df = last_pivots[last_pivots["pivot_type"] == "H"].copy()
        lows_df = last_pivots[last_pivots["pivot_type"] == "L"].copy()

        if len(highs_df) < 2 or len(lows_df) < 2:
            return empty_result

        high_x = highs_df["candle_index"].tolist()
        high_y = highs_df["pivot_price"].tolist()

        low_x = lows_df["candle_index"].tolist()
        low_y = lows_df["pivot_price"].tolist()

        high_pct = fit_line_pct_slope(high_x, high_y)
        low_pct = fit_line_pct_slope(low_x, low_y)

        if high_pct <= 0.00005 or low_pct >= -0.00005:
            return empty_result

        swing_sizes = [
            abs(prices[i + 1] - prices[i])
            for i in range(len(prices) - 1)
        ]

        if len(swing_sizes) < 3:
            return empty_result

        split_idx = max(1, len(swing_sizes) // 2)

        early_avg = sum(swing_sizes[:split_idx]) / max(split_idx, 1)
        late_avg = sum(swing_sizes[split_idx:]) / max(
            len(swing_sizes) - split_idx,
            1,
        )

        expand_ratio = safe_ratio(late_avg, max(early_avg, 1e-9))

        if expand_ratio < 1.05:
            return empty_result

        high_rise_score = min(max(high_pct / 0.0020, 0.0), 1.0)
        low_drop_score = min(max(abs(low_pct) / 0.0020, 0.0), 1.0)

        if expand_ratio >= 1.40:
            expansion_score = 1.0
        elif expand_ratio >= 1.20:
            expansion_score = 0.8
        elif expand_ratio >= 1.05:
            expansion_score = 0.5
        else:
            expansion_score = 0.0

        pivot_richness_score = min(len(last_pivots) / 6.0, 1.0)

        alternation_count = sum(
            1
            for i in range(len(types) - 1)
            if types[i] != types[i + 1]
        )

        alternation_score = safe_ratio(
            alternation_count,
            max(len(types) - 1, 1),
        )

        balance_score = safe_ratio(
            min(len(highs_df), len(lows_df)),
            max(len(highs_df), len(lows_df)),
        )

        rule_score = megaphone_score(
            high_rise_score=high_rise_score,
            low_drop_score=low_drop_score,
            expansion_score=expansion_score,
            pivot_richness_score=pivot_richness_score,
            alternation_score=alternation_score,
            balance_score=balance_score,
        )

        if self._model is not None:
            try:
                ml_score = learned_confidence(
                    high_rise_score=high_rise_score,
                    low_drop_score=low_drop_score,
                    expansion_score=expansion_score,
                    pivot_richness_score=pivot_richness_score,
                    alternation_score=alternation_score,
                    balance_score=balance_score,
                    model=self._model,
                )

                overall_score = 0.7 * ml_score + 0.3 * rule_score
            except Exception:
                overall_score = rule_score
        else:
            overall_score = rule_score

        confidence = clamp_0_100(overall_score)

        upper_bound = max(high_y)
        lower_bound = min(low_y)
        last_close = float(df.iloc[-1]["Close"])

        confirmed = confidence >= 50 and (
            last_close > float(upper_bound)
            or last_close < float(lower_bound)
        )

        if last_close > upper_bound:
            direction = "bullish"
        elif last_close < lower_bound:
            direction = "bearish"
        else:
            direction = "neutral"

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
            "direction": direction,
        }