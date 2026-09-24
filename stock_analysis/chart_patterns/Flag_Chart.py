from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.tools.chart_utils import (
    fit_line_params,
    fit_line_pct_slope,
    safe_ratio,
    detect_pivots,
)
from stock_analysis.confidence_models.chart_patterns.flag_model import (
    load_model,
    learned_confidence,
    flag_score,
)

import pandas as pd


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


class FlagChartPattern(BasePattern):
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
            "direction": "neutral",
        }

        if len(df) < 12:
            return empty_result

        df = df.reset_index(drop=True).copy()

        best_split_idx = None
        best_split_score = float("-inf")

        for split_idx in range(4, len(df) - 3):
            pole_df = df.iloc[:split_idx].copy()
            flag_df = df.iloc[split_idx:].copy()

            if len(pole_df) < 4 or len(flag_df) < 4:
                continue

            pole_start = pole_df.iloc[0]["Close"]
            pole_end = pole_df.iloc[-1]["Close"]
            pole_move = pole_end - pole_start
            pole_move_pct = safe_ratio(
                abs(pole_move),
                max(abs(pole_start), 1e-9),
            )

            pole_range = pole_df["High"].max() - pole_df["Low"].min()
            flag_range = flag_df["High"].max() - flag_df["Low"].min()
            range_ratio = safe_ratio(flag_range, max(pole_range, 1e-9))

            split_score = pole_move_pct - 0.5 * range_ratio

            if pole_move_pct >= 0.015 and split_score > best_split_score:
                best_split_score = split_score
                best_split_idx = split_idx

        if best_split_idx is None:
            return empty_result

        pole_df = df.iloc[:best_split_idx].copy()
        flag_df = df.iloc[best_split_idx:].copy()

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
            return empty_result

        flag_high = flag_df["High"].max()
        flag_low = flag_df["Low"].min()
        flag_range = flag_high - flag_low

        pole_range = pole_df["High"].max() - pole_df["Low"].min()
        range_ratio = safe_ratio(flag_range, max(pole_range, 1e-9))

        pole_x = list(range(len(pole_df)))
        pole_y = pole_df["Close"].tolist()

        flag_x = list(range(len(flag_df)))
        flag_y = flag_df["Close"].tolist()

        pole_slope, _ = fit_line_params(pole_x, pole_y)
        flag_slope, _ = fit_line_params(flag_x, flag_y)

        pole_pct_slope = fit_line_pct_slope(pole_x, pole_y)
        flag_pct_slope = fit_line_pct_slope(flag_x, flag_y)

        slope_strength_ratio = safe_ratio(
            abs(flag_pct_slope),
            max(abs(pole_pct_slope), 1e-9),
        )

        pole_avg_range = pole_df["High"].sub(pole_df["Low"]).mean()
        flag_avg_range = flag_df["High"].sub(flag_df["Low"]).mean()
        avg_range_ratio = safe_ratio(flag_avg_range, max(pole_avg_range, 1e-9))

        flag_len_ratio = safe_ratio(len(flag_df), max(len(pole_df), 1))

        if pole_is_bullish:
            if flag_slope <= 0:
                flag_direction_score = 1.0
            elif slope_strength_ratio <= 0.5:
                flag_direction_score = 0.6
            else:
                flag_direction_score = 0.0
        else:
            if flag_slope >= 0:
                flag_direction_score = 1.0
            elif slope_strength_ratio <= 0.5:
                flag_direction_score = 0.6
            else:
                flag_direction_score = 0.0

        if pole_move_pct < 0.015:
            return empty_result

        if range_ratio > 1.20:
            return empty_result

        if slope_strength_ratio > 1.50:
            return empty_result

        rule_score = flag_score(
            pole_move_pct=pole_move_pct,
            range_ratio=range_ratio,
            slope_strength_ratio=slope_strength_ratio,
            avg_range_ratio=avg_range_ratio,
            flag_len_ratio=flag_len_ratio,
            flag_direction_score=flag_direction_score,
        )

        if self._model is not None:
            try:
                ml_score = learned_confidence(
                    pole_move_pct=pole_move_pct,
                    range_ratio=range_ratio,
                    slope_strength_ratio=slope_strength_ratio,
                    avg_range_ratio=avg_range_ratio,
                    flag_len_ratio=flag_len_ratio,
                    flag_direction_score=flag_direction_score,
                    model=self._model,
                )

                overall_score = 0.7 * ml_score + 0.3 * rule_score
            except Exception:
                overall_score = rule_score
        else:
            overall_score = rule_score

        confidence = clamp_0_100(overall_score)

        if len(flag_df) >= 2:
            flag_body_df = flag_df.iloc[:-1]
            breakout_high = flag_body_df["High"].max()
            breakout_low = flag_body_df["Low"].min()
        else:
            breakout_high = flag_df["High"].max()
            breakout_low = flag_df["Low"].min()

        last_close = float(df.iloc[-1]["Close"])

        confirmed = confidence >= 50 and (
            (pole_is_bullish and last_close > float(breakout_high))
            or (pole_is_bearish and last_close < float(breakout_low))
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

        local_flag_pivots = detect_pivots(
            flag_df.reset_index(drop=True),
            lookback=2,
            min_move_pct=0.003,
        )

        if local_flag_pivots is not None and len(local_flag_pivots) > 0:
            for _, row in local_flag_pivots.iterrows():
                local_idx = int(row["candle_index"])
                global_idx = int(flag_df.index[0] + local_idx)

                best_pivots.append({
                    "type": row["pivot_type"],
                    "price": float(row["pivot_price"]),
                    "time": df.iloc[global_idx]["Datetime"],
                    "label": f"{row['pivot_type']}",
                })
        else:
            flag_high_idx = int(flag_df["High"].idxmax())
            flag_low_idx = int(flag_df["Low"].idxmin())

            best_pivots.append({
                "type": "flag_high",
                "price": float(flag_high),
                "time": df.iloc[flag_high_idx]["Datetime"],
                "label": "FlagHigh",
            })

            best_pivots.append({
                "type": "flag_low",
                "price": float(flag_low),
                "time": df.iloc[flag_low_idx]["Datetime"],
                "label": "FlagLow",
            })

        best_pivots = sorted(
            best_pivots,
            key=lambda x: pd.to_datetime(x["time"]),
        )

        deduped_pivots = []
        seen = set()

        for p in best_pivots:
            if p["time"] not in seen:
                deduped_pivots.append(p)
                seen.add(p["time"])

        return {
            "confirmed": bool(confirmed),
            "confidence": round(float(confidence), 2),
            "pivots": deduped_pivots,
            "direction": direction,
        }
