from abc import ABC, abstractmethod
import pandas as pd
from pydantic import ValidationError
from stock_analysis.models import PatternResult
import stock_analysis.config as config
import logging

logger = logging.getLogger(__name__)

class BasePattern(ABC):
    enabled = True
    category = "unknown"
    confidence = 0
    expected_pretrend = None

    @abstractmethod
    def _run(self, df:pd.DataFrame) -> dict:
        pass

    def get_expected_pretrend(self) -> str | None:
        return self.expected_pretrend

    def is_confidence_sufficient(self, confidence: float) -> bool:
        return confidence >= config.DETECTION_CONFIDENCE_THRESHOLD

    def _trend_strength_scale(self, pre_trend: dict | None) -> float:
        if not pre_trend:
            return 0.0

        strength = float(pre_trend.get("strength", 0.0) or 0.0)
        if strength <= 0:
            return 0.0

        reference = max(float(config.TREND_STRENGTH_REFERENCE), 1e-9)
        return max(0.0, min(1.0, strength / (strength + reference)))

    def _apply_pretrend_adjustment(
        self,
        result: PatternResult,
        pre_trend: dict | None = None,
    ) -> PatternResult:
        confidence = float(result.confidence)
        breakdown = dict(result.confidence_breakdown or {})
        metadata = dict(result.confidence_metadata or {})
        breakdown["base_score"] = round(confidence, 2)

        if not pre_trend:
            result.confidence = round(max(0.0, min(100.0, confidence)), 2)
            result.confidence_breakdown = breakdown
            result.confidence_metadata = metadata
            return result

        trend_label = pre_trend.get("label")
        expected_trend = self.get_expected_pretrend()

        # Keep neutral patterns and unknown trend labels on legacy confirmation logic.
        if expected_trend is None or trend_label in {"unknown", None}:
            result.confidence = round(max(0.0, min(100.0, confidence)), 2)
            result.confidence_breakdown = breakdown
            metadata["pretrend_label"] = trend_label
            metadata["expected_pretrend"] = expected_trend
            result.confidence_metadata = metadata
            return result

        base_adjustment = 0.0
        trend_strength = float(pre_trend.get("strength", 0.0) or 0.0)
        trend_scale = self._trend_strength_scale(pre_trend)

        if expected_trend and trend_label == expected_trend:
            base_adjustment = float(config.TREND_MATCH_BONUS)
        elif expected_trend and trend_label == "sideways":
            base_adjustment = -float(config.TREND_SIDEWAYS_PENALTY)
        elif expected_trend and trend_label not in {expected_trend, "unknown", None}:
            base_adjustment = -float(config.TREND_MISMATCH_PENALTY)

        adjustment = base_adjustment * trend_scale

        adjusted_confidence = max(0.0, min(100.0, confidence + adjustment))
        breakdown["pretrend_base_adjustment"] = round(base_adjustment, 2)
        breakdown["pretrend_adjustment"] = round(adjustment, 2)
        breakdown["pretrend_strength"] = round(trend_strength, 4)
        breakdown["pretrend_strength_scale"] = round(trend_scale, 4)
        breakdown["final_score"] = round(adjusted_confidence, 2)
        metadata["pretrend_label"] = trend_label
        metadata["expected_pretrend"] = expected_trend

        result.confidence = round(adjusted_confidence, 2)
        result.confidence_breakdown = breakdown
        result.confidence_metadata = metadata

        if result.structure_detected and not self.is_confidence_sufficient(result.confidence):
            result.confirmed = False
        elif self.category == "candlestick":
            result.confirmed = result.structure_detected and self.is_confidence_sufficient(result.confidence)
        elif result.confirmed and not self.is_confidence_sufficient(result.confidence):
            result.confirmed = False

        return result

    def execute(self, df: pd.DataFrame, pre_trend: dict | None = None) -> PatternResult:
        try:
            base_result = self._run(df)
            result = PatternResult(**base_result)
            return self._apply_pretrend_adjustment(result, pre_trend=pre_trend)
        except ValidationError as e:
            logger.warning(f"{self.__class__.__name__} returned invalid data: {e}")
            return PatternResult(confirmed=False, confidence=0)
        except Exception as e:
            logger.error(f"{self.__class__.__name__}._run failed: {e}")
            return PatternResult(confirmed=False, confidence=0)

