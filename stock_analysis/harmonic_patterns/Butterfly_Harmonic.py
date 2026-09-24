import pandas as pd
from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.confidence_models.harmonic_patterns.butterfly_model import load_model, learned_confidence
from stock_analysis.tools.chart_utils import detect_pivots
from stock_analysis.tools.harmonic_utils import (
    extract_recent_xabcd_candidates,
    build_harmonic_context,
    harmonic_detect_pivots,
    harmonic_ratio_similarity,
    clip_confidence,
    is_valid_xabcd_alternating
)



class ButterflyHarmonicPattern(BasePattern):
    enabled = True
    category = Category.HARMONIC.value
    
    BUTTERFLY_RANGES = {
        "XAB":   (0.786, 0.786),  
        "ABC":   (0.382, 0.886),
        "BCD":   (1.618, 2.618),  
        "XABCD": (1.272, 1.618)
    }
        
    def __init__(self):
        self._model, self._scaler = load_model()
    
    def is_butterfly_candidate(self, retraces, tolerance: float = 0.08):
        for leg, (lo, hi) in self.BUTTERFLY_RANGES.items():
            val = retraces[leg]
            if not (lo * (1 - tolerance) <= val <= hi * (1 + tolerance)):
                return False
        return True
    
    def butterfly_validate(self, X, A, B, C, D):
        bullish = X < A
        return D < X if bullish else D > X
    
    def _run(self, df: pd.DataFrame) -> dict:
        if len(df) < 15:
            return {
                "confirmed": False,
                "confidence": 0,
            }

        pivots = harmonic_detect_pivots(df)
        
        if len(pivots) < 5:
            return {
                "confirmed": False,
                "confidence": 0,
            }

        candidates = extract_recent_xabcd_candidates(
            pivots,
            max_candidates=100,
            filter_fn=self.is_butterfly_candidate,
            validate_fn=self.butterfly_validate
        )

        if not candidates:
            return {
                "confirmed": False,
                "confidence": 0,
            }

        best_confidence = 0.0
        best_candidate = None
        best_direction = "neutral"
        
        # 3. evaluate each candidate
        for candidate in candidates:
            if not is_valid_xabcd_alternating(candidate):
                continue
            
            ctx = build_harmonic_context(candidate)
            if ctx is None:
                continue

            ratios = ctx["ratios"]
            direction = ctx["direction"]

            if direction == "unknown":
                continue

            ab_xa = ratios["AB_XA"]
            bc_ab = ratios["BC_AB"]
            cd_bc = ratios["CD_BC"]
            xd_xa = ratios["XD_XA"]

            actual = { "XAB": ab_xa, "ABC": bc_ab, "BCD": cd_bc, "XABCD": xd_xa }
            ideals = { "XAB": 0.786, "ABC": 0.618, "BCD": 2.24, "XABCD": 1.272 }
            
            rule_based_conf = harmonic_ratio_similarity(actual, ideals, self.BUTTERFLY_RANGES)

            ml_based_conf = learned_confidence(ab_xa, bc_ab, cd_bc, xd_xa, self._model, self._scaler)
            ml_based_conf = clip_confidence(ml_based_conf)
            
            confidence = (rule_based_conf + ml_based_conf)/2

            if confidence > best_confidence:
                best_confidence = confidence
                best_candidate = candidate
                best_direction = direction

        confirmed = best_confidence > 0
        best_pivots = []
        if best_candidate is not None:
            labels = ['X', 'A', 'B', 'C', 'D']
            for i, label in enumerate(labels):
                best_pivots.append({
                    "type":  best_candidate["types"][i],
                    "price": best_candidate["prices"][i],
                    "time":  best_candidate["times"][i],
                })
            
        return {
            "confirmed": confirmed,
            "pivots": best_pivots,
            "direction": best_direction,
            "confidence": round(best_confidence, 2),
        }