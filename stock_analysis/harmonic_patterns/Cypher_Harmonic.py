import pandas as pd
from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.confidence_models.harmonic_patterns.cypher_model import load_model, learned_confidence
from stock_analysis.tools.chart_utils import detect_pivots
from stock_analysis.tools.harmonic_utils import (
    compute_traces_cypher,
    extract_recent_xabcd_candidates,
    build_extended_harmonic_context,
    harmonic_detect_pivots,
    clip_confidence,
    harmonic_ratio_similarity,
    is_valid_xabcd_alternating,
)


class CypherHarmonicPattern(BasePattern):
    enabled = True
    category = Category.HARMONIC.value

    CYPHER_RANGES = {
        "XAB": (0.382, 0.618),
        "ABC": (1.272, 1.414),
        "BCD": (0.618, 0.786)
    }
    
    def __init__(self):
        self._model, self._scaler = load_model()
    
    def is_cypher_candidate(self, retraces, tolerance: float = 0.08):
        for leg, (lo, hi) in self.CYPHER_RANGES.items():
            val = retraces[leg]
            if not (lo * (1 - tolerance) <= val <= hi * (1 + tolerance)):
                return False
        return True
       
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
            filter_fn=self.is_cypher_candidate,
            traces_fn=compute_traces_cypher
        )

        if not candidates:
            return {
                "confirmed": False,
                "confidence": 0,
            }

        best_confidence = 0.0
        best_candidate = None
        best_direction = "neutral"
        
        for candidate in candidates:
            if not is_valid_xabcd_alternating(candidate):
                continue

            ctx = build_extended_harmonic_context(candidate)
            if ctx is None:
                continue

            ratios = ctx["ratios"]
            direction = ctx["direction"]

            if direction == "unknown":
                continue

            ab_xa = ratios["AB_XA"]
            bc_xa = ratios["BC_XA"]  
            cd_xc = ratios["CD_XC"]
            
            actual = { "XAB": ab_xa, "ABC": bc_xa, "BCD": cd_xc }
            ideals = { "XAB": 0.382, "ABC": 1.272, "BCD": 0.786 }

            rule_based_conf = harmonic_ratio_similarity(actual, ideals, self.CYPHER_RANGES)

            ml_based_conf = learned_confidence(ab_xa, bc_xa, cd_xc, self._model, self._scaler)
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