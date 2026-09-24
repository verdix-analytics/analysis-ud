from stock_analysis.base import BasePattern
from stock_analysis.Enums.category import Category
from stock_analysis.confidence_models.harmonic_patterns.shark_model import load_model, learned_confidence
from stock_analysis.tools.chart_utils import detect_pivots
from stock_analysis.tools.harmonic_utils import (
    compute_traces_shark,
    extract_recent_xabcd_candidates,
    build_shark_context,
    harmonic_detect_pivots,
    harmonic_ratio_similarity,
    is_potential_shark_structure,
    range_score,
    fib_score,
    average_scores,
    clip_confidence,
)

import pandas as pd


class SharkHarmonicPattern(BasePattern):
    enabled = True
    category = Category.HARMONIC.value
    
    SHARK_RANGE = {
        "XAB":   (0.382, 0.618),  
        "ABC":   (1.130, 1.618),  
        "BCD":   (1.618, 2.240),
        "XABCD": (0.886, 1.130),  
    }
    
    def __init__(self):
        self._model, self._scaler = load_model()
    
    def is_shark_candidate(self, retraces, tolerance: float = 0.08):
        for leg, (lo, hi) in self.SHARK_RANGE.items():
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

        # 2. Build recent 5-point candidates
        candidates = extract_recent_xabcd_candidates(
            pivots,
            max_candidates=100,
            filter_fn=self.is_shark_candidate,
            traces_fn=compute_traces_shark
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
            # Shark uses original OXABC definition.
            # In current framework, map:
            # current X,A,B,C,D -> original O,X,A,B,C
            if not is_potential_shark_structure(candidate):
                continue

            ctx = build_shark_context(candidate)
            if ctx is None:
                continue

            ratios = ctx["ratios"]
            direction = ctx["direction"]

            if direction == "unknown":
                continue

            ab_xa = ratios["AB_XA"]   # original XA / OX
            bc_ab = ratios["BC_AB"]   # original AB / XA
            cd_bc = ratios["CD_BC"]   # original BC / AB
            xd_xa = ratios["XD_XA"]   # original OC / OX

            actuals = { "XAB": ab_xa, "ABC": bc_ab, "BCD": cd_bc, "XABCD": xd_xa }
            ideals = { "XAB": 0.446,  "ABC": 1.13, "BCD": 1.618, "XABCD": 0.886 }
            rule_based_conf = harmonic_ratio_similarity(actuals, ideals, self.SHARK_RANGE)

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