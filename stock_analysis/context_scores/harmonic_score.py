import math
import pandas as pd
from typing import Optional
import collections
from .utilities import recency_bias, confidence_weight, signal_age_decay, trend_score, trend_structure_bonus

TECHNICAL_WEIGHTS = {
    "mss_score": 0.3,
    "harmonic_score": 0.7    
}

def direction_majority(formations: list[dict]) -> str:
    if not formations:
        return "neutral"
    directions = [f.get("signal", {}).get("direction", "neutral") for f in formations]
    return collections.Counter(directions).most_common(1)[0][0]
        
    
def harmonic_score(formations: list[dict]) -> dict:
    if not formations:
        return {
            "score": 0,
            "bias": "neutral",
            "agreement": 0,
            "bull_score": 0,
            "bear_score": 0,
            "pattern_count": 0,
            "pre_trend_score": 0,
            "contributing_patterns": []
        }
    
    bull_score, bear_score = 0,0
    total_weight = 0
    contributing_patterns = []
    
    pre_alignment_scores = []
    structure_bonus = []
    
    
    for formation in formations:
        confidence = formation.get("confidence", 0)
        direction = formation["signal"]["direction"]
        window_count = formation["window_count"]
        pattern = formation.get("pattern")
        pre_trend = formation.get("pre_trend")
        
        r_w = recency_bias(window_count)
        c_w = confidence_weight(confidence)
        a_w = signal_age_decay(formation)
        
        pre_score = trend_score(pre_trend, direction)
        s_bonus = trend_structure_bonus(pre_trend, direction)
        
        pre_alignment_scores.append(pre_score)
        structure_bonus.append(s_bonus)
        combined_weight = r_w * c_w * pre_score * a_w
        
        total_weight += combined_weight
        
        if direction == "bullish":
            bull_score += combined_weight
        elif direction == "bearish":
            bear_score += combined_weight
        
        contributing_patterns.append({
            "pattern": pattern,
            "direction": direction,
            "recency_weight": round(r_w, 4),
            "confidence_weight": round(c_w, 4),
            "pre_trend_score": round(pre_score, 4),
            "structure_bonus": round(s_bonus, 4),
            "combined_weight": round(combined_weight, 4),
            "window": window_count
        })
    
    mss_score = formations[0].get("technical_analysis", {}).get("Market Sentiment", 0)
    norm_bull = bull_score / total_weight if total_weight else 0
    norm_bear = bear_score / total_weight if total_weight else 0
    net_diff = norm_bull - norm_bear
    net_sum = norm_bull + norm_bear
    
    agreement = abs(net_diff)/net_sum if net_sum else 0
    
    bias = direction_majority(formations)
    
    agree_count = sum( 1 for formation in formations if formation["signal"]["direction"] == bias)
    
    consensus = agree_count / len(formations)
    agreeing_formations = [formation for formation in formations if formation.get("direction") == bias]
    unique_agreeing = len(set(formation["pattern"] for formation in agreeing_formations))
    diversity_mult = 1 + 0.05 * (unique_agreeing - 1)
    
    avg_pre_score = sum(pre_alignment_scores) / len(pre_alignment_scores)
    avg_s_bonus = sum(structure_bonus) / len(structure_bonus)
    
    raw_score = (abs(net_diff)*0.4 + consensus*0.35 + avg_pre_score*0.25) * agreement * diversity_mult * avg_s_bonus
    final_score = mss_score*TECHNICAL_WEIGHTS["mss_score"] + TECHNICAL_WEIGHTS["harmonic_score"]*min(raw_score, 1)
    
    sign = -1 if bias == 'bearish' else 1
    
    return {
        "score": round(final_score, 3) * sign,
        "bias": bias,
        "agreement": round(agreement, 3),
        "bull_score": round(norm_bull * 100, 2),
        "bear_score": round(norm_bear * 100, 2),
        "pattern_count": len(formations),
        "pre_trend_score": round(avg_pre_score, 4),
        "technical_analysis": formations[0].get("technical_analysis",{}),
        "contributing_patterns": sorted(
            contributing_patterns, key=lambda x: x["combined_weight"], reverse=True
        )
    }
        