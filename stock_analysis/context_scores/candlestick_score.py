from collections import Counter
import math
import numpy as np
from stock_analysis.context_scores.utilities import recency_bias, trend_score, trend_structure_bonus

WEIGHT = {
    "score": 0.35,  
    "confidence": 0.25,  
    "pre_trend": 0.25,  
    "recency": 0.15
}

TECHNICAL_WEIGHTS = {
    "mss_score": 0.85,
    "candlestick_score": 0.15
}

PATTERN_RELIABILITY = {
    "BearishEngulfingCandlestickPattern": 0.9,
    "BearishHaramiCandlestickPattern": 0.7,
    "BullishEngulfingCandlestickPattern": 0.9,
    "BullishHaramiCandlestickPattern":0.7,
    "DojiCandlestickPattern": 0.5,
    "EveningStarCandlestickPattern": 1,
    "HammerCandlestickPattern": 0.75,
    "InvertedHammerCandlestickPattern": 0.65,
    "MorningStarCandlestickPattern": 1,
    "ShootingStarCandlestickPattern": 0.75,
    "SpinningTopCandlestickPattern": 0.45
}

DEFAULT_RELIABILITY = 0.6

def pattern_reliability(pattern: str) -> float:
    for key, value in PATTERN_RELIABILITY.items():
        if key == pattern:
            return value
    return DEFAULT_RELIABILITY

def volatility_gate(formations: list[dict]) -> float:
    if len(formations) < 2:
        return 1
    
    confidences = [f.get("confidence", 50) / 100 for f in formations]
    std = float(np.std(confidences))
    
    gate = 1 - min(std * 2, 0.5)
    return gate

def pattern_count_scale(n: int) -> float:
    if n == 0:
        return 0
    return 1-0.3 * math.exp(-0.5 * (n-1))

def score_recency(formations: list[dict], max_window_count: int = 10) -> float:
    if not formations: 
        return 0
    
    HALF_LIFE_BARS = 5
    lam = math.log(2)/HALF_LIFE_BARS
    
    scores = []
    for f in formations:
        wc = f.get("window_count", 1)
        scores.append(math.exp(-lam * max(0, wc - 1)))
        
    scores_sorted = sorted(scores, reverse = True)
    pos_weights = [1 / (i+1) for i in range(len(scores_sorted))]
    weighted = sum(s * w for s, w in zip(scores_sorted, pos_weights))
    return float(weighted/sum(pos_weights))

def direction_agreement(formations: list[dict]) -> tuple[float, str]:
    if not formations:
        return 0.5, "neutral"
    
    bull = sum(
        pattern_reliability(f.get("pattern", ""))
        for f in formations
        if f.get("signal", {}).get("direction") == "bullish"
    )
    
    bear = sum(
        pattern_reliability(f.get("pattern", ""))
        for f in formations
        if f.get("signal", {}).get("direction") == "bearish"
    )
    
    total = bear + bull
    
    if total == 0:
        return 0.5, "neutral"
    
    bull_ratio = bull / total
    
    k = 8
    sigmoid_score = 1 / ( 1 + math.exp(-k * (bull_ratio - 0.5)))
    
    if bull_ratio >= 0.60:
        bias = "bullish"
    elif bull_ratio <= 0.40:
        bias = "bearish"
    else:
        bias = "neutral"
    
    return sigmoid_score, bias

def score_confidence(formations: list[dict]) -> float:
    if not formations:
        return 0
    
    weighted_sum = 0
    weight_total = 0
    
    for f in formations:
        reliability = pattern_reliability(f.get("pattern", ""))
        confidence = f.get("confidence", 0)/100
        weighted_sum += confidence * reliability
        weight_total += reliability
        
    return weighted_sum / weight_total if weight_total > 0 else 0

def pre_trend_alignment(formations: list[dict], bias: str) -> float:
    if bias == "neutral" or not formations:
        return 0.5
    scores = []
    
    for f in formations:
        pre_trend = f.get("pre_trend")
        direction = f.get("signal", {}).get("direction", bias)
        
        base = trend_score(pre_trend, direction)
        bonus = trend_structure_bonus(pre_trend, direction)
        
        adjusted = min(1, base * bonus)
        scores.append(adjusted)
        
    return float(np.mean(scores)) if scores else 0.5

def candlestick_score(formations: list[dict]) -> dict:
    if not formations:
        return {
            "score": 0.5,
            "bias": "neutral",
            "agreement": 0,
            "bull_score": 0,
            "bear_score": 0,
            "pattern_count": 0,
            "contributing_patterns": []
        }
    
    score, bias = direction_agreement(formations)
    conf_score = score_confidence(formations)
    pre_trend_score = pre_trend_alignment(formations, bias)
    recency_score = score_recency(formations)
    mss_score = formations[0].get("technical_analysis",{}).get("Market Sentiment", 0)
    
    candlestick_score = (score * WEIGHT["score"] + conf_score * WEIGHT["confidence"] 
                    + pre_trend_score * WEIGHT["pre_trend"] + recency_score * WEIGHT["recency"])
    
    count_scale = pattern_count_scale(len(formations))
    candlestick_score *= count_scale
    
    vol_gate = volatility_gate(formations)
    candlestick_score *= vol_gate
    
    candlestick_score = float(np.clip(candlestick_score, 0, 1))
    
    final_score = TECHNICAL_WEIGHTS["mss_score"] * mss_score + TECHNICAL_WEIGHTS["candlestick_score"] * candlestick_score
    
    final_score = -final_score if bias == "bearish" else final_score
    
    raw_bull = score * conf_score * count_scale
    raw_bear = (1-score) * conf_score * count_scale
    bull_score = float(np.clip(raw_bull, 0, 1))
    bear_score = float(np.clip(raw_bear, 0, 1))
    
    contributing_patterns = []
    
    for f in formations:
        contributing_patterns.append({
            "pattern": f.get("pattern"),
            "direction": f.get("signal", {}).get("direction", "neutral"),
            "window": f.get("window_count")
        })
        
    return {
        "score":         round(final_score, 4),
        "bias":          bias,
        "agreement":     round(score, 4),
        "bull_score":    round(bull_score, 4),
        "bear_score":    round(bear_score, 4),
        "pattern_count": len(formations),
        "technical_analysis": formations[0].get("technical_analysis", {}),
        "contributing_patterns": contributing_patterns,
    }