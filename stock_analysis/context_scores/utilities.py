import math
import pandas as pd

def recency_bias(window_count: int, decay: float = 0.3) -> float:
    return math.exp(-decay * (window_count-1))

def confidence_weight(confidence: float) -> float:
    return (confidence / 100) ** 2

def signal_age_days(signal: dict) -> float | None:
    pivots = signal.get("signal", {}).get("pivots") or signal.get("pivots")
    if not pivots:
        return None
    try:
        last_time = pd.Timestamp(pivots[-1]["time"])
        return (pd.Timestamp.now() - last_time).total_seconds() / 86400
    except Exception:
        return None

def signal_age_decay(formation: dict, validity_days: float = 30) -> float:
    age_days = signal_age_days(formation)
    if age_days is None or age_days <= 0: return 1
    
    k = 4 / validity_days
    weight = 1 / (1 + math.exp(k* (age_days - validity_days)))
    return round(max(weight, 0.001), 4)

def trend_score(trend: dict | None, direction: str) -> float:
    #print(trend)
    if not trend or trend.get("label") == "unknown":
        return 0.5
    
    label = trend["label"]
    strength = min(trend.get("strength", 0), 100) / 100
    
    expected_obs = (
        (direction == "bullish" and label == "downtrend") or
        (direction == "bearish" and label == "uptrend")
    )
    
    sideways_obs = label == "sideways"
    
    if expected_obs: return 0.6 + 0.4 * strength
    elif sideways_obs: return 0.4 + 0.1 * strength
    else: return max(0, 0.2 - 0.2 * strength)

def trend_structure_bonus(trend: dict | None, direction: str) -> float:
    if not trend:
        return 1
    
    if direction == "bullish":
        q1 = trend.get("lower_high_ratio", 0)
        q2 = trend.get("lower_low_ratio", 0)
    else:
        q1 = trend.get("higher_high_ratio", 0)
        q2 = trend.get("higher_low_ratio", 0)
    
    structure_quality = (q1 + q2) / 2
    bonus = 1 + 0.15 * max(0, structure_quality - 0.55)
    return round(bonus, 4)
    