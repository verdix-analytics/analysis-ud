from datetime import datetime

CADENCE = {
    "harmonic": 24*60*60, 
    "chart" : 60*60, 
    "candlestick": 60
}

_last_run: dict[str, datetime] = {
    "harmonic":    datetime.min,
    "chart":       datetime.min,
    "candlestick": datetime.min,
}

def is_due(pattern_type: str) -> bool:
    elapsed = (datetime.now() - _last_run[pattern_type]).total_seconds()
    return elapsed >= CADENCE[pattern_type]

def mark_run(pattern_type: str):
    _last_run[pattern_type] = datetime.now()