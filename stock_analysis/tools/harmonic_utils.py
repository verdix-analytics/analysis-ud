import pandas as pd
import numpy as np
from pyharmonics.technicals import Technicals
from itertools import combinations


# 1. Basic numeric helpers
def safe_ratio(numerator: float, denominator: float) -> float:
    """
    Safe division to avoid division by zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def pct_diff(a: float, b: float) -> float:
    """
    Percentage difference between two values.

    Example:
        a=100, b=105 -> 0.05
    """
    if a == 0 and b == 0:
        return 0.0

    base = (abs(a) + abs(b)) / 2
    if base == 0:
        return 0.0

    return abs(a - b) / base


def is_close_pct(a: float, b: float, tol: float = 0.03) -> bool:
    """
    Check whether two values are close within percentage tolerance.
    """
    return pct_diff(a, b) <= tol


def in_range(value: float, low: float, high: float) -> bool:
    """
    Check whether value is inside [low, high].
    """
    return low <= value <= high


# 2. Fibonacci matching helpers
def is_fib_match(value: float, target: float, tol: float = 0.05) -> bool:
    """
    Check whether a ratio is close to a target Fibonacci level.

    Example:
        is_fib_match(0.62, 0.618, tol=0.05)
    """
    return abs(value - target) <= tol


def fib_score(value: float, target: float, tol: float = 0.05) -> float:
    """
    Score how close a ratio is to a Fibonacci target.

    Returns a score in [0, 1]:
        1.0 -> exact match
        0.0 -> outside tolerance
    """
    diff = abs(value - target)
    if diff > tol:
        return 0.0

    return max(0.0, 1 - diff / max(tol, 1e-9))


def range_score(value: float, low: float, high: float) -> float:
    """
    Score how well a value lies inside a target interval.

    Returns score in [0, 1]:
        - 1.0 if exactly at interval midpoint
        - decreases toward boundaries
        - 0.0 if outside the interval
    """
    if low <= value <= high:
        mid = (low + high) / 2
        half = (high - low) / 2
        return 1 - 0.5 * abs(value - mid) / half if half > 0 else 1
    
    width = high - low if high > low else 1
    if value < low:
        overshoot = (low - value) / width
    else:
        overshoot = (value - high) / width
    
    return max(0, 0.5 - overshoot)


# 3. Pivot extraction helpers
def get_last_n_pivots(pivots: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Get last n pivots from pivot dataframe.
    """
    if pivots is None or pivots.empty:
        return pd.DataFrame(columns=getattr(pivots, "columns", None))

    return pivots.tail(n).reset_index(drop=True)


def extract_xabcd_from_pivots(pivots: pd.DataFrame) -> dict | None:
    """
    Extract X, A, B, C, D from exactly 5 pivots.

    Expected pivot dataframe columns:
        ['Datetime', 'pivot_type', 'pivot_price', 'candle_index']

    Returns
    -------
    dict or None
        {
            "types": [...],
            "prices": [...],
            "indices": [...],
            "times": [...]
        }
    """
    if pivots is None or len(pivots) != 5:
        return None

    pivots = pivots.reset_index(drop=True)

    return {
        "types": pivots["pivot_type"].tolist(),
        "prices": pivots["pivot_price"].tolist(),
        "indices": pivots["candle_index"].tolist(),
        "times": pivots["Datetime"].tolist(),
    }

def compute_traces(X, A, B, C, D):
    bullish = X < A  # X is a low, A is a high

    if bullish:
        XA = A - X
        AB = A - B
        BC = C - B
        XAB   = safe_ratio(AB,XA)
        ABC   = safe_ratio(BC,AB)
        BCD   = safe_ratio((C - D), BC)
        XABCD = safe_ratio((A - D), XA)
    else:
        XA = X - A
        AB = B - A
        BC = B - C
        XAB   = safe_ratio(AB, XA)
        ABC   = safe_ratio(BC, AB)
        BCD   = safe_ratio((D - C), BC)
        XABCD = safe_ratio(D - A, XA) 


    if XA == 0 or AB == 0 or BC == 0:
        return None

    return {
        "XAB":   round(XAB,   6),
        "ABC":   round(ABC,   6),
        "BCD":   round(BCD,   6),
        "XABCD": round(XABCD, 6),
    }

def compute_traces_cypher(X, A, B, C, D):
    bullish = X < A

    if bullish:
        XA = A - X
        AB = A - B
        BC = C - B
        XC = C - X  
        CD = C - D
    else:
        XA = X - A
        AB = B - A
        BC = B - C
        XC = abs(X - C)          
        CD = D - C

    if XA == 0 or AB == 0 or BC == 0 or XC == 0:
        return None

    return {
        "XAB":   round(safe_ratio(AB, XA),   6),
        "ABC":   round(safe_ratio(BC, XA),   6),   
        "BCD":   round(safe_ratio(CD, XC),   6)
    }
    
def compute_traces_shark(O, A, B, C, D):
    bullish = O < A

    if bullish:
        OA = A - O
        AB = A - B
        BC = C - B  
        CD = C - D
    else:
        OA = O - A
        AB = B - A
        BC = B - C
        CD = D - C

    if OA == 0 or AB == 0 or BC == 0:
        return None

    return {
        "XAB":   round(safe_ratio(AB, OA),  6),
        "ABC":   round(safe_ratio(BC, OA),  6), 
        "BCD":   round(safe_ratio(CD, BC),  6),
        "XABCD": round(safe_ratio(A - D, OA) if bullish else safe_ratio(D - A, OA), 6),
    }
    
def extract_recent_xabcd_candidates(pivots_df: pd.DataFrame, 
                                    max_candidates: int = 100, 
                                    filter_fn=None, 
                                    search_window: int=20,
                                    traces_fn = compute_traces,
                                    require_alternating: bool=True,
                                    validate_fn=None) -> list:
    """
    Extract rolling 5-pivot XABCD candidates from the most recent pivots.

    Parameters
    ----------
    pivots : pd.DataFrame
    max_candidates : int
        Only keep the most recent few candidate windows for efficiency.

    Returns
    -------
    list[dict]
        Each item has the same structure as extract_xabcd_from_pivots(...)
    """
    if pivots_df is None or len(pivots_df) < 5:
        return []

    pivots = pivots_df.to_dict('records')
    n = len(pivots)
    candidates = []
    
    for m in range(n-1, 3, -1):
        D = pivots[m]
        
        start_idx = max(0, m-search_window)
        
        for indices in combinations(range(start_idx, m), 4):
            i, j, k, l = indices
            
            X, A, B, C = pivots[i], pivots[j], pivots[k], pivots[l]
            
            if X["pivot_type"] == A["pivot_type"]: continue
            
            types = [X["pivot_type"], A["pivot_type"], B["pivot_type"], C["pivot_type"]], D["pivot_type"]
            
            if require_alternating: 
                points = [X, A, B, C, D]
                if any(points[p]["pivot_type"] == points[p + 1]["pivot_type"] for p in range(4)):
                    continue
            else:
                if X["pivot_type"] == A["pivot_type"]:
                    continue
                if A["pivot_type"] == B["pivot_type"]:
                    continue
                
            
            p_x, p_a, p_b, p_c, p_d = X["pivot_price"], A["pivot_price"], B["pivot_price"], C["pivot_price"], D["pivot_price"]
            
            if validate_fn and not validate_fn(p_x, p_a, p_b, p_c, p_d):
                continue
            
            retraces = traces_fn(p_x, p_a, p_b, p_c, p_d)
            
            if retraces is None:
                continue
            
            if filter_fn and not filter_fn(retraces):
                continue
            
            candidates.append({
                "prices": [p_x, p_a, p_b, p_c, p_d],
                "types": [X["pivot_type"], A["pivot_type"], B["pivot_type"], C["pivot_type"], D["pivot_type"]],
                "times": [X["Datetime"], A["Datetime"], B["Datetime"], C["Datetime"], D["Datetime"]],
                "indices": D["candle_index"],
                "retraces": retraces,
            })
    
    return candidates


# 4. XABCD move & ratio computation
def compute_xabcd_moves(candidate: dict) -> dict | None:
    """
    Compute signed and absolute move lengths for XABCD.

    candidate must contain:
        candidate["prices"] = [X, A, B, C, D]

    Returns
    -------
    dict or None
        {
            "X": x, "A": a, "B": b, "C": c, "D": d,
            "XA": ..., "AB": ..., "BC": ..., "CD": ..., "XD": ...,
            "XA_len": ..., "AB_len": ..., "BC_len": ..., "CD_len": ..., "XD_len": ...
        }
    """
    if candidate is None or "prices" not in candidate or len(candidate["prices"]) != 5:
        return None

    x, a, b, c, d = candidate["prices"]

    XA = a - x
    AB = b - a
    BC = c - b
    CD = d - c
    XD = d - x

    return {
        "X": x,
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "XA": XA,
        "AB": AB,
        "BC": BC,
        "CD": CD,
        "XD": XD,
        "XA_len": abs(XA),
        "AB_len": abs(AB),
        "BC_len": abs(BC),
        "CD_len": abs(CD),
        "XD_len": abs(XD),
    }


def compute_xabcd_ratios(moves: dict) -> dict | None:
    """
    Compute common harmonic ratios.

    Returns
    -------
    dict or None
        {
            "AB_XA": AB/XA,
            "BC_AB": BC/AB,
            "CD_BC": CD/BC,
            "XD_XA": XD/XA
        }
    """
    if moves is None:
        return None

    XA_len = moves["XA_len"]
    AB_len = moves["AB_len"]
    BC_len = moves["BC_len"]
    CD_len = moves["CD_len"]
    XD_len = moves["XD_len"]

    return {
        "AB_XA": safe_ratio(AB_len, XA_len),
        "BC_AB": safe_ratio(BC_len, AB_len),
        "CD_BC": safe_ratio(CD_len, BC_len),
        "XD_XA": safe_ratio(XD_len, XA_len),
    }


def compute_extended_xabcd_ratios(moves: dict) -> dict | None:
    """
    Compute extended harmonic ratios for patterns like Cypher.

    Returns
    -------
    dict or None
        {
            "AB_XA": ...,
            "BC_AB": ...,
            "CD_BC": ...,
            "XD_XA": ...,
            "XC_XA": ...,
            "XD_XC": ...,
            "AC_XA": ...,
        }
    """
    if moves is None:
        return None

    X = moves["X"]
    A = moves["A"]
    C = moves["C"]
    D = moves["D"]

    XA_len = moves["XA_len"]
    AB_len = moves["AB_len"]
    BC_len = moves["BC_len"]
    CD_len = moves["CD_len"]
    XD_len = moves["XD_len"]

    XC_len = abs(C - X)
    AC_len = abs(C - A)

    return {
        "AB_XA": safe_ratio(AB_len, XA_len),
        "BC_AB": safe_ratio(BC_len, AB_len),
        "CD_BC": safe_ratio(CD_len, BC_len),
        "XD_XA": safe_ratio(XD_len, XA_len),
        "XC_XA": safe_ratio(XC_len, XA_len),
        "XD_XC": safe_ratio(XD_len, XC_len),
        "AC_XA": safe_ratio(AC_len, XA_len),
        "BC_XA": safe_ratio(BC_len, XA_len),
        "CD_XC": safe_ratio(CD_len, XC_len)
    }


def compute_shark_ratios(candidate: dict) -> dict | None:
    """
    Compute Shark-specific ratios.

    candidate["prices"] should represent [X, A, B, C, D].

    Returns
    -------
    dict or None
        {
            "AB_XA": ...,
            "BC_AB": ...,
            "XC_XA": ...,
            "XD_XC": ...,
            "XD_XA": ...,
            "CD_BC": ...,
        }
    """
    moves = compute_xabcd_moves(candidate)
    if moves is None:
        return None

    X = moves["X"]
    C = moves["C"]
    D = moves["D"]

    XA_len = moves["XA_len"]
    AB_len = moves["AB_len"]
    BC_len = moves["BC_len"]
    CD_len = moves["CD_len"]
    XD_len = moves["XD_len"]

    XC_len = abs(C - X)

    return {
        "AB_XA": safe_ratio(AB_len, XA_len),
        "BC_AB": safe_ratio(BC_len, AB_len),
        "XC_XA": safe_ratio(XC_len, XA_len),
        "XD_XC": safe_ratio(XD_len, XC_len),
        "XD_XA": safe_ratio(XD_len, XA_len),
        "CD_BC": safe_ratio(CD_len, BC_len),
    }


# 5. Structure / direction helpers
def is_valid_xabcd_alternating(candidate: dict) -> bool:
    """
    Check whether pivot types alternate properly.

    Valid alternating structures:
        H-L-H-L-H
        L-H-L-H-L
    """
    if candidate is None or "types" not in candidate:
        return False

    types = candidate["types"]

    return types in [
        ["H", "L", "H", "L", "H"],
        ["L", "H", "L", "H", "L"],
    ]


def is_bullish_xabcd(moves: dict) -> bool:
    """
    Bullish harmonic structure:
        XA down, AB up, BC down, CD up
    """
    if moves is None:
        return False

    return (
        moves["XA"] < 0
        and moves["AB"] > 0
        and moves["BC"] < 0
        and moves["CD"] > 0
    )


def is_bearish_xabcd(moves: dict) -> bool:
    """
    Bearish harmonic structure:
        XA up, AB down, BC up, CD down
    """
    if moves is None:
        return False

    return (
        moves["XA"] > 0
        and moves["AB"] < 0
        and moves["BC"] > 0
        and moves["CD"] < 0
    )


def structure_direction(candidate: dict, moves: dict) -> str:
    """
    Return:
        'bullish' / 'bearish' / 'unknown'
    """
    if not is_valid_xabcd_alternating(candidate):
        return "unknown"

    if is_bullish_xabcd(moves):
        return "bullish"

    if is_bearish_xabcd(moves):
        return "bearish"

    return "unknown"


def is_potential_shark_structure(candidate: dict) -> bool:
    """
    Pre-filter for Shark candidates.
    Ensures the structure is a valid alternating 5-point pattern.
    """
    if candidate is None:
        return False

    if not is_valid_xabcd_alternating(candidate):
        return False

    moves = compute_xabcd_moves(candidate)
    if moves is None:
        return False

    return (
        is_bullish_xabcd(moves)
        or is_bearish_xabcd(moves)
    )


# 6. Scoring helpers
def average_scores(scores: list[float]) -> float:
    """
    Average a list of scores in [0, 1].
    """
    if not scores:
        return 0.0

    return float(sum(scores) / len(scores))


def clip_confidence(score_0_to_1: float) -> float:
    """
    Convert score from [0, 1] to [0, 100], clipped.
    """
    return float(np.clip(score_0_to_1 * 100, 0, 100))


# 7. Context builders
def build_harmonic_context(candidate: dict) -> dict | None:
    """
    Build a compact context dict for standard harmonic detectors.

    Returns
    -------
    dict or None
        {
            "candidate": ...,
            "moves": ...,
            "ratios": ...,
            "direction": ...
        }
    """
    if candidate is None:
        return None

    moves = compute_xabcd_moves(candidate)
    if moves is None:
        return None

    ratios = compute_xabcd_ratios(moves)
    direction = structure_direction(candidate, moves)

    return {
        "candidate": candidate,
        "moves": moves,
        "ratios": ratios,
        "direction": direction,
    }


def build_extended_harmonic_context(candidate: dict) -> dict | None:
    """
    Build harmonic context with extended ratios.
    Useful for Cypher and other custom harmonic patterns.
    """
    if candidate is None:
        return None

    moves = compute_xabcd_moves(candidate)
    if moves is None:
        return None

    ratios = compute_extended_xabcd_ratios(moves)
    direction = structure_direction(candidate, moves)

    return {
        "candidate": candidate,
        "moves": moves,
        "ratios": ratios,
        "direction": direction,
    }


def build_shark_context(candidate: dict) -> dict | None:
    """
    Build Shark-specific harmonic context.
    """
    if candidate is None:
        return None

    moves = compute_xabcd_moves(candidate)
    if moves is None:
        return None

    ratios = compute_shark_ratios(candidate)
    direction = structure_direction(candidate, moves)

    return {
        "candidate": candidate,
        "moves": moves,
        "ratios": ratios,
        "direction": direction,
    }

def harmonic_detect_pivots(df: pd.DataFrame):
    """
    Build Pivots from dataframe and extract them to dataframes

    Args:
        df (pd.DataFrame): Financial Asset OHLCV data
    """
    tech = Technicals(df, "", "", peak_spacing=2)
    
    pivots = []
    for idx, price, ptype in tech.peak_data:
        pivots.append({
            "Datetime": df.loc[df.index[idx], 'Datetime'],
            "pivot_type": "H" if ptype == 1 else "L",
            "pivot_price": float(price),
            "candle_index": int(idx)
        })
    
    pivots = pd.DataFrame(pivots).sort_values("candle_index").reset_index(drop=True)
    
    # records = pivots.to_dict("records")
    # cleaned = [records[0]]
    # for p in records[1:]:
    #     if p['pivot_type'] != cleaned[-1]['pivot_type']:
    #         cleaned.append(p)
    #     else:
    #         # Keep the more extreme pivot of the two
    #         if p['pivot_type'] == 'H':
    #             if p['pivot_price'] > cleaned[-1]['pivot_price']:
    #                 cleaned[-1] = p
    #         else:
    #             if p['pivot_price'] < cleaned[-1]['pivot_price']:
    #                 cleaned[-1] = p
    return pivots

def harmonic_ratio_similarity(actuals: dict, 
                              ideals: dict, 
                              ranges:dict, 
                              tolerance: float = 0.03) -> float:
        """
        Score each leg against the Bat ranges using a tolerance-aware 
        similarity calculation.
        """
        scores = []

        for leg, val in actuals.items():
            lo, hi = ranges[leg]
            
            tol_lo = lo * (1 - tolerance)
            tol_hi = hi * (1 + tolerance)
            in_range = tol_lo <= val <= tol_hi

            if val < tol_lo or val > tol_hi:
                scores.append(0.0)
                continue
        
            range_width = (tol_hi - tol_lo)
            half_width = (range_width / 2) if range_width > 0 else 0.05
            
            ideal = ideals[leg]
            distance = abs(val - ideal)
            
            sim = max(0.0, 1.0 - (distance / half_width))
            scores.append(sim)

        return round((sum(scores) / len(scores)) * 100, 2)