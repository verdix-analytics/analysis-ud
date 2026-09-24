import numpy as np
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from stock_analysis.tools.harmonic_utils import (
    range_score,
    fib_score,
    clip_confidence,
)

np.random.seed(42)

MODEL_DIR   = os.path.join(os.path.dirname(__file__), '..', "pkl_files")
MODEL_PATH  = os.path.join(MODEL_DIR, "cypher_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "cypher_scaler.pkl")


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: RULE-BASED SCORE
#
# Mirrors CypherHarmonicPattern._run() hard constraints + soft scoring.
#
# NOTE: Cypher uses 3 ratios, not 4:
#   ab_xa  — AB leg as fraction of XA
#   xc_xa  — XC leg as fraction of XA  (replaces BC_AB + CD_BC)
#   xd_xc  — XD retracement of XC      (the D-point completion)
#
# Context is built via build_extended_harmonic_context(), not
# build_harmonic_context(), so BC_AB and CD_BC are not available.
# ─────────────────────────────────────────────────────────────────────

def cypher_score(ab_xa: float, bc_xa: float, cd_xc: float) -> float:
    """
    Rule-based confidence score for the Cypher harmonic pattern.

    Hard constraints (broad admissibility gate, taken from CypherHarmonicPattern):
        AB ∈ [0.382, 0.618] XA  → hard: [0.205, 0.795]
        XC ∈ [1.272, 1.414] XA  → hard: [1.1655, 1.5205]
        XD ≈ 0.786 of XC, tol=0.08 → hard: [0.546, 1.026]

    Soft scoring (textbook targets):
        AB ∈ [0.382, 0.618]
        XC ∈ [1.272, 1.414]
        XD ≈ 0.786 of XC
    """
    # Hard constraints
    # if not (0.205  <= ab_xa <= 0.795 ): return 0.0
    # if not (1.1655 <= xc_xa <= 1.5205): return 0.0
    # if not (0.546  <= xd_xc <= 1.026 ): return 0.0

    # Soft scoring
    score_ab = range_score(ab_xa, 0.382, 0.618)
    score_xc = range_score(bc_xa, 1.272, 1.414)
    score_d  = fib_score(cd_xc,  0.786, tol=0.08)

    overall = (score_ab + score_xc + score_d) / 3
    return clip_confidence(overall)


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: MODEL I/O
# ─────────────────────────────────────────────────────────────────────

def load_model():
    """
    Load saved model and scaler.
    Raises FileNotFoundError if not yet trained.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Cypher model not found. Run `python -m stock_analysis.cypher_model` "
            "once to train and save it."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: SYNTHETIC DATASET
#
# Positives      → cypher_ideal, cypher_boundary
# Hard negatives → near_miss, bat, gartley, crab, butterfly
# Noise          → random, outside
#
# Key Cypher distinguishers:
#   XC > 1.0 XA  — the C point EXCEEDS the X origin (impulse extension),
#                  which no other standard harmonic requires.
#   XD ≈ 0.786 XC — D retraces ~78.6% of the XC move (same fib as
#                   Butterfly AB, but applied to a different leg).
#   AB is modest (0.382–0.618) — similar to Bat/Crab, unlike Butterfly.
# ─────────────────────────────────────────────────────────────────────

def _sample(region: str) -> tuple:
    noise = np.random.normal(0, 0.01)

    if region == "cypher_ideal":
        # Dead-centre of all textbook ranges
        ab_xa = np.random.uniform(0.382, 0.618)
        xc_xa = np.random.uniform(1.272, 1.414)
        xd_xc = np.random.uniform(0.706, 0.866)   # tight around 0.786 fib

    elif region == "cypher_boundary":
        # Full hard-constraint space
        ab_xa = np.random.uniform(0.205,  0.795)
        xc_xa = np.random.uniform(1.1655, 1.5205)
        xd_xc = np.random.uniform(0.546,  1.026)

    elif region == "near_miss":
        # Valid cypher_ideal with ONE ratio pushed outside its soft range
        ab_xa = np.random.uniform(0.382, 0.618)
        xc_xa = np.random.uniform(1.272, 1.414)
        xd_xc = np.random.uniform(0.706, 0.866)
        flip = np.random.choice(["ab", "xc", "xd"])
        if flip == "ab":
            ab_xa = np.random.choice([
                np.random.uniform(0.205, 0.381),
                np.random.uniform(0.619, 0.795),
            ])
        elif flip == "xc":
            xc_xa = np.random.choice([
                np.random.uniform(1.1655, 1.271),
                np.random.uniform(1.415,  1.5205),
            ])
        else:
            xd_xc = np.random.choice([
                np.random.uniform(0.546, 0.705),
                np.random.uniform(0.867, 1.026),
            ])

    elif region == "bat":
        # Bat: XC well below 1.0 — fails XC hard gate naturally
        ab_xa = np.random.uniform(0.382, 0.500)
        xc_xa = np.random.uniform(0.618, 0.886)
        xd_xc = np.random.uniform(0.700, 1.000)

    elif region == "gartley":
        # Gartley: AB ~0.618, XC ~0.786 of XA — XC < 1.0, fails hard gate
        ab_xa = np.random.uniform(0.580, 0.660)
        xc_xa = np.random.uniform(0.700, 0.900)
        xd_xc = np.random.uniform(0.700, 0.900)

    elif region == "crab":
        # Crab: XC is a larger extension (1.618+), above Cypher's XC hard gate
        ab_xa = np.random.uniform(0.382, 0.618)
        xc_xa = np.random.uniform(1.550, 2.000)
        xd_xc = np.random.uniform(0.700, 0.900)

    elif region == "butterfly":
        # Butterfly: AB ~0.786 (above Cypher's AB soft range)
        ab_xa = np.random.uniform(0.706, 0.866)
        xc_xa = np.random.uniform(1.1655, 1.5205)
        xd_xc = np.random.uniform(0.546,  1.026)

    elif region == "random":
        # Random inside the broad hard-constraint space
        ab_xa = np.random.uniform(0.205,  0.795)
        xc_xa = np.random.uniform(1.1655, 1.5205)
        xd_xc = np.random.uniform(0.546,  1.026)

    else:  # outside hard constraints entirely — one ratio drawn OOB directly
        flip  = np.random.choice(["ab", "xc", "xd"])
        ab_xa = np.random.uniform(0.205,  0.795)
        xc_xa = np.random.uniform(1.1655, 1.5205)
        xd_xc = np.random.uniform(0.546,  1.026)
        if flip == "ab":
            ab_xa = np.random.choice([
                np.random.uniform(0.001, 0.204),
                np.random.uniform(0.796, 1.500),
            ])
        elif flip == "xc":
            xc_xa = np.random.choice([
                np.random.uniform(0.001, 1.164),
                np.random.uniform(1.521, 2.500),
            ])
        else:
            xd_xc = np.random.choice([
                np.random.uniform(0.001, 0.545),
                np.random.uniform(1.027, 1.800),
            ])

    return ab_xa + noise, xc_xa + noise, xd_xc + noise


REGION_COUNTS = {
    "cypher_ideal":    1500,   # strong positives
    "cypher_boundary": 1000,   # weaker positives + soft negatives
    "near_miss":       1000,   # hard negatives: one ratio just off
    "bat":              800,   # hard negatives (XC fails gate naturally)
    "gartley":          600,   # hard negatives (XC fails gate naturally)
    "crab":             400,   # hard negatives (XC too large)
    "butterfly":        400,   # hard negatives (AB too large)
    "random":           500,   # negatives inside hard-constraint space
    "outside":         2000,   # negatives outside hard constraints (boosted)
}


def build_dataset() -> pd.DataFrame:
    rows = []
    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            ab_xa, xc_xa, xd_xc = _sample(region)
            cyp_conf = cypher_score(ab_xa, xc_xa, xd_xc)
            label = 1 if cyp_conf > 0.5 else 0
            rows.append({
                "AB_XA":           ab_xa,
                "XC_XA":           xc_xa,
                "XD_XC":           xd_xc,
                "rule_confidence": cyp_conf,
                "label":           label,
                "region":          region,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: TRAINING
#
# Features: soft scores only (not raw ratios).
# The model learns optimal weights across the three scores, replacing
# the naive equal-weight average (/ 3) in cypher_score().
# The hard-gate pre-filter in learned_confidence() handles the constraint
# boundary; the model only ever sees inputs that passed the gate.
# ─────────────────────────────────────────────────────────────────────

FEATURES = ["score_ab", "score_xc", "score_d"]


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute soft-score columns and append them to the dataframe."""
    df = df.copy()
    df["score_ab"] = df.apply(lambda r: range_score(r.AB_XA, 0.382, 0.618), axis=1)
    df["score_xc"] = df.apply(lambda r: range_score(r.XC_XA, 1.272, 1.414), axis=1)
    df["score_d"]  = df.apply(lambda r: fib_score(r.XD_XC,  0.786, tol=0.08), axis=1)
    return df


def train(df: pd.DataFrame):
    df = _add_features(df)
    X = df[FEATURES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    y_prob = model.predict_proba(X_test_s)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n  ROC-AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["non-cypher", "cypher"]))

    return model, scaler, X_test, y_test, y_prob


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: INFERENCE
#
# Drop-in replacement for CypherHarmonicPattern._run() scoring:
#
#   # Instead of:
#   overall_score = average_scores([score_ab, score_xc, score_d])
#   confidence = clip_confidence(overall_score)
#
#   # Use:
#   confidence = learned_confidence(ab_xa, xc_xa, xd_xc,
#                                   model, scaler) * 100
# ─────────────────────────────────────────────────────────────────────

def learned_confidence(ab_xa: float, xc_xa: float, xd_xc: float,
                       model: LogisticRegression,
                       scaler: StandardScaler) -> float:
    """
    Returns cypher probability in [0, 1]. Multiply by 100 for [0, 100] scale.

    Pipeline:
      1. Hard-gate pre-filter — any ratio outside its constraint returns 0.0
         immediately, matching cypher_score() and preventing the model from
         extrapolating into structurally impossible regions.
      2. Compute the three soft scores (same as cypher_score() soft scoring).
      3. Pass scores-only to the model, which has learned optimal weights
         across them to replace the naive equal-weight average.
    """
    # Hard-gate pre-filter
    if not (0.205  <= ab_xa <= 0.795 ): return 0.0
    if not (1.1655 <= xc_xa <= 1.5205): return 0.0
    if not (0.546  <= xd_xc <= 1.026 ): return 0.0

    s_ab = range_score(ab_xa, 0.382, 0.618)
    s_xc = range_score(xc_xa, 1.272, 1.414)
    s_d  = fib_score(xd_xc,  0.786, tol=0.08)

    x = np.array([[s_ab, s_xc, s_d]])
    return float(model.predict_proba(scaler.transform(x))[0, 1])


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1 — Build dataset
    df = build_dataset()
    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos

    print(f"\n  Total samples          : {len(df)}")
    print(f"  Cypher confirmed (1)   : {n_pos:>4}  ({100*n_pos/len(df):.1f}%)")
    print(f"  Non-cypher (0)         : {n_neg:>4}  ({100*n_neg/len(df):.1f}%)")
    print(f"\n  Breakdown by region:")
    print(f"  {'region':<22}  {'n':>5}  {'cypher':>6}  {'cyp%':>6}  {'avg_rule_conf':>14}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*14}")
    for region, grp in df.groupby("region"):
        pos = int(grp["label"].sum())
        avg = grp["rule_confidence"].mean()
        print(f"  {region:<22}  {len(grp):>5}  {pos:>6}  "
              f"{100*pos/len(grp):>5.1f}%  {avg:>14.1f}")

    # Step 2 — Train
    model, scaler, X_test, y_test, y_prob = train(df)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  Model saved  → {MODEL_PATH}")
    print(f"  Scaler saved → {SCALER_PATH}")

    # Step 3 — Comparison on canonical ratio combinations
    print(f"\n  {'Pattern':<28}  {'Rule':>8}  {'Learned':>9}  {'Delta':>7}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*9}  {'-'*7}")

    test_cases = [
        # Cypher ideal
        ("Cypher — perfect",           0.500, 1.340, 0.786),
        ("Cypher — good",              0.382, 1.300, 0.800),
        ("Cypher — soft boundary",     0.618, 1.272, 0.786),
        # Borderline
        ("Near miss: AB high",         0.750, 1.340, 0.786),
        ("Near miss: XC low",          0.500, 1.200, 0.786),
        ("Near miss: XD off",          0.500, 1.340, 0.950),
        # Other patterns (should score low)
        ("Bat",                        0.440, 0.800, 0.886),
        ("Gartley",                    0.618, 0.786, 0.786),
        ("Crab",                       0.500, 1.618, 0.786),
        ("Butterfly",                  0.786, 1.340, 0.786),
        # Fails hard constraints
        ("Fails AB gate (too high)",   0.850, 1.340, 0.786),
        ("Fails XC gate (too low)",    0.500, 1.100, 0.786),
        ("Fails XC gate (too high)",   0.500, 1.600, 0.786),
        ("Fails XD gate (too high)",   0.500, 1.340, 1.100),
    ]

    for desc, ab, xc, xd in test_cases:
        rule_c  = cypher_score(ab, xc, xd)
        learn_c = learned_confidence(ab, xc, xd, model, scaler) * 100
        delta   = learn_c - rule_c
        print(f"  {desc:<28}  {rule_c:>7.1f}%  {learn_c:>8.1f}%  {delta:>+6.1f}%")