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
MODEL_PATH  = os.path.join(MODEL_DIR, "butterfly_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "butterfly_scaler.pkl")


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: RULE-BASED SCORE
#
# Mirrors ButterflyHarmonicPattern._run() hard constraints + soft scoring
# so that the model can learn to improve on the equal-weight baseline.
# ─────────────────────────────────────────────────────────────────────

def butterfly_score(ab_xa: float, bc_ab: float,
                    cd_bc: float, xd_xa: float) -> float:
    """
    Rule-based confidence score for the Butterfly harmonic pattern.

    Hard constraints (broad admissibility gate, taken from ButterflyHarmonicPattern):
        AB ≈ 0.786 XA  → [0.546, 1.026]
        BC ∈ [0.382, 0.886] AB  → [0.004, 1.264]   (hard: 0.0 – 1.264)
        CD ∈ [1.618, 2.618] BC  → hard: [0.868, 3.368]
        XD ∈ [1.27,  1.618] XA  → hard: [1.096, 1.792]

    Soft scoring (textbook targets):
        AB ≈ 0.786 XA
        BC ∈ [0.382, 0.886]
        CD ∈ [1.618, 2.618]
        XD ∈ [1.27,  1.618]
    """
    # # Hard constraints
    # if not (0.546 <= ab_xa <= 1.026): return 0.0
    # if not (0.000 <= bc_ab <= 1.264): return 0.0
    # if not (0.868 <= cd_bc <= 3.368): return 0.0
    # if not (1.096 <= xd_xa <= 1.792): return 0.0

    # Soft scoring
    score_ab = fib_score(ab_xa, 0.786, tol=0.08)
    score_bc = range_score(bc_ab, 0.382, 0.886)
    score_cd = range_score(cd_bc, 1.618, 2.618)
    score_xd = range_score(xd_xa, 1.27,  1.618)

    overall = (score_ab + score_bc + score_cd + score_xd) / 4
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
            "Butterfly model not found. Run `python -m stock_analysis.butterfly_model` "
            "once to train and save it."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: SYNTHETIC DATASET
#
# Positives  → butterfly_ideal, butterfly_boundary
# Hard negatives → near_miss, bat, gartley, crab
# Noise      → random, outside
# ─────────────────────────────────────────────────────────────────────

def _sample(region: str) -> tuple:
    noise = np.random.normal(0, 0.01)

    if region == "butterfly_ideal":
        # Dead-centre of all textbook ranges
        ab_xa = np.random.uniform(0.706, 0.866)   # tight around 0.786
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(1.618, 2.618)
        xd_xa = np.random.uniform(1.270, 1.618)

    elif region == "butterfly_boundary":
        # Full hard-constraint space (weaker positives + soft negatives)
        ab_xa = np.random.uniform(0.546, 1.026)
        bc_ab = np.random.uniform(0.000, 1.264)
        cd_bc = np.random.uniform(0.868, 3.368)
        xd_xa = np.random.uniform(1.096, 1.792)

    elif region == "near_miss":
        # Start with a valid butterfly_ideal, push ONE ratio outside soft range
        ab_xa = np.random.uniform(0.706, 0.866)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(1.618, 2.618)
        xd_xa = np.random.uniform(1.270, 1.618)
        flip = np.random.choice(["ab", "bc", "cd", "xd"])
        if flip == "ab":
            ab_xa = np.random.choice([
                np.random.uniform(0.546, 0.705),
                np.random.uniform(0.867, 1.026),
            ])
        elif flip == "bc":
            bc_ab = np.random.choice([
                np.random.uniform(0.000, 0.381),
                np.random.uniform(0.887, 1.264),
            ])
        elif flip == "cd":
            cd_bc = np.random.choice([
                np.random.uniform(0.868, 1.617),
                np.random.uniform(2.619, 3.368),
            ])
        else:
            xd_xa = np.random.choice([
                np.random.uniform(1.096, 1.269),
                np.random.uniform(1.619, 1.792),
            ])

    elif region == "bat":
        # Bat ratios — should score low as butterfly
        ab_xa = np.random.uniform(0.382, 0.500)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(1.618, 2.618)
        xd_xa = np.random.uniform(0.806, 0.966)

    elif region == "gartley":
        ab_xa = np.random.uniform(0.580, 0.660)
        bc_ab = np.random.uniform(0.380, 0.890)
        cd_bc = np.random.uniform(1.270, 1.620)
        xd_xa = np.random.uniform(0.740, 0.820)

    elif region == "crab":
        ab_xa = np.random.uniform(0.380, 0.620)
        bc_ab = np.random.uniform(0.380, 0.890)
        cd_bc = np.random.uniform(2.240, 3.620)
        xd_xa = np.random.uniform(1.550, 1.700)

    elif region == "random":
        # Random inside the broad hard-constraint space
        ab_xa = np.random.uniform(0.546, 1.026)
        bc_ab = np.random.uniform(0.000, 1.264)
        cd_bc = np.random.uniform(0.868, 3.368)
        xd_xa = np.random.uniform(1.096, 1.792)

    else:  # outside hard constraints entirely — sample at least one ratio OOB
        flip = np.random.choice(["ab", "bc", "cd", "xd"])
        ab_xa = np.random.uniform(0.546, 1.026)
        bc_ab = np.random.uniform(0.000, 1.264)
        cd_bc = np.random.uniform(0.868, 3.368)
        xd_xa = np.random.uniform(1.096, 1.792)
        if flip == "ab":
            ab_xa = np.random.choice([
                np.random.uniform(0.01,  0.545),
                np.random.uniform(1.027, 1.800),
            ])
        elif flip == "bc":
            bc_ab = np.random.uniform(1.265, 2.500)
        elif flip == "cd":
            cd_bc = np.random.choice([
                np.random.uniform(0.01,  0.867),
                np.random.uniform(3.369, 5.000),
            ])
        else:
            xd_xa = np.random.choice([
                np.random.uniform(0.01,  1.095),
                np.random.uniform(1.793, 2.500),
            ])

    return ab_xa + noise, bc_ab + noise, cd_bc + noise, xd_xa + noise


REGION_COUNTS = {
    "butterfly_ideal":    1500,   # strong positives
    "butterfly_boundary": 1000,   # weaker positives + soft negatives
    "near_miss":          1000,   # hard negatives: one ratio just off
    "bat":                 800,   # hard negatives: different harmonic
    "gartley":             600,   # hard negatives
    "crab":                400,   # hard negatives
    "random":              500,   # negatives inside hard-constraint space
    "outside":            2000,   # negatives outside hard constraints (boosted for boundary learning)
}


def build_dataset() -> pd.DataFrame:
    rows = []
    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            ab_xa, bc_ab, cd_bc, xd_xa = _sample(region)
            bfly_conf = butterfly_score(ab_xa, bc_ab, cd_bc, xd_xa)
            # Label = 1 when rule-based score is convincingly positive
            # AND the pattern's defining constraint (XD > 1) is satisfied
            label = 1 if (bfly_conf > 0.5 and xd_xa >= 1.096) else 0
            rows.append({
                "AB_XA":           ab_xa,
                "BC_AB":           bc_ab,
                "CD_BC":           cd_bc,
                "XD_XA":           xd_xa,
                "rule_confidence": bfly_conf,
                "label":           label,
                "region":          region,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: TRAINING
# ─────────────────────────────────────────────────────────────────────

FEATURES = ["score_ab", "score_bc", "score_cd", "score_xd"]


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute soft-score columns and append them to the dataframe."""
    df = df.copy()
    df["score_ab"] = df.apply(lambda r: fib_score(r.AB_XA, 0.786, tol=0.08), axis=1)
    df["score_bc"] = df.apply(lambda r: range_score(r.BC_AB, 0.382, 0.886), axis=1)
    df["score_cd"] = df.apply(lambda r: range_score(r.CD_BC, 1.618, 2.618), axis=1)
    df["score_xd"] = df.apply(lambda r: range_score(r.XD_XA, 1.27,  1.618), axis=1)
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
                                target_names=["non-butterfly", "butterfly"]))

    return model, scaler, X_test, y_test, y_prob


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: INFERENCE
#
# Drop-in replacement for ButterflyHarmonicPattern._run() scoring:
#
#   # Instead of:
#   overall_score = average_scores([score_ab, score_bc, score_cd, score_xd])
#   confidence = clip_confidence(overall_score)
#
#   # Use:
#   confidence = learned_confidence(ab_xa, bc_ab, cd_bc, xd_xa,
#                                   model, scaler) * 100
# ─────────────────────────────────────────────────────────────────────

def learned_confidence(ab_xa: float, bc_ab: float,
                       cd_bc: float, xd_xa: float,
                       model: LogisticRegression,
                       scaler: StandardScaler) -> float:
    """
    Returns butterfly probability in [0, 1]. Multiply by 100 for [0, 100] scale.

    Pipeline:
      1. Hard-gate pre-filter — any ratio outside its constraint returns 0.0
         immediately, matching butterfly_score() and preventing the model from
         extrapolating into structurally impossible regions.
      2. Compute the four soft scores (same as butterfly_score() soft scoring).
      3. Pass scores-only to the model, which has learned optimal weights
         across them to replace the naive equal-weight average.
    """
    # Hard-gate pre-filter
    if not (0.546 <= ab_xa <= 1.026): return 0.0
    if not (0.000 <= bc_ab <= 1.264): return 0.0
    if not (0.868 <= cd_bc <= 3.368): return 0.0
    if not (1.096 <= xd_xa <= 1.792): return 0.0

    s_ab = fib_score(ab_xa, 0.786, tol=0.08)
    s_bc = range_score(bc_ab, 0.382, 0.886)
    s_cd = range_score(cd_bc, 1.618, 2.618)
    s_xd = range_score(xd_xa, 1.27,  1.618)

    x = np.array([[s_ab, s_bc, s_cd, s_xd]])
    return float(model.predict_proba(scaler.transform(x))[0, 1])


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1 — Build dataset
    df = build_dataset()
    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos

    print(f"\n  Total samples            : {len(df)}")
    print(f"  Butterfly confirmed (1)  : {n_pos:>4}  ({100*n_pos/len(df):.1f}%)")
    print(f"  Non-butterfly (0)        : {n_neg:>4}  ({100*n_neg/len(df):.1f}%)")
    print(f"\n  Breakdown by region:")
    print(f"  {'region':<22}  {'n':>5}  {'butterfly':>9}  {'bfly%':>6}  {'avg_rule_conf':>14}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*9}  {'-'*6}  {'-'*14}")
    for region, grp in df.groupby("region"):
        pos = int(grp["label"].sum())
        avg = grp["rule_confidence"].mean()
        print(f"  {region:<22}  {len(grp):>5}  {pos:>9}  "
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
        # Butterfly ideal
        ("Butterfly — perfect",        0.786, 0.618, 2.000, 1.618),
        ("Butterfly — good",           0.800, 0.700, 2.200, 1.400),
        ("Butterfly — soft boundary",  0.786, 0.382, 1.618, 1.270),
        # Borderline
        ("Near miss: AB low",          0.600, 0.618, 2.000, 1.500),
        ("Near miss: XD high",         0.786, 0.618, 2.000, 1.750),
        ("Near miss: CD low",          0.786, 0.618, 1.000, 1.500),
        # Other patterns (should score low as butterfly)
        ("Bat",                        0.440, 0.700, 2.200, 0.890),
        ("Gartley",                    0.618, 0.618, 1.618, 0.786),
        ("Crab",                       0.500, 0.886, 3.000, 1.618),
        # Fails hard constraints
        ("Fails AB gate (too low)",    0.400, 0.618, 2.000, 1.500),
        ("Fails XD gate (too low)",    0.786, 0.618, 2.000, 1.000),
        ("Fails CD gate (too high)",   0.786, 0.618, 4.000, 1.500),
    ]

    for desc, ab, bc, cd, xd in test_cases:
        rule_c  = butterfly_score(ab, bc, cd, xd)
        learn_c = learned_confidence(ab, bc, cd, xd, model, scaler) * 100
        delta   = learn_c - rule_c
        print(f"  {desc:<28}  {rule_c:>7.1f}%  {learn_c:>8.1f}%  {delta:>+6.1f}%")