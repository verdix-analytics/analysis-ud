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
MODEL_PATH  = os.path.join(MODEL_DIR, "gartley_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "gartley_scaler.pkl")


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: RULE-BASED SCORE
#
# Mirrors GartleyHarmonicPattern._run() hard constraints + soft scoring
# so that the model can learn to improve on the equal-weight baseline.
#
# Gartley distinguishers vs other harmonics:
#   AB ≈ 0.618 XA  — tightest AB constraint of all harmonics (fib_score)
#   CD ∈ [1.272, 1.618] BC — smaller extension than Bat/Crab/Butterfly
#   XD ≈ 0.786 XA  — same as Butterfly AB, but on a different leg
#
# The combination of AB ≈ 0.618 AND XD ≈ 0.786 AND CD < 1.618 is unique
# to Gartley. Bat shares the XD ~0.886 region (different fib). Butterfly
# has AB ~0.786 (much higher). Crab has XD ~1.618 (much higher).
# ─────────────────────────────────────────────────────────────────────

def gartley_score(ab_xa: float, bc_ab: float,
                  cd_bc: float, xd_xa: float) -> float:
    """
    Rule-based confidence score for the Gartley harmonic pattern.

    Hard constraints (broad admissibility gate, taken from GartleyHarmonicPattern):
        AB ≈ 0.618 XA, tol=0.08   → hard: [0.378, 0.858]
        BC ∈ [0.382, 0.886] AB    → hard: [0.130, 1.138]
        CD ∈ [1.272, 1.618] BC    → hard: [1.099, 1.791]
        XD ≈ 0.786 XA, tol=0.08  → hard: [0.546, 1.026]

    Soft scoring (textbook targets):
        AB ≈ 0.618
        BC ∈ [0.382, 0.886]
        CD ∈ [1.272, 1.618]
        XD ≈ 0.786
    """
    # Hard constraints
    # if not (0.378 <= ab_xa <= 0.858): return 0.0
    # if not (0.130 <= bc_ab <= 1.138): return 0.0
    # if not (1.099 <= cd_bc <= 1.791): return 0.0
    # if not (0.546 <= xd_xa <= 1.026): return 0.0

    # Soft scoring
    score_ab = fib_score(ab_xa,  0.618, tol=0.08)
    score_bc = range_score(bc_ab, 0.382, 0.886)
    score_cd = range_score(cd_bc, 1.272, 1.618)
    score_xd = fib_score(xd_xa,  0.786, tol=0.08)

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
            "Gartley model not found. Run `python -m stock_analysis.gartley_model` "
            "once to train and save it."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: SYNTHETIC DATASET
#
# Positives      → gartley_ideal, gartley_boundary
# Hard negatives → near_miss, bat, crab, butterfly
# Noise          → random, outside
#
# Hardest confusion pairs:
#   Bat     — shares XD region (0.786–0.886), differs on AB (0.382–0.5 vs 0.618)
#             and CD (1.618–2.618 vs 1.272–1.618)
#   Butterfly — AB at 0.786 is above Gartley's AB gate top (0.858), so it
#               fails the hard gate cleanly. Included for training completeness.
#   Crab    — XD at 1.618 fails Gartley XD hard gate cleanly.
# ─────────────────────────────────────────────────────────────────────

def _sample(region: str) -> tuple:
    noise = np.random.normal(0, 0.01)

    if region == "gartley_ideal":
        # Dead-centre of all textbook ranges
        ab_xa = np.random.uniform(0.538, 0.698)   # tight fib_score tol=0.08 around 0.618
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(1.272, 1.618)
        xd_xa = np.random.uniform(0.706, 0.866)   # tight fib_score tol=0.08 around 0.786

    elif region == "gartley_boundary":
        # Full hard-constraint space (weaker positives + soft negatives)
        ab_xa = np.random.uniform(0.378, 0.858)
        bc_ab = np.random.uniform(0.130, 1.138)
        cd_bc = np.random.uniform(1.099, 1.791)
        xd_xa = np.random.uniform(0.546, 1.026)

    elif region == "near_miss":
        # Valid gartley_ideal with ONE ratio pushed outside its soft range
        ab_xa = np.random.uniform(0.538, 0.698)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(1.272, 1.618)
        xd_xa = np.random.uniform(0.706, 0.866)
        flip = np.random.choice(["ab", "bc", "cd", "xd"])
        if flip == "ab":
            ab_xa = np.random.choice([
                np.random.uniform(0.378, 0.537),
                np.random.uniform(0.699, 0.858),
            ])
        elif flip == "bc":
            bc_ab = np.random.choice([
                np.random.uniform(0.130, 0.381),
                np.random.uniform(0.887, 1.138),
            ])
        elif flip == "cd":
            cd_bc = np.random.choice([
                np.random.uniform(1.099, 1.271),
                np.random.uniform(1.619, 1.791),
            ])
        else:
            xd_xa = np.random.choice([
                np.random.uniform(0.546, 0.705),
                np.random.uniform(0.867, 1.026),
            ])

    elif region == "bat":
        # Bat: AB 0.382–0.500 (below Gartley's AB soft range), XD ~0.886,
        # CD 1.618–2.618 (above Gartley's CD hard gate top of 1.791)
        ab_xa = np.random.uniform(0.382, 0.500)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(1.618, 2.618)   # above Gartley CD gate → fails hard gate
        xd_xa = np.random.uniform(0.806, 0.966)

    elif region == "bat_overlap":
        # Bat samples that DO pass Gartley's hard gate — the trickiest negatives.
        # AB in [0.382–0.500] passes Gartley AB gate [0.378, 0.858].
        # CD must be kept inside [1.099, 1.791] to pass the gate.
        # XD at ~0.886 still passes Gartley XD gate [0.546, 1.026].
        ab_xa = np.random.uniform(0.382, 0.500)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(1.099, 1.791)   # inside Gartley CD gate
        xd_xa = np.random.uniform(0.806, 0.966)   # inside Gartley XD gate

    elif region == "crab":
        # Crab: XD ~1.618 — fails Gartley XD gate [0.546, 1.026] cleanly
        ab_xa = np.random.uniform(0.382, 0.618)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(2.240, 3.618)
        xd_xa = np.random.uniform(1.518, 1.718)

    elif region == "butterfly":
        # Butterfly: AB ~0.786 — above Gartley AB hard gate (0.858), fails cleanly
        ab_xa = np.random.uniform(0.706, 0.866)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(1.618, 2.618)
        xd_xa = np.random.uniform(1.270, 1.618)

    elif region == "random":
        # Random inside the broad hard-constraint space
        ab_xa = np.random.uniform(0.378, 0.858)
        bc_ab = np.random.uniform(0.130, 1.138)
        cd_bc = np.random.uniform(1.099, 1.791)
        xd_xa = np.random.uniform(0.546, 1.026)

    else:  # outside hard constraints entirely — one ratio drawn OOB directly
        flip  = np.random.choice(["ab", "bc", "cd", "xd"])
        ab_xa = np.random.uniform(0.378, 0.858)
        bc_ab = np.random.uniform(0.130, 1.138)
        cd_bc = np.random.uniform(1.099, 1.791)
        xd_xa = np.random.uniform(0.546, 1.026)
        if flip == "ab":
            ab_xa = np.random.choice([
                np.random.uniform(0.001, 0.377),
                np.random.uniform(0.859, 1.500),
            ])
        elif flip == "bc":
            bc_ab = np.random.choice([
                np.random.uniform(0.001, 0.129),
                np.random.uniform(1.139, 2.500),
            ])
        elif flip == "cd":
            cd_bc = np.random.choice([
                np.random.uniform(0.001, 1.098),
                np.random.uniform(1.792, 4.000),
            ])
        else:
            xd_xa = np.random.choice([
                np.random.uniform(0.001, 0.545),
                np.random.uniform(1.027, 2.000),
            ])

    return ab_xa + noise, bc_ab + noise, cd_bc + noise, xd_xa + noise


REGION_COUNTS = {
    "gartley_ideal":    1500,   # strong positives
    "gartley_boundary": 1000,   # weaker positives + soft negatives
    "near_miss":        1000,   # hard negatives: one ratio just off
    "bat":               600,   # hard negatives (CD usually fails gate)
    "bat_overlap":       800,   # hardest negatives: bat ratios inside Gartley gate
    "crab":              400,   # hard negatives (XD fails gate cleanly)
    "butterfly":         400,   # hard negatives (AB fails gate cleanly)
    "random":            500,   # negatives inside hard-constraint space
    "outside":          2000,   # negatives outside hard constraints (boosted)
}


def build_dataset() -> pd.DataFrame:
    rows = []
    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            ab_xa, bc_ab, cd_bc, xd_xa = _sample(region)
            gart_conf = gartley_score(ab_xa, bc_ab, cd_bc, xd_xa)
            # Label = 1 when rule-based score is convincingly positive
            # AND the two defining Gartley constraints hold (AB ~0.618, XD ~0.786)
            label = 1 if (
                gart_conf > 0.5
                and 0.378 <= ab_xa <= 0.858
                and 0.546 <= xd_xa <= 1.026
            ) else 0
            rows.append({
                "AB_XA":           ab_xa,
                "BC_AB":           bc_ab,
                "CD_BC":           cd_bc,
                "XD_XA":           xd_xa,
                "rule_confidence": gart_conf,
                "label":           label,
                "region":          region,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: TRAINING
#
# Features: soft scores only (not raw ratios).
# The model learns optimal weights across the four scores, replacing
# the naive equal-weight average (/ 4) in gartley_score().
# The hard-gate pre-filter in learned_confidence() handles the constraint
# boundary; the model only ever sees inputs that passed the gate.
# ─────────────────────────────────────────────────────────────────────

FEATURES = ["score_ab", "score_bc", "score_cd", "score_xd"]


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute soft-score columns and append them to the dataframe."""
    df = df.copy()
    df["score_ab"] = df.apply(lambda r: fib_score(r.AB_XA,  0.618, tol=0.08), axis=1)
    df["score_bc"] = df.apply(lambda r: range_score(r.BC_AB, 0.382, 0.886), axis=1)
    df["score_cd"] = df.apply(lambda r: range_score(r.CD_BC, 1.272, 1.618), axis=1)
    df["score_xd"] = df.apply(lambda r: fib_score(r.XD_XA,  0.786, tol=0.08), axis=1)
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
                                target_names=["non-gartley", "gartley"]))

    return model, scaler, X_test, y_test, y_prob


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: INFERENCE
#
# Drop-in replacement for GartleyHarmonicPattern._run() scoring:
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
    Returns gartley probability in [0, 1]. Multiply by 100 for [0, 100] scale.

    Pipeline:
      1. Hard-gate pre-filter — any ratio outside its constraint returns 0.0
         immediately, matching gartley_score() and preventing the model from
         extrapolating into structurally impossible regions.
      2. Compute the four soft scores (same as gartley_score() soft scoring).
      3. Pass scores-only to the model, which has learned optimal weights
         across them to replace the naive equal-weight average.
    """
    # Hard-gate pre-filter
    if not (0.378 <= ab_xa <= 0.858): return 0.0
    if not (0.130 <= bc_ab <= 1.138): return 0.0
    if not (1.099 <= cd_bc <= 1.791): return 0.0
    if not (0.546 <= xd_xa <= 1.026): return 0.0

    s_ab = fib_score(ab_xa,  0.618, tol=0.08)
    s_bc = range_score(bc_ab, 0.382, 0.886)
    s_cd = range_score(cd_bc, 1.272, 1.618)
    s_xd = fib_score(xd_xa,  0.786, tol=0.08)

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

    print(f"\n  Total samples           : {len(df)}")
    print(f"  Gartley confirmed (1)   : {n_pos:>4}  ({100*n_pos/len(df):.1f}%)")
    print(f"  Non-gartley (0)         : {n_neg:>4}  ({100*n_neg/len(df):.1f}%)")
    print(f"\n  Breakdown by region:")
    print(f"  {'region':<22}  {'n':>5}  {'gartley':>7}  {'gart%':>6}  {'avg_rule_conf':>14}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*14}")
    for region, grp in df.groupby("region"):
        pos = int(grp["label"].sum())
        avg = grp["rule_confidence"].mean()
        print(f"  {region:<22}  {len(grp):>5}  {pos:>7}  "
              f"{100*pos/len(grp):>5.1f}%  {avg:>14.1f}")

    # Step 2 — Train
    model, scaler, X_test, y_test, y_prob = train(df)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  Model saved  → {MODEL_PATH}")
    print(f"  Scaler saved → {SCALER_PATH}")

    # Step 3 — Comparison on canonical ratio combinations
    print(f"\n  {'Pattern':<30}  {'Rule':>8}  {'Learned':>9}  {'Delta':>7}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*9}  {'-'*7}")

    test_cases = [
        # Gartley ideal
        ("Gartley — perfect",            0.618, 0.618, 1.414, 0.786),
        ("Gartley — good",               0.600, 0.700, 1.500, 0.800),
        ("Gartley — soft boundary",      0.618, 0.382, 1.272, 0.786),
        # Borderline / near miss
        ("Near miss: AB low",            0.420, 0.618, 1.414, 0.786),
        ("Near miss: AB high",           0.800, 0.618, 1.414, 0.786),
        ("Near miss: CD high",           0.618, 0.618, 1.700, 0.786),
        ("Near miss: XD off",            0.618, 0.618, 1.414, 0.950),
        # Hardest confusion: Bat ratios inside Gartley gate
        ("Bat (overlap, inside gate)",   0.450, 0.618, 1.400, 0.886),
        ("Bat (overlap, inside gate) 2", 0.480, 0.700, 1.300, 0.900),
        # Other patterns (should score low / fail gate)
        ("Bat (CD fails gate)",          0.440, 0.700, 2.200, 0.890),
        ("Crab",                         0.500, 0.618, 3.000, 1.618),
        ("Butterfly",                    0.786, 0.500, 1.800, 1.270),
        # Fails hard constraints
        ("Fails AB gate (too low)",      0.300, 0.618, 1.414, 0.786),
        ("Fails AB gate (too high)",     0.900, 0.618, 1.414, 0.786),
        ("Fails CD gate (too high)",     0.618, 0.618, 2.000, 0.786),
        ("Fails XD gate (too high)",     0.618, 0.618, 1.414, 1.100),
    ]

    for desc, ab, bc, cd, xd in test_cases:
        rule_c  = gartley_score(ab, bc, cd, xd)
        learn_c = learned_confidence(ab, bc, cd, xd, model, scaler) * 100
        delta   = learn_c - rule_c
        print(f"  {desc:<30}  {rule_c:>7.1f}%  {learn_c:>8.1f}%  {delta:>+6.1f}%")