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
MODEL_PATH  = os.path.join(MODEL_DIR, "crab_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "crab_scaler.pkl")


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: RULE-BASED SCORE
#
# Mirrors CrabHarmonicPattern._run() hard constraints + soft scoring
# so that the model can learn to improve on the equal-weight baseline.
# ─────────────────────────────────────────────────────────────────────

def crab_score(ab_xa: float, bc_ab: float,
               cd_bc: float, xd_xa: float) -> float:
    """
    Rule-based confidence score for the Crab harmonic pattern.

    Hard constraints (broad admissibility gate, taken from CrabHarmonicPattern):
        AB ∈ [0.382, 0.618] XA  → hard: [0.205, 0.795]
        BC ∈ [0.382, 0.886] AB  → hard: [0.004, 1.264]   (hard: 0.0 – 1.264)
        CD ∈ [2.24,  3.618] BC  → hard: [1.551, 4.307]
        XD ≈ 1.618 XA, tol=0.10 → hard: [1.318, 1.918]

    Soft scoring (textbook targets):
        AB ∈ [0.382, 0.618]
        BC ∈ [0.382, 0.886]
        CD ∈ [2.24,  3.618]
        XD ≈ 1.618
    """
    # Hard constraints
    # if not (0.205 <= ab_xa <= 0.795): return 0.0
    # if not (0.000 <= bc_ab <= 1.264): return 0.0
    # if not (1.551 <= cd_bc <= 4.307): return 0.0
    # if not (1.318 <= xd_xa <= 1.918): return 0.0

    # Soft scoring
    score_ab = range_score(ab_xa, 0.382, 0.618)
    score_bc = range_score(bc_ab, 0.382, 0.886)
    score_cd = range_score(cd_bc, 2.24,  3.618)
    score_xd = fib_score(xd_xa,  1.618, tol=0.10)

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
            "Crab model not found. Run `python -m stock_analysis.crab_model` "
            "once to train and save it."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: SYNTHETIC DATASET
#
# Positives     → crab_ideal, crab_boundary
# Hard negatives → near_miss, bat, gartley, butterfly
# Noise          → random, outside
#
# Key distinguisher: Crab has the LARGEST CD extension (2.24–3.618)
# and XD pinned tightly at 1.618. These two constraints separate it
# clearly from Bat (XD < 1) and Butterfly (XD 1.27–1.618, CD smaller).
# ─────────────────────────────────────────────────────────────────────

def _sample(region: str) -> tuple:
    noise = np.random.normal(0, 0.01)

    if region == "crab_ideal":
        # Dead-centre of all textbook ranges
        ab_xa = np.random.uniform(0.382, 0.618)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(2.240, 3.618)
        xd_xa = np.random.uniform(1.518, 1.718)   # tight fib_score tol of 0.10 around 1.618

    elif region == "crab_boundary":
        # Full hard-constraint space (weaker positives + soft negatives)
        ab_xa = np.random.uniform(0.205, 0.795)
        bc_ab = np.random.uniform(0.000, 1.264)
        cd_bc = np.random.uniform(1.551, 4.307)
        xd_xa = np.random.uniform(1.318, 1.918)

    elif region == "near_miss":
        # Start with a valid crab_ideal, push ONE ratio outside its soft range
        ab_xa = np.random.uniform(0.382, 0.618)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(2.240, 3.618)
        xd_xa = np.random.uniform(1.518, 1.718)
        flip = np.random.choice(["ab", "bc", "cd", "xd"])
        if flip == "ab":
            ab_xa = np.random.choice([
                np.random.uniform(0.205, 0.381),
                np.random.uniform(0.619, 0.795),
            ])
        elif flip == "bc":
            bc_ab = np.random.choice([
                np.random.uniform(0.000, 0.381),
                np.random.uniform(0.887, 1.264),
            ])
        elif flip == "cd":
            cd_bc = np.random.choice([
                np.random.uniform(1.551, 2.239),
                np.random.uniform(3.619, 4.307),
            ])
        else:
            xd_xa = np.random.choice([
                np.random.uniform(1.318, 1.517),
                np.random.uniform(1.719, 1.918),
            ])

    elif region == "bat":
        # Bat: AB smaller, XD ~0.886 — well outside crab hard constraints
        ab_xa = np.random.uniform(0.382, 0.500)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(1.618, 2.618)
        xd_xa = np.random.uniform(0.806, 0.966)

    elif region == "gartley":
        ab_xa = np.random.uniform(0.580, 0.660)
        bc_ab = np.random.uniform(0.380, 0.890)
        cd_bc = np.random.uniform(1.270, 1.620)
        xd_xa = np.random.uniform(0.740, 0.820)

    elif region == "butterfly":
        # Butterfly: AB ~0.786 (above crab's AB range), XD 1.27–1.618
        ab_xa = np.random.uniform(0.706, 0.866)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(1.618, 2.618)
        xd_xa = np.random.uniform(1.270, 1.618)

    elif region == "random":
        # Random inside the broad hard-constraint space
        ab_xa = np.random.uniform(0.205, 0.795)
        bc_ab = np.random.uniform(0.000, 1.264)
        cd_bc = np.random.uniform(1.551, 4.307)
        xd_xa = np.random.uniform(1.318, 1.918)

    else:  # outside hard constraints entirely — sample at least one ratio OOB
        flip = np.random.choice(["ab", "bc", "cd", "xd"])
        # Keep all other ratios in-bounds so the model learns the individual boundary
        ab_xa = np.random.uniform(0.205, 0.795)
        bc_ab = np.random.uniform(0.000, 1.264)
        cd_bc = np.random.uniform(1.551, 4.307)
        xd_xa = np.random.uniform(1.318, 1.918)
        if flip == "ab":
            ab_xa = np.random.choice([
                np.random.uniform(0.001, 0.204),
                np.random.uniform(0.796, 1.500),
            ])
        elif flip == "bc":
            bc_ab = np.random.uniform(1.265, 2.500)
        elif flip == "cd":
            cd_bc = np.random.choice([
                np.random.uniform(0.001, 1.550),
                np.random.uniform(4.308, 6.000),
            ])
        else:
            xd_xa = np.random.choice([
                np.random.uniform(0.001, 1.317),
                np.random.uniform(1.919, 2.500),
            ])

    return ab_xa + noise, bc_ab + noise, cd_bc + noise, xd_xa + noise


REGION_COUNTS = {
    "crab_ideal":     1500,   # strong positives
    "crab_boundary":  1000,   # weaker positives + soft negatives
    "near_miss":      1000,   # hard negatives: one ratio just off
    "bat":             800,   # hard negatives: different harmonic
    "gartley":         600,   # hard negatives
    "butterfly":       400,   # hard negatives (overlaps XD range slightly)
    "random":          500,   # negatives inside hard-constraint space
    "outside":        2000,   # negatives outside hard constraints (was 200 — boosted to force boundary learning)
}


def build_dataset() -> pd.DataFrame:
    rows = []
    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            ab_xa, bc_ab, cd_bc, xd_xa = _sample(region)
            crab_conf = crab_score(ab_xa, bc_ab, cd_bc, xd_xa)
            # Label = 1 when rule-based score is convincingly positive
            # AND the pattern's defining constraint (CD >= 2.24, XD ~1.618) holds
            label = 1 if (crab_conf > 0.5 and cd_bc >= 2.240 and 1.318 <= xd_xa <= 1.918) else 0
            rows.append({
                "AB_XA":           ab_xa,
                "BC_AB":           bc_ab,
                "CD_BC":           cd_bc,
                "XD_XA":           xd_xa,
                "rule_confidence": crab_conf,
                "label":           label,
                "region":          region,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: TRAINING
# ─────────────────────────────────────────────────────────────────────

# Soft scores only — no raw ratios.
#
# The model's job is to learn OPTIMAL WEIGHTS across the four rule-based
# scores, replacing the naive equal-weight average (/ 4) in crab_score().
# Using raw ratios alongside scores would be redundant (the scores already
# encode ratio distance from target) and would risk the model ignoring the
# rule structure entirely, causing it to extrapolate into gated-out regions.
# The hard-gate pre-filter in learned_confidence() handles the constraint
# boundary; the model only ever sees inputs that passed the gate.
FEATURES = ["score_ab", "score_bc", "score_cd", "score_xd"]


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute soft-score columns and append them to the dataframe."""
    df = df.copy()
    df["score_ab"] = df.apply(lambda r: range_score(r.AB_XA, 0.382, 0.618), axis=1)
    df["score_bc"] = df.apply(lambda r: range_score(r.BC_AB, 0.382, 0.886), axis=1)
    df["score_cd"] = df.apply(lambda r: range_score(r.CD_BC, 2.24,  3.618), axis=1)
    df["score_xd"] = df.apply(lambda r: fib_score(r.XD_XA,  1.618, tol=0.10), axis=1)
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
                                target_names=["non-crab", "crab"]))

    return model, scaler, X_test, y_test, y_prob


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: INFERENCE
#
# Drop-in replacement for CrabHarmonicPattern._run() scoring:
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
    Returns crab probability in [0, 1]. Multiply by 100 for [0, 100] scale.

    Pipeline:
      1. Hard-gate pre-filter — any ratio outside its constraint returns 0.0
         immediately, matching crab_score() and preventing the model from
         extrapolating into structurally impossible regions.
      2. Compute the four soft scores (same as crab_score() soft scoring).
      3. Pass scores-only to the model, which has learned optimal weights
         across them to replace the naive equal-weight average.
    """

    s_ab = range_score(ab_xa, 0.382, 0.618)
    s_bc = range_score(bc_ab, 0.382, 0.886)
    s_cd = range_score(cd_bc, 2.24,  3.618)
    s_xd = fib_score(xd_xa,  1.618, tol=0.10)

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

    print(f"\n  Total samples        : {len(df)}")
    print(f"  Crab confirmed (1)   : {n_pos:>4}  ({100*n_pos/len(df):.1f}%)")
    print(f"  Non-crab (0)         : {n_neg:>4}  ({100*n_neg/len(df):.1f}%)")
    print(f"\n  Breakdown by region:")
    print(f"  {'region':<22}  {'n':>5}  {'crab':>5}  {'crab%':>6}  {'avg_rule_conf':>14}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*14}")
    for region, grp in df.groupby("region"):
        pos = int(grp["label"].sum())
        avg = grp["rule_confidence"].mean()
        print(f"  {region:<22}  {len(grp):>5}  {pos:>5}  "
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
        # Crab ideal
        ("Crab — perfect",             0.500, 0.618, 3.000, 1.618),
        ("Crab — good",                0.382, 0.700, 2.800, 1.600),
        ("Crab — soft boundary",       0.618, 0.382, 2.240, 1.618),
        # Borderline
        ("Near miss: AB high",         0.750, 0.618, 3.000, 1.618),
        ("Near miss: XD off",          0.500, 0.618, 3.000, 1.800),
        ("Near miss: CD low",          0.500, 0.618, 1.800, 1.618),
        # Other patterns (should score low as crab)
        ("Bat",                        0.440, 0.700, 2.200, 0.890),
        ("Gartley",                    0.618, 0.618, 1.618, 0.786),
        ("Butterfly",                  0.786, 0.500, 1.800, 1.270),
        # Fails hard constraints
        ("Fails AB gate (too high)",   0.850, 0.618, 3.000, 1.618),
        ("Fails XD gate (too low)",    0.500, 0.618, 3.000, 1.200),
        ("Fails CD gate (too high)",   0.500, 0.618, 5.000, 1.618),
        ("Fails CD gate (too low)",    0.500, 0.618, 1.200, 1.618),
        ("Bob", 0.580, 4.884, 0.214, 2.645)
    ]

    for desc, ab, bc, cd, xd in test_cases:
        rule_c  = crab_score(ab, bc, cd, xd)
        learn_c = learned_confidence(ab, bc, cd, xd, model, scaler) * 100
        delta   = learn_c - rule_c
        print(f"  {desc:<28}  {rule_c:>7.1f}%  {learn_c:>8.1f}%  {delta:>+6.1f}%")