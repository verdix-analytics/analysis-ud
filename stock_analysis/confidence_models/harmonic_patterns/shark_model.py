import numpy as np
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stock_analysis.tools.harmonic_utils import (
    range_score,
    fib_score,
    clip_confidence,
)

np.random.seed(42)

MODEL_DIR   = os.path.join(os.path.dirname(__file__), '..', "pkl_files")
MODEL_PATH  = os.path.join(MODEL_DIR, "shark_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "shark_scaler.pkl")

# ---------------------------------------------------------------------------
# SECTION 1: RULE-BASED SCORE
#
# Mirrors the scoring block inside SharkHarmonicPattern._run().
# Ratio naming follows the XABCD→OXABC mapping used in that class:
#
#   ab_xa  → AB_XA  (original XA / OX)   ideal [0.382, 0.618]
#   bc_ab  → BC_AB  (original AB / XA)   ideal [1.13,  1.618]
#   cd_bc  → CD_BC  (original BC / AB)   ideal [1.618, 2.24]
#   xd_xa  → XD_XA  (original OC / OX)   ideal ≈ 0.886
# ---------------------------------------------------------------------------

def shark_score(ab_xa: float, bc_ab: float,
                cd_bc: float, xd_xa: float) -> float:
    """
    Rule-based Shark confidence score in [0, 100].

    Hard constraints are the broadened gates from SharkHarmonicPattern._run().
    Soft scoring uses the ideal Shark Fibonacci ratios.
    Returns 0.0 when any hard constraint is violated.
    """
    # Hard constraints (broadened gate — same as _run())
    if not (0.282 <= ab_xa <= 0.718):   return 0.0
    if not (1.03  <= bc_ab <= 1.718):   return 0.0
    if not (1.518 <= cd_bc <= 2.34):    return 0.0
    if abs(xd_xa - 0.886) > 0.22:       return 0.0   # xd_xa in [0.666, 1.106]

    # Soft scoring — ideal Shark ratios
    score_xa = range_score(ab_xa, 0.382, 0.618)
    score_ab = range_score(bc_ab, 1.13,  1.618)
    score_bc = range_score(cd_bc, 1.618, 2.24)
    score_c  = fib_score(xd_xa,  0.886, tol=0.08)

    overall = (score_xa + score_ab + score_bc + score_c) / 4
    return clip_confidence(overall)


# ---------------------------------------------------------------------------
# SECTION 2: MODEL PERSISTENCE
# ---------------------------------------------------------------------------

def load_model():
    """Load saved model and scaler. Raises FileNotFoundError if not trained."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Shark model not found. Run `python -m stock_analysis.shark_model` "
            "once to train and save it."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ---------------------------------------------------------------------------
# SECTION 3: SYNTHETIC DATASET
#
# Region strategy
# ───────────────
# shark_ideal    — all four ratios squarely inside their ideal Shark ranges
# shark_boundary — inside the broad hard-constraint gate but outside ideal
# near_miss      — one ratio flipped just outside its soft ideal range
# gartley        — distinct harmonic: AB_XA ≈ 0.618, XD ≈ 0.786
# bat            — AB_XA in [0.382, 0.500], XD ≈ 0.886 but CD low
# crab           — high CD_BC (≥ 2.618), XD ≈ 1.618
# cypher         — BC_AB in [1.272, 1.414], XD ≈ 0.786
# random         — uniform inside the hard-constraint envelope
# outside        — one ratio violates a hard constraint entirely
# ---------------------------------------------------------------------------

def _sample(region: str) -> tuple:
    noise = np.random.normal(0, 0.01)

    if region == "shark_ideal":
        ab_xa = np.random.uniform(0.382, 0.618)
        bc_ab = np.random.uniform(1.13,  1.618)
        cd_bc = np.random.uniform(1.618, 2.24)
        xd_xa = np.random.uniform(0.806, 0.966)   # within fib_score tol of 0.886

    elif region == "shark_boundary":
        ab_xa = np.random.uniform(0.282, 0.718)
        bc_ab = np.random.uniform(1.03,  1.718)
        cd_bc = np.random.uniform(1.518, 2.34)
        xd_xa = np.random.uniform(0.666, 1.106)

    elif region == "near_miss":
        # Start with an ideal Shark candidate, then push one ratio outside ideal
        ab_xa = np.random.uniform(0.382, 0.618)
        bc_ab = np.random.uniform(1.13,  1.618)
        cd_bc = np.random.uniform(1.618, 2.24)
        xd_xa = np.random.uniform(0.806, 0.966)
        flip  = np.random.choice(["ab", "bc", "cd", "xd"])
        if flip == "ab":
            ab_xa = np.random.choice([
                np.random.uniform(0.282, 0.381),
                np.random.uniform(0.619, 0.718),
            ])
        elif flip == "bc":
            bc_ab = np.random.choice([
                np.random.uniform(1.030, 1.129),
                np.random.uniform(1.619, 1.718),
            ])
        elif flip == "cd":
            cd_bc = np.random.choice([
                np.random.uniform(1.518, 1.617),
                np.random.uniform(2.241, 2.340),
            ])
        else:
            xd_xa = np.random.choice([
                np.random.uniform(0.666, 0.805),
                np.random.uniform(0.967, 1.106),
            ])

    elif region == "gartley":
        # Gartley: AB_XA ≈ 0.618, XD ≈ 0.786
        ab_xa = np.random.uniform(0.580, 0.660)
        bc_ab = np.random.uniform(0.380, 0.890)
        cd_bc = np.random.uniform(1.270, 1.620)
        xd_xa = np.random.uniform(0.740, 0.830)

    elif region == "bat":
        # Bat: AB_XA in [0.382, 0.500], XD ≈ 0.886 but BC_AB and CD_BC outside Shark
        ab_xa = np.random.uniform(0.382, 0.500)
        bc_ab = np.random.uniform(0.382, 0.886)   # too low for Shark
        cd_bc = np.random.uniform(1.618, 2.618)
        xd_xa = np.random.uniform(0.826, 0.946)

    elif region == "crab":
        # Crab: very high CD_BC (≥ 2.618), XD ≈ 1.618
        ab_xa = np.random.uniform(0.382, 0.618)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(2.618, 3.618)
        xd_xa = np.random.uniform(1.550, 1.700)

    elif region == "cypher":
        # Cypher: BC_AB in [1.272, 1.414], XD ≈ 0.786
        ab_xa = np.random.uniform(0.382, 0.618)
        bc_ab = np.random.uniform(1.272, 1.414)
        cd_bc = np.random.uniform(0.618, 1.130)   # too low for Shark
        xd_xa = np.random.uniform(0.706, 0.866)

    elif region == "random":
        # Uniform draw inside the hard-constraint envelope
        ab_xa = np.random.uniform(0.282, 0.718)
        bc_ab = np.random.uniform(1.030, 1.718)
        cd_bc = np.random.uniform(1.518, 2.340)
        xd_xa = np.random.uniform(0.666, 1.106)

    else:  # outside — violate at least one hard constraint
        ab_xa = np.random.uniform(0.282, 0.718)
        bc_ab = np.random.uniform(1.030, 1.718)
        cd_bc = np.random.uniform(1.518, 2.340)
        xd_xa = np.random.uniform(0.666, 1.106)
        flip  = np.random.choice(["ab", "bc", "cd", "xd"])
        if flip == "ab":
            ab_xa = np.random.choice([
                np.random.uniform(0.01,  0.281),
                np.random.uniform(0.719, 1.300),
            ])
        elif flip == "bc":
            bc_ab = np.random.choice([
                np.random.uniform(0.01,  1.029),
                np.random.uniform(1.719, 2.800),
            ])
        elif flip == "cd":
            cd_bc = np.random.choice([
                np.random.uniform(0.01,  1.517),
                np.random.uniform(2.341, 4.000),
            ])
        else:
            xd_xa = np.random.choice([
                np.random.uniform(0.01,  0.665),
                np.random.uniform(1.107, 2.000),
            ])

    return ab_xa + noise, bc_ab + noise, cd_bc + noise, xd_xa + noise


REGION_COUNTS = {
    "shark_ideal":    1500,   # strong positives
    "shark_boundary": 1000,   # weaker positives + soft negatives
    "near_miss":      1000,   # hard negatives: look like Shark but one ratio off
    "gartley":         600,   # hard negatives: different harmonic
    "bat":             600,   # hard negatives: overlapping XD but wrong BC/CD
    "crab":            400,   # hard negatives: high CD
    "cypher":          400,   # hard negatives: low CD
    "random":          500,   # negatives inside hard-constraint space
    "outside":         200,   # negatives outside hard constraints entirely
}


def build_dataset() -> pd.DataFrame:
    rows = []
    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            ab_xa, bc_ab, cd_bc, xd_xa = _sample(region)
            shark_conf = shark_score(ab_xa, bc_ab, cd_bc, xd_xa)
            label = 1 if shark_conf > 0.5 else 0
            rows.append({
                "AB_XA":           ab_xa,
                "BC_AB":           bc_ab,
                "CD_BC":           cd_bc,
                "XD_XA":           xd_xa,
                "rule_confidence": shark_conf,
                "label":           label,
                "region":          region,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


FEATURES = ["AB_XA", "BC_AB", "CD_BC", "XD_XA"]


# ---------------------------------------------------------------------------
# SECTION 4: TRAINING
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame):
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

    return model, scaler, X_test, y_test, y_prob


# ---------------------------------------------------------------------------
# SECTION 5: INFERENCE
#
# Drop-in replacement for the scoring block in SharkHarmonicPattern._run():
#
#   # Instead of:
#   overall_score = average_scores([score_xa, score_ab, score_bc, score_c])
#   confidence = clip_confidence(overall_score)
#
#   # Use:
#   confidence = learned_confidence(ab_xa, bc_ab, cd_bc, xd_xa,
#                                   model, scaler) * 100
# ---------------------------------------------------------------------------

def learned_confidence(ab_xa: float, bc_ab: float,
                       cd_bc: float, xd_xa: float,
                       model: LogisticRegression,
                       scaler: StandardScaler) -> float:
    """Returns Shark probability in [0, 1]. Multiply by 100 for [0, 100] scale."""
    x = np.array([[ab_xa, bc_ab, cd_bc, xd_xa]])
    return float(model.predict_proba(scaler.transform(x))[0, 1])


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Step 1 — Dataset
    df = build_dataset()
    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos

    print(f"\n  Total samples        : {len(df)}")
    print(f"  Shark confirmed (1)  : {n_pos:>4}  ({100*n_pos/len(df):.1f}%)")
    print(f"  Non-shark (0)        : {n_neg:>4}  ({100*n_neg/len(df):.1f}%)")
    print(f"\n  Breakdown by region:")
    print(f"  {'region':<20}  {'n':>5}  {'shark':>5}  {'shark%':>7}  {'avg_rule_conf':>14}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*14}")
    for region, grp in df.groupby("region"):
        pos = int(grp["label"].sum())
        avg = grp["rule_confidence"].mean()
        print(f"  {region:<20}  {len(grp):>5}  {pos:>5}  "
              f"{100*pos/len(grp):>6.1f}%  {avg:>14.1f}")

    # Step 2 — Train
    model, scaler, X_test, y_test, y_prob = train(df)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")
    print(f"  Scaler saved → {SCALER_PATH}")

    # Step 3 — Comparison on canonical Shark ratio combinations
    print(f"\n  {'Pattern':<26}  {'Rule':>8}  {'Learned':>9}  {'Delta':>7}")
    print(f"  {'-'*26}  {'-'*8}  {'-'*9}  {'-'*7}")

    test_cases = [
        # Shark ideal
        ("Shark — perfect",         0.500, 1.13,  1.618, 0.886),
        ("Shark — good",            0.450, 1.400, 2.000, 0.890),
        ("Shark — soft boundary",   0.382, 1.618, 2.240, 0.886),
        # Borderline / near misses
        ("Near miss: AB low",       0.300, 1.300, 1.900, 0.886),
        ("Near miss: XD far",       0.500, 1.300, 1.900, 0.980),
        ("Near miss: BC_AB low",    0.500, 1.050, 1.900, 0.886),
        ("Near miss: CD_BC low",    0.500, 1.300, 1.540, 0.886),
        # Other patterns (should score low)
        ("Gartley",                 0.618, 0.786, 1.272, 0.786),
        ("Bat",                     0.382, 0.618, 2.000, 0.886),
        ("Crab",                    0.500, 0.886, 3.000, 1.618),
        ("Cypher",                  0.500, 1.350, 0.786, 0.786),
        # Fails hard constraints
        ("Fails AB gate (too low)", 0.200, 1.300, 1.900, 0.886),
        ("Fails BC_AB gate",        0.500, 0.800, 1.900, 0.886),
        ("Fails XD gate",           0.500, 1.300, 1.900, 1.200),
    ]

    for desc, ab, bc, cd, xd in test_cases:
        rule_c  = shark_score(ab, bc, cd, xd)
        learn_c = learned_confidence(ab, bc, cd, xd, model, scaler) * 100
        delta   = learn_c - rule_c
        print(f"  {desc:<26}  {rule_c:>7.1f}%  {learn_c:>8.1f}%  {delta:>+6.1f}%")