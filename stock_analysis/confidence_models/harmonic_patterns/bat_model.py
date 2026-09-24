import numpy as np
import pandas as pd
import os
import sys
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from stock_analysis.tools.harmonic_utils import (
    range_score, 
    fib_score,
    clip_confidence
)
 
np.random.seed(42)

MODEL_DIR  = os.path.join(os.path.dirname(__file__), '..', "pkl_files")
MODEL_PATH  = os.path.join(MODEL_DIR, "bat_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "bat_scaler.pkl")

def bat_score(ab_xa: float, bc_ab: float,
                     cd_bc: float, xd_xa: float) -> float:
    """
    Calculating score of AB, BC, CD and XD using range score for AB, BC, CD 
    and fibonnaci score for XD
    the implementation is same as what is written in _run() for BatHarmonic pattern
    but i just moved it here, so that the model can calculate the confidence score rather 
    than using weighted scores.
    """
    # Hard constraints (broad gate from your code)
    # if not (0.264 <= ab_xa <= 0.618): return 0.0
    # if not (0.000 <= bc_ab <= 1.390): return 0.0
    # if not (0.618 <= cd_bc <= 3.618): return 0.0
    # if not (0.566 <= xd_xa <= 1.206): return 0.0
 
    # Soft scoring with equal weights (what the model will learn to replace)
    score_ab = range_score(ab_xa, 0.382, 0.500)
    score_bc = range_score(bc_ab, 0.382, 0.886)
    score_cd = range_score(cd_bc, 1.618, 2.618)
    score_xd = fib_score(xd_xa,  0.886, tol=0.08)
 
    overall = (score_ab + score_bc + score_cd + score_xd) / 4
    return clip_confidence(overall)

def load_model():
    """
    Load saved model and scaler.
    Raises FileNotFoundError if not yet trained.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Bat model not found. Run `python -m stock_analysis.bat_model` "
            "once to train and save it."
        )

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def _sample(region: str) -> tuple:
    noise = np.random.normal(0,0.01)
    
    if region == "bat_ideal":
        ab_xa = np.random.uniform(0.382, 0.500)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(1.618, 2.618)
        xd_xa = np.random.uniform(0.806, 0.966)   # within fib_score tol of 0.886
 
    elif region == "bat_boundary":
        ab_xa = np.random.uniform(0.264, 0.618)
        bc_ab = np.random.uniform(0.000, 1.390)
        cd_bc = np.random.uniform(0.618, 3.618)
        xd_xa = np.random.uniform(0.566, 1.206)
 
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
 
    elif region == "butterfly":
        ab_xa = np.random.uniform(0.700, 0.800)
        bc_ab = np.random.uniform(0.380, 0.890)
        cd_bc = np.random.uniform(1.618, 2.240)
        xd_xa = np.random.uniform(1.270, 1.420)
 
    elif region == "near_miss":
        # Start with a valid bat_ideal, push one ratio outside its soft range
        ab_xa = np.random.uniform(0.382, 0.500)
        bc_ab = np.random.uniform(0.382, 0.886)
        cd_bc = np.random.uniform(1.618, 2.618)
        xd_xa = np.random.uniform(0.806, 0.966)
        flip = np.random.choice(["ab", "bc", "cd", "xd"])
        if flip == "ab":
            ab_xa = np.random.choice([
                np.random.uniform(0.264, 0.381),
                np.random.uniform(0.501, 0.618),
            ])
        elif flip == "bc":
            bc_ab = np.random.choice([
                np.random.uniform(0.000, 0.381),
                np.random.uniform(0.887, 1.390),
            ])
        elif flip == "cd":
            cd_bc = np.random.choice([
                np.random.uniform(0.618, 1.617),
                np.random.uniform(2.619, 3.618),
            ])
        else:
            xd_xa = np.random.choice([
                np.random.uniform(0.566, 0.805),
                np.random.uniform(0.967, 1.206),
            ])
 
    elif region == "random":
        ab_xa = np.random.uniform(0.264, 0.618)
        bc_ab = np.random.uniform(0.000, 1.390)
        cd_bc = np.random.uniform(0.618, 3.618)
        xd_xa = np.random.uniform(0.566, 1.206)
 
    else:  # outside hard constraints entirely
        ab_xa = np.random.uniform(0.264, 0.618)
        bc_ab = np.random.uniform(0.000, 1.390)
        cd_bc = np.random.uniform(0.618, 3.618)
        xd_xa = np.random.uniform(0.566, 1.206)
        flip = np.random.choice(["ab", "bc", "cd", "xd"])
        if flip == "ab":
            ab_xa = np.random.choice([
                np.random.uniform(0.01,  0.263),
                np.random.uniform(0.619, 1.300),
            ])
        elif flip == "bc":
            bc_ab = np.random.uniform(1.40, 2.50)
        elif flip == "cd":
            cd_bc = np.random.choice([
                np.random.uniform(0.01,  0.617),
                np.random.uniform(3.619, 5.000),
            ])
        else:
            xd_xa = np.random.choice([
                np.random.uniform(0.01,  0.565),
                np.random.uniform(1.207, 2.000),
            ])
 
    return ab_xa+noise, bc_ab+noise, cd_bc+noise, xd_xa+noise
 
 
REGION_COUNTS = {
    "bat_ideal":    1500,   # strong positives
    "bat_boundary": 1000,   # weaker positives + negatives (hard constraint only)
    "near_miss":    1000,   # hard negatives: look like bat but one ratio off
    "gartley":       800,   # hard negatives: different harmonic
    "crab":          600,   # hard negatives
    "butterfly":     400,   # hard negatives
    "random":        500,   # negatives inside hard constraint space
    "outside":       200,   # negatives outside hard constraints entirely
}
 
 
def build_dataset() -> pd.DataFrame:
    rows = []
    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            ab_xa, bc_ab, cd_bc, xd_xa = _sample(region)
            bat_conf = bat_score(ab_xa, bc_ab, cd_bc, xd_xa)
            label = 1 if (bat_conf > 0.5 and xd_xa <= 1.206) else 0
            rows.append({
                "AB_XA":           ab_xa,
                "BC_AB":           bc_ab,
                "CD_BC":           cd_bc,
                "XD_XA":           xd_xa,
                "rule_confidence": bat_conf,
                "label":           label,
                "region":          region,
            })
 
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

FEATURES = ["AB_XA", "BC_AB", "CD_BC", "XD_XA"]
 
 
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
        class_weight="balanced",   # corrects for bat/non-bat imbalance
        random_state=42,
    )
    model.fit(X_train_s, y_train)
 
    y_prob = model.predict_proba(X_test_s)[:, 1]
 
    return model, scaler, X_test, y_test, y_prob
 
 
# ─────────────────────────────────────────────────────────────────────
# SECTION 4: INFERENCE
#
# Plug learned_confidence() into BatHarmonicPattern._run()
# to replace the equal-weight average_scores() call:
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
    """Returns bat probability in [0, 1]. Multiply by 100 for [0, 100] scale."""
    x = np.array([[ab_xa, bc_ab, cd_bc, xd_xa]])
    return float(model.predict_proba(scaler.transform(x))[0, 1])
 
 
# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
 
    df = build_dataset()
    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos
 
    print(f"\n  Total samples     : {len(df)}")
    print(f"  Bat confirmed (1) : {n_pos:>4}  ({100*n_pos/len(df):.1f}%)")
    print(f"  Non-bat (0)       : {n_neg:>4}  ({100*n_neg/len(df):.1f}%)")
    print(f"\n  Breakdown by region:")
    print(f"  {'region':<18}  {'n':>5}  {'bat':>5}  {'bat%':>6}  {'avg_rule_conf':>14}")
    print(f"  {'-'*18}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*14}")
    for region, grp in df.groupby("region"):
        pos = int(grp["label"].sum())
        avg = grp["rule_confidence"].mean()
        print(f"  {region:<18}  {len(grp):>5}  {pos:>5}  "
              f"{100*pos/len(grp):>5.1f}%  {avg:>14.1f}")
 
    # Step 2 — Train 
    model, scaler, X_test, y_test, y_prob = train(df)
    os.makedirs(MODEL_DIR, exist_ok=True)  # add this line
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
 
    # Step 3 — Comparison on canonical ratio combinations
    print(f"\n  {'Pattern':<22}  {'Rule':>8}  {'Learned':>9}  {'Delta':>7}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*9}  {'-'*7}")
 
    test_cases = [
        # Bat ideal
        ("Bat — perfect",       0.382, 0.618, 2.000, 0.886),
        ("Bat — good",          0.440, 0.700, 2.200, 0.890),
        ("Bat — soft boundary", 0.382, 0.382, 1.618, 0.886),
        # Borderline
        ("Near miss: AB high",  0.560, 0.618, 2.000, 0.886),
        ("Near miss: XD far",   0.382, 0.618, 2.000, 0.960),
        ("Near miss: CD low",   0.382, 0.618, 0.900, 0.886),
        # Other patterns (should score low)
        ("Gartley",             0.618, 0.618, 1.618, 0.786),
        ("Crab",                0.500, 0.886, 3.000, 1.618),
        ("Butterfly",           0.786, 0.500, 1.800, 1.270),
        # Fails hard constraints
        ("Fails AB gate",       0.700, 0.618, 2.000, 0.886),
        ("Fails XD gate",       0.382, 0.618, 2.000, 1.400),
    ]
 
    for desc, ab, bc, cd, xd in test_cases:
        rule_c  = bat_score(ab, bc, cd, xd)
        learn_c = learned_confidence(ab, bc, cd, xd, model, scaler) * 100
        delta   = learn_c - rule_c
        print(f"  {desc:<22}  {rule_c:>7.1f}%  {learn_c:>8.1f}%  {delta:>+6.1f}%")