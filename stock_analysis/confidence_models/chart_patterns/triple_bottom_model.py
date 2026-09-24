import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pkl_files")
MODEL_PATH = os.path.join(MODEL_DIR, "triple_bottom_model.pkl")

FEATURES = [
    "bottom_spread_pct",
    "rebound_height",
    "span_score",
    "symmetry_ratio",
    "peak_balance_ratio",
    "total_span",
]


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def triple_bottom_score(
    bottom_spread_pct: float,
    rebound_height: float,
    span_score: float,
    symmetry_ratio: float,
    peak_balance_ratio: float,
    total_span: float,
) -> float:
    """
    Strict rule-based quality score for a textbook-style Triple Bottom.
    Output: [0, 100]
    """

    # 1) Bottom consistency
    if bottom_spread_pct <= 0.008:
        score_bottoms = 100
    elif bottom_spread_pct <= 0.015:
        score_bottoms = 80
    elif bottom_spread_pct <= 0.025:
        score_bottoms = 55
    elif bottom_spread_pct <= 0.035:
        score_bottoms = 25
    else:
        score_bottoms = 0

    # 2) Rebound height
    if rebound_height >= 0.060:
        score_rebound = 100
    elif rebound_height >= 0.045:
        score_rebound = 80
    elif rebound_height >= 0.030:
        score_rebound = 55
    elif rebound_height >= 0.020:
        score_rebound = 25
    else:
        score_rebound = 0

    # 3) Span adequacy
    if span_score >= 0.95:
        score_span = 100
    elif span_score >= 0.75:
        score_span = 80
    elif span_score >= 0.55:
        score_span = 55
    elif span_score >= 0.35:
        score_span = 25
    else:
        score_span = 0

    # 4) Structural symmetry
    if symmetry_ratio >= 0.90:
        score_sym = 100
    elif symmetry_ratio >= 0.75:
        score_sym = 80
    elif symmetry_ratio >= 0.60:
        score_sym = 55
    elif symmetry_ratio >= 0.40:
        score_sym = 25
    else:
        score_sym = 0

    # 5) Peak balance
    if peak_balance_ratio >= 0.95:
        score_peak_balance = 100
    elif peak_balance_ratio >= 0.80:
        score_peak_balance = 80
    elif peak_balance_ratio >= 0.65:
        score_peak_balance = 55
    elif peak_balance_ratio >= 0.45:
        score_peak_balance = 25
    else:
        score_peak_balance = 0

    # 6) Total span smoothness
    if total_span >= 10:
        score_total_span = 100
    elif total_span >= 8:
        score_total_span = 80
    elif total_span >= 6:
        score_total_span = 55
    elif total_span >= 4:
        score_total_span = 25
    else:
        score_total_span = 0

    overall = (
        0.30 * score_bottoms +
        0.22 * score_rebound +
        0.14 * score_span +
        0.14 * score_sym +
        0.10 * score_peak_balance +
        0.10 * score_total_span
    )

    return clamp_0_100(overall)


def _sample(region: str):
    if region == "ideal":
        bottom_spread_pct = np.random.uniform(0.000, 0.008)
        rebound_height = np.random.uniform(0.050, 0.080)
        span_score = np.random.uniform(0.85, 1.00)
        symmetry_ratio = np.random.uniform(0.80, 1.00)
        peak_balance_ratio = np.random.uniform(0.85, 1.00)
        total_span = np.random.uniform(8, 14)

    elif region == "good":
        bottom_spread_pct = np.random.uniform(0.008, 0.018)
        rebound_height = np.random.uniform(0.035, 0.060)
        span_score = np.random.uniform(0.65, 0.90)
        symmetry_ratio = np.random.uniform(0.65, 0.90)
        peak_balance_ratio = np.random.uniform(0.70, 0.90)
        total_span = np.random.uniform(6, 12)

    elif region == "borderline":
        bottom_spread_pct = np.random.uniform(0.018, 0.035)
        rebound_height = np.random.uniform(0.020, 0.040)
        span_score = np.random.uniform(0.40, 0.75)
        symmetry_ratio = np.random.uniform(0.40, 0.75)
        peak_balance_ratio = np.random.uniform(0.45, 0.75)
        total_span = np.random.uniform(4, 10)

    elif region == "bad_bottoms":
        bottom_spread_pct = np.random.uniform(0.035, 0.100)
        rebound_height = np.random.uniform(0.030, 0.060)
        span_score = np.random.uniform(0.60, 0.90)
        symmetry_ratio = np.random.uniform(0.55, 0.90)
        peak_balance_ratio = np.random.uniform(0.60, 0.90)
        total_span = np.random.uniform(6, 12)

    elif region == "weak_rebounds":
        bottom_spread_pct = np.random.uniform(0.005, 0.020)
        rebound_height = np.random.uniform(0.003, 0.020)
        span_score = np.random.uniform(0.60, 0.90)
        symmetry_ratio = np.random.uniform(0.55, 0.90)
        peak_balance_ratio = np.random.uniform(0.60, 0.90)
        total_span = np.random.uniform(6, 12)

    elif region == "asymmetric":
        bottom_spread_pct = np.random.uniform(0.005, 0.020)
        rebound_height = np.random.uniform(0.025, 0.060)
        span_score = np.random.uniform(0.55, 0.85)
        symmetry_ratio = np.random.uniform(0.10, 0.45)
        peak_balance_ratio = np.random.uniform(0.50, 0.90)
        total_span = np.random.uniform(6, 12)

    elif region == "uneven_peaks":
        bottom_spread_pct = np.random.uniform(0.005, 0.020)
        rebound_height = np.random.uniform(0.025, 0.060)
        span_score = np.random.uniform(0.55, 0.85)
        symmetry_ratio = np.random.uniform(0.55, 0.90)
        peak_balance_ratio = np.random.uniform(0.10, 0.45)
        total_span = np.random.uniform(6, 12)

    elif region == "compressed":
        bottom_spread_pct = np.random.uniform(0.005, 0.020)
        rebound_height = np.random.uniform(0.020, 0.050)
        span_score = np.random.uniform(0.10, 0.35)
        symmetry_ratio = np.random.uniform(0.55, 0.95)
        peak_balance_ratio = np.random.uniform(0.55, 0.95)
        total_span = np.random.uniform(3, 5)

    elif region == "random":
        bottom_spread_pct = np.random.uniform(0.000, 0.120)
        rebound_height = np.random.uniform(0.000, 0.100)
        span_score = np.random.uniform(0.00, 1.00)
        symmetry_ratio = np.random.uniform(0.00, 1.00)
        peak_balance_ratio = np.random.uniform(0.00, 1.00)
        total_span = np.random.uniform(3, 15)

    else:  # outside
        bottom_spread_pct = np.random.uniform(0.060, 0.200)
        rebound_height = np.random.uniform(0.000, 0.015)
        span_score = np.random.uniform(0.00, 0.30)
        symmetry_ratio = np.random.uniform(0.00, 0.30)
        peak_balance_ratio = np.random.uniform(0.00, 0.30)
        total_span = np.random.uniform(3, 6)

    return (
        float(bottom_spread_pct),
        float(rebound_height),
        float(span_score),
        float(symmetry_ratio),
        float(peak_balance_ratio),
        float(total_span),
    )


REGION_COUNTS = {
    "ideal": 1400,
    "good": 1200,
    "borderline": 1000,
    "bad_bottoms": 900,
    "weak_rebounds": 900,
    "asymmetric": 800,
    "uneven_peaks": 700,
    "compressed": 700,
    "random": 700,
    "outside": 300,
}


def build_dataset() -> pd.DataFrame:
    rows = []

    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            (
                bottom_spread_pct,
                rebound_height,
                span_score,
                symmetry_ratio,
                peak_balance_ratio,
                total_span,
            ) = _sample(region)

            base_score = triple_bottom_score(
                bottom_spread_pct=bottom_spread_pct,
                rebound_height=rebound_height,
                span_score=span_score,
                symmetry_ratio=symmetry_ratio,
                peak_balance_ratio=peak_balance_ratio,
                total_span=total_span,
            )

            noise = np.random.normal(0, 2.0)
            quality = clamp_0_100(base_score + noise)

            rows.append({
                "bottom_spread_pct": bottom_spread_pct,
                "rebound_height": rebound_height,
                "span_score": span_score,
                "symmetry_ratio": symmetry_ratio,
                "peak_balance_ratio": peak_balance_ratio,
                "total_span": total_span,
                "quality": quality,
                "region": region,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def train(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df["quality"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, mae, r2


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Triple Bottom regression model not found. Run "
            "`python -m stock_analysis.confidence_models.chart_patterns.triple_bottom_model` "
            "to train and save it first."
        )
    model = joblib.load(MODEL_PATH)
    return model


def learned_confidence(
    bottom_spread_pct: float,
    rebound_height: float,
    span_score: float,
    symmetry_ratio: float,
    peak_balance_ratio: float,
    total_span: float,
    model: RandomForestRegressor,
) -> float:
    x = np.array([[
        bottom_spread_pct,
        rebound_height,
        span_score,
        symmetry_ratio,
        peak_balance_ratio,
        total_span,
    ]])

    pred = float(model.predict(x)[0])
    return clamp_0_100(pred)


if __name__ == "__main__":
    df = build_dataset()

    print(f"\nTotal samples: {len(df)}")
    print("\nQuality distribution:")
    print(df["quality"].describe())

    print("\nUnique rounded quality values (sample):")
    print(sorted(df["quality"].round(2).unique()[:20]))
    print(f"Total unique rounded qualities: {len(np.unique(df['quality'].round(2)))}")

    model, X_test, y_test, y_pred, mae, r2 = train(df)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"\nTest MAE: {mae:.3f}")
    print(f"Test R^2: {r2:.3f}")

    test_cases = [
        ("Ideal",           0.005, 0.060, 0.95, 0.90, 0.95, 10),
        ("Good",            0.012, 0.045, 0.80, 0.80, 0.82, 8),
        ("Borderline",      0.025, 0.028, 0.60, 0.60, 0.62, 6),
        ("Bad bottoms",     0.070, 0.040, 0.80, 0.80, 0.82, 8),
        ("Weak rebound",    0.012, 0.010, 0.80, 0.80, 0.82, 8),
        ("Asymmetric",      0.012, 0.040, 0.75, 0.25, 0.80, 8),
        ("Uneven peaks",    0.012, 0.040, 0.75, 0.80, 0.25, 8),
        ("Compressed",      0.012, 0.035, 0.20, 0.80, 0.80, 4),
        ("User example",    0.018, 0.032, 0.62, 0.58, 0.70, 7),
    ]

    print(f"\n{'Pattern':<16} {'RuleScore':>10} {'Predicted':>10} {'Delta':>10}")
    print("-" * 54)

    for desc, bottom_spread_pct, rebound_height, span_score, symmetry_ratio, peak_balance_ratio, total_span in test_cases:
        rule_score = triple_bottom_score(
            bottom_spread_pct=bottom_spread_pct,
            rebound_height=rebound_height,
            span_score=span_score,
            symmetry_ratio=symmetry_ratio,
            peak_balance_ratio=peak_balance_ratio,
            total_span=total_span,
        )
        pred_score = learned_confidence(
            bottom_spread_pct=bottom_spread_pct,
            rebound_height=rebound_height,
            span_score=span_score,
            symmetry_ratio=symmetry_ratio,
            peak_balance_ratio=peak_balance_ratio,
            total_span=total_span,
            model=model,
        )
        delta = pred_score - rule_score
        print(f"{desc:<16} {rule_score:>10.2f} {pred_score:>10.2f} {delta:>10.2f}")