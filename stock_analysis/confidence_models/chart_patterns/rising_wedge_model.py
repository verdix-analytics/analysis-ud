import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pkl_files")
MODEL_PATH = os.path.join(MODEL_DIR, "rising_wedge_model.pkl")

FEATURES = [
    "high_pct_slope",
    "low_pct_slope",
    "slope_divergence_score",
    "convergence_score",
    "touch_richness_score",
    "touch_balance_ratio",
]


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def rising_wedge_score(
    high_pct_slope: float,
    low_pct_slope: float,
    slope_divergence_score: float,
    convergence_score: float,
    touch_richness_score: float,
    touch_balance_ratio: float,
) -> float:
    """
    Strict rule-based quality score for a textbook-style Rising Wedge.
    Output: [0, 100]
    """

    avg_rise = (high_pct_slope + low_pct_slope) / 2

    # 1) Both lines rising
    if avg_rise >= 0.0010:
        score_rise = 100
    elif avg_rise >= 0.0006:
        score_rise = 80
    elif avg_rise >= 0.0003:
        score_rise = 55
    elif avg_rise >= 0.00015:
        score_rise = 25
    else:
        score_rise = 0

    # 2) Lower line should rise faster than upper line
    if slope_divergence_score >= 0.95:
        score_divergence = 100
    elif slope_divergence_score >= 0.80:
        score_divergence = 80
    elif slope_divergence_score >= 0.60:
        score_divergence = 55
    elif slope_divergence_score >= 0.40:
        score_divergence = 25
    else:
        score_divergence = 0

    # 3) Wedge should narrow
    if convergence_score >= 0.95:
        score_convergence = 100
    elif convergence_score >= 0.80:
        score_convergence = 80
    elif convergence_score >= 0.60:
        score_convergence = 55
    elif convergence_score >= 0.40:
        score_convergence = 25
    else:
        score_convergence = 0

    # 4) Pivot richness
    if touch_richness_score >= 0.95:
        score_touches = 100
    elif touch_richness_score >= 0.80:
        score_touches = 80
    elif touch_richness_score >= 0.60:
        score_touches = 55
    elif touch_richness_score >= 0.40:
        score_touches = 25
    else:
        score_touches = 0

    # 5) High/low touch balance
    if touch_balance_ratio >= 0.95:
        score_balance = 100
    elif touch_balance_ratio >= 0.80:
        score_balance = 80
    elif touch_balance_ratio >= 0.65:
        score_balance = 55
    elif touch_balance_ratio >= 0.45:
        score_balance = 25
    else:
        score_balance = 0

    overall = (
        0.28 * score_rise +
        0.26 * score_divergence +
        0.22 * score_convergence +
        0.14 * score_touches +
        0.10 * score_balance
    )

    return clamp_0_100(overall)


def _sample(region: str):
    if region == "ideal":
        high_pct_slope = np.random.uniform(0.0003, 0.0010)
        low_pct_slope = np.random.uniform(0.0008, 0.0020)
        slope_divergence_score = np.random.uniform(0.85, 1.00)
        convergence_score = np.random.uniform(0.85, 1.00)
        touch_richness_score = np.random.uniform(0.85, 1.00)
        touch_balance_ratio = np.random.uniform(0.85, 1.00)

    elif region == "good":
        high_pct_slope = np.random.uniform(0.0002, 0.0008)
        low_pct_slope = np.random.uniform(0.0005, 0.0015)
        slope_divergence_score = np.random.uniform(0.70, 0.92)
        convergence_score = np.random.uniform(0.70, 0.92)
        touch_richness_score = np.random.uniform(0.70, 0.92)
        touch_balance_ratio = np.random.uniform(0.70, 0.92)

    elif region == "borderline":
        high_pct_slope = np.random.uniform(0.00015, 0.0006)
        low_pct_slope = np.random.uniform(0.0002, 0.0010)
        slope_divergence_score = np.random.uniform(0.45, 0.75)
        convergence_score = np.random.uniform(0.45, 0.75)
        touch_richness_score = np.random.uniform(0.45, 0.75)
        touch_balance_ratio = np.random.uniform(0.45, 0.75)

    elif region == "weak_rise":
        high_pct_slope = np.random.uniform(0.00000, 0.00018)
        low_pct_slope = np.random.uniform(0.00000, 0.00020)
        slope_divergence_score = np.random.uniform(0.60, 0.90)
        convergence_score = np.random.uniform(0.60, 0.90)
        touch_richness_score = np.random.uniform(0.60, 0.90)
        touch_balance_ratio = np.random.uniform(0.60, 0.90)

    elif region == "bad_divergence":
        high_pct_slope = np.random.uniform(0.0002, 0.0010)
        low_pct_slope = np.random.uniform(0.0002, 0.0010)
        slope_divergence_score = np.random.uniform(0.00, 0.35)
        convergence_score = np.random.uniform(0.60, 0.90)
        touch_richness_score = np.random.uniform(0.60, 0.90)
        touch_balance_ratio = np.random.uniform(0.60, 0.90)

    elif region == "non_converging":
        high_pct_slope = np.random.uniform(0.0002, 0.0010)
        low_pct_slope = np.random.uniform(0.0005, 0.0015)
        slope_divergence_score = np.random.uniform(0.60, 0.90)
        convergence_score = np.random.uniform(0.00, 0.35)
        touch_richness_score = np.random.uniform(0.60, 0.90)
        touch_balance_ratio = np.random.uniform(0.60, 0.90)

    elif region == "few_touches":
        high_pct_slope = np.random.uniform(0.0002, 0.0010)
        low_pct_slope = np.random.uniform(0.0005, 0.0015)
        slope_divergence_score = np.random.uniform(0.60, 0.90)
        convergence_score = np.random.uniform(0.60, 0.90)
        touch_richness_score = np.random.uniform(0.00, 0.35)
        touch_balance_ratio = np.random.uniform(0.60, 0.90)

    elif region == "imbalanced_touches":
        high_pct_slope = np.random.uniform(0.0002, 0.0010)
        low_pct_slope = np.random.uniform(0.0005, 0.0015)
        slope_divergence_score = np.random.uniform(0.60, 0.90)
        convergence_score = np.random.uniform(0.60, 0.90)
        touch_richness_score = np.random.uniform(0.55, 0.90)
        touch_balance_ratio = np.random.uniform(0.00, 0.35)

    elif region == "random":
        high_pct_slope = np.random.uniform(-0.0010, 0.0020)
        low_pct_slope = np.random.uniform(-0.0010, 0.0025)
        slope_divergence_score = np.random.uniform(0.00, 1.00)
        convergence_score = np.random.uniform(0.00, 1.00)
        touch_richness_score = np.random.uniform(0.00, 1.00)
        touch_balance_ratio = np.random.uniform(0.00, 1.00)

    else:  # outside
        high_pct_slope = np.random.uniform(-0.0020, 0.00005)
        low_pct_slope = np.random.uniform(-0.0020, 0.00005)
        slope_divergence_score = np.random.uniform(0.00, 0.25)
        convergence_score = np.random.uniform(0.00, 0.25)
        touch_richness_score = np.random.uniform(0.00, 0.25)
        touch_balance_ratio = np.random.uniform(0.00, 0.25)

    return (
        float(high_pct_slope),
        float(low_pct_slope),
        float(slope_divergence_score),
        float(convergence_score),
        float(touch_richness_score),
        float(touch_balance_ratio),
    )


REGION_COUNTS = {
    "ideal": 1400,
    "good": 1200,
    "borderline": 1000,
    "weak_rise": 900,
    "bad_divergence": 900,
    "non_converging": 800,
    "few_touches": 800,
    "imbalanced_touches": 700,
    "random": 700,
    "outside": 300,
}


def build_dataset() -> pd.DataFrame:
    rows = []

    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            (
                high_pct_slope,
                low_pct_slope,
                slope_divergence_score,
                convergence_score,
                touch_richness_score,
                touch_balance_ratio,
            ) = _sample(region)

            base_score = rising_wedge_score(
                high_pct_slope=high_pct_slope,
                low_pct_slope=low_pct_slope,
                slope_divergence_score=slope_divergence_score,
                convergence_score=convergence_score,
                touch_richness_score=touch_richness_score,
                touch_balance_ratio=touch_balance_ratio,
            )

            noise = np.random.normal(0, 2.0)
            quality = clamp_0_100(base_score + noise)

            rows.append({
                "high_pct_slope": high_pct_slope,
                "low_pct_slope": low_pct_slope,
                "slope_divergence_score": slope_divergence_score,
                "convergence_score": convergence_score,
                "touch_richness_score": touch_richness_score,
                "touch_balance_ratio": touch_balance_ratio,
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
            "Rising Wedge regression model not found. Run "
            "`python -m stock_analysis.confidence_models.chart_patterns.rising_wedge_model` "
            "to train and save it first."
        )
    model = joblib.load(MODEL_PATH)
    return model


def learned_confidence(
    high_pct_slope: float,
    low_pct_slope: float,
    slope_divergence_score: float,
    convergence_score: float,
    touch_richness_score: float,
    touch_balance_ratio: float,
    model: RandomForestRegressor,
) -> float:
    x = np.array([[
        high_pct_slope,
        low_pct_slope,
        slope_divergence_score,
        convergence_score,
        touch_richness_score,
        touch_balance_ratio,
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
        ("Ideal",            0.0005, 0.0012, 0.95, 0.95, 0.95, 0.95),
        ("Good",             0.00035, 0.0008, 0.82, 0.82, 0.80, 0.80),
        ("Borderline",       0.0002, 0.00035, 0.60, 0.60, 0.60, 0.60),
        ("Weak rise",        0.00008, 0.00010, 0.80, 0.80, 0.80, 0.80),
        ("Bad divergence",   0.0006, 0.00065, 0.20, 0.80, 0.80, 0.80),
        ("No convergence",   0.0004, 0.0010, 0.82, 0.20, 0.80, 0.80),
        ("Few touches",      0.0004, 0.0010, 0.82, 0.82, 0.20, 0.80),
        ("Bad balance",      0.0004, 0.0010, 0.82, 0.82, 0.80, 0.20),
        ("User example",     0.00032, 0.00071, 0.72, 0.70, 0.68, 0.66),
    ]

    print(f"\n{'Pattern':<16} {'RuleScore':>10} {'Predicted':>10} {'Delta':>10}")
    print("-" * 54)

    for desc, high_pct_slope, low_pct_slope, slope_divergence_score, convergence_score, touch_richness_score, touch_balance_ratio in test_cases:
        rule_score = rising_wedge_score(
            high_pct_slope=high_pct_slope,
            low_pct_slope=low_pct_slope,
            slope_divergence_score=slope_divergence_score,
            convergence_score=convergence_score,
            touch_richness_score=touch_richness_score,
            touch_balance_ratio=touch_balance_ratio,
        )
        pred_score = learned_confidence(
            high_pct_slope=high_pct_slope,
            low_pct_slope=low_pct_slope,
            slope_divergence_score=slope_divergence_score,
            convergence_score=convergence_score,
            touch_richness_score=touch_richness_score,
            touch_balance_ratio=touch_balance_ratio,
            model=model,
        )
        delta = pred_score - rule_score
        print(f"{desc:<16} {rule_score:>10.2f} {pred_score:>10.2f} {delta:>10.2f}")