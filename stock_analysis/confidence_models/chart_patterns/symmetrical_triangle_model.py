import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pkl_files")
MODEL_PATH = os.path.join(MODEL_DIR, "symmetrical_triangle_model.pkl")

FEATURES = [
    "high_neg_pct_slope",
    "low_pct_slope",
    "slope_balance",
    "gap_pct_now",
    "pivot_count",
    "touch_balance_ratio",
]


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def symmetrical_triangle_score(
    high_neg_pct_slope: float,
    low_pct_slope: float,
    slope_balance: float,
    gap_pct_now: float,
    pivot_count: float,
    touch_balance_ratio: float,
) -> float:
    """
    Strict rule-based quality score for a textbook-style Symmetrical Triangle.
    Output: [0, 100]
    """

    # 1) Upper line should descend
    if high_neg_pct_slope >= 0.0008:
        score_high_drop = 100
    elif high_neg_pct_slope >= 0.0005:
        score_high_drop = 80
    elif high_neg_pct_slope >= 0.00025:
        score_high_drop = 55
    elif high_neg_pct_slope >= 0.00010:
        score_high_drop = 25
    else:
        score_high_drop = 0

    # 2) Lower line should rise
    if low_pct_slope >= 0.0008:
        score_low_rise = 100
    elif low_pct_slope >= 0.0005:
        score_low_rise = 80
    elif low_pct_slope >= 0.00025:
        score_low_rise = 55
    elif low_pct_slope >= 0.00010:
        score_low_rise = 25
    else:
        score_low_rise = 0

    # 3) Slope balance
    if slope_balance >= 0.90:
        score_balance = 100
    elif slope_balance >= 0.75:
        score_balance = 80
    elif slope_balance >= 0.60:
        score_balance = 55
    elif slope_balance >= 0.40:
        score_balance = 25
    else:
        score_balance = 0

    # 4) Convergence
    if gap_pct_now <= 0.015:
        score_gap = 100
    elif gap_pct_now <= 0.025:
        score_gap = 80
    elif gap_pct_now <= 0.040:
        score_gap = 55
    elif gap_pct_now <= 0.060:
        score_gap = 25
    else:
        score_gap = 0

    # 5) Pivot richness
    if pivot_count >= 8:
        score_pivots = 100
    elif pivot_count >= 6:
        score_pivots = 80
    elif pivot_count >= 5:
        score_pivots = 55
    elif pivot_count >= 4:
        score_pivots = 25
    else:
        score_pivots = 0

    # 6) High/low touch balance
    if touch_balance_ratio >= 0.90:
        score_touch_balance = 100
    elif touch_balance_ratio >= 0.75:
        score_touch_balance = 80
    elif touch_balance_ratio >= 0.60:
        score_touch_balance = 55
    elif touch_balance_ratio >= 0.40:
        score_touch_balance = 25
    else:
        score_touch_balance = 0

    overall = (
        0.24 * score_high_drop +
        0.24 * score_low_rise +
        0.20 * score_balance +
        0.14 * score_gap +
        0.10 * score_pivots +
        0.08 * score_touch_balance
    )

    return clamp_0_100(overall)


def _sample(region: str):
    if region == "ideal":
        high_neg_pct_slope = np.random.uniform(0.0008, 0.0020)
        low_pct_slope = np.random.uniform(0.0008, 0.0020)
        slope_balance = np.random.uniform(0.85, 1.00)
        gap_pct_now = np.random.uniform(0.005, 0.018)
        pivot_count = np.random.randint(6, 10)
        n_highs = np.random.randint(3, 5)
        n_lows = np.random.randint(3, 5)

    elif region == "good":
        high_neg_pct_slope = np.random.uniform(0.0005, 0.0012)
        low_pct_slope = np.random.uniform(0.0005, 0.0012)
        slope_balance = np.random.uniform(0.70, 0.90)
        gap_pct_now = np.random.uniform(0.012, 0.028)
        pivot_count = np.random.randint(5, 9)
        n_highs = np.random.randint(2, 5)
        n_lows = np.random.randint(2, 5)

    elif region == "borderline":
        high_neg_pct_slope = np.random.uniform(0.00015, 0.0006)
        low_pct_slope = np.random.uniform(0.00015, 0.0006)
        slope_balance = np.random.uniform(0.45, 0.75)
        gap_pct_now = np.random.uniform(0.020, 0.045)
        pivot_count = np.random.randint(4, 7)
        n_highs = np.random.randint(2, 4)
        n_lows = np.random.randint(2, 4)

    elif region == "weak_high_drop":
        high_neg_pct_slope = np.random.uniform(0.0000, 0.00015)
        low_pct_slope = np.random.uniform(0.0004, 0.0012)
        slope_balance = np.random.uniform(0.40, 0.80)
        gap_pct_now = np.random.uniform(0.020, 0.050)
        pivot_count = np.random.randint(4, 8)
        n_highs = np.random.randint(2, 4)
        n_lows = np.random.randint(2, 4)

    elif region == "weak_low_rise":
        high_neg_pct_slope = np.random.uniform(0.0004, 0.0012)
        low_pct_slope = np.random.uniform(0.0000, 0.00015)
        slope_balance = np.random.uniform(0.40, 0.80)
        gap_pct_now = np.random.uniform(0.020, 0.050)
        pivot_count = np.random.randint(4, 8)
        n_highs = np.random.randint(2, 4)
        n_lows = np.random.randint(2, 4)

    elif region == "unbalanced_slopes":
        high_neg_pct_slope = np.random.uniform(0.0004, 0.0020)
        low_pct_slope = np.random.uniform(0.0004, 0.0020)
        slope_balance = np.random.uniform(0.10, 0.45)
        gap_pct_now = np.random.uniform(0.015, 0.040)
        pivot_count = np.random.randint(4, 8)
        n_highs = np.random.randint(2, 4)
        n_lows = np.random.randint(2, 4)

    elif region == "wide_gap":
        high_neg_pct_slope = np.random.uniform(0.0004, 0.0012)
        low_pct_slope = np.random.uniform(0.0004, 0.0012)
        slope_balance = np.random.uniform(0.55, 0.90)
        gap_pct_now = np.random.uniform(0.050, 0.120)
        pivot_count = np.random.randint(4, 8)
        n_highs = np.random.randint(2, 4)
        n_lows = np.random.randint(2, 4)

    elif region == "imbalanced_touches":
        high_neg_pct_slope = np.random.uniform(0.0004, 0.0012)
        low_pct_slope = np.random.uniform(0.0004, 0.0012)
        slope_balance = np.random.uniform(0.55, 0.90)
        gap_pct_now = np.random.uniform(0.015, 0.035)
        pivot_count = np.random.randint(4, 8)
        if np.random.rand() < 0.5:
            n_highs = 2
            n_lows = np.random.randint(5, 8)
        else:
            n_highs = np.random.randint(5, 8)
            n_lows = 2

    elif region == "compressed":
        high_neg_pct_slope = np.random.uniform(0.0004, 0.0012)
        low_pct_slope = np.random.uniform(0.0004, 0.0012)
        slope_balance = np.random.uniform(0.60, 0.95)
        gap_pct_now = np.random.uniform(0.003, 0.012)
        pivot_count = 4
        n_highs = 2
        n_lows = 2

    elif region == "random":
        high_neg_pct_slope = np.random.uniform(0.0000, 0.0100)
        low_pct_slope = np.random.uniform(0.0000, 0.0100)
        slope_balance = np.random.uniform(0.00, 1.00)
        gap_pct_now = np.random.uniform(0.003, 0.150)
        pivot_count = np.random.randint(4, 10)
        n_highs = np.random.randint(2, 8)
        n_lows = np.random.randint(2, 8)

    else:  # outside
        high_neg_pct_slope = np.random.uniform(0.0000, 0.0001)
        low_pct_slope = np.random.uniform(0.0000, 0.0001)
        slope_balance = np.random.uniform(0.00, 0.30)
        gap_pct_now = np.random.uniform(0.080, 0.200)
        pivot_count = np.random.randint(4, 7)
        n_highs = np.random.randint(2, 5)
        n_lows = np.random.randint(2, 5)

    touch_balance_ratio = min(n_highs, n_lows) / max(n_highs, n_lows)

    return (
        float(high_neg_pct_slope),
        float(low_pct_slope),
        float(slope_balance),
        float(gap_pct_now),
        float(pivot_count),
        float(touch_balance_ratio),
    )


REGION_COUNTS = {
    "ideal": 1400,
    "good": 1200,
    "borderline": 1000,
    "weak_high_drop": 900,
    "weak_low_rise": 900,
    "unbalanced_slopes": 800,
    "wide_gap": 700,
    "imbalanced_touches": 700,
    "compressed": 600,
    "random": 700,
    "outside": 300,
}


def build_dataset() -> pd.DataFrame:
    rows = []

    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            (
                high_neg_pct_slope,
                low_pct_slope,
                slope_balance,
                gap_pct_now,
                pivot_count,
                touch_balance_ratio,
            ) = _sample(region)

            base_score = symmetrical_triangle_score(
                high_neg_pct_slope=high_neg_pct_slope,
                low_pct_slope=low_pct_slope,
                slope_balance=slope_balance,
                gap_pct_now=gap_pct_now,
                pivot_count=pivot_count,
                touch_balance_ratio=touch_balance_ratio,
            )

            noise = np.random.normal(0, 2.0)
            quality = clamp_0_100(base_score + noise)

            rows.append({
                "high_neg_pct_slope": high_neg_pct_slope,
                "low_pct_slope": low_pct_slope,
                "slope_balance": slope_balance,
                "gap_pct_now": gap_pct_now,
                "pivot_count": pivot_count,
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
            "Symmetrical Triangle regression model not found. Run "
            "`python -m stock_analysis.confidence_models.chart_patterns.symmetrical_triangle_model` "
            "to train and save it first."
        )

    model = joblib.load(MODEL_PATH)
    return model


def learned_confidence(
    high_neg_pct_slope: float,
    low_pct_slope: float,
    slope_balance: float,
    gap_pct_now: float,
    pivot_count: float,
    touch_balance_ratio: float,
    model: RandomForestRegressor,
) -> float:
    x = np.array([[
        high_neg_pct_slope,
        low_pct_slope,
        slope_balance,
        gap_pct_now,
        pivot_count,
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
        ("Ideal",             0.0012, 0.0012, 0.95, 0.010, 8, 1.00),
        ("Good",              0.0007, 0.0007, 0.80, 0.020, 6, 0.75),
        ("Borderline",        0.0003, 0.0003, 0.55, 0.035, 5, 0.67),
        ("Weak high drop",    0.00005, 0.0008, 0.60, 0.020, 6, 1.00),
        ("Weak low rise",     0.0008, 0.00005, 0.60, 0.020, 6, 1.00),
        ("Unbalanced slope",  0.0015, 0.0003, 0.20, 0.020, 6, 1.00),
        ("Wide gap",          0.0008, 0.0008, 0.80, 0.090, 6, 1.00),
        ("Imbalanced touch",  0.0008, 0.0008, 0.80, 0.020, 7, 0.33),
        ("Compressed",        0.0008, 0.0008, 0.85, 0.008, 4, 1.00),
        ("User example",      0.00042, 0.00055, 0.58, 0.028, 6, 0.60),
    ]

    print(f"\n{'Pattern':<18} {'RuleScore':>10} {'Predicted':>10} {'Delta':>10}")
    print("-" * 56)

    for desc, high_neg_pct_slope, low_pct_slope, slope_balance, gap_pct_now, pivot_count, touch_balance_ratio in test_cases:
        rule_score = symmetrical_triangle_score(
            high_neg_pct_slope=high_neg_pct_slope,
            low_pct_slope=low_pct_slope,
            slope_balance=slope_balance,
            gap_pct_now=gap_pct_now,
            pivot_count=pivot_count,
            touch_balance_ratio=touch_balance_ratio,
        )
        pred_score = learned_confidence(
            high_neg_pct_slope=high_neg_pct_slope,
            low_pct_slope=low_pct_slope,
            slope_balance=slope_balance,
            gap_pct_now=gap_pct_now,
            pivot_count=pivot_count,
            touch_balance_ratio=touch_balance_ratio,
            model=model,
        )
        delta = pred_score - rule_score
        print(f"{desc:<18} {rule_score:>10.2f} {pred_score:>10.2f} {delta:>10.2f}")