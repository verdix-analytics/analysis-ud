import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pkl_files")
MODEL_PATH = os.path.join(MODEL_DIR, "ascending_channel_model.pkl")

FEATURES = [
    "upper_pct_slope",
    "lower_pct_slope",
    "slope_parallel_score",
    "touch_balance_ratio",
    "width_stability_score",
    "touch_richness_score",
]


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def ascending_channel_score(
    upper_pct_slope: float,
    lower_pct_slope: float,
    slope_parallel_score: float,
    touch_balance_ratio: float,
    width_stability_score: float,
    touch_richness_score: float,
) -> float:
    """
    Strict rule-based quality score for a textbook-style Ascending Channel.
    Output: [0, 100]
    """

    # 1) Both lines should clearly rise
    avg_rise = (upper_pct_slope + lower_pct_slope) / 2

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

    # 2) Parallelism
    if slope_parallel_score >= 0.95:
        score_parallel = 100
    elif slope_parallel_score >= 0.80:
        score_parallel = 80
    elif slope_parallel_score >= 0.60:
        score_parallel = 55
    elif slope_parallel_score >= 0.40:
        score_parallel = 25
    else:
        score_parallel = 0

    # 3) Touch balance
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

    # 4) Width stability
    if width_stability_score >= 0.95:
        score_width = 100
    elif width_stability_score >= 0.80:
        score_width = 80
    elif width_stability_score >= 0.60:
        score_width = 55
    elif width_stability_score >= 0.40:
        score_width = 25
    else:
        score_width = 0

    # 5) Touch richness
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

    overall = (
        0.30 * score_rise +
        0.24 * score_parallel +
        0.16 * score_balance +
        0.16 * score_width +
        0.14 * score_touches
    )

    return clamp_0_100(overall)


def _sample(region: str):
    if region == "ideal":
        upper_pct_slope = np.random.uniform(0.0008, 0.0020)
        lower_pct_slope = np.random.uniform(0.0008, 0.0020)
        slope_parallel_score = np.random.uniform(0.88, 1.00)
        touch_balance_ratio = np.random.uniform(0.85, 1.00)
        width_stability_score = np.random.uniform(0.85, 1.00)
        touch_richness_score = np.random.uniform(0.85, 1.00)

    elif region == "good":
        upper_pct_slope = np.random.uniform(0.0004, 0.0012)
        lower_pct_slope = np.random.uniform(0.0004, 0.0012)
        slope_parallel_score = np.random.uniform(0.70, 0.92)
        touch_balance_ratio = np.random.uniform(0.70, 0.92)
        width_stability_score = np.random.uniform(0.70, 0.92)
        touch_richness_score = np.random.uniform(0.70, 0.92)

    elif region == "borderline":
        upper_pct_slope = np.random.uniform(0.00015, 0.0006)
        lower_pct_slope = np.random.uniform(0.00015, 0.0006)
        slope_parallel_score = np.random.uniform(0.45, 0.75)
        touch_balance_ratio = np.random.uniform(0.45, 0.75)
        width_stability_score = np.random.uniform(0.45, 0.75)
        touch_richness_score = np.random.uniform(0.45, 0.75)

    elif region == "weak_rise":
        upper_pct_slope = np.random.uniform(0.00000, 0.00018)
        lower_pct_slope = np.random.uniform(0.00000, 0.00018)
        slope_parallel_score = np.random.uniform(0.65, 0.95)
        touch_balance_ratio = np.random.uniform(0.65, 0.95)
        width_stability_score = np.random.uniform(0.65, 0.95)
        touch_richness_score = np.random.uniform(0.65, 0.95)

    elif region == "non_parallel":
        upper_pct_slope = np.random.uniform(0.0003, 0.0014)
        lower_pct_slope = np.random.uniform(0.0003, 0.0014)
        slope_parallel_score = np.random.uniform(0.00, 0.35)
        touch_balance_ratio = np.random.uniform(0.60, 0.90)
        width_stability_score = np.random.uniform(0.50, 0.85)
        touch_richness_score = np.random.uniform(0.60, 0.90)

    elif region == "imbalanced_touches":
        upper_pct_slope = np.random.uniform(0.0003, 0.0014)
        lower_pct_slope = np.random.uniform(0.0003, 0.0014)
        slope_parallel_score = np.random.uniform(0.65, 0.95)
        touch_balance_ratio = np.random.uniform(0.00, 0.35)
        width_stability_score = np.random.uniform(0.65, 0.95)
        touch_richness_score = np.random.uniform(0.55, 0.90)

    elif region == "unstable_width":
        upper_pct_slope = np.random.uniform(0.0003, 0.0014)
        lower_pct_slope = np.random.uniform(0.0003, 0.0014)
        slope_parallel_score = np.random.uniform(0.55, 0.90)
        touch_balance_ratio = np.random.uniform(0.55, 0.90)
        width_stability_score = np.random.uniform(0.00, 0.35)
        touch_richness_score = np.random.uniform(0.55, 0.90)

    elif region == "few_touches":
        upper_pct_slope = np.random.uniform(0.0003, 0.0014)
        lower_pct_slope = np.random.uniform(0.0003, 0.0014)
        slope_parallel_score = np.random.uniform(0.60, 0.90)
        touch_balance_ratio = np.random.uniform(0.55, 0.90)
        width_stability_score = np.random.uniform(0.55, 0.90)
        touch_richness_score = np.random.uniform(0.00, 0.35)

    elif region == "random":
        upper_pct_slope = np.random.uniform(-0.0010, 0.0030)
        lower_pct_slope = np.random.uniform(-0.0010, 0.0030)
        slope_parallel_score = np.random.uniform(0.00, 1.00)
        touch_balance_ratio = np.random.uniform(0.00, 1.00)
        width_stability_score = np.random.uniform(0.00, 1.00)
        touch_richness_score = np.random.uniform(0.00, 1.00)

    else:  # outside
        upper_pct_slope = np.random.uniform(-0.0020, 0.00005)
        lower_pct_slope = np.random.uniform(-0.0020, 0.00005)
        slope_parallel_score = np.random.uniform(0.00, 0.25)
        touch_balance_ratio = np.random.uniform(0.00, 0.25)
        width_stability_score = np.random.uniform(0.00, 0.25)
        touch_richness_score = np.random.uniform(0.00, 0.25)

    return (
        float(upper_pct_slope),
        float(lower_pct_slope),
        float(slope_parallel_score),
        float(touch_balance_ratio),
        float(width_stability_score),
        float(touch_richness_score),
    )


REGION_COUNTS = {
    "ideal": 1400,
    "good": 1200,
    "borderline": 1000,
    "weak_rise": 900,
    "non_parallel": 900,
    "imbalanced_touches": 800,
    "unstable_width": 800,
    "few_touches": 700,
    "random": 700,
    "outside": 300,
}


def build_dataset() -> pd.DataFrame:
    rows = []

    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            (
                upper_pct_slope,
                lower_pct_slope,
                slope_parallel_score,
                touch_balance_ratio,
                width_stability_score,
                touch_richness_score,
            ) = _sample(region)

            base_score = ascending_channel_score(
                upper_pct_slope=upper_pct_slope,
                lower_pct_slope=lower_pct_slope,
                slope_parallel_score=slope_parallel_score,
                touch_balance_ratio=touch_balance_ratio,
                width_stability_score=width_stability_score,
                touch_richness_score=touch_richness_score,
            )

            noise = np.random.normal(0, 2.0)
            quality = clamp_0_100(base_score + noise)

            rows.append({
                "upper_pct_slope": upper_pct_slope,
                "lower_pct_slope": lower_pct_slope,
                "slope_parallel_score": slope_parallel_score,
                "touch_balance_ratio": touch_balance_ratio,
                "width_stability_score": width_stability_score,
                "touch_richness_score": touch_richness_score,
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
            "Ascending Channel regression model not found. Run "
            "`python -m stock_analysis.confidence_models.chart_patterns.ascending_channel_model` "
            "to train and save it first."
        )
    model = joblib.load(MODEL_PATH)
    return model


def learned_confidence(
    upper_pct_slope: float,
    lower_pct_slope: float,
    slope_parallel_score: float,
    touch_balance_ratio: float,
    width_stability_score: float,
    touch_richness_score: float,
    model: RandomForestRegressor,
) -> float:
    x = np.array([[
        upper_pct_slope,
        lower_pct_slope,
        slope_parallel_score,
        touch_balance_ratio,
        width_stability_score,
        touch_richness_score,
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
        ("Ideal",            0.0012, 0.0011, 0.95, 0.95, 0.95, 0.95),
        ("Good",             0.0007, 0.00065, 0.82, 0.80, 0.82, 0.80),
        ("Borderline",       0.00025, 0.00022, 0.60, 0.62, 0.60, 0.58),
        ("Weak rise",        0.00008, 0.00006, 0.85, 0.85, 0.85, 0.85),
        ("Non-parallel",     0.0012, 0.0004, 0.20, 0.75, 0.65, 0.75),
        ("Bad touches",      0.0008, 0.00075, 0.82, 0.20, 0.82, 0.40),
        ("Unstable width",   0.0008, 0.00075, 0.80, 0.80, 0.20, 0.75),
        ("Few touches",      0.0008, 0.00075, 0.80, 0.80, 0.80, 0.20),
        ("User example",     0.00052, 0.00049, 0.72, 0.68, 0.74, 0.66),
    ]

    print(f"\n{'Pattern':<16} {'RuleScore':>10} {'Predicted':>10} {'Delta':>10}")
    print("-" * 54)

    for desc, upper_pct_slope, lower_pct_slope, slope_parallel_score, touch_balance_ratio, width_stability_score, touch_richness_score in test_cases:
        rule_score = ascending_channel_score(
            upper_pct_slope=upper_pct_slope,
            lower_pct_slope=lower_pct_slope,
            slope_parallel_score=slope_parallel_score,
            touch_balance_ratio=touch_balance_ratio,
            width_stability_score=width_stability_score,
            touch_richness_score=touch_richness_score,
        )
        pred_score = learned_confidence(
            upper_pct_slope=upper_pct_slope,
            lower_pct_slope=lower_pct_slope,
            slope_parallel_score=slope_parallel_score,
            touch_balance_ratio=touch_balance_ratio,
            width_stability_score=width_stability_score,
            touch_richness_score=touch_richness_score,
            model=model,
        )
        delta = pred_score - rule_score
        print(f"{desc:<16} {rule_score:>10.2f} {pred_score:>10.2f} {delta:>10.2f}")