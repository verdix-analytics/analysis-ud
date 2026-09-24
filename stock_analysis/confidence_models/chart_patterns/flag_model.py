import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pkl_files")
MODEL_PATH = os.path.join(MODEL_DIR, "flag_model.pkl")

FEATURES = [
    "pole_move_pct",
    "range_ratio",
    "slope_strength_ratio",
    "avg_range_ratio",
    "flag_len_ratio",
    "flag_direction_score",
]


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def flag_score(
    pole_move_pct: float,
    range_ratio: float,
    slope_strength_ratio: float,
    avg_range_ratio: float,
    flag_len_ratio: float,
    flag_direction_score: float,
) -> float:
    """
    Strict rule-based quality score for a textbook-style Flag pattern.
    Output: [0, 100]

    flag_direction_score:
        1.0   = ideal counter-trend / flat consolidation
        0.5   = acceptable weak same-direction consolidation
        0.0   = bad flag direction
    """

    # 1) Pole strength
    if pole_move_pct >= 0.040:
        score_pole = 100
    elif pole_move_pct >= 0.030:
        score_pole = 80
    elif pole_move_pct >= 0.020:
        score_pole = 55
    elif pole_move_pct >= 0.015:
        score_pole = 25
    else:
        score_pole = 0

    # 2) Flag range should be much smaller than pole range
    if range_ratio <= 0.35:
        score_range = 100
    elif range_ratio <= 0.50:
        score_range = 80
    elif range_ratio <= 0.70:
        score_range = 55
    elif range_ratio <= 0.90:
        score_range = 25
    else:
        score_range = 0

    # 3) Flag slope should be weaker than pole slope
    if slope_strength_ratio <= 0.20:
        score_slope_strength = 100
    elif slope_strength_ratio <= 0.35:
        score_slope_strength = 80
    elif slope_strength_ratio <= 0.50:
        score_slope_strength = 55
    elif slope_strength_ratio <= 0.70:
        score_slope_strength = 25
    else:
        score_slope_strength = 0

    # 4) Candle compression
    if avg_range_ratio <= 0.60:
        score_compression = 100
    elif avg_range_ratio <= 0.80:
        score_compression = 80
    elif avg_range_ratio <= 1.00:
        score_compression = 55
    elif avg_range_ratio <= 1.20:
        score_compression = 25
    else:
        score_compression = 0

    # 5) Flag duration relative to pole
    if 0.40 <= flag_len_ratio <= 1.20:
        score_len = 100
    elif 0.25 <= flag_len_ratio <= 1.50:
        score_len = 80
    elif 0.15 <= flag_len_ratio <= 2.00:
        score_len = 55
    else:
        score_len = 25

    # 6) Flag direction quality
    if flag_direction_score >= 0.95:
        score_direction = 100
    elif flag_direction_score >= 0.70:
        score_direction = 80
    elif flag_direction_score >= 0.45:
        score_direction = 55
    elif flag_direction_score >= 0.20:
        score_direction = 25
    else:
        score_direction = 0

    overall = (
        0.28 * score_pole +
        0.22 * score_range +
        0.18 * score_slope_strength +
        0.12 * score_compression +
        0.08 * score_len +
        0.12 * score_direction
    )

    return clamp_0_100(overall)


def _sample(region: str):
    if region == "ideal":
        pole_move_pct = np.random.uniform(0.035, 0.080)
        range_ratio = np.random.uniform(0.15, 0.40)
        slope_strength_ratio = np.random.uniform(0.05, 0.25)
        avg_range_ratio = np.random.uniform(0.35, 0.70)
        flag_len_ratio = np.random.uniform(0.40, 1.10)
        flag_direction_score = np.random.uniform(0.90, 1.00)

    elif region == "good":
        pole_move_pct = np.random.uniform(0.025, 0.050)
        range_ratio = np.random.uniform(0.30, 0.55)
        slope_strength_ratio = np.random.uniform(0.15, 0.40)
        avg_range_ratio = np.random.uniform(0.55, 0.90)
        flag_len_ratio = np.random.uniform(0.30, 1.40)
        flag_direction_score = np.random.uniform(0.70, 0.95)

    elif region == "borderline":
        pole_move_pct = np.random.uniform(0.015, 0.030)
        range_ratio = np.random.uniform(0.50, 0.80)
        slope_strength_ratio = np.random.uniform(0.35, 0.60)
        avg_range_ratio = np.random.uniform(0.80, 1.10)
        flag_len_ratio = np.random.uniform(0.20, 1.80)
        flag_direction_score = np.random.uniform(0.40, 0.75)

    elif region == "weak_pole":
        pole_move_pct = np.random.uniform(0.005, 0.018)
        range_ratio = np.random.uniform(0.20, 0.60)
        slope_strength_ratio = np.random.uniform(0.10, 0.45)
        avg_range_ratio = np.random.uniform(0.50, 0.90)
        flag_len_ratio = np.random.uniform(0.30, 1.30)
        flag_direction_score = np.random.uniform(0.70, 1.00)

    elif region == "wide_flag":
        pole_move_pct = np.random.uniform(0.020, 0.050)
        range_ratio = np.random.uniform(0.80, 1.40)
        slope_strength_ratio = np.random.uniform(0.15, 0.50)
        avg_range_ratio = np.random.uniform(0.80, 1.30)
        flag_len_ratio = np.random.uniform(0.30, 1.50)
        flag_direction_score = np.random.uniform(0.60, 1.00)

    elif region == "strong_flag_slope":
        pole_move_pct = np.random.uniform(0.020, 0.050)
        range_ratio = np.random.uniform(0.25, 0.60)
        slope_strength_ratio = np.random.uniform(0.70, 1.40)
        avg_range_ratio = np.random.uniform(0.60, 1.00)
        flag_len_ratio = np.random.uniform(0.30, 1.50)
        flag_direction_score = np.random.uniform(0.10, 0.50)

    elif region == "no_compression":
        pole_move_pct = np.random.uniform(0.020, 0.050)
        range_ratio = np.random.uniform(0.30, 0.70)
        slope_strength_ratio = np.random.uniform(0.15, 0.50)
        avg_range_ratio = np.random.uniform(1.00, 1.80)
        flag_len_ratio = np.random.uniform(0.30, 1.40)
        flag_direction_score = np.random.uniform(0.50, 1.00)

    elif region == "bad_duration":
        pole_move_pct = np.random.uniform(0.020, 0.050)
        range_ratio = np.random.uniform(0.25, 0.60)
        slope_strength_ratio = np.random.uniform(0.15, 0.50)
        avg_range_ratio = np.random.uniform(0.50, 0.95)
        if np.random.rand() < 0.5:
            flag_len_ratio = np.random.uniform(0.05, 0.15)
        else:
            flag_len_ratio = np.random.uniform(2.00, 4.00)
        flag_direction_score = np.random.uniform(0.60, 1.00)

    elif region == "same_direction_flag":
        pole_move_pct = np.random.uniform(0.020, 0.050)
        range_ratio = np.random.uniform(0.25, 0.60)
        slope_strength_ratio = np.random.uniform(0.15, 0.60)
        avg_range_ratio = np.random.uniform(0.50, 0.95)
        flag_len_ratio = np.random.uniform(0.30, 1.40)
        flag_direction_score = np.random.uniform(0.00, 0.30)

    elif region == "random":
        pole_move_pct = np.random.uniform(0.000, 0.100)
        range_ratio = np.random.uniform(0.10, 1.80)
        slope_strength_ratio = np.random.uniform(0.00, 2.00)
        avg_range_ratio = np.random.uniform(0.20, 2.00)
        flag_len_ratio = np.random.uniform(0.05, 4.00)
        flag_direction_score = np.random.uniform(0.00, 1.00)

    else:  # outside
        pole_move_pct = np.random.uniform(0.000, 0.010)
        range_ratio = np.random.uniform(1.00, 2.50)
        slope_strength_ratio = np.random.uniform(1.00, 3.00)
        avg_range_ratio = np.random.uniform(1.20, 2.50)
        flag_len_ratio = np.random.uniform(0.05, 5.00)
        flag_direction_score = np.random.uniform(0.00, 0.20)

    return (
        float(pole_move_pct),
        float(range_ratio),
        float(slope_strength_ratio),
        float(avg_range_ratio),
        float(flag_len_ratio),
        float(flag_direction_score),
    )


REGION_COUNTS = {
    "ideal": 1400,
    "good": 1200,
    "borderline": 1000,
    "weak_pole": 900,
    "wide_flag": 900,
    "strong_flag_slope": 800,
    "no_compression": 700,
    "bad_duration": 700,
    "same_direction_flag": 700,
    "random": 700,
    "outside": 300,
}


def build_dataset() -> pd.DataFrame:
    rows = []

    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            (
                pole_move_pct,
                range_ratio,
                slope_strength_ratio,
                avg_range_ratio,
                flag_len_ratio,
                flag_direction_score,
            ) = _sample(region)

            base_score = flag_score(
                pole_move_pct=pole_move_pct,
                range_ratio=range_ratio,
                slope_strength_ratio=slope_strength_ratio,
                avg_range_ratio=avg_range_ratio,
                flag_len_ratio=flag_len_ratio,
                flag_direction_score=flag_direction_score,
            )

            noise = np.random.normal(0, 2.0)
            quality = clamp_0_100(base_score + noise)

            rows.append({
                "pole_move_pct": pole_move_pct,
                "range_ratio": range_ratio,
                "slope_strength_ratio": slope_strength_ratio,
                "avg_range_ratio": avg_range_ratio,
                "flag_len_ratio": flag_len_ratio,
                "flag_direction_score": flag_direction_score,
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
            "Flag regression model not found. Run "
            "`python -m stock_analysis.confidence_models.chart_patterns.flag_model` "
            "to train and save it first."
        )
    model = joblib.load(MODEL_PATH)
    return model


def learned_confidence(
    pole_move_pct: float,
    range_ratio: float,
    slope_strength_ratio: float,
    avg_range_ratio: float,
    flag_len_ratio: float,
    flag_direction_score: float,
    model: RandomForestRegressor,
) -> float:
    x = np.array([[
        pole_move_pct,
        range_ratio,
        slope_strength_ratio,
        avg_range_ratio,
        flag_len_ratio,
        flag_direction_score,
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
        ("Ideal",              0.050, 0.30, 0.15, 0.55, 0.70, 1.00),
        ("Good",               0.030, 0.45, 0.30, 0.75, 0.90, 0.85),
        ("Borderline",         0.018, 0.70, 0.50, 0.95, 1.40, 0.55),
        ("Weak pole",          0.010, 0.35, 0.25, 0.60, 0.70, 0.95),
        ("Wide flag",          0.030, 1.10, 0.25, 0.90, 0.80, 0.85),
        ("Strong flag slope",  0.030, 0.45, 1.10, 0.80, 0.90, 0.20),
        ("No compression",     0.030, 0.45, 0.30, 1.50, 0.90, 0.80),
        ("Bad duration",       0.030, 0.45, 0.30, 0.80, 2.80, 0.80),
        ("Same direction",     0.030, 0.45, 0.30, 0.80, 0.90, 0.10),
        ("User example",       0.024, 0.52, 0.42, 0.88, 1.10, 0.70),
    ]

    print(f"\n{'Pattern':<18} {'RuleScore':>10} {'Predicted':>10} {'Delta':>10}")
    print("-" * 56)

    for desc, pole_move_pct, range_ratio, slope_strength_ratio, avg_range_ratio, flag_len_ratio, flag_direction_score in test_cases:
        rule_score = flag_score(
            pole_move_pct=pole_move_pct,
            range_ratio=range_ratio,
            slope_strength_ratio=slope_strength_ratio,
            avg_range_ratio=avg_range_ratio,
            flag_len_ratio=flag_len_ratio,
            flag_direction_score=flag_direction_score,
        )
        pred_score = learned_confidence(
            pole_move_pct=pole_move_pct,
            range_ratio=range_ratio,
            slope_strength_ratio=slope_strength_ratio,
            avg_range_ratio=avg_range_ratio,
            flag_len_ratio=flag_len_ratio,
            flag_direction_score=flag_direction_score,
            model=model,
        )
        delta = pred_score - rule_score
        print(f"{desc:<18} {rule_score:>10.2f} {pred_score:>10.2f} {delta:>10.2f}")