import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pkl_files")
MODEL_PATH = os.path.join(MODEL_DIR, "cup_and_handle_model.pkl")

FEATURES = [
    "rim_similarity_score",
    "cup_depth_ratio",
    "handle_shallow_score",
    "handle_len_score",
    "cup_roundness_score",
    "handle_slope_score",
]


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def cup_and_handle_score(
    rim_similarity_score: float,
    cup_depth_ratio: float,
    handle_shallow_score: float,
    handle_len_score: float,
    cup_roundness_score: float,
    handle_slope_score: float,
) -> float:
    """
    Strict rule-based quality score for a textbook-style Cup and Handle.
    Output: [0, 100]
    """

    if rim_similarity_score >= 0.95:
        score_rims = 100
    elif rim_similarity_score >= 0.80:
        score_rims = 80
    elif rim_similarity_score >= 0.60:
        score_rims = 55
    elif rim_similarity_score >= 0.40:
        score_rims = 25
    else:
        score_rims = 0

    if cup_depth_ratio >= 0.12:
        score_depth = 100
    elif cup_depth_ratio >= 0.08:
        score_depth = 80
    elif cup_depth_ratio >= 0.05:
        score_depth = 55
    elif cup_depth_ratio >= 0.03:
        score_depth = 25
    else:
        score_depth = 0

    if handle_shallow_score >= 0.95:
        score_handle = 100
    elif handle_shallow_score >= 0.80:
        score_handle = 80
    elif handle_shallow_score >= 0.60:
        score_handle = 55
    elif handle_shallow_score >= 0.40:
        score_handle = 25
    else:
        score_handle = 0

    if handle_len_score >= 0.95:
        score_len = 100
    elif handle_len_score >= 0.80:
        score_len = 80
    elif handle_len_score >= 0.60:
        score_len = 55
    elif handle_len_score >= 0.40:
        score_len = 25
    else:
        score_len = 0

    if cup_roundness_score >= 0.95:
        score_round = 100
    elif cup_roundness_score >= 0.80:
        score_round = 80
    elif cup_roundness_score >= 0.60:
        score_round = 55
    elif cup_roundness_score >= 0.40:
        score_round = 25
    else:
        score_round = 0

    if handle_slope_score >= 0.95:
        score_slope = 100
    elif handle_slope_score >= 0.80:
        score_slope = 80
    elif handle_slope_score >= 0.60:
        score_slope = 55
    elif handle_slope_score >= 0.40:
        score_slope = 25
    else:
        score_slope = 0

    overall = (
        0.24 * score_rims +
        0.22 * score_depth +
        0.18 * score_handle +
        0.12 * score_len +
        0.14 * score_round +
        0.10 * score_slope
    )

    return clamp_0_100(overall)


def _sample(region: str):
    if region == "ideal":
        rim_similarity_score = np.random.uniform(0.88, 1.00)
        cup_depth_ratio = np.random.uniform(0.08, 0.18)
        handle_shallow_score = np.random.uniform(0.85, 1.00)
        handle_len_score = np.random.uniform(0.85, 1.00)
        cup_roundness_score = np.random.uniform(0.85, 1.00)
        handle_slope_score = np.random.uniform(0.85, 1.00)

    elif region == "good":
        rim_similarity_score = np.random.uniform(0.70, 0.92)
        cup_depth_ratio = np.random.uniform(0.05, 0.12)
        handle_shallow_score = np.random.uniform(0.70, 0.92)
        handle_len_score = np.random.uniform(0.70, 0.92)
        cup_roundness_score = np.random.uniform(0.70, 0.92)
        handle_slope_score = np.random.uniform(0.70, 0.92)

    elif region == "borderline":
        rim_similarity_score = np.random.uniform(0.45, 0.75)
        cup_depth_ratio = np.random.uniform(0.03, 0.08)
        handle_shallow_score = np.random.uniform(0.45, 0.75)
        handle_len_score = np.random.uniform(0.45, 0.75)
        cup_roundness_score = np.random.uniform(0.45, 0.75)
        handle_slope_score = np.random.uniform(0.45, 0.75)

    elif region == "bad_rims":
        rim_similarity_score = np.random.uniform(0.00, 0.35)
        cup_depth_ratio = np.random.uniform(0.05, 0.14)
        handle_shallow_score = np.random.uniform(0.60, 0.95)
        handle_len_score = np.random.uniform(0.60, 0.95)
        cup_roundness_score = np.random.uniform(0.60, 0.95)
        handle_slope_score = np.random.uniform(0.60, 0.95)

    elif region == "shallow_cup":
        rim_similarity_score = np.random.uniform(0.60, 0.95)
        cup_depth_ratio = np.random.uniform(0.005, 0.035)
        handle_shallow_score = np.random.uniform(0.60, 0.95)
        handle_len_score = np.random.uniform(0.60, 0.95)
        cup_roundness_score = np.random.uniform(0.60, 0.95)
        handle_slope_score = np.random.uniform(0.60, 0.95)

    elif region == "deep_handle":
        rim_similarity_score = np.random.uniform(0.60, 0.95)
        cup_depth_ratio = np.random.uniform(0.05, 0.14)
        handle_shallow_score = np.random.uniform(0.00, 0.35)
        handle_len_score = np.random.uniform(0.60, 0.95)
        cup_roundness_score = np.random.uniform(0.60, 0.95)
        handle_slope_score = np.random.uniform(0.60, 0.95)

    elif region == "long_handle":
        rim_similarity_score = np.random.uniform(0.60, 0.95)
        cup_depth_ratio = np.random.uniform(0.05, 0.14)
        handle_shallow_score = np.random.uniform(0.60, 0.95)
        handle_len_score = np.random.uniform(0.00, 0.35)
        cup_roundness_score = np.random.uniform(0.60, 0.95)
        handle_slope_score = np.random.uniform(0.60, 0.95)

    elif region == "flat_bottom_issue":
        rim_similarity_score = np.random.uniform(0.60, 0.95)
        cup_depth_ratio = np.random.uniform(0.05, 0.14)
        handle_shallow_score = np.random.uniform(0.60, 0.95)
        handle_len_score = np.random.uniform(0.60, 0.95)
        cup_roundness_score = np.random.uniform(0.00, 0.35)
        handle_slope_score = np.random.uniform(0.60, 0.95)

    elif region == "bad_handle_slope":
        rim_similarity_score = np.random.uniform(0.60, 0.95)
        cup_depth_ratio = np.random.uniform(0.05, 0.14)
        handle_shallow_score = np.random.uniform(0.60, 0.95)
        handle_len_score = np.random.uniform(0.60, 0.95)
        cup_roundness_score = np.random.uniform(0.60, 0.95)
        handle_slope_score = np.random.uniform(0.00, 0.35)

    elif region == "random":
        rim_similarity_score = np.random.uniform(0.00, 1.00)
        cup_depth_ratio = np.random.uniform(0.00, 0.25)
        handle_shallow_score = np.random.uniform(0.00, 1.00)
        handle_len_score = np.random.uniform(0.00, 1.00)
        cup_roundness_score = np.random.uniform(0.00, 1.00)
        handle_slope_score = np.random.uniform(0.00, 1.00)

    else:  # outside
        rim_similarity_score = np.random.uniform(0.00, 0.25)
        cup_depth_ratio = np.random.uniform(0.00, 0.03)
        handle_shallow_score = np.random.uniform(0.00, 0.25)
        handle_len_score = np.random.uniform(0.00, 0.25)
        cup_roundness_score = np.random.uniform(0.00, 0.25)
        handle_slope_score = np.random.uniform(0.00, 0.25)

    return (
        float(rim_similarity_score),
        float(cup_depth_ratio),
        float(handle_shallow_score),
        float(handle_len_score),
        float(cup_roundness_score),
        float(handle_slope_score),
    )


REGION_COUNTS = {
    "ideal": 1400,
    "good": 1200,
    "borderline": 1000,
    "bad_rims": 900,
    "shallow_cup": 900,
    "deep_handle": 800,
    "long_handle": 800,
    "flat_bottom_issue": 700,
    "bad_handle_slope": 700,
    "random": 700,
    "outside": 300,
}


def build_dataset() -> pd.DataFrame:
    rows = []

    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            (
                rim_similarity_score,
                cup_depth_ratio,
                handle_shallow_score,
                handle_len_score,
                cup_roundness_score,
                handle_slope_score,
            ) = _sample(region)

            base_score = cup_and_handle_score(
                rim_similarity_score=rim_similarity_score,
                cup_depth_ratio=cup_depth_ratio,
                handle_shallow_score=handle_shallow_score,
                handle_len_score=handle_len_score,
                cup_roundness_score=cup_roundness_score,
                handle_slope_score=handle_slope_score,
            )

            noise = np.random.normal(0, 2.0)
            quality = clamp_0_100(base_score + noise)

            rows.append({
                "rim_similarity_score": rim_similarity_score,
                "cup_depth_ratio": cup_depth_ratio,
                "handle_shallow_score": handle_shallow_score,
                "handle_len_score": handle_len_score,
                "cup_roundness_score": cup_roundness_score,
                "handle_slope_score": handle_slope_score,
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
            "Cup and Handle regression model not found. Run "
            "`python -m stock_analysis.confidence_models.chart_patterns.cup_and_handle_model` "
            "to train and save it first."
        )
    model = joblib.load(MODEL_PATH)
    return model


def learned_confidence(
    rim_similarity_score: float,
    cup_depth_ratio: float,
    handle_shallow_score: float,
    handle_len_score: float,
    cup_roundness_score: float,
    handle_slope_score: float,
    model: RandomForestRegressor,
) -> float:
    x = np.array([[
        rim_similarity_score,
        cup_depth_ratio,
        handle_shallow_score,
        handle_len_score,
        cup_roundness_score,
        handle_slope_score,
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
        ("Ideal",            0.95, 0.12, 0.95, 0.95, 0.95, 0.95),
        ("Good",             0.82, 0.08, 0.82, 0.80, 0.82, 0.82),
        ("Borderline",       0.60, 0.05, 0.60, 0.60, 0.60, 0.60),
        ("Bad rims",         0.20, 0.10, 0.80, 0.80, 0.80, 0.80),
        ("Shallow cup",      0.80, 0.02, 0.80, 0.80, 0.80, 0.80),
        ("Deep handle",      0.80, 0.10, 0.20, 0.80, 0.80, 0.80),
        ("Long handle",      0.80, 0.10, 0.80, 0.20, 0.80, 0.80),
        ("Flat bottom",      0.80, 0.10, 0.80, 0.80, 0.20, 0.80),
        ("Bad slope",        0.80, 0.10, 0.80, 0.80, 0.80, 0.20),
        ("User example",     0.72, 0.07, 0.74, 0.68, 0.70, 0.66),
    ]

    print(f"\n{'Pattern':<16} {'RuleScore':>10} {'Predicted':>10} {'Delta':>10}")
    print("-" * 54)

    for desc, rim_similarity_score, cup_depth_ratio, handle_shallow_score, handle_len_score, cup_roundness_score, handle_slope_score in test_cases:
        rule_score = cup_and_handle_score(
            rim_similarity_score=rim_similarity_score,
            cup_depth_ratio=cup_depth_ratio,
            handle_shallow_score=handle_shallow_score,
            handle_len_score=handle_len_score,
            cup_roundness_score=cup_roundness_score,
            handle_slope_score=handle_slope_score,
        )
        pred_score = learned_confidence(
            rim_similarity_score=rim_similarity_score,
            cup_depth_ratio=cup_depth_ratio,
            handle_shallow_score=handle_shallow_score,
            handle_len_score=handle_len_score,
            cup_roundness_score=cup_roundness_score,
            handle_slope_score=handle_slope_score,
            model=model,
        )
        delta = pred_score - rule_score
        print(f"{desc:<16} {rule_score:>10.2f} {pred_score:>10.2f} {delta:>10.2f}")