import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pkl_files")
MODEL_PATH = os.path.join(MODEL_DIR, "megaphone_model.pkl")

FEATURES = [
    "high_rise_score",
    "low_drop_score",
    "expansion_score",
    "pivot_richness_score",
    "alternation_score",
    "balance_score",
]


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def megaphone_score(
    high_rise_score: float,
    low_drop_score: float,
    expansion_score: float,
    pivot_richness_score: float,
    alternation_score: float,
    balance_score: float,
) -> float:
    """
    Strict rule-based quality score for a textbook-style Megaphone pattern.
    Output: [0, 100]
    """

    if high_rise_score >= 0.95:
        score_high = 100
    elif high_rise_score >= 0.80:
        score_high = 80
    elif high_rise_score >= 0.60:
        score_high = 55
    elif high_rise_score >= 0.40:
        score_high = 25
    else:
        score_high = 0

    if low_drop_score >= 0.95:
        score_low = 100
    elif low_drop_score >= 0.80:
        score_low = 80
    elif low_drop_score >= 0.60:
        score_low = 55
    elif low_drop_score >= 0.40:
        score_low = 25
    else:
        score_low = 0

    if expansion_score >= 0.95:
        score_expand = 100
    elif expansion_score >= 0.80:
        score_expand = 80
    elif expansion_score >= 0.60:
        score_expand = 55
    elif expansion_score >= 0.40:
        score_expand = 25
    else:
        score_expand = 0

    if pivot_richness_score >= 0.95:
        score_pivots = 100
    elif pivot_richness_score >= 0.80:
        score_pivots = 80
    elif pivot_richness_score >= 0.60:
        score_pivots = 55
    elif pivot_richness_score >= 0.40:
        score_pivots = 25
    else:
        score_pivots = 0

    if alternation_score >= 0.95:
        score_alt = 100
    elif alternation_score >= 0.80:
        score_alt = 80
    elif alternation_score >= 0.60:
        score_alt = 55
    elif alternation_score >= 0.40:
        score_alt = 25
    else:
        score_alt = 0

    if balance_score >= 0.95:
        score_balance = 100
    elif balance_score >= 0.80:
        score_balance = 80
    elif balance_score >= 0.60:
        score_balance = 55
    elif balance_score >= 0.40:
        score_balance = 25
    else:
        score_balance = 0

    overall = (
        0.22 * score_high +
        0.22 * score_low +
        0.24 * score_expand +
        0.12 * score_pivots +
        0.10 * score_alt +
        0.10 * score_balance
    )

    return clamp_0_100(overall)


def _sample(region: str):
    if region == "ideal":
        high_rise_score = np.random.uniform(0.88, 1.00)
        low_drop_score = np.random.uniform(0.88, 1.00)
        expansion_score = np.random.uniform(0.88, 1.00)
        pivot_richness_score = np.random.uniform(0.85, 1.00)
        alternation_score = np.random.uniform(0.85, 1.00)
        balance_score = np.random.uniform(0.85, 1.00)

    elif region == "good":
        high_rise_score = np.random.uniform(0.70, 0.92)
        low_drop_score = np.random.uniform(0.70, 0.92)
        expansion_score = np.random.uniform(0.70, 0.92)
        pivot_richness_score = np.random.uniform(0.70, 0.92)
        alternation_score = np.random.uniform(0.70, 0.92)
        balance_score = np.random.uniform(0.70, 0.92)

    elif region == "borderline":
        high_rise_score = np.random.uniform(0.45, 0.75)
        low_drop_score = np.random.uniform(0.45, 0.75)
        expansion_score = np.random.uniform(0.45, 0.75)
        pivot_richness_score = np.random.uniform(0.45, 0.75)
        alternation_score = np.random.uniform(0.45, 0.75)
        balance_score = np.random.uniform(0.45, 0.75)

    elif region == "weak_high_rise":
        high_rise_score = np.random.uniform(0.00, 0.35)
        low_drop_score = np.random.uniform(0.60, 0.95)
        expansion_score = np.random.uniform(0.60, 0.95)
        pivot_richness_score = np.random.uniform(0.60, 0.95)
        alternation_score = np.random.uniform(0.60, 0.95)
        balance_score = np.random.uniform(0.60, 0.95)

    elif region == "weak_low_drop":
        high_rise_score = np.random.uniform(0.60, 0.95)
        low_drop_score = np.random.uniform(0.00, 0.35)
        expansion_score = np.random.uniform(0.60, 0.95)
        pivot_richness_score = np.random.uniform(0.60, 0.95)
        alternation_score = np.random.uniform(0.60, 0.95)
        balance_score = np.random.uniform(0.60, 0.95)

    elif region == "no_expansion":
        high_rise_score = np.random.uniform(0.60, 0.95)
        low_drop_score = np.random.uniform(0.60, 0.95)
        expansion_score = np.random.uniform(0.00, 0.35)
        pivot_richness_score = np.random.uniform(0.60, 0.95)
        alternation_score = np.random.uniform(0.60, 0.95)
        balance_score = np.random.uniform(0.60, 0.95)

    elif region == "few_pivots":
        high_rise_score = np.random.uniform(0.60, 0.95)
        low_drop_score = np.random.uniform(0.60, 0.95)
        expansion_score = np.random.uniform(0.60, 0.95)
        pivot_richness_score = np.random.uniform(0.00, 0.35)
        alternation_score = np.random.uniform(0.60, 0.95)
        balance_score = np.random.uniform(0.60, 0.95)

    elif region == "bad_alternation":
        high_rise_score = np.random.uniform(0.60, 0.95)
        low_drop_score = np.random.uniform(0.60, 0.95)
        expansion_score = np.random.uniform(0.60, 0.95)
        pivot_richness_score = np.random.uniform(0.60, 0.95)
        alternation_score = np.random.uniform(0.00, 0.35)
        balance_score = np.random.uniform(0.60, 0.95)

    elif region == "imbalanced":
        high_rise_score = np.random.uniform(0.60, 0.95)
        low_drop_score = np.random.uniform(0.60, 0.95)
        expansion_score = np.random.uniform(0.60, 0.95)
        pivot_richness_score = np.random.uniform(0.60, 0.95)
        alternation_score = np.random.uniform(0.60, 0.95)
        balance_score = np.random.uniform(0.00, 0.35)

    elif region == "random":
        high_rise_score = np.random.uniform(0.00, 1.00)
        low_drop_score = np.random.uniform(0.00, 1.00)
        expansion_score = np.random.uniform(0.00, 1.00)
        pivot_richness_score = np.random.uniform(0.00, 1.00)
        alternation_score = np.random.uniform(0.00, 1.00)
        balance_score = np.random.uniform(0.00, 1.00)

    else:  # outside
        high_rise_score = np.random.uniform(0.00, 0.25)
        low_drop_score = np.random.uniform(0.00, 0.25)
        expansion_score = np.random.uniform(0.00, 0.25)
        pivot_richness_score = np.random.uniform(0.00, 0.25)
        alternation_score = np.random.uniform(0.00, 0.25)
        balance_score = np.random.uniform(0.00, 0.25)

    return (
        float(high_rise_score),
        float(low_drop_score),
        float(expansion_score),
        float(pivot_richness_score),
        float(alternation_score),
        float(balance_score),
    )


REGION_COUNTS = {
    "ideal": 1400,
    "good": 1200,
    "borderline": 1000,
    "weak_high_rise": 900,
    "weak_low_drop": 900,
    "no_expansion": 800,
    "few_pivots": 800,
    "bad_alternation": 700,
    "imbalanced": 700,
    "random": 700,
    "outside": 300,
}


def build_dataset() -> pd.DataFrame:
    rows = []

    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            (
                high_rise_score,
                low_drop_score,
                expansion_score,
                pivot_richness_score,
                alternation_score,
                balance_score,
            ) = _sample(region)

            base_score = megaphone_score(
                high_rise_score=high_rise_score,
                low_drop_score=low_drop_score,
                expansion_score=expansion_score,
                pivot_richness_score=pivot_richness_score,
                alternation_score=alternation_score,
                balance_score=balance_score,
            )

            noise = np.random.normal(0, 2.0)
            quality = clamp_0_100(base_score + noise)

            rows.append({
                "high_rise_score": high_rise_score,
                "low_drop_score": low_drop_score,
                "expansion_score": expansion_score,
                "pivot_richness_score": pivot_richness_score,
                "alternation_score": alternation_score,
                "balance_score": balance_score,
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
            "Megaphone regression model not found. Run "
            "`python -m stock_analysis.confidence_models.chart_patterns.megaphone_model` "
            "to train and save it first."
        )
    model = joblib.load(MODEL_PATH)
    return model


def learned_confidence(
    high_rise_score: float,
    low_drop_score: float,
    expansion_score: float,
    pivot_richness_score: float,
    alternation_score: float,
    balance_score: float,
    model: RandomForestRegressor,
) -> float:
    x = np.array([[
        high_rise_score,
        low_drop_score,
        expansion_score,
        pivot_richness_score,
        alternation_score,
        balance_score,
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
        ("Ideal",            0.95, 0.95, 0.95, 0.95, 0.95, 0.95),
        ("Good",             0.82, 0.82, 0.82, 0.80, 0.80, 0.80),
        ("Borderline",       0.60, 0.60, 0.60, 0.60, 0.60, 0.60),
        ("Weak high rise",   0.20, 0.80, 0.80, 0.80, 0.80, 0.80),
        ("Weak low drop",    0.80, 0.20, 0.80, 0.80, 0.80, 0.80),
        ("No expansion",     0.80, 0.80, 0.20, 0.80, 0.80, 0.80),
        ("Few pivots",       0.80, 0.80, 0.80, 0.20, 0.80, 0.80),
        ("Bad alternation",  0.80, 0.80, 0.80, 0.80, 0.20, 0.80),
        ("Imbalanced",       0.80, 0.80, 0.80, 0.80, 0.80, 0.20),
        ("User example",     0.72, 0.70, 0.68, 0.66, 0.74, 0.62),
    ]

    print(f"\n{'Pattern':<16} {'RuleScore':>10} {'Predicted':>10} {'Delta':>10}")
    print("-" * 54)

    for desc, high_rise_score, low_drop_score, expansion_score, pivot_richness_score, alternation_score, balance_score in test_cases:
        rule_score = megaphone_score(
            high_rise_score=high_rise_score,
            low_drop_score=low_drop_score,
            expansion_score=expansion_score,
            pivot_richness_score=pivot_richness_score,
            alternation_score=alternation_score,
            balance_score=balance_score,
        )
        pred_score = learned_confidence(
            high_rise_score=high_rise_score,
            low_drop_score=low_drop_score,
            expansion_score=expansion_score,
            pivot_richness_score=pivot_richness_score,
            alternation_score=alternation_score,
            balance_score=balance_score,
            model=model,
        )
        delta = pred_score - rule_score
        print(f"{desc:<16} {rule_score:>10.2f} {pred_score:>10.2f} {delta:>10.2f}")