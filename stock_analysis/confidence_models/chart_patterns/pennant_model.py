import numpy as np
import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pkl_files")
MODEL_PATH = os.path.join(MODEL_DIR, "pennant_model.pkl")

FEATURES = [
    "pole_strength_score",
    "convergence_score",
    "compactness_score",
    "pivot_richness_score",
    "balance_score",
    "symmetry_score",
]


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def pennant_score(
    pole_strength_score: float,
    convergence_score: float,
    compactness_score: float,
    pivot_richness_score: float,
    balance_score: float,
    symmetry_score: float,
) -> float:
    """
    Strict rule-based quality score for a textbook-style Pennant.
    Output: [0, 100]
    """

    if pole_strength_score >= 0.95:
        score_pole = 100
    elif pole_strength_score >= 0.80:
        score_pole = 80
    elif pole_strength_score >= 0.60:
        score_pole = 55
    elif pole_strength_score >= 0.40:
        score_pole = 25
    else:
        score_pole = 0

    if convergence_score >= 0.95:
        score_conv = 100
    elif convergence_score >= 0.80:
        score_conv = 80
    elif convergence_score >= 0.60:
        score_conv = 55
    elif convergence_score >= 0.40:
        score_conv = 25
    else:
        score_conv = 0

    if compactness_score >= 0.95:
        score_compact = 100
    elif compactness_score >= 0.80:
        score_compact = 80
    elif compactness_score >= 0.60:
        score_compact = 55
    elif compactness_score >= 0.40:
        score_compact = 25
    else:
        score_compact = 0

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

    if symmetry_score >= 0.95:
        score_sym = 100
    elif symmetry_score >= 0.80:
        score_sym = 80
    elif symmetry_score >= 0.60:
        score_sym = 55
    elif symmetry_score >= 0.40:
        score_sym = 25
    else:
        score_sym = 0

    overall = (
        0.26 * score_pole +
        0.24 * score_conv +
        0.18 * score_compact +
        0.12 * score_pivots +
        0.10 * score_balance +
        0.10 * score_sym
    )

    return clamp_0_100(overall)


def _sample(region: str):
    if region == "ideal":
        pole_strength_score = np.random.uniform(0.88, 1.00)
        convergence_score = np.random.uniform(0.88, 1.00)
        compactness_score = np.random.uniform(0.88, 1.00)
        pivot_richness_score = np.random.uniform(0.85, 1.00)
        balance_score = np.random.uniform(0.85, 1.00)
        symmetry_score = np.random.uniform(0.85, 1.00)

    elif region == "good":
        pole_strength_score = np.random.uniform(0.70, 0.92)
        convergence_score = np.random.uniform(0.70, 0.92)
        compactness_score = np.random.uniform(0.70, 0.92)
        pivot_richness_score = np.random.uniform(0.70, 0.92)
        balance_score = np.random.uniform(0.70, 0.92)
        symmetry_score = np.random.uniform(0.70, 0.92)

    elif region == "borderline":
        pole_strength_score = np.random.uniform(0.45, 0.75)
        convergence_score = np.random.uniform(0.45, 0.75)
        compactness_score = np.random.uniform(0.45, 0.75)
        pivot_richness_score = np.random.uniform(0.45, 0.75)
        balance_score = np.random.uniform(0.45, 0.75)
        symmetry_score = np.random.uniform(0.45, 0.75)

    elif region == "weak_pole":
        pole_strength_score = np.random.uniform(0.00, 0.35)
        convergence_score = np.random.uniform(0.60, 0.95)
        compactness_score = np.random.uniform(0.60, 0.95)
        pivot_richness_score = np.random.uniform(0.60, 0.95)
        balance_score = np.random.uniform(0.60, 0.95)
        symmetry_score = np.random.uniform(0.60, 0.95)

    elif region == "bad_convergence":
        pole_strength_score = np.random.uniform(0.60, 0.95)
        convergence_score = np.random.uniform(0.00, 0.35)
        compactness_score = np.random.uniform(0.60, 0.95)
        pivot_richness_score = np.random.uniform(0.60, 0.95)
        balance_score = np.random.uniform(0.60, 0.95)
        symmetry_score = np.random.uniform(0.60, 0.95)

    elif region == "wide_pennant":
        pole_strength_score = np.random.uniform(0.60, 0.95)
        convergence_score = np.random.uniform(0.60, 0.95)
        compactness_score = np.random.uniform(0.00, 0.35)
        pivot_richness_score = np.random.uniform(0.60, 0.95)
        balance_score = np.random.uniform(0.60, 0.95)
        symmetry_score = np.random.uniform(0.60, 0.95)

    elif region == "few_pivots":
        pole_strength_score = np.random.uniform(0.60, 0.95)
        convergence_score = np.random.uniform(0.60, 0.95)
        compactness_score = np.random.uniform(0.60, 0.95)
        pivot_richness_score = np.random.uniform(0.00, 0.35)
        balance_score = np.random.uniform(0.60, 0.95)
        symmetry_score = np.random.uniform(0.60, 0.95)

    elif region == "imbalanced":
        pole_strength_score = np.random.uniform(0.60, 0.95)
        convergence_score = np.random.uniform(0.60, 0.95)
        compactness_score = np.random.uniform(0.60, 0.95)
        pivot_richness_score = np.random.uniform(0.60, 0.95)
        balance_score = np.random.uniform(0.00, 0.35)
        symmetry_score = np.random.uniform(0.60, 0.95)

    elif region == "asymmetric":
        pole_strength_score = np.random.uniform(0.60, 0.95)
        convergence_score = np.random.uniform(0.60, 0.95)
        compactness_score = np.random.uniform(0.60, 0.95)
        pivot_richness_score = np.random.uniform(0.60, 0.95)
        balance_score = np.random.uniform(0.60, 0.95)
        symmetry_score = np.random.uniform(0.00, 0.35)

    elif region == "random":
        pole_strength_score = np.random.uniform(0.00, 1.00)
        convergence_score = np.random.uniform(0.00, 1.00)
        compactness_score = np.random.uniform(0.00, 1.00)
        pivot_richness_score = np.random.uniform(0.00, 1.00)
        balance_score = np.random.uniform(0.00, 1.00)
        symmetry_score = np.random.uniform(0.00, 1.00)

    else:  # outside
        pole_strength_score = np.random.uniform(0.00, 0.25)
        convergence_score = np.random.uniform(0.00, 0.25)
        compactness_score = np.random.uniform(0.00, 0.25)
        pivot_richness_score = np.random.uniform(0.00, 0.25)
        balance_score = np.random.uniform(0.00, 0.25)
        symmetry_score = np.random.uniform(0.00, 0.25)

    return (
        float(pole_strength_score),
        float(convergence_score),
        float(compactness_score),
        float(pivot_richness_score),
        float(balance_score),
        float(symmetry_score),
    )


REGION_COUNTS = {
    "ideal": 1400,
    "good": 1200,
    "borderline": 1000,
    "weak_pole": 900,
    "bad_convergence": 900,
    "wide_pennant": 800,
    "few_pivots": 800,
    "imbalanced": 700,
    "asymmetric": 700,
    "random": 700,
    "outside": 300,
}


def build_dataset() -> pd.DataFrame:
    rows = []

    for region, n in REGION_COUNTS.items():
        for _ in range(n):
            (
                pole_strength_score,
                convergence_score,
                compactness_score,
                pivot_richness_score,
                balance_score,
                symmetry_score,
            ) = _sample(region)

            base_score = pennant_score(
                pole_strength_score=pole_strength_score,
                convergence_score=convergence_score,
                compactness_score=compactness_score,
                pivot_richness_score=pivot_richness_score,
                balance_score=balance_score,
                symmetry_score=symmetry_score,
            )

            noise = np.random.normal(0, 2.0)
            quality = clamp_0_100(base_score + noise)

            rows.append({
                "pole_strength_score": pole_strength_score,
                "convergence_score": convergence_score,
                "compactness_score": compactness_score,
                "pivot_richness_score": pivot_richness_score,
                "balance_score": balance_score,
                "symmetry_score": symmetry_score,
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
            "Pennant regression model not found. Run "
            "`python -m stock_analysis.confidence_models.chart_patterns.pennant_model` "
            "to train and save it first."
        )
    model = joblib.load(MODEL_PATH)
    return model


def learned_confidence(
    pole_strength_score: float,
    convergence_score: float,
    compactness_score: float,
    pivot_richness_score: float,
    balance_score: float,
    symmetry_score: float,
    model: RandomForestRegressor,
) -> float:
    x = np.array([[
        pole_strength_score,
        convergence_score,
        compactness_score,
        pivot_richness_score,
        balance_score,
        symmetry_score,
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
        ("Weak pole",        0.20, 0.80, 0.80, 0.80, 0.80, 0.80),
        ("Bad convergence",  0.80, 0.20, 0.80, 0.80, 0.80, 0.80),
        ("Wide pennant",     0.80, 0.80, 0.20, 0.80, 0.80, 0.80),
        ("Few pivots",       0.80, 0.80, 0.80, 0.20, 0.80, 0.80),
        ("Imbalanced",       0.80, 0.80, 0.80, 0.80, 0.20, 0.80),
        ("Asymmetric",       0.80, 0.80, 0.80, 0.80, 0.80, 0.20),
        ("User example",     0.72, 0.70, 0.68, 0.66, 0.74, 0.62),
    ]

    print(f"\n{'Pattern':<16} {'RuleScore':>10} {'Predicted':>10} {'Delta':>10}")
    print("-" * 54)

    for desc, pole_strength_score, convergence_score, compactness_score, pivot_richness_score, balance_score, symmetry_score in test_cases:
        rule_score = pennant_score(
            pole_strength_score=pole_strength_score,
            convergence_score=convergence_score,
            compactness_score=compactness_score,
            pivot_richness_score=pivot_richness_score,
            balance_score=balance_score,
            symmetry_score=symmetry_score,
        )
        pred_score = learned_confidence(
            pole_strength_score=pole_strength_score,
            convergence_score=convergence_score,
            compactness_score=compactness_score,
            pivot_richness_score=pivot_richness_score,
            balance_score=balance_score,
            symmetry_score=symmetry_score,
            model=model,
        )
        delta = pred_score - rule_score
        print(f"{desc:<16} {rule_score:>10.2f} {pred_score:>10.2f} {delta:>10.2f}")