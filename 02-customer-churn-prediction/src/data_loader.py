"""
Data Ingestion and Synthetic Telecommunications Churn Generator.

Generates realistic subscriber account attributes, usage patterns, service subscriptions,
and contractual behaviors with genuine probabilistic class imbalance (~26% churn rate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def generate_churn_dataset(
    n_samples: int = 3500,
    seed: int = 42,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Generate a realistic telecommunications customer churn dataset.

    Parameters
    ----------
    n_samples : int, default=3500
        Number of customer records to generate.
    seed : int, default=42
        Reproducible random seed.
    output_path : Path or str, optional
        Target filepath to save the CSV.

    Returns
    -------
    pd.DataFrame
        Dataset of subscriber attributes, services, and binary churn label.
    """
    if n_samples < 100:
        raise ValueError(f"n_samples must be at least 100; got {n_samples}")

    rng = np.random.default_rng(seed)

    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    gender = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes", rng.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45]), "No")

    # Tenure in months (1 to 72)
    tenure_months = rng.integers(1, 73, size=n_samples)

    # Contract distribution: correlated with tenure
    contract_probs = np.zeros((n_samples, 3))
    for i, t in enumerate(tenure_months):
        if t < 12:
            contract_probs[i] = [0.82, 0.12, 0.06]
        elif t < 36:
            contract_probs[i] = [0.45, 0.35, 0.20]
        else:
            contract_probs[i] = [0.18, 0.32, 0.50]

    contract = [rng.choice(CONTRACT_TYPES, p=contract_probs[i]) for i in range(n_samples)]

    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "Yes",
        rng.choice(["Yes", "No"], size=n_samples, p=[0.44, 0.56]),
        "No phone service",
    )

    internet_service = rng.choice(INTERNET_SERVICES, size=n_samples, p=[0.44, 0.34, 0.22])

    def assign_addon(service_type: np.ndarray, p_yes: float) -> np.ndarray:
        return np.where(
            service_type != "No",
            rng.choice(["Yes", "No"], size=n_samples, p=[p_yes, 1.0 - p_yes]),
            "No internet service",
        )

    online_security = assign_addon(internet_service, 0.38)
    online_backup = assign_addon(internet_service, 0.44)
    device_protection = assign_addon(internet_service, 0.42)
    tech_support = assign_addon(internet_service, 0.39)
    streaming_tv = assign_addon(internet_service, 0.49)
    streaming_movies = assign_addon(internet_service, 0.50)

    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    payment_method = rng.choice(PAYMENT_METHODS, size=n_samples, p=[0.34, 0.22, 0.22, 0.22])

    # Monthly charges calculation based on active services
    base_charge = np.where(phone_service == "Yes", 20.0, 5.0)
    internet_charge = np.where(
        internet_service == "Fiber optic",
        50.0,
        np.where(internet_service == "DSL", 30.0, 0.0),
    )
    addons_count = (
        (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    )
    monthly_charges = np.round(
        base_charge + internet_charge + (addons_count * 5.5) + rng.normal(0, 3.0, size=n_samples),
        2,
    )
    monthly_charges = np.clip(monthly_charges, 18.5, 125.0)

    total_charges = np.round(
        monthly_charges * tenure_months + rng.normal(0, 20.0, size=n_samples),
        2,
    )
    total_charges = np.clip(total_charges, 18.5, 9000.0)

    # Probabilistic Churn Modeling (Log-Odds function calibrated to ~26.5% churn)
    # High churn drivers: Month-to-month contract, low tenure, high monthly charges, Fiber without tech support
    log_odds = (
        -0.85
        + (np.array([1.1 if c == "Month-to-month" else (-0.7 if c == "Two year" else -0.2) for c in contract]))
        - (tenure_months * 0.045)
        + ((monthly_charges - 65.0) * 0.022)
        + (np.where(internet_service == "Fiber optic", 0.45, 0.0))
        - (np.where(tech_support == "Yes", 0.55, 0.0))
        - (np.where(online_security == "Yes", 0.45, 0.0))
        + (np.where(payment_method == "Electronic check", 0.35, -0.15))
        + (senior_citizen * 0.25)
        + rng.logistic(0, 0.35, size=n_samples)
    )

    churn_prob = 1.0 / (1.0 + np.exp(-log_odds))
    churn = (rng.uniform(0, 1, size=n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure_months,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "churn": churn,
    })

    # Inject real-world data imperfections:
    # 1. Blank string spaces in total_charges (common in Kaggle Telco data for tenure == 0)
    blank_mask = rng.uniform(0, 1, size=n_samples) < 0.015
    df.loc[blank_mask, "total_charges"] = np.nan

    # 2. Append 15 duplicate customer records for deduplication testing
    dupes = df.iloc[:15].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)

    return df


def load_churn_data(
    filepath: Optional[Path | str] = None,
    ensure_exists: bool = True,
) -> pd.DataFrame:
    """Load telecommunications customer churn dataset from disk, generating if missing.

    Parameters
    ----------
    filepath : Path or str, optional
        Target CSV path.
    ensure_exists : bool, default=True
        Whether to generate synthetic dataset if not found.

    Returns
    -------
    pd.DataFrame
        Cleanable customer churn dataframe.
    """
    if filepath is None:
        project_root = Path(__file__).resolve().parent.parent
        filepath = project_root / "data" / "customer_churn.csv"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        if ensure_exists:
            return generate_churn_dataset(n_samples=3500, seed=42, output_path=filepath)
        raise FileNotFoundError(f"Customer churn dataset not found at: {filepath.resolve()}")

    return pd.read_csv(filepath)
