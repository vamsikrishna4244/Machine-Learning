"""
Credit Card Fraud Dataset Loader and Transaction Simulator.

Generates realistic payment card transaction streams with extreme class imbalance (~0.8% fraud),
heavy-tailed transaction amounts, and PCA-projected feature vectors modeled after the ULB/Kaggle benchmark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


FEATURE_NAMES = [f"V{i}" for i in range(1, 29)]
ALL_COLUMNS = ["Time", "Amount"] + FEATURE_NAMES + ["Class"]


def generate_credit_card_data(
    n_samples: int = 6000,
    fraud_rate: float = 0.008,
    seed: int = 42,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Generate synthetic credit card transaction records with realistic fraud dynamics.

    Parameters
    ----------
    n_samples : int, default=6000
        Total transactions.
    fraud_rate : float, default=0.008
        Fraction of fraudulent transactions (e.g., 0.8% = 48 fraud cases in 6,000 rows).
    seed : int, default=42
        Reproducible random seed.
    output_path : Path or str, optional
        Target path to save CSV.

    Returns
    -------
    pd.DataFrame
        Dataframe containing 'Time', 'Amount', 'V1'-'V28', and 'Class'.
    """
    if n_samples < 500:
        raise ValueError(f"n_samples must be at least 500; got {n_samples}")
    if not (0.0005 <= fraud_rate <= 0.20):
        raise ValueError(f"fraud_rate must be between 0.0005 and 0.20; got {fraud_rate}")

    rng = np.random.default_rng(seed)
    n_fraud = max(10, int(np.round(n_samples * fraud_rate)))
    n_legit = n_samples - n_fraud

    # 1. Generate Timestamps (seconds over 48 hours: 0 to 172,800)
    time_legit = np.sort(rng.uniform(0, 172800, size=n_legit))
    time_fraud = rng.choice(time_legit, size=n_fraud, replace=False)

    # 2. Generate Transaction Amounts (heavy-tailed lognormal distribution)
    # Legitimate: median ~$45, occasionally reaching $1,500+
    amount_legit = np.round(rng.lognormal(mean=3.6, sigma=1.2, size=n_legit), 2)
    # Fraud: often small authorization test amounts ($1-$10) or large fraudulent transfers ($400-$2,500)
    fraud_modes = rng.choice(["small_test", "large_drain"], size=n_fraud, p=[0.40, 0.60])
    amount_fraud = np.where(
        fraud_modes == "small_test",
        np.round(rng.uniform(1.0, 15.0, size=n_fraud), 2),
        np.round(rng.uniform(350.0, 2400.0, size=n_fraud), 2),
    )

    # 3. Generate Latent PCA Features V1 - V28
    # Legitimate transactions: standard normal distributions around zero
    v_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))

    # Fraud transactions: key discriminative components have distinct mean shifts
    # In empirical ULB dataset: V14, V12, V10, V17 have negative shifts; V4, V11 have positive shifts
    v_fraud = rng.normal(loc=0.0, scale=1.4, size=(n_fraud, 28))
    # V4 (index 3) is elevated in fraud
    v_fraud[:, 3] += rng.normal(loc=3.5, scale=1.0, size=n_fraud)
    # V11 (index 10) is elevated in fraud
    v_fraud[:, 10] += rng.normal(loc=2.8, scale=0.9, size=n_fraud)
    # V10 (index 9) is depressed in fraud
    v_fraud[:, 9] -= rng.normal(loc=3.2, scale=1.1, size=n_fraud)
    # V12 (index 11) is depressed in fraud
    v_fraud[:, 11] -= rng.normal(loc=3.8, scale=1.0, size=n_fraud)
    # V14 (index 13) is strongly depressed in fraud
    v_fraud[:, 13] -= rng.normal(loc=4.5, scale=1.2, size=n_fraud)
    # V17 (index 16) is depressed in fraud
    v_fraud[:, 16] -= rng.normal(loc=3.0, scale=1.1, size=n_fraud)

    # Build DataFrames
    df_legit = pd.DataFrame(v_legit, columns=FEATURE_NAMES)
    df_legit["Time"] = time_legit
    df_legit["Amount"] = amount_legit
    df_legit["Class"] = 0

    df_fraud = pd.DataFrame(v_fraud, columns=FEATURE_NAMES)
    df_fraud["Time"] = time_fraud
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Reorder columns: Time, Amount, V1-V28, Class
    df = df[["Time", "Amount"] + FEATURE_NAMES + ["Class"]]

    # Inject realistic noise: 10 duplicate rows
    dupes = df.iloc[:10].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    return df


def load_fraud_data(
    filepath: Optional[Path | str] = None,
    ensure_exists: bool = True,
) -> pd.DataFrame:
    """Load transaction dataset from disk or generate if missing."""
    if filepath is None:
        project_root = Path(__file__).resolve().parent.parent
        filepath = project_root / "data" / "credit_card_fraud.csv"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        if ensure_exists:
            return generate_credit_card_data(n_samples=6000, fraud_rate=0.008, seed=42, output_path=filepath)
        raise FileNotFoundError(f"Fraud dataset not found at: {filepath.resolve()}")

    return pd.read_csv(filepath)
