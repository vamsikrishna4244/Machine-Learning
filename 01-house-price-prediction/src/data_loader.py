"""
Data Loading and Generation Module for House Price Prediction.

Provides automated generation of realistic residential real estate data
based on hedonic pricing theory, with reproducible random seeds and schema validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


NEIGHBORHOODS = [
    "Downtown",
    "Suburbs",
    "Northside",
    "Westside",
    "Eastside",
    "Lakeside",
    "Hillcrest",
    "Greenwood",
]

NEIGHBORHOOD_PREMIUMS = {
    "Downtown": 120_000.0,
    "Lakeside": 160_000.0,
    "Hillcrest": 90_000.0,
    "Westside": 65_000.0,
    "Eastside": 50_000.0,
    "Northside": 30_000.0,
    "Greenwood": 20_000.0,
    "Suburbs": 0.0,
}


def generate_housing_dataset(
    n_samples: int = 2500,
    seed: int = 42,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Generate a realistic residential housing dataset using hedonic property pricing models.

    Parameters
    ----------
    n_samples : int, default=2500
        Number of residential housing records to generate.
    seed : int, default=42
        Random seed for reproducibility.
    output_path : Path or str, optional
        Filepath to persist the generated CSV dataset.

    Returns
    -------
    pd.DataFrame
        Synthetic housing records with numerical and categorical attributes.
    """
    if n_samples < 50:
        raise ValueError(f"n_samples must be at least 50; got {n_samples}")

    rng = np.random.default_rng(seed)

    # Core structural features
    bedrooms = rng.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.05, 0.20, 0.45, 0.22, 0.06, 0.02])
    bathrooms = np.round(bedrooms * 0.5 + rng.choice([0.5, 0.75, 1.0, 1.25, 1.5], size=n_samples), 2)
    bathrooms = np.clip(bathrooms, 1.0, 5.0)

    # Living square footage strongly correlated with bedrooms
    sqft_living = np.clip(
        np.round(bedrooms * 550.0 + bathrooms * 300.0 + rng.normal(0, 350.0, size=n_samples)),
        550.0,
        5800.0,
    )
    sqft_lot = np.round(sqft_living * rng.uniform(1.8, 6.0, size=n_samples) + rng.exponential(1500.0, size=n_samples))

    floors = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], size=n_samples, p=[0.45, 0.10, 0.38, 0.04, 0.03])
    waterfront = rng.choice([0, 1], size=n_samples, p=[0.94, 0.06])
    view_quality = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.70, 0.12, 0.08, 0.06, 0.04])
    condition = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.03, 0.07, 0.60, 0.22, 0.08])
    grade = np.clip(
        np.round(4.0 + (sqft_living / 650.0) + condition * 0.3 + rng.normal(0, 0.7, size=n_samples)).astype(int),
        4,
        12,
    )

    yr_built = rng.integers(1930, 2024, size=n_samples)
    is_renovated = rng.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    yr_renovated = np.where(
        is_renovated == 1,
        np.clip(yr_built + rng.integers(5, 45, size=n_samples), yr_built, 2024),
        0,
    )

    neighborhood = rng.choice(NEIGHBORHOODS, size=n_samples)

    # Hedonic Valuation Formula (Ground Truth Process + Gaussian Noise)
    base_val = 140_000.0
    sqft_val = sqft_living * 185.0
    grade_val = (grade ** 1.65) * 6_200.0
    bath_val = bathrooms * 28_000.0
    bed_val = bedrooms * 14_000.0
    wf_val = waterfront * 210_000.0
    view_val = view_quality * 24_000.0
    cond_val = (condition - 3) * 18_000.0
    age_penalty = (2024 - yr_built) * 450.0
    renovation_bonus = np.where(yr_renovated > 0, (yr_renovated - yr_built) * 350.0 + 20_000.0, 0.0)
    neigh_prem = np.array([NEIGHBORHOOD_PREMIUMS[n] for n in neighborhood])

    noise = rng.normal(0, 28_000.0, size=n_samples)

    price = np.round(
        base_val
        + sqft_val
        + grade_val
        + bath_val
        + bed_val
        + wf_val
        + view_val
        + cond_val
        - age_penalty
        + renovation_bonus
        + neigh_prem
        + noise,
        -2,  # round to nearest $100
    )
    price = np.clip(price, 85_000.0, 1_850_000.0)

    df = pd.DataFrame({
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view_quality": view_quality,
        "condition": condition,
        "grade": grade,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "neighborhood": neighborhood,
        "price": price,
    })

    # Intentionally inject controlled missing values and duplicates for cleaning demonstration
    # 1. 2% missing in yr_renovated (represented as NaN)
    nan_mask = rng.uniform(0, 1, size=n_samples) < 0.025
    df.loc[nan_mask, "yr_renovated"] = np.nan

    # 2. 12 duplicate records appended at the end
    duplicates = df.iloc[:12].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)

    return df


def load_housing_data(
    filepath: Optional[Path | str] = None,
    ensure_exists: bool = True,
) -> pd.DataFrame:
    """Load housing dataset from disk, generating it automatically if missing.

    Parameters
    ----------
    filepath : Path or str, optional
        Target CSV path. Defaults to `data/housing_data.csv` relative to project root.
    ensure_exists : bool, default=True
        Whether to generate synthetic dataset if file is not found.

    Returns
    -------
    pd.DataFrame
        Housing dataset ready for inspection and preprocessing.
    """
    if filepath is None:
        project_root = Path(__file__).resolve().parent.parent
        filepath = project_root / "data" / "housing_data.csv"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        if ensure_exists:
            return generate_housing_dataset(n_samples=2500, seed=42, output_path=filepath)
        raise FileNotFoundError(f"Housing data file not found at: {filepath.resolve()}")

    df = pd.read_csv(filepath)
    return df
