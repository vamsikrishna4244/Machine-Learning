"""
Customer Segmentation Dataset Loader and Synthetic Retail Data Generator.

Generates realistic retail and e-commerce customer behavior attributes:
Age, Annual Income ($k), Spending Score (1-100), Purchase Frequency, and Average Basket Value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


CUSTOMER_FEATURES = [
    "Age",
    "Annual_Income_k",
    "Spending_Score",
    "Purchase_Frequency",
    "Average_Basket_Value",
]


def generate_customer_data(
    n_samples: int = 2000,
    seed: int = 42,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Generate realistic retail customer behavioral dataset with natural clusters.

    Parameters
    ----------
    n_samples : int, default=2000
        Number of customer records to generate.
    seed : int, default=42
        Reproducible random seed.
    output_path : Path or str, optional
        Target path to save CSV.

    Returns
    -------
    pd.DataFrame
        Dataframe with CustomerID and numerical behavioral attributes.
    """
    if n_samples < 200:
        raise ValueError(f"n_samples must be at least 200; got {n_samples}")

    rng = np.random.default_rng(seed)

    # Archetype proportions:
    # 1. VIP Big Spenders (20%)
    # 2. Conservative Affluent (20%)
    # 3. Young Impulse Trendsetters (20%)
    # 4. Frugal Budget (20%)
    # 5. Mainstream Core (20%)
    n_per_cluster = n_samples // 5
    rem = n_samples % 5

    clusters = []

    # Cluster 1: VIP Big Spenders (High Income, High Spending Score)
    n1 = n_per_cluster + rem
    age1 = rng.integers(28, 55, size=n1)
    income1 = np.round(rng.normal(95.0, 12.0, size=n1), 1)
    spending1 = np.round(rng.normal(82.0, 7.0, size=n1), 1)
    freq1 = np.round(rng.normal(32.0, 5.0, size=n1), 0)
    basket1 = np.round(rng.normal(210.0, 30.0, size=n1), 2)
    clusters.append(pd.DataFrame({
        "Age": age1,
        "Annual_Income_k": income1,
        "Spending_Score": spending1,
        "Purchase_Frequency": freq1,
        "Average_Basket_Value": basket1,
    }))

    # Cluster 2: Conservative Affluent (High Income, Low Spending Score)
    n2 = n_per_cluster
    age2 = rng.integers(35, 68, size=n2)
    income2 = np.round(rng.normal(92.0, 11.0, size=n2), 1)
    spending2 = np.round(rng.normal(22.0, 6.0, size=n2), 1)
    freq2 = np.round(rng.normal(8.0, 3.0, size=n2), 0)
    basket2 = np.round(rng.normal(160.0, 25.0, size=n2), 2)
    clusters.append(pd.DataFrame({
        "Age": age2,
        "Annual_Income_k": income2,
        "Spending_Score": spending2,
        "Purchase_Frequency": freq2,
        "Average_Basket_Value": basket2,
    }))

    # Cluster 3: Young Impulse Trendsetters (Low/Mid Income, High Spending Score)
    n3 = n_per_cluster
    age3 = rng.integers(18, 32, size=n3)
    income3 = np.round(rng.normal(32.0, 8.0, size=n3), 1)
    spending3 = np.round(rng.normal(84.0, 6.0, size=n3), 1)
    freq3 = np.round(rng.normal(24.0, 4.0, size=n3), 0)
    basket3 = np.round(rng.normal(85.0, 15.0, size=n3), 2)
    clusters.append(pd.DataFrame({
        "Age": age3,
        "Annual_Income_k": income3,
        "Spending_Score": spending3,
        "Purchase_Frequency": freq3,
        "Average_Basket_Value": basket3,
    }))

    # Cluster 4: Frugal Budget (Low Income, Low Spending Score)
    n4 = n_per_cluster
    age4 = rng.integers(22, 65, size=n4)
    income4 = np.round(rng.normal(28.0, 7.0, size=n4), 1)
    spending4 = np.round(rng.normal(18.0, 5.0, size=n4), 1)
    freq4 = np.round(rng.normal(5.0, 2.0, size=n4), 0)
    basket4 = np.round(rng.normal(45.0, 10.0, size=n4), 2)
    clusters.append(pd.DataFrame({
        "Age": age4,
        "Annual_Income_k": income4,
        "Spending_Score": spending4,
        "Purchase_Frequency": freq4,
        "Average_Basket_Value": basket4,
    }))

    # Cluster 5: Mainstream Core (Moderate Income, Moderate Spending Score)
    n5 = n_per_cluster
    age5 = rng.integers(25, 58, size=n5)
    income5 = np.round(rng.normal(58.0, 9.0, size=n5), 1)
    spending5 = np.round(rng.normal(50.0, 6.0, size=n5), 1)
    freq5 = np.round(rng.normal(16.0, 3.0, size=n5), 0)
    basket5 = np.round(rng.normal(110.0, 18.0, size=n5), 2)
    clusters.append(pd.DataFrame({
        "Age": age5,
        "Annual_Income_k": income5,
        "Spending_Score": spending5,
        "Purchase_Frequency": freq5,
        "Average_Basket_Value": basket5,
    }))

    df = pd.concat(clusters, ignore_index=True)

    # Post-process bounds: Spending_Score in [1, 100], income > 10, basket > 10, freq >= 1
    df["Spending_Score"] = np.clip(df["Spending_Score"], 1.0, 100.0)
    df["Annual_Income_k"] = np.clip(df["Annual_Income_k"], 10.0, 160.0)
    df["Average_Basket_Value"] = np.clip(df["Average_Basket_Value"], 10.0, 450.0)
    df["Purchase_Frequency"] = np.clip(df["Purchase_Frequency"], 1.0, 60.0)

    # Shuffle rows deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "CustomerID", [f"CUST-{1000 + i}" for i in range(len(df))])

    # Inject 10 duplicate records for deduplication pipeline testing
    dupes = df.iloc[:10].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    return df


def load_customer_data(
    filepath: Optional[Path | str] = None,
    ensure_exists: bool = True,
) -> pd.DataFrame:
    """Load customer segmentation dataset from disk or generate if missing."""
    if filepath is None:
        project_root = Path(__file__).resolve().parent.parent
        filepath = project_root / "data" / "customer_segmentation.csv"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        if ensure_exists:
            return generate_customer_data(n_samples=2000, seed=42, output_path=filepath)
        raise FileNotFoundError(f"Customer dataset not found at: {filepath.resolve()}")

    return pd.read_csv(filepath)
