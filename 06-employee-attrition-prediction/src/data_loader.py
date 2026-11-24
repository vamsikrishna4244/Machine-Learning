"""
Employee Attrition Dataset Loader and Synthetic HR Data Generator.

Generates realistic employee records modeled after the IBM HR Analytics benchmark,
capturing compensation, job satisfaction, work-life balance, and attrition hazard dynamics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


DEPARTMENTS = ["Research & Development", "Sales", "Human Resources"]
JOB_ROLES = [
    "Software Engineer",
    "Research Scientist",
    "Sales Executive",
    "Laboratory Technician",
    "Manufacturing Director",
    "Healthcare Representative",
    "Sales Representative",
    "Manager",
    "Human Resources Specialist",
]
BUSINESS_TRAVELS = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
MARITAL_STATUSES = ["Married", "Single", "Divorced"]


def generate_attrition_dataset(
    n_samples: int = 2000,
    attrition_rate: float = 0.16,
    seed: int = 42,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Generate realistic employee records with probabilistic attrition drivers.

    Parameters
    ----------
    n_samples : int, default=2000
        Number of employee records.
    attrition_rate : float, default=0.16
        Target attrition prevalence (~16% matches IBM HR baseline).
    seed : int, default=42
        Reproducible random seed.
    output_path : Path or str, optional
        Target file path.

    Returns
    -------
    pd.DataFrame
        Complete employee records dataframe.
    """
    if n_samples < 200:
        raise ValueError(f"n_samples must be at least 200; got {n_samples}")
    if not (0.01 <= attrition_rate <= 0.50):
        raise ValueError(f"attrition_rate must be between 0.01 and 0.50; got {attrition_rate}")

    rng = np.random.default_rng(seed)

    emp_ids = [f"EMP-{1000 + i}" for i in range(n_samples)]
    age = rng.integers(21, 61, size=n_samples)
    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.60, 0.40])
    marital = rng.choice(MARITAL_STATUSES, size=n_samples, p=[0.46, 0.32, 0.22])

    department = rng.choice(DEPARTMENTS, size=n_samples, p=[0.65, 0.30, 0.05])
    job_role = rng.choice(JOB_ROLES, size=n_samples)
    job_level = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.37, 0.36, 0.15, 0.07, 0.05])

    # Tenures & Experience
    total_working_years = np.clip(age - 20 - rng.integers(0, 5, size=n_samples), 1, 40)
    years_at_company = np.clip(
        [rng.integers(0, max(1, twy + 1)) for twy in total_working_years], 0, 30
    )
    years_in_current_role = np.clip(
        [rng.integers(0, max(1, yac + 1)) for yac in years_at_company], 0, 18
    )
    years_since_last_promotion = np.clip(
        [rng.integers(0, max(1, yac + 1)) for yac in years_at_company], 0, 15
    )
    years_with_curr_manager = np.clip(
        [rng.integers(0, max(1, yac + 1)) for yac in years_at_company], 0, 18
    )

    # Ordinal Satisfaction & Balance Metrics (1 = Low to 4 = High)
    job_satisfaction = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.19, 0.19, 0.31, 0.31])
    environment_satisfaction = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.19, 0.20, 0.30, 0.31])
    work_life_balance = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.08, 0.23, 0.60, 0.09])
    job_involvement = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.06, 0.25, 0.59, 0.10])

    # Compensation & Work Conditions
    base_salaries = {1: 2800, 2: 5200, 3: 8800, 4: 13500, 5: 18500}
    monthly_income = np.array([
        int(rng.normal(base_salaries[lvl], base_salaries[lvl] * 0.15))
        for lvl in job_level
    ])
    monthly_income = np.clip(monthly_income, 1800, 25000)

    overtime = rng.choice(["Yes", "No"], size=n_samples, p=[0.28, 0.72])
    business_travel = rng.choice(BUSINESS_TRAVELS, size=n_samples, p=[0.70, 0.20, 0.10])
    stock_option_level = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.43, 0.40, 0.11, 0.06])
    distance_from_home = rng.integers(1, 30, size=n_samples)
    num_companies_worked = rng.integers(0, 9, size=n_samples)

    # Probabilistic Attrition Logic (calculating log-odds hazard)
    log_odds = -2.80  # Baseline intercept (~6% baseline)
    log_odds += np.where(overtime == "Yes", 1.45, 0.0)
    log_odds += np.where(work_life_balance == 1, 1.25, 0.0)
    log_odds += np.where(job_satisfaction == 1, 1.10, 0.0)
    log_odds += np.where(years_since_last_promotion >= 5, 0.75, 0.0)
    log_odds += np.where(monthly_income < 3500, 0.90, 0.0)
    log_odds += np.where(business_travel == "Travel_Frequently", 0.85, 0.0)
    log_odds += np.where(marital == "Single", 0.55, 0.0)
    log_odds += np.where(distance_from_home > 20, 0.45, 0.0)

    # Protective retention factors
    log_odds -= np.where(stock_option_level >= 1, 0.80, 0.0)
    log_odds -= np.where(years_at_company > 7, 0.65, 0.0)
    log_odds -= np.where(job_satisfaction >= 3, 0.70, 0.0)
    log_odds -= np.where(work_life_balance >= 3, 0.50, 0.0)
    log_odds -= np.where(monthly_income > 10000, 0.90, 0.0)

    probs = 1.0 / (1.0 + np.exp(-log_odds))
    attrition = (rng.uniform(0, 1, size=n_samples) < probs).astype(int)

    # Calibrate overall attrition rate to target
    current_rate = float(np.mean(attrition))
    if current_rate > attrition_rate:
        # Flip excess positives to negatives
        excess = int((current_rate - attrition_rate) * n_samples)
        pos_indices = np.where(attrition == 1)[0]
        flip_idx = rng.choice(pos_indices, size=min(excess, len(pos_indices)), replace=False)
        attrition[flip_idx] = 0
    elif current_rate < attrition_rate:
        # Flip excess negatives with highest probability to positives
        defect = int((attrition_rate - current_rate) * n_samples)
        neg_indices = np.where(attrition == 0)[0]
        # Rank by probability
        neg_sorted = neg_indices[np.argsort(probs[neg_indices])[::-1]]
        flip_idx = neg_sorted[:min(defect, len(neg_sorted))]
        attrition[flip_idx] = 1

    df = pd.DataFrame({
        "EmployeeID": emp_ids,
        "Age": age,
        "Gender": gender,
        "MaritalStatus": marital,
        "Department": department,
        "JobRole": job_role,
        "JobLevel": job_level,
        "MonthlyIncome": monthly_income,
        "TotalWorkingYears": total_working_years,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
        "DistanceFromHome": distance_from_home,
        "NumCompaniesWorked": num_companies_worked,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": environment_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "JobInvolvement": job_involvement,
        "OverTime": overtime,
        "BusinessTravel": business_travel,
        "StockOptionLevel": stock_option_level,
        "Attrition": attrition,
    })

    # Shuffle rows deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Inject realistic noise: 10 duplicate rows
    dupes = df.iloc[:10].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    return df


def load_attrition_data(
    filepath: Optional[Path | str] = None,
    ensure_exists: bool = True,
) -> pd.DataFrame:
    """Load employee attrition dataset from disk or generate if missing."""
    if filepath is None:
        project_root = Path(__file__).resolve().parent.parent
        filepath = project_root / "data" / "employee_attrition.csv"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        if ensure_exists:
            return generate_attrition_dataset(n_samples=2000, seed=42, output_path=filepath)
        raise FileNotFoundError(f"Attrition dataset not found at: {filepath.resolve()}")

    return pd.read_csv(filepath)
