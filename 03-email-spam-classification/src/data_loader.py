"""
Email Spam Dataset Loader and Synthetic Email Generator.

Generates realistic corporate communication (Ham) and unsolicited promotional/phishing (Spam)
email messages with varied vocabulary and natural linguistic patterns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


HAM_GREETINGS = [
    "Hi team,", "Hello all,", "Good morning,", "Hi Alex,", "Hey everyone,",
    "Dear colleagues,", "Hi Jordan,", "Team,", "Hope you had a great weekend,"
]

HAM_TOPICS = [
    "regarding the upcoming sprint planning session and roadmap review",
    "the Q3 budget allocation has been finalized by the finance committee",
    "we observed a slight latency spike on the database replica cluster",
    "the client onboarding documentation has been posted to our internal wiki",
    "please make sure to review the pull request for the authentication service",
    "the staging server will undergo scheduled maintenance this Saturday night",
    "our team offsite itinerary and travel arrangements have been confirmed",
    "the candidate interview feedback is due in the recruiting portal today",
    "please find attached the quarterly KPI and customer retention report",
    "we are rolling out updated security guidelines for multi-factor login",
    "the UX designer shared the new mobile wireframes for client feedback",
    "let us sync on the API rate limiting thresholds before production rollout",
]

HAM_DETAILS = [
    "Let me know if you encounter any blockers or need technical assistance.",
    "Please add your agenda items to the shared meeting notes by noon.",
    "All unit tests and integration tests are currently passing in CI.",
    "Finance has already processed the vendor invoice for the software license.",
    "We can discuss the open architecture questions during tomorrow's standup.",
    "Please verify that your development environment is updated to Python 3.11.",
    "The metrics show a 14% improvement in page load speeds across all regions.",
    "Feel free to leave comments directly on the Figma design prototypes.",
]

HAM_SIGNOFFS = [
    "Best regards,\nEngineering Lead",
    "Thanks,\nProject Manager",
    "Cheers,\nProduct Team",
    "Warmly,\nOperations",
    "Best,\nDevOps Team",
]

SPAM_GREETINGS = [
    "URGENT ATTENTION:", "CONGRATULATIONS WINNER!", "Dear Valued Customer,",
    "FINAL WARNING:", "Dear Beneficiary,", "Official Notification:",
    "Exclusive Invitation:", "Security Alert Notification:"
]

SPAM_HOOKS = [
    "you have been selected to receive a guaranteed cash prize of $2,500,000",
    "your bank account has been flagged for unauthorized login attempts",
    "pre-approved for an instant $45,000 personal loan with zero credit check",
    "claim your free Apple iPhone 15 Pro Max lottery prize right now",
    "massive 85% discount clearance on authentic Swiss replica luxury watches",
    "earn $7,500 every week from home using this automated cryptocurrency bot",
    "an unclaimed FedEx package is waiting for your immediate confirmation",
    "our medical doctor reveals a secret 10-second fat-burning bedtime routine",
    "barrister seeks confidential partner to transfer £9.4 million offshore estate",
    "refinance your mortgage at guaranteed 1.25% fixed interest rates today",
]

SPAM_ACTIONS = [
    "Click the link below immediately to verify your credentials and claim funds.",
    "Reply with your full banking account numbers and passport copy to transfer.",
    "Limited time offer expires in 2 hours! Do not miss this once-in-a-lifetime opportunity.",
    "Call our toll-free hotline right now to confirm delivery before return to sender.",
    "Deposit only $100 today to start receiving guaranteed passive daily returns.",
    "Click here to decrypt your confidential payroll attachment file securely.",
]


def generate_spam_dataset(
    n_samples: int = 2500,
    spam_ratio: float = 0.18,
    seed: int = 42,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Generate a synthetic email corpus with realistic Ham and Spam distribution.

    Parameters
    ----------
    n_samples : int, default=2500
        Total email messages to generate.
    spam_ratio : float, default=0.18
        Proportion of spam emails.
    seed : int, default=42
        Reproducible random seed.
    output_path : Path or str, optional
        Target path to save CSV.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'subject', 'body', 'message', 'label', and 'label_text'.
    """
    if n_samples < 100:
        raise ValueError(f"n_samples must be at least 100; got {n_samples}")
    if not (0.01 <= spam_ratio <= 0.99):
        raise ValueError(f"spam_ratio must be between 0.01 and 0.99; got {spam_ratio}")

    rng = np.random.default_rng(seed)
    n_spam = int(np.round(n_samples * spam_ratio))
    n_ham = n_samples - n_spam

    records = []

    # Generate Ham emails
    for i in range(n_ham):
        greeting = rng.choice(HAM_GREETINGS)
        topic = rng.choice(HAM_TOPICS)
        detail = rng.choice(HAM_DETAILS)
        signoff = rng.choice(HAM_SIGNOFFS)

        # Subjects
        subj_prefix = rng.choice(["Update:", "Meeting:", "Notice:", "Sync:", "Fwd:", ""])
        subject = f"{subj_prefix} {topic.capitalize()}".strip()
        body = f"{greeting}\n\nThis is a note {topic}. {detail}\n\n{signoff}"
        message = f"{subject} {body}"

        records.append({
            "email_id": f"HAM-{10000 + i}",
            "subject": subject,
            "body": body,
            "message": message,
            "label": 0,
            "label_text": "ham",
        })

    # Generate Spam emails
    for i in range(n_spam):
        greeting = rng.choice(SPAM_GREETINGS)
        hook = rng.choice(SPAM_HOOKS)
        action = rng.choice(SPAM_ACTIONS)

        subj_prefix = rng.choice(["***", "URGENT:", "ACT NOW:", "WINNER:", "ALERT:", ""])
        subject = f"{subj_prefix} {hook.capitalize()}".strip()

        # Add link or token
        link = f"http://secure-verify-{rng.integers(100, 999)}.net/claim"
        body = f"{greeting}\n\nNotice: {hook.capitalize()}. {action} Link: {link}"
        message = f"{subject} {body}"

        records.append({
            "email_id": f"SPAM-{20000 + i}",
            "subject": subject,
            "body": body,
            "message": message,
            "label": 1,
            "label_text": "spam",
        })

    df = pd.DataFrame(records)
    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Inject 10 duplicate rows for deduplication pipeline testing
    dupes = df.iloc[:10].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    return df


def load_email_data(
    filepath: Optional[Path | str] = None,
    ensure_exists: bool = True,
) -> pd.DataFrame:
    """Load email spam dataset from CSV or generate if missing."""
    if filepath is None:
        project_root = Path(__file__).resolve().parent.parent
        filepath = project_root / "data" / "spam_emails.csv"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        if ensure_exists:
            return generate_spam_dataset(n_samples=2500, spam_ratio=0.18, seed=42, output_path=filepath)
        raise FileNotFoundError(f"Spam dataset not found at: {filepath.resolve()}")

    return pd.read_csv(filepath)
