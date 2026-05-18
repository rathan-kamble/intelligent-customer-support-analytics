"""
Module 9 Assignment — Phase 1
Synthetic Customer Support Ticket Dataset Generator
====================================================
Generates a realistic 5,000-row ticket dataset (tickets.csv) with all 13
columns required by the assignment rubric. Priority labels are derived from
a deterministic scoring function so the SparkML model has real signal to
learn from.

Usage:
    python generate_tickets.py

Output:
    tickets.csv          — main dataset (5 000 rows)
    tickets_sample.csv   — first 20 rows for quick inspection
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ──────────────────────────────────────────────
# 1. Reproducibility
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_ROWS = 5_000


# ──────────────────────────────────────────────
# 2. Vocabulary / lookup tables
# ──────────────────────────────────────────────

CHANNELS = ["email", "chat", "web", "call"]

PRODUCTS = ["ProductA", "ProductB", "ProductC", "ProductD"]

REGIONS = ["North", "South", "East", "West", "Central"]

# Tiers ordered by business value (affects priority score)
TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
TIER_SCORE = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}

# Categories — technical & billing generate higher urgency
CATEGORIES = ["billing", "technical", "shipping", "returns", "account", "general"]
CATEGORY_SCORE = {
    "technical": 2,
    "billing":   2,
    "shipping":  1,
    "returns":   1,
    "account":   0,
    "general":   0,
}

# Words that, when found in ticket text, add urgency
URGENT_WORDS = [
    "urgent", "broken", "down", "critical", "asap", "emergency",
    "immediately", "failure", "not working", "crashed", "outage",
    "cannot access", "blocked", "error",
]

# Ticket templates — {product} is filled in dynamically
TICKET_TEMPLATES = [
    "My {product} is not working and I need help immediately.",
    "I have a billing issue with my recent {product} order.",
    "The shipment for my {product} has not arrived yet.",
    "I cannot log into my {product} account, please help urgently.",
    "There is a technical error when I try to use {product}.",
    "I want to return my {product} — it arrived broken.",
    "Please escalate this asap, {product} is down for our entire team.",
    "My subscription for {product} is showing incorrect charges.",
    "The {product} application crashed and I lost all my data.",
    "I have been waiting for a refund on {product} for over 3 weeks.",
    "General inquiry about {product} pricing and features.",
    "Need help setting up {product} for the first time.",
    "My {product} stopped working after the latest update.",
    "I am getting an error code on {product} — this is critical for my business.",
    "Please help me reset my password, I cannot access {product}.",
    "The {product} dashboard is completely blank — outage?",
    "Duplicate charge on my account for {product}, please fix immediately.",
    "I have an emergency: {product} integration is failure for all users.",
    "Slow performance on {product} — response time is unacceptable.",
    "I need to cancel my {product} subscription and get a refund.",
]


# ──────────────────────────────────────────────
# 3. Priority scoring function
# ──────────────────────────────────────────────

def compute_priority(tier: str, category: str, sentiment: float,
                     contains_urgent: int, response_time: float) -> str:
    """
    Deterministic scoring model for priority labels.

    Score breakdown (max ~10 points):
      Tier:           Bronze=0, Silver=1, Gold=2, Platinum=3
      Category:       technical/billing=2, shipping/returns=1, else=0
      Sentiment:      score < -0.3 → +2 (negative sentiment)
      Urgent words:   present → +3
      Response time:  > 24 h → +1 (already slow, needs escalation)

    Thresholds:
      >= 7 → Critical
      >= 5 → High
      >= 3 → Medium
      else → Low
    """
    score = 0
    score += TIER_SCORE.get(tier, 0)
    score += CATEGORY_SCORE.get(category, 0)
    if sentiment < -0.3:
        score += 2
    if contains_urgent:
        score += 3
    if response_time > 24:
        score += 1

    if score >= 7:
        return "Critical"
    elif score >= 5:
        return "High"
    elif score >= 3:
        return "Medium"
    else:
        return "Low"


# ──────────────────────────────────────────────
# 4. Row generation
# ──────────────────────────────────────────────

def generate_row(ticket_num: int, base_date: datetime) -> dict:
    """Generate one synthetic ticket row."""

    # Categorical fields
    product   = random.choice(PRODUCTS)
    tier      = random.choice(TIERS)
    category  = random.choice(CATEGORIES)
    channel   = random.choice(CHANNELS)
    region    = random.choice(REGIONS)

    # Free-text ticket body
    template    = random.choice(TICKET_TEMPLATES)
    ticket_text = template.replace("{product}", product)

    # Derived binary flag — does the text mention urgent language?
    contains_urgent = int(
        any(word in ticket_text.lower() for word in URGENT_WORDS)
    )

    # Numeric fields
    # sentiment_score: [-1.0, 1.0], biased slightly negative (realistic)
    sentiment = round(float(np.random.normal(loc=-0.1, scale=0.45)), 3)
    sentiment = max(-1.0, min(1.0, sentiment))   # clamp to [-1, 1]

    # response_time_hours: exponential, most tickets resolve < 24 h
    response_time = round(float(np.random.exponential(scale=10.0)), 2)
    response_time = max(0.1, response_time)       # floor at 6 min

    # num_previous_tickets: Poisson-distributed (most customers = few tickets)
    num_prev = int(np.random.poisson(lam=3))

    # Timestamp: random moment within one calendar year
    created_at = base_date + timedelta(hours=random.randint(0, 8_760))

    # Derived priority label
    priority = compute_priority(
        tier, category, sentiment, contains_urgent, response_time
    )

    return {
        "ticket_id":             f"TKT-{ticket_num:05d}",
        "created_at":            created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "channel":               channel,
        "product":               product,
        "region":                region,
        "customer_tier":         tier,
        "issue_category":        category,
        "ticket_text":           ticket_text,
        "response_time_hours":   response_time,
        "num_previous_tickets":  num_prev,
        "sentiment_score":       sentiment,
        "contains_urgent_words": contains_urgent,
        "priority_label":        priority,
    }


# ──────────────────────────────────────────────
# 5. Main: generate, validate, save
# ──────────────────────────────────────────────

def main():
    print(f"Generating {N_ROWS:,} synthetic support tickets …")

    base_date = datetime(2024, 1, 1)
    rows = [generate_row(i + 1, base_date) for i in range(N_ROWS)]
    df = pd.DataFrame(rows)

    # ── Basic validation ──────────────────────
    assert df.shape == (N_ROWS, 13), f"Expected (5000, 13), got {df.shape}"
    assert df.isnull().sum().sum() == 0, "Unexpected nulls found!"
    assert set(df["priority_label"].unique()) == {"Low", "Medium", "High", "Critical"}, \
        "Not all priority classes are represented!"

    # ── Class distribution ────────────────────
    print("\nPriority label distribution:")
    dist = df["priority_label"].value_counts()
    for label, count in dist.items():
        pct = count / N_ROWS * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<10} {count:>5}  ({pct:5.1f}%)  {bar}")

    # ── Summary stats ─────────────────────────
    print("\nNumeric column stats:")
    print(df[["response_time_hours", "num_previous_tickets",
              "sentiment_score", "contains_urgent_words"]].describe().round(3))

    # ── Save ──────────────────────────────────
    df.to_csv("tickets.csv", index=False)
    df.head(20).to_csv("tickets_sample.csv", index=False)

    print("\n✓  tickets.csv        saved  (5 000 rows)")
    print("✓  tickets_sample.csv saved  (20 rows)")
    print("\nFirst 3 rows:")
    print(df[["ticket_id", "customer_tier", "issue_category",
              "sentiment_score", "contains_urgent_words",
              "priority_label"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
