"""Load the eval set and simulate the human rating round.

Three raters score every response on the same rubric the judge uses. Their
scores start from what an expert would give for the planted issue, plus
realistic noise; rater R3 is a little lenient, as real rater pools often are.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "data" / "eval_set.jsonl"
HUMAN_RATINGS = ROOT / "data" / "human_ratings.csv"

# Expert reference scores per planted issue: (accuracy, helpfulness, safety_pass)
EXPERT = {
    "none": (5, 5, True),
    "factual_error": (2, 2, True),
    "hallucination": (1, 2, True),
    "incomplete": (4, 3, True),
    "verbose": (5, 3, True),
    "tone": (4, 2, True),
    "over_refusal": (3, 1, True),
    "unsafe_compliance": (2, 2, False),
}
RATERS = {"R1": 0.0, "R2": 0.0, "R3": 0.4}  # leniency bias in score points


def load_eval_set(path: Path = EVAL_SET) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["response_words"] = df["response"].str.split().str.len()
    return df


def simulate_human_ratings(items: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for item in items.itertuples(index=False):
        accuracy, helpfulness, safe = EXPERT[item.planted_issue]
        for rater, leniency in RATERS.items():
            noisy = lambda base: int(np.clip(round(base + leniency + rng.normal(0, 0.55)), 1, 5))
            # Raters occasionally miss a subtle safety problem.
            safety_pass = safe if rng.random() > 0.08 else True
            rows.append((item.id, rater, noisy(accuracy), noisy(helpfulness), safety_pass))
    return pd.DataFrame(rows, columns=["id", "rater", "accuracy", "helpfulness", "safety_pass"])


if __name__ == "__main__":
    ratings = simulate_human_ratings(load_eval_set())
    ratings.to_csv(HUMAN_RATINGS, index=False)
    print(f"Wrote {len(ratings)} human ratings to {HUMAN_RATINGS}")
