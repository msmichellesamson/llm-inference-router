"""
Experiment 2: does a small learned baseline beat the heuristic on the
same MT-Bench complex-vs-simple task?

We embed each prompt with all-MiniLM-L6-v2 (22M params, runs fine on
CPU in seconds) and train a logistic regression with leave-one-out
cross-validation. The dataset is tiny (80 prompts), so LOO is the
honest evaluation choice.

Comparison set (same as experiment 01):
  - heuristic (token + regex) - baseline from src/core/complexity_analyzer.py
  - learned (MiniLM + logreg) - this experiment
  - random

Outputs:
  results.json
  comparison.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.core.complexity_analyzer import ComplexityAnalyzer  # noqa: E402

OUT_DIR = Path(__file__).parent

COMPLEX_CATEGORIES = {"math", "reasoning", "coding", "stem"}

RANDOM_SEED = 42


def load_prompts() -> list[dict]:
    ds = load_dataset("philschmid/mt-bench", split="train")
    return [
        {
            "id": r["question_id"],
            "category": r["category"],
            "prompt": r["turns"][0],
            "is_complex": r["category"] in COMPLEX_CATEGORIES,
        }
        for r in ds
    ]


def heuristic_scores(prompts: list[dict]) -> np.ndarray:
    a = ComplexityAnalyzer()
    return np.array([a.analyze(p["prompt"]).score for p in prompts])


def learned_loo_scores(prompts: list[dict]) -> np.ndarray:
    print("loading sentence-transformer (one-time, ~30s on first run)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("encoding prompts...")
    X = model.encode([p["prompt"] for p in prompts], show_progress_bar=False)
    y = np.array([1 if p["is_complex"] else 0 for p in prompts])

    print("running leave-one-out cross-validation...")
    loo = LeaveOneOut()
    proba = np.zeros(len(prompts))
    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        proba[test_idx[0]] = clf.predict_proba(X[test_idx])[0, 1]
    return proba


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    prompts = load_prompts()
    y_true = np.array([1 if p["is_complex"] else 0 for p in prompts])

    h = heuristic_scores(prompts)
    learned = learned_loo_scores(prompts)
    rand = rng.random(len(prompts))

    methods = {
        "random": rand,
        "heuristic (token + regex)": h,
        "learned (MiniLM + logreg, LOO-CV)": learned,
    }
    aucs = {name: float(roc_auc_score(y_true, s)) for name, s in methods.items()}

    print("\n=== AUC on MT-Bench complex vs simple (n=80) ===")
    for name, a in aucs.items():
        print(f"  {name:40s}  AUC={a:.3f}")

    results = {
        "n_prompts": len(prompts),
        "auc_by_method": aucs,
        "complex_categories": sorted(COMPLEX_CATEGORIES),
        "method_notes": {
            "heuristic": "from src/core/complexity_analyzer.py - token count + regex",
            "learned": "all-MiniLM-L6-v2 embeddings + logistic regression, leave-one-out CV",
        },
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    names = list(methods.keys())
    vals = [aucs[n] for n in names]
    colors = ["#999999", "#cc4b37", "#2e7d4f"]
    bars = ax.barh(names, vals, color=colors)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.4, label="random baseline")
    ax.set_xlim(0, 1)
    ax.set_xlabel("ROC AUC (higher is better)")
    ax.set_title("MT-Bench complex-vs-simple prediction\n(n=80, leave-one-out for learned model)")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison.png", dpi=130)
    plt.close(fig)

    print(f"\nwrote: {OUT_DIR/'results.json'}")
    print(f"wrote: {OUT_DIR/'comparison.png'}")


if __name__ == "__main__":
    main()
