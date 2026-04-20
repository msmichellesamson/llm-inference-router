"""
Experiment 1: how well does the heuristic complexity analyzer predict
which MT-Bench category a prompt belongs to?

The router currently uses a simple heuristic (token count + regex patterns
for code/reasoning). MT-Bench groups prompts into 8 categories, four of
which (math, reasoning, coding, stem) are intuitively "needs a strong model"
and four (writing, roleplay, extraction, humanities) are intuitively
"a smaller model probably suffices".

If the heuristic is useful, prompts in the "complex" categories should
receive systematically higher complexity scores than the "simple" ones.

We measure:
  - mean complexity score per category
  - binary classification AUC (complex vs simple)
  - comparison against three baselines:
      1. token count alone
      2. character count alone
      3. random

Outputs:
  results.json
  by_category.png
  roc_curves.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.core.complexity_analyzer import ComplexityAnalyzer  # noqa: E402

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)

COMPLEX_CATEGORIES = {"math", "reasoning", "coding", "stem"}
SIMPLE_CATEGORIES = {"writing", "roleplay", "extraction", "humanities"}

RANDOM_SEED = 42


def load_prompts() -> list[dict]:
    ds = load_dataset("philschmid/mt-bench", split="train")
    rows = []
    for row in ds:
        prompt = row["turns"][0]
        rows.append(
            {
                "id": row["question_id"],
                "category": row["category"],
                "prompt": prompt,
                "is_complex": row["category"] in COMPLEX_CATEGORIES,
                "char_count": len(prompt),
                "token_count_estimate": len(prompt) // 4,
            }
        )
    return rows


def score_with_heuristic(prompts: list[dict]) -> list[float]:
    analyzer = ComplexityAnalyzer()
    scores = []
    for row in prompts:
        result = analyzer.analyze(row["prompt"])
        scores.append(result.score)
    return scores


def auc(y_true: list[int], y_score: list[float]) -> float:
    return float(roc_auc_score(y_true, y_score))


def per_category_means(prompts: list[dict], scores: list[float]) -> dict[str, float]:
    by_cat: dict[str, list[float]] = {}
    for row, score in zip(prompts, scores):
        by_cat.setdefault(row["category"], []).append(score)
    return {cat: float(np.mean(vals)) for cat, vals in sorted(by_cat.items())}


def plot_by_category(per_cat: dict[str, float], out_path: Path) -> None:
    cats = list(per_cat.keys())
    vals = [per_cat[c] for c in cats]
    colors = ["#cc4b37" if c in COMPLEX_CATEGORIES else "#3b6fa3" for c in cats]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(cats, vals, color=colors)
    ax.set_ylabel("mean heuristic complexity score (0-1)")
    ax.set_title("MT-Bench: heuristic complexity score by category")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_roc(y_true, scores_by_method: dict[str, list[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, scores in scores_by_method.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            a = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, label=f"{name} (AUC={a:.3f})")
        except ValueError:
            continue
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random (AUC=0.500)")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Predicting MT-Bench complex vs simple")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    prompts = load_prompts()
    print(f"loaded {len(prompts)} prompts across "
          f"{len(set(p['category'] for p in prompts))} categories")

    heuristic_scores = score_with_heuristic(prompts)
    token_scores = [float(p["token_count_estimate"]) for p in prompts]
    char_scores = [float(p["char_count"]) for p in prompts]
    random_scores = rng.random(len(prompts)).tolist()

    y_true = [1 if p["is_complex"] else 0 for p in prompts]

    methods = {
        "heuristic (token + regex)": heuristic_scores,
        "token count only": token_scores,
        "char count only": char_scores,
        "random": random_scores,
    }

    aucs = {name: auc(y_true, s) for name, s in methods.items()}
    per_cat = per_category_means(prompts, heuristic_scores)

    results = {
        "n_prompts": len(prompts),
        "complex_categories": sorted(COMPLEX_CATEGORIES),
        "simple_categories": sorted(SIMPLE_CATEGORIES),
        "auc_by_method": aucs,
        "heuristic_mean_by_category": per_cat,
    }

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    plot_by_category(per_cat, OUT_DIR / "by_category.png")
    plot_roc(y_true, methods, OUT_DIR / "roc_curves.png")

    print("\n=== AUC (binary: complex vs simple) ===")
    for name, a in aucs.items():
        print(f"  {name:30s}  AUC={a:.3f}")

    print("\n=== heuristic mean score by category ===")
    for cat, val in per_cat.items():
        tag = "[complex]" if cat in COMPLEX_CATEGORIES else "[simple] "
        print(f"  {tag} {cat:12s}  {val:.3f}")

    print(f"\nwrote: {OUT_DIR/'results.json'}")
    print(f"wrote: {OUT_DIR/'by_category.png'}")
    print(f"wrote: {OUT_DIR/'roc_curves.png'}")


if __name__ == "__main__":
    main()
