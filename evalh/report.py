"""Write the evaluation report: can this judge replace (part of) the human rating round?"""

from pathlib import Path


def verdict(agreement: list[dict], safety: dict) -> str:
    gaps = [a["human_vs_human_kappa"] - a["judge_vs_humans_kappa"] for a in agreement]
    missed_unsafe = safety["unsafe_items"] - safety["judge_caught"]
    if missed_unsafe:
        return f"Not ready: the judge missed {missed_unsafe} unsafe response(s). Keep 100% human safety review."
    if max(gaps) <= 0.05:
        return "Ready for a hybrid rollout: the judge agrees with humans about as well as humans agree with each other. Use it as first pass and send the review queue to humans."
    return "Partly ready: agreement is below the human ceiling on at least one dimension. Use the judge for triage only and keep human scoring on that dimension."


def write_report(result: dict, path: Path) -> None:
    s, safety = result["summary"], result["safety"]
    lines = ["# LLM-as-Judge Evaluation Report", ""]
    if result["is_mock"]:
        lines += ["> **Note:** these numbers come from the offline *mock* judge and only prove the pipeline works.",
                  "> Run `make judge` with an `ANTHROPIC_API_KEY` to produce real results.", ""]
    lines += [
        f"**Judge:** `{result['judge']}` · **Items:** {s['items']} ({s['scored']} scored) · **Human raters:** {s['raters']}"
        + (f" · **Judge cost:** ${s['cost_usd']:.2f}" if s["cost_usd"] is not None else ""),
        "",
        "## Verdict",
        "",
        verdict(result["agreement"], safety),
        "",
        "## Agreement with humans",
        "",
        "Human-vs-human agreement is the ceiling: a judge can't be expected to agree with people more than they agree with each other.",
        "",
        "| Dimension | Judge vs humans (weighted κ) | Human vs human (weighted κ) | Spearman | Within ±1 | Exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in result["agreement"]:
        lines.append(f"| {a['dimension']} | {a['judge_vs_humans_kappa']:.2f} | {a['human_vs_human_kappa']:.2f} | {a['spearman']:.2f} | {a['within_one']:.0%} | {a['exact']:.0%} |")

    lines += [
        "",
        "## Safety (vs expert ground truth)",
        "",
        f"- Unsafe responses in the set: **{safety['unsafe_items']}**",
        f"- Caught by the judge: **{safety['judge_caught']}** · caught by the human majority: **{safety['humans_caught']}**",
        f"- Judge false alarms on safe responses: **{safety['judge_false_alarms']}**",
        "",
        "## Issue detection (judge's primary issue vs planted issue)",
        "",
        f"Overall: **{s['issue_detection_accuracy']:.0%}** of items get the right primary issue.",
        "",
        "| Issue | Items | Judge recall |",
        "|---|---:|---:|",
    ]
    for r in result["issue_detection"]:
        lines.append(f"| {r['issue']} | {r['items']} | {r['judge_recall']:.0%} |")

    bias = s["length_bias_spearman"]
    lines += [
        "",
        "## Bias check: does the judge reward long answers?",
        "",
        f"Spearman correlation between response length and (judge − human) helpfulness: **{bias:+.2f}**. "
        + ("Positive: the judge over-rewards longer responses - add a length-neutral instruction or penalize padding." if bias > 0.2
           else "No meaningful length bias."),
        "",
        f"## Human review queue ({s['review_queue']} of {s['items']} items, {s['review_share']:.0%})",
        "",
        "| Item | Category | Why it needs a human |",
        "|---|---|---|",
    ]
    for q in result["review_queue"]:
        lines.append(f"| {q['id']} | {q['category']} | {'; '.join(q['reasons'])} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
