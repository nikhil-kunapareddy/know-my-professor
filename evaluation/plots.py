"""Figures for a stored evaluation run.

    python -m evaluation.plots evaluation/results/<run-dir>

Reads the JSON a run wrote (``report.json``, or ``data.json`` for the two probe
experiments) and writes PNGs beside it. Static images on purpose: these live in
git next to the numbers they came from and have to render in a diff view, on
GitHub, and in an editor, where no chart runtime exists. The interactive layer
is ``report.json`` itself — every score carries the judge's own reason.

Colour follows one rule set, applied in this order: the form comes from the
data's job, the palette is assigned by that job (categorical for identity,
sequential for magnitude), and the values are the validated defaults — checked
with the palette validator rather than eyeballed. Two series is the most any
figure here needs, and blue/orange clears every gate including contrast.

One polarity note that the figures carry explicitly: a metric where lower is
better is marked with a down arrow wherever it appears beside the others.
``INVERTED`` still lists ``noise_sensitivity`` even though the current metric
table has no inverted metric — this module renders *stored* reports, and the
runs recorded before the move from Ragas to DeepEval contain it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in this environment; write files only
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# --- design tokens ---------------------------------------------------------
# Light surface only. A PNG carries its own background, so it stays legible on a
# dark page; a dark variant would be a second artifact with no new information.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8981"
GRID = "#e6e5e2"

#: Categorical slots 1 and 2. Validated all-pairs, both modes: CVD ΔE 24.7,
#: normal-vision ΔE 33.6, both >= 3:1 on this surface.
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

#: Slots 1-4 for a model comparison. Validated on the adjacent pairlist (worst
#: CVD ΔE 9.1, normal-vision 22.9); slots 3 and 4 sit under 3:1 on this surface,
#: so every bar carries a visible value label — the documented relief.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

#: Sequential blue, light -> dark, for magnitude.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
             "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
#: Neutral for "not measured" — a skip is not a zero, and must not read as one.
NEUTRAL = "#f0efec"

STAGE_LABEL = {"retrieval": "RETRIEVAL", "generation": "GENERATION", "end_to_end": "END TO END"}
#: Kept deliberately broader than the live metric table: see the module
#: docstring. A metric named here that a report does not contain costs nothing.
INVERTED = {"noise_sensitivity"}


def _style(ax, *, grid_axis: str | None = "x") -> None:
    """Hairline solid grid, no box, recessive axes."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-", zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)


def _figure(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path, title: str, subtitle: str = "") -> Path:
    # Offsets in inches, converted to figure fractions. These figures grow with
    # the number of cases, so a fixed fraction would put the subtitle on top of
    # the title in the tall ones.
    height = fig.get_figheight()
    fig.text(0.01, 1 - 0.30 / height, title, ha="left", va="center",
             fontsize=12, color=INK, weight="medium")
    if subtitle:
        fig.text(0.01, 1 - 0.58 / height, subtitle, ha="left", va="center",
                 fontsize=8.5, color=INK_SECONDARY)
    # Headroom is reserved in inches for the same reason the offsets above are.
    # Matplotlib's default top is a *fraction* (0.88), so a short figure puts
    # the axes — and whatever is drawn at the top of it — straight through the
    # subtitle. Dropping four metrics made every figure short enough to show it.
    reserved = 0.95 if subtitle else 0.70
    fig.subplots_adjust(top=min(fig.subplotpars.top, 1 - reserved / height))
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _polarity_note(report: dict) -> str:
    """The "lower is better" sentence, but only when it is true of this report.

    ``INVERTED`` is deliberately broader than the live metric table (it still
    names metrics only older stored runs contain), so the caption has to be
    decided per report rather than per module. A legend for a mark that appears
    nowhere in the figure is just a thing to puzzle over.
    """
    metrics = report.get("metrics", {})
    inverted = any(s.get("lower_is_better") or name in INVERTED
                   for name, s in metrics.items())
    return " ↓ = lower is better." if inverted else ""


def _legend_below(ax, handles, pad_inches: float = 0.62) -> None:
    """Put the legend under the plot.

    A legend inside the axes lands on the marks — in the dumbbell figures it
    covered the last row's value labels, which is the one row a reader checks.
    """
    offset = pad_inches / ax.figure.get_figheight()
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0, -offset),
        ncol=len(handles), frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY,
        handletextpad=0.5, columnspacing=1.6, borderaxespad=0,
    )


def _label(name: str) -> str:
    return f"{name} ↓" if name in INVERTED else name


# --- figure 1: every score at once ----------------------------------------


def heatmap(report: dict, out: Path) -> Path:
    """Case × metric magnitude, with skips held out as a neutral.

    The cell values double as the table view, so identity and magnitude are
    never carried by colour alone.
    """
    metrics = list(report["metrics"])
    cases = list(dict.fromkeys(s["case_id"] for s in report["scores"]))
    lookup = {(s["case_id"], s["metric"]): s for s in report["scores"]}

    fig, ax = _figure(1.05 * len(metrics) + 3.2, 0.42 * len(cases) + 2.2)
    _style(ax, grid_axis=None)

    for row, case in enumerate(cases):
        for col, metric in enumerate(metrics):
            score = lookup.get((case, metric))
            value = score and score["value"]
            if value is None:
                colour, text, ink = NEUTRAL, "—", INK_MUTED
            else:
                # Sequential position, then ink that stays readable on it.
                colour = BLUE_RAMP[min(int(value * len(BLUE_RAMP)), len(BLUE_RAMP) - 1)]
                text = f"{value:.2f}"
                ink = "#ffffff" if value >= 0.62 else INK
            # 2px surface gap rather than a border around each mark.
            ax.add_patch(plt.Rectangle((col + 0.03, row + 0.03), 0.94, 0.94,
                                       facecolor=colour, edgecolor=SURFACE, linewidth=1.5))
            ax.text(col + 0.5, row + 0.5, text, ha="center", va="center",
                    fontsize=7.5, color=ink)

    ax.set_xlim(0, len(metrics))
    ax.set_ylim(0, len(cases))
    ax.set_xticks([i + 0.5 for i in range(len(metrics))])
    ax.set_xticklabels([_label(m) for m in metrics], rotation=40, ha="right", fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(cases))])
    ax.set_yticklabels(cases, fontsize=8)
    ax.invert_yaxis()

    _legend_below(
        ax, [Patch(facecolor=NEUTRAL, edgecolor=SURFACE, label="not measured (skipped)")],
        pad_inches=1.05,
    )
    return _save(fig, out, "Every score in the run",
                 "Darker is higher." + _polarity_note(report)
                 + " A skipped cell is a missing input, not a zero.")


# --- figure 2: the headline per metric ------------------------------------


def metric_means(report: dict, out: Path) -> Path:
    """Mean per metric, grouped by the stage it probes.

    One series, so one colour: bar length already carries magnitude, and a ramp
    here would burn the colour channel on information the chart shows twice.
    """
    summaries = report["metrics"]
    ordered = [
        (stage, name, s) for stage in STAGE_LABEL
        for name, s in summaries.items() if s["stage"] == stage
    ]

    fig, ax = _figure(7.8, 0.42 * len(ordered) + 2.0)
    _style(ax)

    labels, positions = [], []
    y = 0.0
    last_stage = None
    for stage, name, summary in ordered:
        if stage != last_stage:
            if last_stage is not None:
                y += 0.7  # a gap between stages, not a divider line
            ax.text(-0.075, y - 0.05, STAGE_LABEL[stage], fontsize=7.5,
                    color=INK_MUTED, weight="medium", va="center")
            y += 0.55
            last_stage = stage
        mean = summary["mean"]
        if mean is not None:
            ax.barh(y, mean, height=0.62, color=SERIES_1, zorder=2)
            ax.text(mean + 0.012, y, f"{mean:.3f}   n={summary['scored']}",
                    va="center", fontsize=8, color=INK_SECONDARY)
        else:
            ax.text(0.012, y, f"not measured   skipped={summary['skipped']}",
                    va="center", fontsize=8, color=INK_MUTED)
        labels.append(_label(name))
        positions.append(y)
        y += 1.0

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.invert_yaxis()
    return _save(fig, out, "Mean score by metric",
                 "n is how many cases produced a number." + _polarity_note(report))


# --- figures 3 and 4: two metrics that should agree, and do not -----------


def paired_metrics(report: dict, left: str, right: str, out: Path,
                   title: str, subtitle: str) -> Path:
    """One case per row, both metrics on a shared scale, joined by a rule.

    Two series, so a legend is present and both ends are direct-labelled: the
    gap between the two dots IS the finding, and it has to be readable without
    relying on hue.
    """
    lookup = {(s["case_id"], s["metric"]): s["value"] for s in report["scores"]}
    cases = [c for c in dict.fromkeys(s["case_id"] for s in report["scores"])
             if lookup.get((c, left)) is not None and lookup.get((c, right)) is not None]
    cases.sort(key=lambda c: lookup[(c, left)])

    fig, ax = _figure(8.2, 0.46 * len(cases) + 2.0)
    _style(ax)

    for row, case in enumerate(cases):
        a, b = lookup[(case, left)], lookup[(case, right)]
        ax.plot([a, b], [row, row], color=GRID, linewidth=1.6, zorder=1,
                solid_capstyle="round")
        # 2px surface ring so overlapping markers stay separable.
        ax.plot(a, row, "o", markersize=9, color=SERIES_1, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.6)
        ax.plot(b, row, "o", markersize=9, color=SERIES_2, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.6)
        low, high = (a, b) if a <= b else (b, a)
        ax.text(low - 0.02, row, f"{low:.2f}", ha="right", va="center",
                fontsize=7.5, color=INK_SECONDARY)
        ax.text(high + 0.02, row, f"{high:.2f}", ha="left", va="center",
                fontsize=7.5, color=INK_SECONDARY)

    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(cases, fontsize=8.5)
    ax.set_xlim(-0.12, 1.14)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.invert_yaxis()
    _legend_below(ax, [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   color=SERIES_1, label=_label(left)),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   color=SERIES_2, label=_label(right)),
    ])
    return _save(fig, out, title, subtitle)


# --- figure 5: the score floor --------------------------------------------


def score_floor(data: dict, out: Path, floor: float = 0.35) -> Path:
    """Both populations on one axis — the overlap is the whole point.

    Two series (answerable vs out-of-scope), so a legend is present; the marks
    are jittered dots rather than a box plot because n is small enough that
    every case should be visible.
    """
    answerable = [r["first_correct_score"] for r in data["answerable"]
                  if r.get("first_correct_score") is not None]
    out_of_scope = [r["top_score"] for r in data["out_of_scope"]]

    fig, ax = _figure(8.6, 3.4)
    _style(ax, grid_axis="x")

    rng = random.Random(0)
    for values, y, colour, label in (
        (answerable, 1.0, SERIES_1, f"answerable — first correct chunk (n={len(answerable)})"),
        (out_of_scope, 0.0, SERIES_2, f"out of scope — top chunk (n={len(out_of_scope)})"),
    ):
        ax.plot(values, [y + rng.uniform(-0.28, 0.28) for _ in values], "o",
                markersize=9, color=colour, alpha=0.75, markeredgecolor=SURFACE,
                markeredgewidth=1.4, zorder=3, label=label)
        lo, hi = min(values), max(values)
        ax.plot([lo, hi], [y + 0.34, y + 0.34], color=colour, linewidth=2,
                solid_capstyle="round", zorder=2)
        ax.text((lo + hi) / 2, y + 0.46, f"{lo:.3f} – {hi:.3f}", ha="center",
                fontsize=8, color=INK_SECONDARY)

    ax.axvline(floor, color=INK_MUTED, linewidth=1.4, zorder=1)
    # The label sits in the gap the finding itself opens up: everything between
    # the floor and 0.70 is empty, which is the point being made.
    ax.text(floor + 0.012, 0.5, f"MIN_RETRIEVAL_SCORE = {floor}\nnothing ever scores this low",
            fontsize=8.5, color=INK_SECONDARY, ha="left", va="center", linespacing=1.6)

    ax.set_yticks([])
    ax.set_ylim(-0.55, 1.75)
    ax.set_xlim(0.2, 0.92)
    ax.set_xlabel("cosine similarity", fontsize=8.5, color=INK_SECONDARY)
    _legend_below(ax, ax.get_legend_handles_labels()[0])
    return _save(fig, out, "No threshold separates in-scope from out-of-scope",
                 f"{data['meta']['embed_model']} on {data['meta']['index_name']}. "
                 "The two ranges overlap, so a cosine floor cannot decide when to refuse.")


# --- figure 6: what a judged case costs -----------------------------------


def judge_calls(data: dict, out: Path) -> Path:
    """Calls per metric at two values of top_k — which metrics scale, and how."""
    metrics = [m["metric"] for m in data["metrics"]]
    small = [m["judge_calls_3_chunks"] for m in data["metrics"]]
    large = [m["judge_calls_8_chunks"] for m in data["metrics"]]

    fig, ax = _figure(8.0, 0.52 * len(metrics) + 2.0)
    _style(ax)

    height = 0.33
    for row, (a, b) in enumerate(zip(small, large, strict=True)):
        # 2px surface gap between the adjacent bars of a pair.
        ax.barh(row - height / 2 - 0.05, a, height=height, color=SERIES_1, zorder=2)
        ax.barh(row + height / 2 + 0.05, b, height=height, color=SERIES_2, zorder=2)
        ax.text(b + 0.35, row + height / 2 + 0.05, str(b), va="center",
                fontsize=8, color=INK_SECONDARY)
        if b != a:
            ax.text(a + 0.35, row - height / 2 - 0.05, str(a), va="center",
                    fontsize=8, color=INK_MUTED)

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([_label(m) for m in metrics], fontsize=8.5)
    ax.set_xlabel("judge calls for one case", fontsize=8.5, color=INK_SECONDARY)
    ax.invert_yaxis()
    _legend_below(ax, [
        Patch(facecolor=SERIES_1, label="top_k = 3"),
        Patch(facecolor=SERIES_2, label="top_k = 8"),
    ])
    return _save(
        fig, out, "Judge calls per case, and what scales with top_k",
        f"{sum(small)} calls at 3 chunks, {sum(large)} at 8. The three metrics that "
        "grow ask the judge about each retrieved chunk separately.",
    )


# --- comparing several runs (model selection) -----------------------------


def load_runs(root: Path) -> list[dict]:
    """Every subdirectory of ``root`` that holds a finished run, name-ordered."""
    runs = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "plots"):
        report = directory / "report.json"
        if not report.exists():
            continue
        samples_path = directory / "samples.jsonl"
        samples = []
        if samples_path.exists():
            for line in samples_path.read_text().splitlines():
                row = json.loads(line)
                if "_run" not in row:
                    samples.append(row)
        runs.append({"label": directory.name, "report": json.loads(report.read_text()),
                     "samples": samples})
    return runs


def _generate_ms(run: dict) -> list[float]:
    return [s["timings_ms"]["generate"] for s in run["samples"]
            if s.get("timings_ms", {}).get("generate")]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def model_quality(runs: list[dict], out: Path) -> Path:
    """Every metric, every model. Grouped bars, one hue per model."""
    metrics = list(runs[0]["report"]["metrics"])
    fig, ax = _figure(9.0, 0.62 * len(metrics) + 2.4)
    _style(ax)

    span = 0.78
    bar = span / len(runs)
    for index, run in enumerate(runs):
        offset = -span / 2 + bar * (index + 0.5)
        for row, metric in enumerate(metrics):
            mean = run["report"]["metrics"].get(metric, {}).get("mean")
            if mean is None:
                continue
            # 2px surface gap between the bars of a group.
            ax.barh(row + offset, mean, height=bar * 0.86, color=SERIES[index % len(SERIES)],
                    zorder=2)
            ax.text(mean + 0.008, row + offset, f"{mean:.2f}", va="center",
                    fontsize=7, color=INK_SECONDARY)

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([_label(m) for m in metrics], fontsize=8.5)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.invert_yaxis()
    _legend_below(ax, [Patch(facecolor=SERIES[i % len(SERIES)], label=r["label"])
                       for i, r in enumerate(runs)], pad_inches=0.78)
    return _save(fig, out, "Answer quality by model",
                 "Same 10 cases, same retrieved context, same prompt — only the "
                 "generator differs. ↓ = lower is better.")


def model_latency(runs: list[dict], out: Path) -> Path:
    """Generate-stage latency: every case as a dot, the median as a rule.

    The generate stage only — embed and retrieve are identical across models, so
    including them would dilute the difference being measured.
    """
    fig, ax = _figure(8.6, 0.85 * len(runs) + 2.4)
    _style(ax, grid_axis="x")

    rng = random.Random(0)
    for index, run in enumerate(runs):
        values = _generate_ms(run)
        if not values:
            continue
        y = len(runs) - index - 1
        colour = SERIES[index % len(SERIES)]
        ax.plot(values, [y + rng.uniform(-0.17, 0.17) for _ in values], "o",
                markersize=9, color=colour, alpha=0.75, markeredgecolor=SURFACE,
                markeredgewidth=1.4, zorder=3)
        median = _median(values)
        ax.plot([median, median], [y - 0.32, y + 0.32], color=INK, linewidth=2, zorder=4)
        ax.text(max(values) + 250, y, f"median {median / 1000:.1f}s", va="center",
                fontsize=8.5, color=INK_SECONDARY)

    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([r["label"] for r in reversed(runs)], fontsize=8.5)
    ax.set_xlabel("generate stage, milliseconds", fontsize=8.5, color=INK_SECONDARY)
    return _save(fig, out, "Answer latency by model",
                 "One dot per case; the vertical rule is the median. Single "
                 "measurements from one machine, so read the ranking, not the values.")


def quality_vs_latency(runs: list[dict], out: Path, metric: str = "faithfulness") -> Path:
    """The decision plot: is the slower model buying anything?

    Identity is carried by a direct label rather than by hue, which is what lets
    this stay readable past the three-slot all-pairs cap for scatter forms.
    """
    fig, ax = _figure(8.0, 5.0)
    _style(ax, grid_axis="both")

    for index, run in enumerate(runs):
        values = _generate_ms(run)
        mean = run["report"]["metrics"].get(metric, {}).get("mean")
        if not values or mean is None:
            continue
        x = _median(values) / 1000
        ax.plot(x, mean, "o", markersize=13, color=SERIES[index % len(SERIES)],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.annotate(run["label"], (x, mean), textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=8.5, color=INK)

    ax.set_xlabel("median generate latency (seconds)", fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel(metric, fontsize=8.5, color=INK_SECONDARY)
    ax.margins(0.22)
    return _save(fig, out, f"{metric} against latency",
                 "Up and to the left is better. Points are labelled directly, so "
                 "identity never depends on colour.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="A directory under evaluation/results/")
    args = parser.parse_args()

    plots = args.run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    print(f"plotting {args.run_dir}")

    report_path = args.run_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        heatmap(report, plots / "01-all-scores.png")
        metric_means(report, plots / "02-metric-means.png")
        paired_metrics(
            report, "context_recall", "faithfulness",
            plots / "03-recall-vs-faithfulness.png",
            "Whether a weak answer is the retriever's fault or the generator's",
            "Both low is a retrieval problem; faithfulness alone low is the "
            "prompt or the model. This is the split the stage framework exists for.",
        )
        paired_metrics(
            report, "answer_correctness", "semantic_similarity",
            plots / "04-correctness-vs-similarity.png",
            "The answers say the right thing; the references are shorter",
            "Semantic similarity is high while claim-level correctness is not — the "
            "answers name more true people than the reference does.",
        )

    runs = load_runs(args.run_dir) if args.run_dir.is_dir() else []
    if len(runs) > 1:
        model_quality(runs, plots / "01-quality-by-model.png")
        model_latency(runs, plots / "02-latency-by-model.png")
        quality_vs_latency(runs, plots / "03-quality-vs-latency.png")

    data_path = args.run_dir / "data.json"
    if data_path.exists():
        data = json.loads(data_path.read_text())
        if "out_of_scope" in data:
            score_floor(data, plots / "01-score-floor-overlap.png")
        if "metrics" in data and data["metrics"] and "judge_calls_8_chunks" in data["metrics"][0]:
            judge_calls(data, plots / "01-judge-calls.png")


if __name__ == "__main__":
    main()
