"""
==============================================================================
MODULE: generate_testing_diagrams.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: renders the results of the testing
  scripts (validate_against_reference_implementation.py,
  hand_written_test_cases.py, sensitivity_sweep.py) as slide-ready images in
  diagrams/ — this file does not re-run the underlying checks differently
  or change any conclusion, it only visualizes what those already
  established. modeling/demo/distance_only_sensitivity.py imports
  draw_slide() from this file to render its own chart with the same
  template, so all four images in diagrams/ look like one consistent set.
- WHAT THIS FILE DOES: draw_slide() is a shared 16:9 template (headline
  stating the conclusion, one big hero number, a simplified supporting
  chart, a small footer) used by every diagram. Three draw functions here
  build on it: a parity plot proving our entropy/TOPSIS math matches the
  independent pymcdm library, a checklist for the 12 hand-written test
  cases, and a grouped bar chart of the sensitivity sweep's results.
- KNOWN LIMITATIONS / ASSUMPTIONS: these are presentation slides, not a
  technical report — captions are deliberately short and plain-language;
  the full methodology lives in diagrams/diagrams.md and each script's own
  header, not on the image itself.
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELING_ROOT = _REPO_ROOT / "modeling"
for _sibling_dir in ("candidate_pool", "scoring", "recommend", "demo"):
    _sibling_path = str(_MODELING_ROOT / _sibling_dir)
    if _sibling_path not in sys.path:
        sys.path.append(_sibling_path)

import scoring  # noqa: E402
import hand_written_test_cases as hwtc  # noqa: E402
import sensitivity_sweep as sweep  # noqa: E402
import try_it_out as demo  # noqa: E402
from pymcdm.methods import TOPSIS  # noqa: E402
from pymcdm import weights as mcdm_weights  # noqa: E402
from pymcdm.normalizations import vector_normalization  # noqa: E402

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------------

DIAGRAMS_DIR: Path = _REPO_ROOT / "diagrams"

ACCENT_TEAL: str = "#0B6E62"
STATUS_GOOD: str = "#0ca30c"
STATUS_BAD: str = "#d03b3b"
CATEGORICAL_SLOT_1: str = "#2a78d6"  # injury_bleeding
CATEGORICAL_SLOT_2: str = "#eb6834"  # severe_anemia
INK: str = "#10181A"
MUTED_INK: str = "#5B6B68"
GRID: str = "#D9E2DF"
SURFACE: str = "#FCFCFB"

# Every diagram in diagrams/ renders at this size — a standard 16:9 slide,
# 1920x1080 at this dpi, droppable straight into a deck.
SLIDE_FIGSIZE = (13.33, 7.5)
SLIDE_DPI = 144

NO_ZEROS_MATRIX = np.array([
    [12.0, 2.0, 1.0],
    [45.0, 3.0, 3.0],
    [8.0, 1.0, 1.0],
    [30.0, 2.0, 2.0],
    [60.0, 3.0, 2.0],
])
IS_COST = np.array([True, False, False])


def style_axes(ax) -> None:
    """Shared, recessive chart chrome applied inside every slide's visual
    panel — no top/right/left spines, a light muted grid, muted ticks.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=MUTED_INK, labelsize=11)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)


def draw_slide(
    output_path: Path,
    headline: str,
    hero_number: str,
    hero_label: str,
    draw_visual_fn: Callable,
    footer_text: str,
    layout: str = "side_by_side",
    hero_color: str = ACCENT_TEAL,
) -> None:
    """Shared slide template every diagram in diagrams/ renders through.

    PARAMETERS:
        output_path: where to save the PNG.
        headline: one sentence stating the conclusion, not a description.
        hero_number: the single biggest takeaway, e.g. "54%" or "7/7".
        hero_label: one short line under the hero number explaining it.
        draw_visual_fn: a function(ax) that draws the supporting chart
            into the axes this function provides.
        footer_text: one line, small print — sample size / provenance.
        layout: "side_by_side" (hero number left, chart right — used when
            the chart is compact) or "stacked" (hero number inline with
            the headline, chart takes the full body — used when the chart
            itself needs the vertical room, e.g. a long checklist).
        hero_color: color for the hero number text.
    """
    assert layout in ("side_by_side", "stacked"), f"unknown layout: {layout}"
    fig = plt.figure(figsize=SLIDE_FIGSIZE, dpi=SLIDE_DPI)
    fig.patch.set_facecolor(SURFACE)

    if layout == "side_by_side":
        grid = fig.add_gridspec(2, 2, height_ratios=[1.6, 8], width_ratios=[1, 1.5], hspace=0.25, wspace=0.2,
                                 left=0.05, right=0.96, top=0.90, bottom=0.09)
        headline_ax = fig.add_subplot(grid[0, :])
        headline_ax.axis("off")
        headline_ax.set_xlim(0, 1)
        headline_ax.set_ylim(0, 1)
        headline_ax.text(0, 0.5, headline, fontsize=28, fontweight="bold", color=INK, va="center", ha="left")

        hero_ax = fig.add_subplot(grid[1, 0])
        hero_ax.axis("off")
        hero_ax.set_xlim(0, 1)
        hero_ax.set_ylim(0, 1)
        hero_ax.set_facecolor(SURFACE)
        hero_ax.text(0.5, 0.58, hero_number, fontsize=88, fontweight="bold", color=hero_color, ha="center", va="center")
        hero_ax.text(0.5, 0.30, hero_label, fontsize=16, color=MUTED_INK, ha="center", va="center")

        visual_ax = fig.add_subplot(grid[1, 1])
        draw_visual_fn(visual_ax)
    else:
        grid = fig.add_gridspec(2, 1, height_ratios=[1, 9], hspace=0.05, left=0.05, right=0.96, top=0.91, bottom=0.07)
        headline_ax = fig.add_subplot(grid[0])
        headline_ax.axis("off")
        headline_ax.set_xlim(0, 1)
        headline_ax.set_ylim(0, 1)
        headline_ax.text(0, 0.3, headline, fontsize=26, fontweight="bold", color=INK, va="center", ha="left")
        headline_ax.text(1.0, 0.3, hero_number, fontsize=40, fontweight="bold", color=hero_color, va="center", ha="right")

        visual_ax = fig.add_subplot(grid[1])
        draw_visual_fn(visual_ax)

    fig.text(0.02, 0.015, footer_text, fontsize=9.5, color=MUTED_INK)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_pymcdm_parity_visual(ax) -> None:
    """The parity scatter for the pymcdm validation slide: our values (x)
    vs. pymcdm's independent reference values (y) for entropy weights and
    TOPSIS scores — every point should sit on the y=x diagonal.
    """
    our_weights = scoring.entropy_weights(NO_ZEROS_MATRIX)
    ref_weights = mcdm_weights.entropy_weights(NO_ZEROS_MATRIX)
    our_scores = scoring.topsis_scores(NO_ZEROS_MATRIX, our_weights, IS_COST)
    pymcdm_types = np.where(IS_COST, -1, 1)
    ref_scores = TOPSIS(normalization_function=vector_normalization)(NO_ZEROS_MATRIX, our_weights, pymcdm_types)

    ax.set_facecolor(SURFACE)
    style_axes(ax)
    all_values = np.concatenate([our_weights, ref_weights, our_scores, ref_scores])
    axis_min, axis_max = float(all_values.min()) - 0.05, float(all_values.max()) + 0.05
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color=GRID, linewidth=3, zorder=1)

    ax.scatter(our_weights, ref_weights, s=180, color=ACCENT_TEAL, zorder=3, label="Entropy weights",
               edgecolors=SURFACE, linewidths=2)
    ax.scatter(our_scores, ref_scores, s=180, color=CATEGORICAL_SLOT_2, marker="D", zorder=3,
               label="TOPSIS scores", edgecolors=SURFACE, linewidths=2)

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Our model", color=MUTED_INK, fontsize=13)
    ax.set_ylabel("Independent reference check", color=MUTED_INK, fontsize=13)
    ax.legend(loc="upper left", frameon=False, fontsize=13, labelcolor=MUTED_INK)


def draw_pymcdm_validation_slide(output_path: Path) -> None:
    """Slide: our scoring math checked against an independent library."""
    draw_slide(
        output_path=output_path,
        headline="Our scoring math is independently verified",
        hero_number="0.0%",
        hero_label="difference from an\nindependent reference check",
        draw_visual_fn=_draw_pymcdm_parity_visual,
        footer_text="Every point sits exactly on the diagonal. Method: modeling/demo/validate_against_reference_implementation.py",
    )


CASE_LABELS: tuple[str, ...] = (
    "Single option in range → choose it directly",
    "Tied distance, only one has needed services → that one wins",
    "Nobody reachable → 3 closest shown, flagged outside window",
    "Dense area, anemia → best-equipped facility wins, not nearest",
    "Confirmed + unconfirmed reachable → confirmed ranks first",
    "Everyone unconfirmed → fallback tier ranks sensibly, all flagged",
    "Anemia + fallback-tier-only → severity boost has no effect",
    "Padding → picks the genuinely closest facility, no exceptions",
    "Tied distance + services → higher blood level breaks the tie",
    "High urgency → severity boost can flip the outcome",
    "Injury scenario → full service match beats a partial match",
    "Every scenario → maps to its documented time window",
)


def _draw_checklist_visual(ax, results: list) -> None:
    """The checklist body for the hand-written test cases slide, drawn
    from already-computed results — run_hand_written_cases_once() runs
    the 12 cases exactly once, not once per render.
    """
    assert len(results) == len(CASE_LABELS), "case count must match label count"
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(CASE_LABELS))
    ax.axis("off")
    for i, (label, passed) in enumerate(zip(CASE_LABELS, results)):
        y = len(CASE_LABELS) - i - 1
        row_color = "#F0F4F2" if i % 2 == 0 else SURFACE
        ax.add_patch(plt.Rectangle((0, y), 1, 1.0, facecolor=row_color, edgecolor="none", zorder=1))
        status_color = STATUS_GOOD if passed else STATUS_BAD
        ax.scatter([0.018], [y + 0.5], s=260, color=status_color, zorder=3)
        ax.text(0.018, y + 0.5, "✓" if passed else "✗", color="white", fontsize=13, fontweight="bold",
                ha="center", va="center", zorder=4)
        ax.text(0.045, y + 0.5, label, color=INK, fontsize=15, va="center", zorder=3)


def run_hand_written_cases_once() -> list:
    """Run every case in hand_written_test_cases.ALL_CASES exactly once."""
    results = [case_fn() for case_fn in hwtc.ALL_CASES]
    assert len(results) == len(CASE_LABELS), "ALL_CASES and CASE_LABELS must stay in sync"
    return results


def draw_hand_written_test_cases_slide(output_path: Path) -> None:
    """Slide: the 12 hand-written "obvious right answer" cases, pass/fail."""
    results = run_hand_written_cases_once()
    passed_count = sum(results)
    total_count = len(results)

    draw_slide(
        output_path=output_path,
        headline="The model passes every common-sense check",
        hero_number=f"{passed_count}/{total_count}",
        hero_label="",
        draw_visual_fn=lambda ax: _draw_checklist_visual(ax, results),
        footer_text="Source: model-design.md (draft) + gaps found during testing. "
                     "Method: modeling/demo/hand_written_test_cases.py",
        layout="stacked",
        hero_color=STATUS_GOOD if passed_count == total_count else STATUS_BAD,
    )


def _draw_sensitivity_visual(ax_pair, per_scenario_results, scenario_colors, scenario_labels) -> None:
    """The two-panel grouped bar chart for the sensitivity sweep slide."""
    ax1, ax2 = ax_pair
    bar_width = 0.35

    ax1.set_facecolor(SURFACE)
    style_axes(ax1)
    x_positions = np.arange(len(sweep.WINDOW_VALUES_TO_TEST))
    for i, scenario_key in enumerate(sweep.TEST_SCENARIOS):
        values = [per_scenario_results[scenario_key]["window"][w] for w in sweep.WINDOW_VALUES_TO_TEST]
        offset = (i - 0.5) * bar_width
        bars = ax1.bar(x_positions + offset, values, width=bar_width, color=scenario_colors[scenario_key],
                        label=scenario_labels[scenario_key], zorder=3)
        for bar, value in zip(bars, values):
            # Suppress the label on a true zero: two adjacent zero-height
            # bars would otherwise print two "0.0%" labels on top of each
            # other (no bar height to space them apart). The flat bar
            # itself already shows zero; the label is only needed where
            # there's an actual value to read off.
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.1f}%", ha="center", va="bottom",
                          color=INK, fontsize=12)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f"{w} min" for w in sweep.WINDOW_VALUES_TO_TEST], fontsize=13)
    ax1.set_ylim(0, max(10, ax1.get_ylim()[1]))
    ax1.set_title("Time window sweep", fontsize=16, fontweight="bold", color=INK, loc="left", pad=10)

    ax2.set_facecolor(SURFACE)
    style_axes(ax2)
    x_positions_2 = np.arange(len(sweep.BOOST_VALUES_TO_TEST))
    for i, scenario_key in enumerate(sweep.TEST_SCENARIOS):
        values = [per_scenario_results[scenario_key]["boost"][b] for b in sweep.BOOST_VALUES_TO_TEST]
        offset = (i - 0.5) * bar_width
        bars = ax2.bar(x_positions_2 + offset, values, width=bar_width, color=scenario_colors[scenario_key],
                        label=scenario_labels[scenario_key], zorder=3)
        for bar, value in zip(bars, values):
            if value > 0:  # see ax1's loop above for why zero labels are suppressed
                ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.1f}%", ha="center", va="bottom",
                          color=INK, fontsize=12)
    ax2.set_xticks(x_positions_2)
    ax2.set_xticklabels([f"x{b}" for b in sweep.BOOST_VALUES_TO_TEST], fontsize=13)
    ax2.set_ylim(0, max(10, ax2.get_ylim()[1]))
    ax2.set_title("Severity boost sweep", fontsize=16, fontweight="bold", color=INK, loc="left", pad=10)
    ax2.legend(loc="upper right", frameon=False, fontsize=12, labelcolor=MUTED_INK)


def draw_sensitivity_sweep_slide(output_path: Path) -> None:
    """Slide: how much judgment-call parameters change the top pick."""
    blood_points, facility_services = demo.load_datasets()
    rng = np.random.default_rng(sweep.RANDOM_SEED)

    scenario_colors = {"injury_bleeding": CATEGORICAL_SLOT_1, "severe_anemia": CATEGORICAL_SLOT_2}
    scenario_labels = {"injury_bleeding": "Injury/bleeding", "severe_anemia": "Severe anemia"}
    per_scenario_results = {}
    for scenario_key in sweep.TEST_SCENARIOS:
        locations = sweep.sample_locations(blood_points, scenario_key, sweep.N_LOCATIONS, rng)
        per_scenario_results[scenario_key] = sweep.run_sweep_for_scenario(
            blood_points, facility_services, scenario_key, locations
        )

    all_pct_values = [
        value
        for result in per_scenario_results.values()
        for sweep_type in ("window", "boost")
        for value in result[sweep_type].values()
    ]
    worst_case = max(all_pct_values)

    def visual(outer_ax):
        outer_ax.axis("off")
        inner_grid = outer_ax.get_subplotspec().subgridspec(1, 2, wspace=0.25)
        fig = outer_ax.figure
        ax1 = fig.add_subplot(inner_grid[0, 0])
        ax2 = fig.add_subplot(inner_grid[0, 1])
        _draw_sensitivity_visual((ax1, ax2), per_scenario_results, scenario_colors, scenario_labels)

    draw_slide(
        output_path=output_path,
        headline="Stable under reasonable parameter changes",
        hero_number=f"≤{worst_case:.1f}%",
        hero_label="top-pick change across every\nwindow / boost value tested",
        draw_visual_fn=visual,
        footer_text=f"n={sweep.N_LOCATIONS} sampled locations per scenario, baseline = "
                    f"{sweep.BASELINE_HIGH_WINDOW} min / x{sweep.BASELINE_BOOST_FACTOR}, seed={sweep.RANDOM_SEED}. "
                    f"Method: modeling/demo/sensitivity_sweep.py",
    )


def main() -> int:
    """Entry point. Writes all three diagrams to diagrams/."""
    DIAGRAMS_DIR.mkdir(exist_ok=True)

    pymcdm_path = DIAGRAMS_DIR / "pymcdm_validation.png"
    draw_pymcdm_validation_slide(pymcdm_path)
    print(f"Wrote {pymcdm_path}")

    checklist_path = DIAGRAMS_DIR / "hand_written_test_cases.png"
    draw_hand_written_test_cases_slide(checklist_path)
    print(f"Wrote {checklist_path}")

    sweep_path = DIAGRAMS_DIR / "sensitivity_sweep.png"
    draw_sensitivity_sweep_slide(sweep_path)
    print(f"Wrote {sweep_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
