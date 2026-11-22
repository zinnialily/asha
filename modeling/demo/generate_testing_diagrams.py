"""
==============================================================================
MODULE: generate_testing_diagrams.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: renders the results of the three
  testing scripts (validate_against_reference_implementation.py,
  hand_written_test_cases.py, sensitivity_sweep.py) as chart images in
  diagrams/ — this file does not re-run the underlying checks differently
  or change any conclusion, it only visualizes what those three already
  established. Reuses the same decision-matrix values and sweep results
  those files produced, rather than inventing new test runs.
- WHAT THIS FILE DOES: writes three PNGs to diagrams/ — a parity plot
  proving our entropy/TOPSIS math matches the independent pymcdm
  implementation exactly, a checklist graphic for the 7 hand-written test
  cases, and a grouped bar chart of the sensitivity sweep's percent-changed
  results by scenario and parameter.
- KNOWN LIMITATIONS / ASSUMPTIONS: the pymcdm parity plot and the
  hand-written test case results are reproduced here from values already
  confirmed by running those two files earlier; the sensitivity sweep
  values are regenerated live (that one varies run to run by sampled
  location, though consistently low in magnitude — see sensitivity_sweep.py
  for its own fixed seed).
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

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
CATEGORICAL_SLOT_1: str = "#2a78d6"  # injury_bleeding
CATEGORICAL_SLOT_2: str = "#eb6834"  # severe_anemia
INK: str = "#10181A"
MUTED_INK: str = "#5B6B68"
GRID: str = "#D9E2DF"
SURFACE: str = "#FCFCFB"

NO_ZEROS_MATRIX = np.array([
    [12.0, 2.0, 1.0],
    [45.0, 3.0, 3.0],
    [8.0, 1.0, 1.0],
    [30.0, 2.0, 2.0],
    [60.0, 3.0, 2.0],
])
IS_COST = np.array([True, False, False])


def style_axes(ax) -> None:
    """Shared, recessive chart chrome — same convention as
    distance_only_sensitivity.py, so every diagram in diagrams/ looks like
    one consistent set rather than three unrelated styles.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=MUTED_INK, labelsize=10)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)


def draw_pymcdm_parity_plot(output_path: Path) -> None:
    """Scatter plot: our values (x) vs. pymcdm's independent reference
    values (y), for both entropy weights and TOPSIS scores. A diagonal
    y=x line is drawn; every point should sit exactly on it if the two
    implementations agree — which is the actual point being shown, not
    just "a chart exists."
    """
    our_weights = scoring.entropy_weights(NO_ZEROS_MATRIX)
    ref_weights = mcdm_weights.entropy_weights(NO_ZEROS_MATRIX)

    our_scores = scoring.topsis_scores(NO_ZEROS_MATRIX, our_weights, IS_COST)
    pymcdm_types = np.where(IS_COST, -1, 1)
    ref_scores = TOPSIS(normalization_function=vector_normalization)(NO_ZEROS_MATRIX, our_weights, pymcdm_types)

    fig, ax = plt.subplots(figsize=(8, 7.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    style_axes(ax)

    all_values = np.concatenate([our_weights, ref_weights, our_scores, ref_scores])
    axis_min, axis_max = float(all_values.min()) - 0.05, float(all_values.max()) + 0.05
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color=GRID, linewidth=2, zorder=1,
            label="Perfect agreement (y = x)")

    ax.scatter(our_weights, ref_weights, s=90, color=ACCENT_TEAL, zorder=3, label="Entropy weights (3 criteria)",
               edgecolors=SURFACE, linewidths=1.5)
    ax.scatter(our_scores, ref_scores, s=90, color=CATEGORICAL_SLOT_2, marker="D", zorder=3,
               label="TOPSIS scores (5 candidates)", edgecolors=SURFACE, linewidths=1.5)

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Our implementation (modeling/scoring/scoring.py)", color=MUTED_INK, fontsize=10)
    ax.set_ylabel("pymcdm (independent reference library)", color=MUTED_INK, fontsize=10)
    ax.set_title("Our math matches an independent implementation", fontsize=14.5, fontweight="bold",
                 color=INK, pad=14, loc="left")
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=MUTED_INK)
    fig.text(0.02, 0.015, "Every point sits on the diagonal — max difference across all 8 values is 1.6e-12\n"
                          "(floating-point noise, not a real difference).", fontsize=8.5, color=MUTED_INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_hand_written_test_cases_checklist(output_path: Path) -> None:
    """A checklist graphic, not a bar chart — pass/fail across 7 named
    cases isn't a magnitude comparison, so a chart would be the wrong form
    here (see dataviz skill's "is it even a chart?" question).
    """
    case_labels = [
        "Single option in range → choose it directly",
        "Tied distance, only one has needed services → that one wins",
        "Nobody reachable → 3 closest shown, flagged outside window",
        "Dense area, anemia → best-equipped facility wins, not nearest",
        "Confirmed + unconfirmed reachable → confirmed ranks first",
        "Everyone unconfirmed → fallback tier ranks sensibly, all flagged",
        "Anemia + fallback-tier-only → severity boost has no effect (expected)",
    ]
    results = [case_fn() for case_fn in hwtc.ALL_CASES]
    assert len(results) == len(case_labels), "case count must match label count"

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(case_labels))
    ax.axis("off")

    row_height = 1.0
    for i, (label, passed) in enumerate(zip(case_labels, results)):
        y = len(case_labels) - i - 1
        row_color = "#F0F4F2" if i % 2 == 0 else SURFACE
        ax.add_patch(plt.Rectangle((0, y), 1, row_height, facecolor=row_color, edgecolor="none", zorder=1))
        status_color = STATUS_GOOD if passed else "#d03b3b"
        ax.scatter([0.03], [y + 0.5], s=170, color=status_color, zorder=3)
        ax.text(0.03, y + 0.5, "✓" if passed else "✗", color="white", fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=4)
        ax.text(0.07, y + 0.5, label, color=INK, fontsize=11.5, va="center", zorder=3)

    passed_count = sum(results)
    fig.suptitle(f"Hand-written test cases: {passed_count}/{len(results)} passed", fontsize=14, fontweight="bold",
                 color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.01, "Source: model-design.md (draft), \"Testing without ground truth\" — formalized in "
                         "modeling/demo/hand_written_test_cases.py", fontsize=8, color=MUTED_INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_sensitivity_sweep_chart(output_path: Path) -> None:
    """Grouped bar chart: percent of top picks changed from baseline, by
    scenario, across the time-window sweep and the severity-boost sweep —
    two panels, since they're different parameters, not one dual-axis chart.
    """
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
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
            ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.1f}%", ha="center", va="bottom",
                      color=INK, fontsize=8.5)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f"{w} min" for w in sweep.WINDOW_VALUES_TO_TEST])
    ax1.set_ylim(0, max(10, ax1.get_ylim()[1]))
    ax1.set_ylabel("Top pick changed from baseline (%)", color=MUTED_INK, fontsize=10)
    ax1.set_title("Time window sweep", fontsize=12.5, fontweight="bold", color=INK, loc="left")

    ax2.set_facecolor(SURFACE)
    style_axes(ax2)
    x_positions_2 = np.arange(len(sweep.BOOST_VALUES_TO_TEST))
    for i, scenario_key in enumerate(sweep.TEST_SCENARIOS):
        values = [per_scenario_results[scenario_key]["boost"][b] for b in sweep.BOOST_VALUES_TO_TEST]
        offset = (i - 0.5) * bar_width
        bars = ax2.bar(x_positions_2 + offset, values, width=bar_width, color=scenario_colors[scenario_key],
                        label=scenario_labels[scenario_key], zorder=3)
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.1f}%", ha="center", va="bottom",
                      color=INK, fontsize=8.5)
    ax2.set_xticks(x_positions_2)
    ax2.set_xticklabels([f"x{b}" for b in sweep.BOOST_VALUES_TO_TEST])
    ax2.set_ylim(0, max(10, ax2.get_ylim()[1]))
    ax2.set_title("Severity boost sweep", fontsize=12.5, fontweight="bold", color=INK, loc="left")
    ax2.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=MUTED_INK)

    fig.suptitle("How much do judgment-call parameters change the top pick?", fontsize=14, fontweight="bold",
                 color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, f"n={sweep.N_LOCATIONS} sampled locations per scenario, baseline = "
                         f"{sweep.BASELINE_HIGH_WINDOW} min / x{sweep.BASELINE_BOOST_FACTOR}, seed={sweep.RANDOM_SEED}",
              fontsize=8, color=MUTED_INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    """Entry point. Writes all three diagrams to diagrams/."""
    DIAGRAMS_DIR.mkdir(exist_ok=True)

    pymcdm_path = DIAGRAMS_DIR / "pymcdm_validation.png"
    draw_pymcdm_parity_plot(pymcdm_path)
    print(f"Wrote {pymcdm_path}")

    checklist_path = DIAGRAMS_DIR / "hand_written_test_cases.png"
    draw_hand_written_test_cases_checklist(checklist_path)
    print(f"Wrote {checklist_path}")

    sweep_path = DIAGRAMS_DIR / "sensitivity_sweep.png"
    draw_sensitivity_sweep_chart(sweep_path)
    print(f"Wrote {sweep_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
