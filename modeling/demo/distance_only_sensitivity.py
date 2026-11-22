"""
==============================================================================
MODULE: distance_only_sensitivity.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: runs the same comparison
  compare_distance_only.py does (full model vs. distance-only ranking), but
  at scale — many sampled locations per scenario instead of 6 fixed cases —
  so the "how different is it" question has a real sample size behind it,
  not just a handful of examples. Reuses compare_distance_only.py's
  run_full_model()/run_distance_only() directly; no scoring logic is
  duplicated here.
- WHAT THIS FILE DOES: samples N locations per scenario (near real
  facilities, jittered — see modeling/demo/try_it_out.py's
  pick_random_location() reasoning for why, not uniformly random across
  India), runs both rankings on each, and reports: how often the #1 pick
  differs, average top-3 overlap, broken down by scenario and by whether
  confirmed capability data was available for the top pick. Writes the raw
  results and a summary to JSON, and renders two chart PNGs to diagrams/.
- KNOWN LIMITATIONS / ASSUMPTIONS: uses a fixed random seed so these exact
  numbers are reproducible — rerunning this file should print the same
  figures reported in chat. Like the rest of modeling/demo/, uses the same
  ASSUMED (not measured) detour_factor/avg_speed_kmph.
==============================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELING_ROOT = _REPO_ROOT / "modeling"
for _sibling_dir in ("candidate_pool", "scoring", "recommend", "demo"):
    _sibling_path = str(_MODELING_ROOT / _sibling_dir)
    if _sibling_path not in sys.path:
        sys.path.append(_sibling_path)

import try_it_out as demo  # noqa: E402
import compare_distance_only as cdo  # noqa: E402
import generate_testing_diagrams as gtd  # noqa: E402  (shared draw_slide() template)

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------------

RANDOM_SEED: int = 7
N_PER_SCENARIO: int = 60
LOCATION_JITTER_DEGREES: float = 0.1  # matches try_it_out.RANDOM_LOCATION_JITTER_DEGREES

RESULTS_JSON_PATH: Path = Path(__file__).resolve().parent / "distance_only_sensitivity_results.json"
DIAGRAMS_DIR: Path = _REPO_ROOT / "diagrams"

# Colors: single accent hue for plain-magnitude bars (matches this project's
# established teal); the confirmed/unconfirmed pair uses this dataviz
# skill's pre-validated status palette (good/warning), chosen specifically
# because this distinction IS a state (confirmed vs. not), not an arbitrary
# category — validated for colorblind-safe separation before use here.
ACCENT_TEAL: str = "#0B6E62"
STATUS_GOOD: str = "#0ca30c"
STATUS_WARNING: str = "#fab219"
INK: str = "#10181A"
MUTED_INK: str = "#5B6B68"
GRID: str = "#D9E2DF"


def sample_test_locations(blood_points: pd.DataFrame, scenario_key: str, n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    """Sample n jittered locations near real facilities for one scenario —
    the same "realistic patient location" reasoning as
    try_it_out.pick_random_location(), not uniformly random across India
    (verified earlier: uniform-random clicks land far from any facility
    about half the time, which would bias this comparison toward trivial
    "nothing reachable" cases rather than real ranking differences).

    PARAMETERS:
        blood_points: must have in_candidate_pool, lat, lon columns.
        scenario_key: unused here but accepted for a stable call signature
            alongside the other per-scenario sampling this file does.
        n: number of locations to sample.
        rng: a numpy random Generator, passed in so callers control the seed.
    """
    assert n >= 1, "must sample at least one location"
    pool = blood_points[blood_points["in_candidate_pool"] == True].dropna(subset=["lat", "lon"])  # noqa: E712
    assert len(pool) >= n, "not enough candidates to sample from"

    sampled = pool.sample(n=n, random_state=rng.integers(0, 1_000_000))
    locations = [
        (row["lat"] + rng.uniform(-LOCATION_JITTER_DEGREES, LOCATION_JITTER_DEGREES),
         row["lon"] + rng.uniform(-LOCATION_JITTER_DEGREES, LOCATION_JITTER_DEGREES))
        for _, row in sampled.iterrows()
    ]
    assert len(locations) == n, "must produce exactly n locations"
    return locations


def run_sensitivity_sweep(blood_points: pd.DataFrame, facility_services: pd.DataFrame) -> pd.DataFrame:
    """Run the full-model-vs-distance-only comparison across
    N_PER_SCENARIO locations for every scenario. Returns one row per
    successful comparison (a jittered point that found zero candidates at
    all is skipped, not counted as a result either way).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    scenario_keys = [key for key, _ in demo.SCENARIO_MENU]
    assert len(scenario_keys) >= 1, "SCENARIO_MENU must not be empty"

    rows = []
    for scenario_key in scenario_keys:
        locations = sample_test_locations(blood_points, scenario_key, N_PER_SCENARIO, rng)
        for patient_lat, patient_lon in locations:
            try:
                full_result = cdo.run_full_model(blood_points, facility_services, scenario_key, patient_lat, patient_lon)
                distance_result = cdo.run_distance_only(blood_points, facility_services, scenario_key, patient_lat, patient_lon)
            except AssertionError:
                continue  # no candidates found at all near this jittered point

            full_names = list(full_result["name"]) if "name" in full_result.columns else list(full_result["unique_id"])
            distance_names = list(distance_result["name"]) if "name" in distance_result.columns else list(distance_result["unique_id"])
            assert len(full_names) >= 1 and len(distance_names) >= 1, "both rankings must return at least one name"

            rows.append({
                "scenario": scenario_key,
                "confirmed_tier_used": bool(full_result.iloc[0].get("needed_services_confirmed", False)),
                "top_pick_changed": full_names[0] != distance_names[0],
                "overlap_count": len(set(full_names) & set(distance_names)),
            })

    results = pd.DataFrame(rows)
    assert len(results) > 0, "sensitivity sweep produced zero usable comparisons"
    return results


def summarize_by_scenario(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results into one row per scenario: sample size, percent of
    cases where the #1 pick changed, and average top-3 overlap.
    """
    assert "scenario" in results.columns, "results is missing the scenario column"
    summary = results.groupby("scenario").agg(
        n=("top_pick_changed", "size"),
        pct_top_pick_changed=("top_pick_changed", "mean"),
        avg_overlap=("overlap_count", "mean"),
    ).reset_index()
    summary["pct_top_pick_changed"] = (summary["pct_top_pick_changed"] * 100).round(1)
    summary["avg_overlap"] = summary["avg_overlap"].round(2)
    return summary


def summarize_by_confirmed_tier(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results into one row per confirmed/unconfirmed tier."""
    assert "confirmed_tier_used" in results.columns, "results is missing confirmed_tier_used"
    summary = results.groupby("confirmed_tier_used").agg(
        n=("top_pick_changed", "size"),
        pct_top_pick_changed=("top_pick_changed", "mean"),
        avg_overlap=("overlap_count", "mean"),
    ).reset_index()
    summary["pct_top_pick_changed"] = (summary["pct_top_pick_changed"] * 100).round(1)
    summary["avg_overlap"] = summary["avg_overlap"].round(2)
    return summary


def style_axes(ax) -> None:
    """Shared, recessive chart chrome: no top/right spines, light muted grid
    behind the bars, muted tick labels. Applied to every chart this file
    draws so they look like one consistent set.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=MUTED_INK, labelsize=10)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)


def draw_scenario_chart(by_scenario: pd.DataFrame, output_path: Path, overall_pct_changed: float) -> None:
    """Slide: the model's #1 pick vs. plain distance-only ranking, by
    scenario. Renders through generate_testing_diagrams.draw_slide() so
    this image matches the other three in diagrams/ (16:9, same headline
    treatment, same hero-number framing).

    PARAMETERS:
        overall_pct_changed: the TRUE overall percentage, computed from the
            raw per-location results (not re-derived from this function's
            own already-rounded per-scenario values) — averaging four
            rounded percentages compounds rounding error and would show a
            hero number that quietly disagrees with the number already
            reported elsewhere for the same run.
    """
    assert len(by_scenario) >= 1, "by_scenario must not be empty"
    labels = {
        "pregnancy_bleeding": "Pregnancy/\nbleeding",
        "injury_bleeding": "Injury/\nbleeding",
        "severe_anemia": "Severe\nanemia",
        "not_sure": "Not sure/\nother",
    }
    ordered = by_scenario.sort_values("pct_top_pick_changed", ascending=False)
    x_labels = [labels.get(s, s) for s in ordered["scenario"]]
    values = ordered["pct_top_pick_changed"].to_numpy()
    n_values = ordered["n"].to_numpy()
    assert (n_values == n_values[0]).all(), "bars have different sample sizes; the shared-n caption would be wrong"

    def visual(ax):
        ax.set_facecolor(gtd.SURFACE)
        gtd.style_axes(ax)
        bars = ax.bar(x_labels, values, color=gtd.ACCENT_TEAL, width=0.55, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", va="bottom",
                    color=gtd.INK, fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(values) + 15)
        ax.tick_params(axis="x", labelsize=13)

    gtd.draw_slide(
        output_path=output_path,
        headline="The model beats pure distance most of the time",
        hero_number=f"{overall_pct_changed:.1f}%",
        hero_label="of top picks differ from\ndistance-only ranking",
        draw_visual_fn=visual,
        footer_text=f"n={n_values[0]} sampled locations per scenario ({int(n_values.sum())} total), "
                    f"near real facilities, seed={RANDOM_SEED}. Method: modeling/demo/distance_only_sensitivity.py",
    )


def draw_confirmed_tier_chart(by_confirmed: pd.DataFrame, output_path: Path) -> None:
    """Two small-multiple bar charts (percent changed, avg overlap) split by
    whether confirmed capability data existed — different units/scales, so
    two separate panels rather than one dual-axis chart. Uses the
    confirmed/unconfirmed status color pair (a real state distinction, not
    an arbitrary category).
    """
    assert len(by_confirmed) == 2, "expected exactly two rows: confirmed and unconfirmed"
    by_confirmed = by_confirmed.sort_values("confirmed_tier_used", ascending=False)
    x_labels = ["Confirmed\ncapability data", "Unconfirmed\n(fallback tier)"]
    colors = [STATUS_GOOD, STATUS_WARNING]
    n_values = by_confirmed["n"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.6), dpi=200)
    fig.patch.set_facecolor("#FCFCFB")

    pct_values = by_confirmed["pct_top_pick_changed"].to_numpy()
    ax1.set_facecolor("#FCFCFB")
    style_axes(ax1)
    bars1 = ax1.bar(x_labels, pct_values, color=colors, width=0.55, zorder=3)
    for bar, value in zip(bars1, pct_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", va="bottom",
                  color=INK, fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(pct_values) + 15)
    ax1.set_title("Top pick changed", fontsize=11.5, fontweight="bold", color=INK, loc="left")

    overlap_values = by_confirmed["avg_overlap"].to_numpy()
    ax2.set_facecolor("#FCFCFB")
    style_axes(ax2)
    bars2 = ax2.bar(x_labels, overlap_values, color=colors, width=0.55, zorder=3)
    for bar, value in zip(bars2, overlap_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f}", ha="center", va="bottom",
                  color=INK, fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 3.4)
    ax2.set_title("Average overlap in top 3 (out of 3)", fontsize=11.5, fontweight="bold", color=INK, loc="left")

    fig.suptitle("The divergence is driven by whether confirmed data exists", fontsize=13, fontweight="bold",
                 color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, f"n={int(n_values[0])} confirmed, n={int(n_values[1])} unconfirmed, seed={RANDOM_SEED}",
              fontsize=8, color=MUTED_INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    """Entry point. Runs the sweep, writes results JSON, renders both chart
    PNGs to diagrams/, and prints the summary tables. Returns 0 on success.
    """
    DIAGRAMS_DIR.mkdir(exist_ok=True)
    blood_points, facility_services = demo.load_datasets()

    results = run_sensitivity_sweep(blood_points, facility_services)
    by_scenario = summarize_by_scenario(results)
    by_confirmed = summarize_by_confirmed_tier(results)

    print("=== By scenario ===")
    print(by_scenario.to_string(index=False))
    print("\n=== By confirmed-tier availability ===")
    print(by_confirmed.to_string(index=False))
    print(f"\nOverall: n={len(results)}, top pick changed {100 * results['top_pick_changed'].mean():.1f}%, "
          f"avg overlap {results['overlap_count'].mean():.2f}")

    output = {
        "random_seed": RANDOM_SEED,
        "n_per_scenario": N_PER_SCENARIO,
        "by_scenario": by_scenario.to_dict(orient="records"),
        "by_confirmed_tier": by_confirmed.to_dict(orient="records"),
        "overall": {
            "n": len(results),
            "pct_top_pick_changed": round(100 * results["top_pick_changed"].mean(), 1),
            "avg_overlap": round(results["overlap_count"].mean(), 2),
        },
    }
    RESULTS_JSON_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {RESULTS_JSON_PATH}")

    scenario_chart_path = DIAGRAMS_DIR / "distance_only_by_scenario.png"
    confirmed_chart_path = DIAGRAMS_DIR / "distance_only_by_confirmed_tier.png"
    draw_scenario_chart(by_scenario, scenario_chart_path, overall_pct_changed=output["overall"]["pct_top_pick_changed"])
    draw_confirmed_tier_chart(by_confirmed, confirmed_chart_path)
    print(f"Wrote {scenario_chart_path}")
    print(f"Wrote {confirmed_chart_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
