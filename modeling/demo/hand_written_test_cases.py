"""
==============================================================================
MODULE: hand_written_test_cases.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: formalizes model-design.md (draft)'s
  "Testing without ground truth" section 1 ("Hand-written test cases") as
  actual, runnable assertions against the real recommend() pipeline — that
  section listed 7 cases in prose; none of them existed as code before this
  file. Calls modeling/recommend/recommend.py directly, the same way
  modeling/demo/try_it_out.py does. Cases 8-12 extend beyond
  model-design.md's original list, covering gaps the first 7 didn't:
  the NaN-unique_id padding bug fixed earlier in this project, blood level
  as a tiebreaker in isolation, the severity boost actually flipping an
  outcome (not just having no effect), service-match behavior outside the
  pregnancy scenario, and the scenario-to-urgency-window mapping itself.
- WHAT THIS FILE DOES: builds small, fixed, synthetic candidate tables for
  each of the 12 cases and asserts the documented or otherwise obvious
  "right answer" actually happens. Each case is its own function,
  independent of the others, printing PASS/FAIL per case plus a final
  summary.
- KNOWN LIMITATIONS / ASSUMPTIONS: while writing case 5, found that
  model-design.md (draft)'s description ("the unconfirmed one should not
  even enter the ranking while a confirmed candidate exists") describes an
  EARLIER version of recommend.py's tier logic, before the later padding fix
  (made in this same project) that lets a confirmed tier with fewer than
  TOP_N_RESULTS rows pull in reachable-but-unconfirmed candidates to fill
  remaining slots, rather than reaching outside the time window. This file
  tests the CURRENT, correct behavior (confirmed always ranks ahead of
  unconfirmed, but unconfirmed can still fill out the remaining slots) —
  model-design.md (draft)'s case 5 wording should be updated to match.
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_MODELING_ROOT = Path(__file__).resolve().parent.parent
for _sibling_dir in ("candidate_pool", "scoring", "recommend"):
    _sibling_path = str(_MODELING_ROOT / _sibling_dir)
    if _sibling_path not in sys.path:
        sys.path.append(_sibling_path)

import candidate_pool as cp  # noqa: E402
import scoring  # noqa: E402
import recommend as R  # noqa: E402

# -----------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------------

DETOUR_FACTOR: float = 1.3
AVG_SPEED_KMPH: float = 35.0
EMPTY_SERVICES = pd.DataFrame({"unique_id": [], "service": []})


def case_1_single_option_within_range() -> bool:
    """"One blood bank within 100 km -> choose it." A single reachable
    candidate should be recommended directly, no ranking performed.
    """
    candidates = pd.DataFrame({"unique_id": ["only_one"], "lat": [19.00], "lon": [73.00], "blood_level": [2]})
    result, weights = R.recommend(candidates, EMPTY_SERVICES, 19.00, 73.00, "pregnancy_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = (
        len(result) == 1
        and result.iloc[0]["unique_id"] == "only_one"
        and result.iloc[0]["note"] == R.NOTE_ONLY_OPTION
        and weights == {}
    )
    print(f"case_1_single_option_within_range: {'PASS' if passed else 'FAIL'}")
    return passed


def case_2_only_matching_service_wins() -> bool:
    """"Two equally close, only one does C-sections, pregnant patient ->
    choose that one." With drive times tied, the confirmed facility with
    the matching services should outrank the one with none.
    """
    candidates = pd.DataFrame({
        "unique_id": ["has_services", "no_services"],
        "lat": [19.00, 19.00], "lon": [73.00, 73.00],  # identical location: drive time is tied
        "blood_level": [2, 2],
    })
    services = pd.DataFrame({
        "unique_id": ["has_services", "has_services", "has_services", "no_services"],
        "service": ["c_section", "obstetrics", "icu", "general_medicine"],
    })
    result, weights = R.recommend(candidates, services, 19.00, 73.00, "pregnancy_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = result.iloc[0]["unique_id"] == "has_services" and "needed_services" in weights
    print(f"case_2_only_matching_service_wins: {'PASS' if passed else 'FAIL'}")
    return passed


def case_3_none_in_window_shows_closest_with_warning() -> bool:
    """"None within the window -> three closest with a warning." When
    nobody is reachable, the closest candidates overall should still be
    returned, each flagged as outside the window.
    """
    candidates = pd.DataFrame({
        "unique_id": ["a", "b", "c"],
        "lat": [19.00, 19.00, 19.00], "lon": [73.00, 73.05, 73.10],
        "blood_level": [2, 2, 2],
    })
    # Patient far enough away that nothing is within the 60-minute window.
    result, weights = R.recommend(candidates, EMPTY_SERVICES, 5.00, 73.00, "pregnancy_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    expected_note = R.NOTE_OUTSIDE_WINDOW.format(top_n=R.TOP_N_RESULTS)
    passed = len(result) == 3 and (result["note"] == expected_note).all() and weights == {}
    print(f"case_3_none_in_window_shows_closest_with_warning: {'PASS' if passed else 'FAIL'}")
    return passed


def case_4_best_equipped_wins_in_dense_area() -> bool:
    """"City with many options, anemia -> best-equipped wins." With several
    similarly-close confirmed candidates, the one with more of the
    scenario's needed services should rank first, not just the nearest one.
    """
    candidates = pd.DataFrame({
        "unique_id": ["best_equipped", "partial", "nearest_but_bare"],
        "lat": [19.00, 19.00, 19.00], "lon": [73.02, 73.03, 73.00],  # nearest_but_bare is closest
        "blood_level": [2, 2, 2],
    })
    services = pd.DataFrame({
        "unique_id": ["best_equipped", "best_equipped", "partial"],
        "service": ["general_medicine", "icu", "general_medicine"],
    })
    result, weights = R.recommend(candidates, services, 19.00, 73.00, "severe_anemia", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = result.iloc[0]["unique_id"] == "best_equipped"
    print(f"case_4_best_equipped_wins_in_dense_area: {'PASS' if passed else 'FAIL'}")
    return passed


def case_5_confirmed_always_ranks_ahead_of_unconfirmed() -> bool:
    """"One confirmed facility (even with only 1 of 3 needed services) and
    one unconfirmed facility, both reachable -> the confirmed one should
    always be preferred." Verifies ranking PRIORITY, not exclusion — see
    this file's header for why "exclusion" is the stale, pre-padding-fix
    expectation.
    """
    candidates = pd.DataFrame({
        "unique_id": ["confirmed_partial", "unconfirmed"],
        "lat": [19.00, 19.00], "lon": [73.00, 73.02],
        "blood_level": [2, 2],
    })
    services = pd.DataFrame({"unique_id": ["confirmed_partial"], "service": ["icu"]})  # 1 of 3 needed services
    result, weights = R.recommend(candidates, services, 19.00, 73.00, "pregnancy_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = (
        result.iloc[0]["unique_id"] == "confirmed_partial"
        and bool(result.iloc[0]["needed_services_confirmed"])
        and (len(result) < 2 or not bool(result.iloc[1]["needed_services_confirmed"]))
    )
    print(f"case_5_confirmed_always_ranks_ahead_of_unconfirmed: {'PASS' if passed else 'FAIL'}")
    return passed


def case_6_all_unconfirmed_fallback_ranks_sensibly() -> bool:
    """"Every reachable candidate is unconfirmed -> check the fallback tier
    ranks sensibly on drive time and blood level alone, every result
    labeled 'not confirmed'." Closer + better blood level should still win
    even with no service data anywhere.
    """
    candidates = pd.DataFrame({
        "unique_id": ["near_good_blood", "far_poor_blood", "mid"],
        "lat": [19.00, 19.00, 19.00], "lon": [73.00, 73.08, 73.04],
        "blood_level": [3, 1, 2],
    })
    result, weights = R.recommend(candidates, EMPTY_SERVICES, 19.00, 73.00, "injury_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = (
        result.iloc[0]["unique_id"] == "near_good_blood"
        and (result["note"] == R.NOTE_UNCONFIRMED_FALLBACK).all()
        and "needed_services" not in weights
    )
    print(f"case_6_all_unconfirmed_fallback_ranks_sensibly: {'PASS' if passed else 'FAIL'}")
    return passed


def case_7_severity_boost_has_no_effect_in_fallback_tier() -> bool:
    """"Lower-urgency (anemia) case where only the fallback tier is
    reachable -> confirm the severity boost has no effect, documented, not
    a bug." The anemia boost targets needed_services, which the fallback
    tier never uses.
    """
    candidates = pd.DataFrame({
        "unique_id": ["a", "b", "c"],
        "lat": [19.00, 19.00, 19.00], "lon": [73.00, 73.05, 73.10],
        "blood_level": [3, 2, 1],
    })
    result, weights = R.recommend(candidates, EMPTY_SERVICES, 19.00, 73.00, "severe_anemia", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = "needed_services" not in weights and set(weights.keys()) == {"drive_min", "blood_level"}
    print(f"case_7_severity_boost_has_no_effect_in_fallback_tier: {'PASS' if passed else 'FAIL'}")
    return passed


def case_8_padding_picks_the_genuinely_closest_facility() -> bool:
    """Padding must fill remaining slots with the genuinely closest
    facility, never wrongly skip one just because it shares a NaN
    unique_id with an already-picked candidate. Narrative version of the
    NaN-unique_id padding bug fixed earlier in this project (see this
    file's header) — a directory-sourced facility with no unique_id must
    not make every OTHER directory-sourced facility look "already chosen."
    """
    candidates = pd.DataFrame({
        "unique_id": [np.nan, np.nan, np.nan, "confirmed_far"],
        "lat": [19.00, 19.00, 19.00, 19.00],
        "lon": [73.000, 73.005, 73.286, 74.430],  # 3rd is genuinely closer than the 4th
        "blood_level": [2, 2, 2, 2],
    })
    services = pd.DataFrame({"unique_id": ["confirmed_far"], "service": ["icu"]})
    result, _ = R.recommend(candidates, services, 19.00, 73.00, "injury_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = len(result) == 3 and 73.286 in set(result["lon"]) and 74.430 not in set(result["lon"])
    print(f"case_8_padding_picks_the_genuinely_closest_facility: {'PASS' if passed else 'FAIL'}")
    return passed


def case_9_blood_level_breaks_a_tie() -> bool:
    """Two facilities, identical drive time and identical confirmed
    services -> the one with the higher blood level should win, isolating
    that criterion from the other two (which have zero variance here and
    so carry zero weight).
    """
    candidates = pd.DataFrame({
        "unique_id": ["high_blood", "low_blood"],
        "lat": [19.00, 19.00], "lon": [73.00, 73.00],  # tied distance
        "blood_level": [3, 1],
    })
    services = pd.DataFrame({"unique_id": ["high_blood", "low_blood"], "service": ["icu", "icu"]})  # tied services
    result, weights = R.recommend(candidates, services, 19.00, 73.00, "pregnancy_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = result.iloc[0]["unique_id"] == "high_blood" and weights.get("blood_level", 0) > 0.9
    print(f"case_9_blood_level_breaks_a_tie: {'PASS' if passed else 'FAIL'}")
    return passed


def case_10_severity_boost_actually_flips_the_outcome() -> bool:
    """Positive counterpart to case 7 (which shows the boost doing
    nothing): here, a closer-but-less-equipped facility should beat a
    farther-but-fully-equipped one ONLY because high urgency boosts drive
    time — verified by comparing the un-boosted entropy-only ranking
    against the real, boosted recommend() ranking and confirming they
    actually disagree (concrete values below were checked to flip before
    being hardcoded as this case's expectation).
    """
    criteria_matrix = np.array([[5.0, 2.0, 0.0], [50.0, 2.0, 3.0]])  # [drive_min, blood_level, needed_services]
    is_cost = np.array([True, False, False])
    unboosted_weights = scoring.entropy_weights(criteria_matrix)
    unboosted_scores = scoring.topsis_scores(criteria_matrix, unboosted_weights, is_cost)
    unboosted_winner = "closer_bare" if unboosted_scores[0] > unboosted_scores[1] else "farther_equipped"

    candidates = pd.DataFrame({
        "unique_id": ["closer_bare", "farther_equipped"],
        "lat": [19.00, 19.00], "lon": [73.01, 73.50],
        "blood_level": [2, 2],
    })
    services = pd.DataFrame({
        "unique_id": ["farther_equipped"] * 3,
        "service": ["emergency_surgery", "trauma_care", "icu"],
    })
    result, _ = R.recommend(candidates, services, 19.00, 73.00, "injury_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    boosted_winner = result.iloc[0]["unique_id"]

    passed = unboosted_winner == "farther_equipped" and boosted_winner == "closer_bare"
    print(f"case_10_severity_boost_actually_flips_the_outcome: {'PASS' if passed else 'FAIL'}")
    return passed


def case_11_injury_full_match_beats_partial_match() -> bool:
    """Broadens case 2's "has-what's-needed wins" check beyond the
    pregnancy scenario: tied distance, injury_bleeding needs
    emergency_surgery + trauma_care + icu -> the facility confirmed for
    all three should outrank one confirmed for only one of them.
    """
    candidates = pd.DataFrame({
        "unique_id": ["full_match", "partial_match"],
        "lat": [19.00, 19.00], "lon": [73.00, 73.00],
        "blood_level": [2, 2],
    })
    services = pd.DataFrame({
        "unique_id": ["full_match", "full_match", "full_match", "partial_match"],
        "service": ["emergency_surgery", "trauma_care", "icu", "icu"],
    })
    result, weights = R.recommend(candidates, services, 19.00, 73.00, "injury_bleeding", DETOUR_FACTOR, AVG_SPEED_KMPH)
    passed = result.iloc[0]["unique_id"] == "full_match" and "needed_services" in weights
    print(f"case_11_injury_full_match_beats_partial_match: {'PASS' if passed else 'FAIL'}")
    return passed


def case_12_every_scenario_maps_to_its_documented_window() -> bool:
    """Every scenario must map to the time window model-design.md (draft)
    documents: pregnancy/injury/not_sure -> 60 min (high urgency),
    severe_anemia -> 120 min (lower urgency). Checks the mapping itself,
    not a recommend() call — catches a scenario-to-urgency typo none of
    the other 11 cases would.
    """
    expected_window_minutes = {
        "pregnancy_bleeding": 60,
        "injury_bleeding": 60,
        "not_sure": 60,
        "severe_anemia": 120,
    }
    actual_window_minutes = {
        scenario: cp.TIME_WINDOW_MINUTES_BY_URGENCY[cp.URGENCY_BY_SCENARIO[scenario]]
        for scenario in expected_window_minutes
    }
    passed = actual_window_minutes == expected_window_minutes
    print(f"case_12_every_scenario_maps_to_its_documented_window: {'PASS' if passed else 'FAIL'}")
    return passed


ALL_CASES = (
    case_1_single_option_within_range,
    case_2_only_matching_service_wins,
    case_3_none_in_window_shows_closest_with_warning,
    case_4_best_equipped_wins_in_dense_area,
    case_5_confirmed_always_ranks_ahead_of_unconfirmed,
    case_6_all_unconfirmed_fallback_ranks_sensibly,
    case_7_severity_boost_has_no_effect_in_fallback_tier,
    case_8_padding_picks_the_genuinely_closest_facility,
    case_9_blood_level_breaks_a_tie,
    case_10_severity_boost_actually_flips_the_outcome,
    case_11_injury_full_match_beats_partial_match,
    case_12_every_scenario_maps_to_its_documented_window,
)


def main() -> int:
    """Entry point. Runs every case in ALL_CASES, prints a summary, and
    returns a process exit code (0 if every case passed, 1 otherwise).
    """
    assert len(ALL_CASES) >= 1, "ALL_CASES must not be empty"
    results = [case_fn() for case_fn in ALL_CASES]
    passed_count = sum(results)
    print(f"\n{passed_count}/{len(results)} hand-written test cases passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
