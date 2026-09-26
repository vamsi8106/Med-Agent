import pytest

from medagent.infra.verification import find_unverified_figures, unverified_figures_warning

_EVIDENCE = (
    "Metformin reduced HbA1c by 1.0-1.5% versus placebo (n=1,200). Avoid if eGFR <30 "
    "mL/min/1.73 m2. UKPDS 2000 showed 32% fewer events. Dose up to 2,000 mg daily."
)
_PATIENT = "conditions=['type 2 diabetes'] HbA1c 9.8 % eGFR 52 mL/min/1.73m². age 68"


def _flagged(answer: str) -> list[str]:
    return find_unverified_figures(answer, _EVIDENCE, _PATIENT)


def test_a_faithful_answer_that_reformats_figures_is_not_flagged() -> None:
    answer = (
        "Metformin lowers HbA1c by 1.0–1.5%; stop if eGFR falls below 30 mL/min. "
        "Up to 2000 mg/day. Her HbA1c of 9.8% is high."
    )
    assert _flagged(answer) == []


def test_an_invented_dose_and_statistic_are_flagged() -> None:
    answer = "Start metformin 850 mg twice daily; this cuts CV events by 45% (UKPDS)."
    assert _flagged(answer) == ["850 mg", "45%"]


def test_a_half_invented_range_is_flagged() -> None:
    assert _flagged("HbA1c improves 1.0-3.5% in trials.") == ["1.0-3.5%"]


def test_years_stages_and_counts_without_units_are_ignored() -> None:
    answer = "Per ADA 2024 standards, stage 3 CKD in a type 2 patient, 3 trials, 2 drugs."
    assert _flagged(answer) == []


def test_figures_from_the_patient_record_count_as_verified() -> None:
    assert find_unverified_figures("Her eGFR is 52 mL/min.", "no numbers here", _PATIENT) == []


@pytest.mark.parametrize(
    ("figure", "source"),
    [
        ("2000 mg", "up to 2,000 mg"),
        ("2,000 mg", "up to 2000 mg"),
        ("2.0 mg", "2 mg"),
        ("2 mg", "2.0 mg"),
        ("5 MG", "5 mg"),
    ],
)
def test_number_formatting_differences_are_not_flagged(figure: str, source: str) -> None:
    assert find_unverified_figures(f"Give {figure}.", source) == []


@pytest.mark.parametrize(
    "figure",
    ["25 mg", "40%", "30 mL/min", "7.5 mg/dL", "140 mmHg", "12 mcg", "5 units", "80 bpm", "0.5 g"],
)
def test_recognises_common_clinical_units(figure: str) -> None:
    assert find_unverified_figures(f"It is {figure} here.", "nothing relevant") == [figure]


def test_repeated_figures_are_reported_once() -> None:
    assert find_unverified_figures("Give 850 mg now and 850 mg later.", "") == ["850 mg"]


def test_no_sources_flags_every_figure() -> None:
    assert find_unverified_figures("Take 25 mg and expect 10%.") == ["25 mg", "10%"]


def test_an_answer_with_no_figures_is_never_flagged() -> None:
    assert find_unverified_figures("Metformin is first-line therapy.", "") == []


def test_a_number_embedded_in_a_word_is_not_a_figure() -> None:
    assert find_unverified_figures("Use the B12 supplement and CD4 counts.", "") == []


def test_warning_names_the_figures() -> None:
    warning = unverified_figures_warning(["850 mg", "45%"])
    assert "UNVERIFIED FIGURES" in warning
    assert "850 mg, 45%" in warning


def test_narrow_no_break_space_between_number_and_unit_is_understood() -> None:
    """Real model output (gpt-oss) writes "7 %" and "2500 mg" with U+202F."""
    assert find_unverified_figures("Give 2500 mg for 7 %.", "") == [
        "2500 mg",
        "7 %",
    ]
    assert find_unverified_figures("Give 2500 mg.", "up to 2,500 mg") == []


def test_trailing_punctuation_is_not_swallowed_into_the_reported_figure() -> None:
    """Seen live: figures were reported as "30 mL/min/1.73 m²." and "…m²;"."""
    answer = "eGFR below 30 mL/min/1.73 m²; or 45 mL/min/1.73 m2."
    assert find_unverified_figures(answer, "") == ["30 mL/min/1.73 m²", "45 mL/min/1.73 m2"]


def test_a_range_of_egfr_values_reports_the_whole_range_once() -> None:
    assert find_unverified_figures("Reduce dose at 30–45 mL/min/1.73 m².", "only 30 here") == [
        "30–45 mL/min/1.73 m²"
    ]
