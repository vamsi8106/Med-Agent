"""Golden retrieval cases: a doctor-style question and where its answer is.

Ground truth is a short phrase copied verbatim from the guideline's extracted
text. A case is a hit at rank k when one of the top-k retrieved chunks contains
that phrase (whitespace-normalised), so the cases stay valid whatever the
chunking -- and they encode where the source says something, not a clinical
judgement of ours. Queries paraphrase the text instead of quoting it: the point
is to test semantic retrieval, not string search.

Sources are local PDFs in guidelines_pdf/ (gitignored). The ADA Standards of
Care PDF there is deliberately absent: its licence forbids use "for text or
data mining, machine learning, or similar technologies" without permission.
KDIGO is CC BY-NC-ND (non-commercial); NICE is subject to its notice of rights.
Phrases avoid words pypdf mangles (it drops the "fi"/"fl" ligatures:
"ltration", "ofce").
"""

from typing import NamedTuple


class RetrievalCase(NamedTuple):
    id: str
    source: str
    query: str
    expected_phrase: str


SOURCES: dict[str, str] = {
    "kdigo-2024-ckd": "KDIGO-2024-CKD-Guideline.pdf",
    "nice-ng136-hypertension": (
        "NG136 Hypertension in adults_ Visual summary 23_07_2025 - "
        "visual-summary-pdf-6899919517.pdf"
    ),
}

CASES: list[RetrievalCase] = [
    RetrievalCase(
        "kdigo-egfr-equation",
        "kdigo-2024-ckd",
        "Which kidney function estimate should I use in an adult at risk of CKD?",
        "we recommend using creatinine-based estimated",
    ),
    RetrievalCase(
        "kdigo-cystatin",
        "kdigo-2024-ckd",
        "When is creatinine alone not accurate enough, so cystatin C should be added?",
        "We recommend using eGFRcr-cys in clinical situations when eGFRcr is less accurate",
    ),
    RetrievalCase(
        "kdigo-risk-equation",
        "kdigo-2024-ckd",
        "How do I predict whether a patient with stage 3 to 5 CKD will progress to dialysis?",
        "externally validated risk equation to estimate the absolute risk of kidney failure",
    ),
    RetrievalCase(
        "kdigo-referral-threshold",
        "kdigo-2024-ckd",
        "At what predicted kidney failure risk should I refer to a nephrologist?",
        "5-year kidney failure risk of 3%–5% can be used to determine need for nephrology referral",
    ),
    RetrievalCase(
        "kdigo-exercise",
        "kdigo-2024-ckd",
        "How much physical exercise per week is advised for someone with chronic kidney disease?",
        "cumulative duration of at least 150 minutes per week",
    ),
    RetrievalCase(
        "kdigo-protein",
        "kdigo-2024-ckd",
        "How much dietary protein should adults with advanced CKD eat?",
        "protein intake of 0.8 g/kg body weight/d in adults with CKD G3–G5",
    ),
    RetrievalCase(
        "kdigo-sodium",
        "kdigo-2024-ckd",
        "What is the daily salt limit for a patient with kidney disease?",
        "sodium intake be <2 g of sodium per day",
    ),
    RetrievalCase(
        "kdigo-bp-target",
        "kdigo-2024-ckd",
        "What systolic blood pressure should I aim for in CKD with hypertension?",
        "target systolic blood pressure (SBP) of <120 mm Hg",
    ),
    RetrievalCase(
        "kdigo-dual-ras",
        "kdigo-2024-ckd",
        "Can I combine an ACE inhibitor with an angiotensin receptor blocker in CKD?",
        "We recommend avoiding any combination of ACEi, ARB, and direct renin inhibitor",
    ),
    RetrievalCase(
        "kdigo-sglt2i",
        "kdigo-2024-ckd",
        "Which patients with chronic kidney disease should be started on an SGLT2 inhibitor?",
        "We recommend treating adults with CKD with an SGLT2i",
    ),
    RetrievalCase(
        "kdigo-ns-mra",
        "kdigo-2024-ckd",
        "When should a nonsteroidal MRA be added for diabetic kidney disease with albuminuria?",
        "nonsteroidal mineralocorticoid receptor antagonist with proven kidney",
    ),
    RetrievalCase(
        "kdigo-glp1",
        "kdigo-2024-ckd",
        "Glucose still high despite metformin and an SGLT2 inhibitor in a CKD patient: what next?",
        "who have not achieved individualized glycemic targets despite use of metformin and "
        "SGLT2 inhibitor",
    ),
    RetrievalCase(
        "kdigo-asymptomatic-urate",
        "kdigo-2024-ckd",
        "Should I treat high uric acid without gout to slow kidney disease progression?",
        "not using agents to lower serum uric acid in people with CKD and "
        "asymptomatic hyperuricemia",
    ),
    RetrievalCase(
        "kdigo-statin",
        "kdigo-2024-ckd",
        "Should a 60-year-old with eGFR below 60 who is not on dialysis take a statin?",
        "but not treated with chronic dialysis or kidney transplantation (GFR categories G3a–G5)",
    ),
    RetrievalCase(
        "kdigo-aspirin",
        "kdigo-2024-ckd",
        "Is aspirin advised after a heart attack in someone with CKD?",
        "We recommend oral low-dose aspirin for prevention of recurrent ischemic",
    ),
    RetrievalCase(
        "kdigo-anticoagulant-af",
        "kdigo-2024-ckd",
        "Which anticoagulant for atrial fibrillation in CKD: a NOAC or warfarin?",
        "in preference to vitamin K antagonists (e.g., warfarin) for thromboprophylaxis",
    ),
    RetrievalCase(
        "nice-target-over-80",
        "nice-ng136-hypertension",
        "What clinic blood pressure target applies to a hypertensive patient over 80?",
        "Clinic BP less than 150/90 mmHg",
    ),
    RetrievalCase(
        "nice-first-line-older",
        "nice-ng136-hypertension",
        "First antihypertensive drug for a 65-year-old without diabetes?",
        "(any age) Step 1 CCB",
    ),
    RetrievalCase(
        "nice-resistant-low-k",
        "nice-ng136-hypertension",
        "Blood pressure uncontrolled on three drugs and potassium is 4.2: what do I add?",
        "low-dose spironolactone if blood potassium level is 4.5 mmol/l or less",
    ),
    RetrievalCase(
        "nice-same-day-referral",
        "nice-ng136-hypertension",
        "When does very high blood pressure need same-day specialist review?",
        "Refer for same-day specialist review if",
    ),
    RetrievalCase(
        "nice-recheck-interval",
        "nice-ng136-hypertension",
        "How often should blood pressure be rechecked if it is normal in clinic?",
        "Check BP at least every 5 years",
    ),
]
