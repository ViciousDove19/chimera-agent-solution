"""Task-3 clinical feature vector, extracted from the case's own input files.

The Cox head reads two things and nothing else: the structured prompt
(``prompt.json`` — age, PSA, DRE) and the clinical-data socket (radiology,
biopsy and radical-prostatectomy report prose).  ``previous_notes`` and
``family_history`` are deliberately excluded: on the 75 labelled cases their
univariate C-indices were 0.49-0.51, i.e. noise.

The same extractor runs at training time (``scripts/train_task3_head.py``) and
at inference, so the artifact's ``clinical_features`` order is authoritative and
a drift between the two is impossible by construction.  Missing values are
``nan`` here and are filled from the artifact's training medians downstream.
"""

from __future__ import annotations

import math
import re
from typing import Any

from chimera_agent_baseline.tools.task3_recurrence import _stage_rank, parse_surgical_pathology

NAN = float("nan")

CT_RANK = {
    "ct1": 0.0,
    "ct1c": 0.0,
    "ct2": 1.0,
    "ct2a": 1.0,
    "ct2b": 1.0,
    "ct2c": 1.0,
    "ct3": 2.0,
    "ct3a": 2.0,
    "ct3b": 3.0,
    "ct4": 4.0,
}

#: Artifact-order feature names. Appending is safe; reordering invalidates the head.
FEATURE_NAMES: tuple[str, ...] = (
    "log_psa",
    "age",
    "isup",
    "gleason_sum",
    "tertiary",
    "pt_rank",
    "margin_pos",
    "epe",
    "svi",
    "lvi",
    "node_pos",
    "node_not_sampled",
    "dre_abnormal",
    "ct_rank",
    "vol",
    "psad",
    "pirads",
    "cspca",
    "bx_isup",
    "cribriform",
    "pni",
    "intraductal",
    "bx_missing",
)


def _num(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def biopsy_isup(report: str | None) -> float | None:
    """Highest ISUP grade group across all biopsy sessions in *report*.

    Fourteen of the 75 cases record more than one session, and the sessions are
    listed oldest first -- so reading the first match returns a superseded
    grade, and for a negative session it returns the literal "ISUP grade group
    0", which is not a grade group at all. The prognostically relevant value is
    the highest grade ever found; a report with only grade-0 sessions has no
    grade.
    """
    grades = [int(g) for g in re.findall(r"ISUP grade group (\d)", report or "")]
    positive = [g for g in grades if g >= 1]
    return float(max(positive)) if positive else None


def clinical_features(prompt: dict[str, Any], clinical: dict[str, Any]) -> dict[str, float]:
    """Return the named feature map for one case; unavailable values are ``nan``."""
    evidence = parse_surgical_pathology(clinical.get("surgical_pathology_report"))
    rad = clinical.get("radiology_report") or ""
    bx = clinical.get("pathology_report") or ""
    dre = prompt.get("dre") or ""

    psa = prompt.get("psa")
    age = prompt.get("age")
    gleason_sum = (
        float(evidence.gleason_primary + evidence.gleason_secondary)
        if evidence.gleason_primary is not None and evidence.gleason_secondary is not None
        else NAN
    )
    ct = re.search(r"\bc(T[0-9][a-c]?|Tx)\b", dre)
    bx_isup = biopsy_isup(bx)

    return {
        # --- structured prompt ---
        "log_psa": math.log1p(float(psa)) if psa is not None else NAN,
        "age": float(age) if age is not None else NAN,
        "dre_abnormal": 1.0 if "abnormal" in dre.lower() else 0.0,
        "ct_rank": CT_RANK.get(ct.group(0).lower() if ct else "", NAN),
        # --- radical-prostatectomy report ---
        "isup": float(evidence.isup_grade) if evidence.isup_grade is not None else NAN,
        "gleason_sum": gleason_sum,
        "tertiary": 1.0 if evidence.gleason_tertiary is not None else 0.0,
        "pt_rank": float(_stage_rank(evidence.pathological_stage)),
        "margin_pos": 1.0 if evidence.margin_status == "positive" else 0.0,
        "epe": 1.0 if evidence.extraprostatic_extension == "present" else 0.0,
        "svi": 1.0 if evidence.seminal_vesicle_invasion == "present" else 0.0,
        "lvi": 1.0 if evidence.lymphovascular_invasion == "present" else 0.0,
        "node_pos": 1.0 if evidence.lymph_node_status == "positive" else 0.0,
        "node_not_sampled": 1.0 if evidence.lymph_node_status == "not_sampled" else 0.0,
        # --- radiology report ---
        "vol": _num(r"Prostate volume:\s*([0-9]+(?:\.[0-9]+)?)", rad) or NAN,
        "psad": _num(r"PSA density:\s*([0-9]+(?:\.[0-9]+)?)", rad) or NAN,
        "pirads": _num(r"PI-RADS:\s*([0-9]+)", rad) or NAN,
        "cspca": _num(r"cancer \(0.1\):\s*([0-9]+(?:\.[0-9]+)?)", rad) or NAN,
        # --- biopsy report ---
        "bx_isup": bx_isup if bx_isup is not None else NAN,
        "cribriform": 1.0 if "cribriform pattern was present" in bx else 0.0,
        "pni": 1.0 if "perineural invasion was present" in bx else 0.0,
        "intraductal": 1.0 if re.search(r"intraductal carcinoma (?:was|were) (?:present|identified)", bx) else 0.0,
        "bx_missing": 1.0 if "ISUP report missing" in bx else 0.0,
    }


#: The subset the head is fitted on: every variable a report the judge can see
#: actually states. The twelve dropped variables (``log_psa``, ``age``,
#: ``gleason_sum``, ``dre_abnormal``, ``ct_rank``, ``vol``, ``cspca``,
#: ``cribriform``, ``pni``, ``intraductal``, ``bx_missing``,
#: ``node_not_sampled``) are still extracted -- the rationale layer reads them
#: -- but they no longer enter the Cox fit.
#: The 2026-09-11 constrained refit improves one-seed nested CV from 0.846 to
#: 0.866. A five-seed fixed-L2 diagnostic also favours it over the unconstrained
#: fusion (0.850 vs 0.842), though seed variance remains much larger than the
#: difference. The constraint makes the clinical prior explicit in the model.
#: ``node_not_sampled`` is deliberately excluded. Its negative fitted effect
#: is a treatment-selection proxy -- nodes go unsampled in lower-risk
#: operations -- rather than biological protection. Every retained variable is
#: an adverse factor whose Stage-B coefficient is constrained non-negative.
HEAD_FEATURE_NAMES: tuple[str, ...] = (
    "isup",
    "tertiary",
    "pt_rank",
    "margin_pos",
    "epe",
    "svi",
    "lvi",
    "node_pos",
    "pirads",
    "psad",
    "bx_isup",
)


def feature_vector(
    prompt: dict[str, Any],
    clinical: dict[str, Any],
    names: tuple[str, ...] | list[str] = HEAD_FEATURE_NAMES,
) -> list[float]:
    """``clinical_features`` in *names* order (default :data:`HEAD_FEATURE_NAMES`).

    Inference passes the artifact's own ``clinical.features`` list, so an
    artifact fitted on a different subset keeps scoring correctly.
    """
    feats = clinical_features(prompt, clinical)
    return [feats[name] for name in names]
