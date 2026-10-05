"""Offline deterministic reference policies for CHIMERA Tasks 1 and 2.

These policies provide reproducible baselines and disagreement sets for the
policy-guided LLM. They remain independent of LangChain so they can be scored
without loading a model. The new inference path must not enforce their outputs
or use them as a clinical fallback.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from chimera_agent_baseline.task1 import apply_policy as apply_task1_stages
from chimera_agent_baseline.task1 import load_config
from chimera_agent_baseline.task2 import apply_policy as apply_task2_stages
from chimera_agent_baseline.task2 import load_config as load_task2_config

_TASK1_CFG: dict[str, Any] | None = None
_TASK2_CFG: dict[str, Any] | None = None


def _task2_cfg() -> dict[str, Any]:
    global _TASK2_CFG
    if _TASK2_CFG is None:
        _TASK2_CFG = load_task2_config()
    return _TASK2_CFG


def _task1_cfg() -> dict[str, Any]:
    """Loaded lazily so an unreadable config surfaces at the call site rather
    than breaking module import."""
    global _TASK1_CFG
    if _TASK1_CFG is None:
        _TASK1_CFG = load_config()
    return _TASK1_CFG


ConfidenceValue = Literal["clear", "borderline", "uncertain"]

# The tool that serves each revealable section, for the reported reveal_sequence.
_SECTION_TOOLS = {
    "radiology_report": "get_mri_report",
    "psa_trend": "get_psa_trend",
    "laboratory_results": "get_lab_results",
    "previous_notes": "get_previous_notes",
    "family_history": "get_family_history",
    "pathology_report": "get_pathology_report",
}

# Where each Task 1 variable's value is read from, for the rationale labels.
_FIELDS = {
    "pirads": "pirads",
    "psa": "psa",
    "psad": "psad",
    "vol": "vol",
    "dre": "dre",
    "age": "age",
    "comorbidity": "pmhx",
    "fh": "family_history",
    "cspca": "cspca",
    "bx": "bx",
}


@dataclass(frozen=True)
class PolicyFactor:
    """One observable fact the rationale should name."""

    variable: str
    text: str


@dataclass(frozen=True)
class FrozenPolicyResult:
    """Reference result plus an explanation of the matched rule."""

    task: int
    decision: str
    confidence: ConfidenceValue
    rule: str
    state: str
    weights: dict[str, str]
    factors: tuple[PolicyFactor, ...]
    missing_evidence: tuple[str, ...] = ()
    # What would make a `no` a `yes`. Empty on a `yes`, where the recommendation
    # already names the next step.
    trigger: str = ""
    required_tools: tuple[str, ...] = ()
    # Tools whose sections we report having consulted. Empty means "whatever
    # the model called", which is still the Task 2 behaviour.
    reveal_tools: tuple[str, ...] = ()
    # A fact about the record that changes no decision: the digital-pathology
    # grade disagrees sharply with the reported one. Audit only -- see
    # `task2_policy.yaml -> pathology_review`.
    review_flag: str = ""

    def to_audit_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip().lower() in {"", "—", "na", "n/a", "unknown"}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None else None


def _revealed_value(revealed: dict[str, Any], tool: str, field: str) -> Any:
    payload = revealed.get(tool)
    return payload.get(field) if isinstance(payload, dict) else None


def _prior_grade(revealed: dict[str, Any]) -> int | None:
    """ISUP grade of the previous biopsy, if the retrieval step found one.

    The grade is prose in the previous notes -- "low-grade disease", "Gleason
    3+4", "ISUP grade group 3" -- so reading it is the model's job, not a
    pattern's.  The retrieval layer is expected to leave an integer 1-5 here;
    until it does, this returns None and the grade rule simply never fires.
    """
    value = _revealed_value(revealed, "get_previous_notes", "prior_isup_grade")
    if value is None:
        value = revealed.get("prior_isup_grade")
    grade = _as_int(value)
    return grade if grade is not None and 1 <= grade <= 5 else None


_AI_GRADE = re.compile(r"AI-predicted ISUP grade group[^0-9]{0,40}?(\d)", re.IGNORECASE)


def _ai_predicted_grade(revealed: dict[str, Any]) -> int | None:
    """The digital-pathology grade the report carries beside the reported one.

    Read only to be *flagged*. It is higher than the signed grade in 54 of the
    64 reports carrying one -- including every active-surveillance case -- so it
    is plainly miscalibrated for this cohort and must never move a risk group.
    """
    report = _revealed_value(revealed, "get_pathology_report", "pathology_report")
    match = _AI_GRADE.search(str(report or ""))
    return int(match.group(1)) if match else None


def _family_history(revealed: dict[str, Any]) -> str:
    value = _revealed_value(revealed, "get_family_history", "family_history")
    return str(value or "unknown").strip().lower()


def apply_task1_policy(prompt: dict[str, Any], revealed: dict[str, Any]) -> FrozenPolicyResult:
    """Task 1 asks three questions in order (``resources/task1_policy.yaml``).

    1. does this person have cancer?   2. how sure are we?   3. if we are not
    sure, is the tissue worth taking?  Each variable's reported weight is the
    role it played in the branch this case reached.
    """
    cfg = _task1_cfg()
    verdict = apply_task1_stages(prompt, cfg, _family_history(revealed), _prior_grade(revealed))
    # The findings the rationale must name are the ones we reported as important
    # or decisive and that we can actually point at: a revealed document, or the
    # patient card. Encounter type and comorbidity count are neither.
    observable = set(cfg["patient_card"])
    factors = tuple(
        PolicyFactor(name, cfg["labels"][name].format(value=prompt.get(_FIELDS[name])))
        for name, weight in verdict.weights.items()
        if weight in ("important", "decisive") and (cfg["sections"].get(name) or name in observable)
        and prompt.get(_FIELDS[name]) is not None
    )
    return FrozenPolicyResult(
        task=1,
        decision=verdict.decision,
        confidence=verdict.confidence,
        rule="; ".join(verdict.reasons),
        state=verdict.stage,
        weights=verdict.weights,
        factors=factors,
        missing_evidence=_task1_undocumented(prompt, cfg, _prior_grade(revealed)),
        trigger=cfg["triggers"].get(verdict.stage, "") if verdict.decision == "no" else "",
        required_tools=("get_mri_report", "get_psa_trend", "get_previous_notes"),
        reveal_tools=tuple(_SECTION_TOOLS[section] for section in verdict.reveals),
    )


def join_clauses(items) -> str:
    """ "a", "a and b", "a, b and c" -- so a limitation sentence reads as one."""
    items = list(items)
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _task1_undocumented(prompt: dict[str, Any], cfg: dict, prior_grade: int | None) -> tuple[str, ...]:
    """The evidence the reference rationales keep asking for and the record lacks.

    Both claims are assertions about the record, so each is made only where it
    has actually been checked: the grade claim needs the notes to have been read
    and to have come back empty, and the lesion claim rests on the corpus read
    that found prior-size language in 0 of 195 Task 1 records.
    """
    encounter = str(prompt.get("enc_type") or "")
    pirads = _as_float(prompt.get("pirads"))
    missing: list[str] = []

    grade_rule = cfg["undocumented"]["prior_grade"]
    if encounter in grade_rule["encounters"] and prior_grade is None:
        missing.append(grade_rule["text"])

    lesion_rule = cfg["undocumented"]["lesion_change"]
    if encounter in lesion_rule["encounters"] and (pirads or 0) >= lesion_rule["min_pirads"]:
        missing.append(lesion_rule["text"])

    return tuple(missing)


def apply_task2_policy(prompt: dict[str, Any], revealed: dict[str, Any]) -> FrozenPolicyResult:
    """Task 2 asks two questions (``resources/task2_policy.yaml``).

    What do we know about this cancer, and does anything else change what the
    patient actually needs? The reported ISUP grade answers the first and is
    authoritative. LUTS/BPO is represented but provisionally managed as a
    separate problem rather than an indication to treat low-risk cancer.
    """
    cfg = _task2_cfg()
    staged_prompt = dict(prompt)
    if _has_adverse_as_histology(revealed):
        staged_prompt["_adverse_as_histology"] = True
    if _cribriform_excluded(revealed):
        staged_prompt["_cribriform_excluded"] = True
    if _histology_stable_across_timepoints(revealed):
        staged_prompt["_histology_stable_timepoints"] = True
    if _single_biopsy_timepoint(revealed):
        staged_prompt["_single_biopsy_timepoint"] = True
    if _psa_confounder(revealed):
        staged_prompt["_psa_confounder"] = True
    verdict = apply_task2_stages(staged_prompt, cfg)
    review = _pathology_review(prompt, revealed, cfg)
    # Only findings the judge can actually find. See `assertable` in the YAML:
    # its view is the clinical-data socket, so a fact that lives only in
    # `prompt.json` -- age, the comorbidity count, the encounter type -- reads
    # as a fabrication however true it is.
    assertable = set(cfg["assertable"])
    # Some facts define the branch without carrying its weight -- see
    # `present_regardless`. The weight stays where the reference puts it; the
    # fact is presented anyway, because the rationale is a description of the
    # case and not a read-out of `variable_weights`.
    regardless = set(cfg.get("present_regardless", {}).get(verdict.stage, ()))
    factors = tuple(
        PolicyFactor(name, cfg["labels"][name].format(value=prompt.get(name)))
        for name, weight in verdict.weights.items()
        if (weight in ("important", "decisive") or name in regardless)
        and name in assertable
        and prompt.get(name) is not None
    )
    return FrozenPolicyResult(
        task=2,
        decision=verdict.decision,
        confidence=verdict.confidence,
        rule="; ".join(verdict.reasons),
        state=verdict.stage,
        weights=verdict.weights,
        factors=factors,
        # A surveillance recommendation that does not say what would end
        # surveillance is a refusal to treat rather than a plan; 10 of the 14
        # reference rationales on these two branches name the trigger outright.
        # The table in the YAML is the gate -- a branch with no entry says
        # nothing. Task 2 declares no `missing_evidence`: no Task 2 reference
        # rationale asks for anything the record lacks.
        trigger=cfg.get("triggers", {}).get(verdict.stage, ""),
        required_tools=("get_pathology_report", "get_previous_notes"),
        # Every reference reveal sequence in Task 2 is empty, and an empty
        # sequence scores full tool precision. Retrieval still happens; this is
        # what we report having consulted.
        reveal_tools=(),
        review_flag=review,
    )


def _has_adverse_as_histology(revealed: dict[str, Any]) -> bool:
    """Detect a positive cribriform/IDC statement without negation.

    Kept separate from the policy switch so the strong EAU exclusion can be
    enabled or disabled as an independent provisional ablation.
    """
    payload = revealed.get("get_pathology_report") or {}
    report = str(payload.get("pathology_report") or "").lower()
    for clause in re.split(r"[.\n]+", report):
        if not re.search(r"\bcribriform\b|\bidc-p\b|\bintraductal carcinoma\b", clause):
            continue
        if re.search(r"\b(?:no|not|without|absent|absence)\b", clause):
            continue
        return True
    return False


def _cribriform_excluded(revealed: dict[str, Any]) -> bool:
    """Is cribriform/IDC *positively reported as absent*?

    Deliberately not the negation of ``_has_adverse_as_histology``: a report
    that never mentions the pattern has not excluded it, and the surveillance
    side of the GG2 band has to be earned by a written statement rather than by
    silence. 147 of the 153 reports mention the pattern either way.
    """
    payload = revealed.get("get_pathology_report") or {}
    report = str(payload.get("pathology_report") or "").lower()
    if not report or _has_adverse_as_histology(revealed):
        return False
    return bool(
        re.search(
            r"\b(?:no|not|without|absent|absence of)\b[^.\n]*"
            r"(?:\bcribriform\b|\bidc-p\b|\bintraductal\b)",
            report,
        )
    )


def _histology_stable_across_timepoints(revealed: dict[str, Any]) -> bool:
    """Does the pathology state stability across MORE THAN ONE biopsy?

    ⚠️ The corpus writes a stability summary for single-biopsy patients too
    ("stable across the single available timepoint"), which is vacuous -- one
    sample cannot demonstrate a trajectory. Those are excluded here, and any
    sentence reporting an upgrade or progression vetoes the flag outright.
    """
    payload = revealed.get("get_pathology_report") or {}
    report = str(payload.get("pathology_report") or "").lower()
    if not report:
        return False
    stable = False
    for clause in re.split(r"[.\n]+", report):
        if re.search(r"\bupgrade\w*\b|\bprogression\b", clause) and not re.search(
            r"\b(?:no|not|without|absent)\b", clause
        ):
            return False
        if not re.search(r"\bstabilit\w*\b|\bstable\b", clause):
            continue
        if re.search(r"\bsingle\b", clause):
            continue
        if re.search(r"\bboth\b|\btwo\b|\bacross timepoints\b|\bacross the documented\b"
                     r"|\bacross the available timepoints\b|\bacross both\b", clause):
            stable = True
    return stable


def _single_biopsy_timepoint(revealed: dict[str, Any]) -> bool:
    """Does the pathology state that only ONE biopsy timepoint exists?

    Positively stated, like the stability read: 27 of the 153 reports say so in
    as many words ("across the single available timepoint"). A report that
    simply describes one specimen has not told us the sampling is complete.
    """
    payload = revealed.get("get_pathology_report") or {}
    report = str(payload.get("pathology_report") or "").lower()
    if not report:
        return False
    return bool(re.search(r"\bsingle\b[^.\n]*\btimepoint\b", report))


_PSA_CONFOUNDERS = (
    r"prostatitis",
    r"urinary tract infection|\buti\b",
    r"catheter\w*|instrumentation",
    r"(?:acute )?urinary retention",
    r"recent (?:biopsy|turp|resection)",
)


def _psa_confounder(revealed: dict[str, Any]) -> bool:
    """Is a non-malignant explanation for a raised PSA documented anywhere?

    PSA is not cancer-specific. Prostatitis, infection, instrumentation and
    retention all raise it, and the EAU asks for the PSA to be repeated under
    standardised conditions before an elevation is treated as a risk feature.
    The branches that read this do not discard the PSA -- they decline to let a
    single confounded measurement force an irreversible recommendation.

    A negated mention ("no evidence of prostatitis") does not count, and neither
    does silence: this must be positively documented, like the other pathology
    reads above.
    """
    for payload in revealed.values():
        if not isinstance(payload, dict):
            continue
        for value in payload.values():
            for clause in re.split(r"[.\n]+", str(value or "").lower()):
                if not any(re.search(term, clause) for term in _PSA_CONFOUNDERS):
                    continue
                if re.search(r"\b(?:no|not|without|absent|absence|negative for)\b", clause):
                    continue
                return True
    return False


def _pathology_review(prompt: dict[str, Any], revealed: dict[str, Any], cfg: dict) -> str:
    """Flag a large reported-vs-predicted grade disagreement. Decides nothing.

    The reported grade stays authoritative; discarding the disagreement in
    silence is what this replaces. Audit only while `in_rationale` is false --
    rationale prose is judged and this claim is unmeasurable offline.
    """
    rules = cfg.get("pathology_review") or {}
    if not rules.get("in_audit"):
        return ""
    reported, predicted = _as_int(prompt.get("bx_isup")), _ai_predicted_grade(revealed)
    if reported is None or predicted is None:
        return ""
    gap = predicted - reported
    return rules["text"].format(gap=gap) if gap >= rules["discordance_at_least"] else ""


def apply_frozen_policy(task: int, prompt: dict[str, Any], revealed: dict[str, Any]) -> FrozenPolicyResult:
    if task == 1:
        return apply_task1_policy(prompt, revealed)
    if task == 2:
        return apply_task2_policy(prompt, revealed)
    raise ValueError(f"Frozen policy is available only for Tasks 1 and 2, got {task}")


def enforce_frozen_policy(
    judgment: dict[str, Any],
    policy: FrozenPolicyResult,
    eligible_variables: list[str],
) -> dict[str, Any]:
    """Overwrite policy-owned fields while retaining the LLM's prose/detail."""
    enforced = dict(judgment)
    if policy.task == 1:
        enforced["biopsy_decision"] = policy.decision
    else:
        recommendation = dict(enforced.get("treatment_recommendation") or {})
        recommendation["primary"] = policy.decision
        if policy.decision != "active_treatment":
            recommendation["modalities"] = []
        else:
            recommendation["as_protocol"] = None
            recommendation["as_trigger"] = None
        enforced["treatment_recommendation"] = recommendation
    enforced["confidence"] = policy.confidence
    enforced["variable_weights"] = {key: policy.weights[key] for key in eligible_variables}
    return enforced


def build_policy_audit(
    *,
    policy: FrozenPolicyResult,
    raw_judgment: dict[str, Any],
    eligible_variables: list[str],
    observed_tools: list[str],
    form_fill_warnings: list[str],
    llm_parse_success: bool = True,
    rationale_only: bool = False,
    rationale_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record evidence acquisition and generated-rationale quality metrics."""
    if policy.task == 1:
        raw_decision = raw_judgment.get("biopsy_decision")
    else:
        recommendation = raw_judgment.get("treatment_recommendation") or {}
        raw_decision = recommendation.get("primary") if isinstance(recommendation, dict) else None

    raw_confidence = raw_judgment.get("confidence")
    raw_weights = raw_judgment.get("variable_weights") or {}
    if not isinstance(raw_weights, dict):
        raw_weights = {}

    ordinal = {"not_used": 0, "noted": 1, "important": 2, "decisive": 3}
    weight_errors: list[float] = []
    for key in eligible_variables:
        expected = policy.weights[key]
        observed = raw_weights.get(key)
        if observed in ordinal:
            weight_errors.append(abs(ordinal[expected] - ordinal[observed]) / 3.0)
        else:
            weight_errors.append(1.0)
    weight_score = 1.0 - sum(weight_errors) / len(weight_errors) if weight_errors else 1.0

    expected_important = {key for key in eligible_variables if policy.weights[key] in {"important", "decisive"}}
    raw_important = {key for key in eligible_variables if raw_weights.get(key) in {"important", "decisive"}}
    if not expected_important and not raw_important:
        factor_f1 = 1.0
    elif not expected_important or not raw_important:
        factor_f1 = 0.0
    else:
        overlap = len(expected_important & raw_important)
        factor_f1 = 2 * overlap / (len(expected_important) + len(raw_important))

    expected_tools = set(policy.required_tools)
    called_tools = set(observed_tools)
    tool_recall = len(expected_tools & called_tools) / len(expected_tools) if expected_tools else 1.0
    clinical_calls = {tool for tool in called_tools if tool.startswith("get_")}
    tool_precision = (
        len(expected_tools & clinical_calls) / len(clinical_calls)
        if clinical_calls
        else (1.0 if not expected_tools else 0.0)
    )

    confidence_ordinal = {"uncertain": 0, "borderline": 1, "clear": 2}
    confidence_distance = None
    if raw_confidence in confidence_ordinal:
        confidence_distance = abs(confidence_ordinal[policy.confidence] - confidence_ordinal[raw_confidence])

    rationale = str(raw_judgment.get("free_text") or "").lower()
    factor_terms = {
        "pirads": ("pi-rads", "pirads"),
        "psad": ("psa density", "psad"),
        "psa": ("psa",),
        "dre": ("dre", "digital rectal", "rectal examination"),
        "fh": ("family history", "familial"),
        "bx": ("biopsy", "tissue", "grade", "gleason"),
        "bx_isup": ("isup", "grade group", "biopsy"),
        "bx_gl_prim": ("gleason", "primary pattern", "pattern 3", "pattern 4", "pattern 5"),
        "bx_gl_sec": ("gleason", "secondary pattern"),
        "ct": ("clinical stage", "ct1", "ct2", "ct3", "ct4"),
        "comorbidity": ("life expectancy", "curative", "palliative", "comorbid"),
        "age": ("age", "year-old", "years old"),
    }
    mentions = []
    for factor in policy.factors:
        terms = factor_terms.get(factor.variable, (factor.variable.replace("_", " "),))
        mentions.append(any(term in rationale for term in terms))
    factor_coverage = sum(mentions) / len(mentions) if mentions else 1.0

    if policy.task == 1:
        decision_terms = (
            ("recommend biopsy", "proceed with biopsy", "biopsy is indicated", "biopsy is recommended")
            if policy.decision == "yes"
            else ("no biopsy", "omit biopsy", "against biopsy", "biopsy is not", "biopsy is unnecessary")
        )
    else:
        decision_terms = (policy.decision.replace("_", " "),)
    rationale_decision_mention = float(any(term in rationale for term in decision_terms))
    missing_disclosure = 1.0
    if policy.missing_evidence:
        missing_disclosure = float(
            any(term in rationale for term in ("missing", "unknown", "unavailable", "not documented", "insufficient"))
        )

    structured_metrics: dict[str, float | int | None]
    if rationale_only:
        structured_metrics = {
            "decision_adherence": None,
            "confidence_adherence": None,
            "confidence_ordinal_distance": None,
            "weight_ordinal_score": None,
            "important_decisive_f1": None,
        }
    else:
        structured_metrics = {
            "decision_adherence": float(raw_decision == policy.decision),
            "confidence_adherence": float(raw_confidence == policy.confidence),
            "confidence_ordinal_distance": confidence_distance,
            "weight_ordinal_score": weight_score,
            "important_decisive_f1": factor_f1,
        }

    validation = rationale_validation or {}
    return {
        "policy": policy.to_audit_dict(),
        "llm_pre_enforcement": {
            "decision": None if rationale_only else raw_decision,
            "confidence": None if rationale_only else raw_confidence,
            "variable_weights": {} if rationale_only else raw_weights,
            "free_text": raw_judgment.get("free_text"),
        },
        "llm_generated_fields": ["free_text"]
        if rationale_only
        else [
            "decision",
            "confidence",
            "variable_weights",
            "free_text",
        ],
        "observed_tools": observed_tools,
        "eligible_variables": eligible_variables,
        "form_fill_warnings": form_fill_warnings,
        "llm_parse_success": llm_parse_success,
        "metrics": {
            **structured_metrics,
            "required_tool_recall": tool_recall,
            "required_tool_precision": tool_precision,
            "form_fill_retry_count": len(form_fill_warnings),
            "llm_parse_success": float(llm_parse_success),
            "rationale_factor_coverage": factor_coverage,
            "rationale_decision_mention": rationale_decision_mention,
            "missing_evidence_disclosure": missing_disclosure,
            "rationale_internal_language_free": float(validation.get("internal_language_free", False)),
            "rationale_recommendation_consistent": float(validation.get("recommendation_consistent", False)),
            "rationale_numeric_grounding": float(validation.get("numeric_grounding", False)),
            "rationale_clinical_term_grounding": float(validation.get("clinical_term_grounding", False)),
            "rationale_evidence_corroboration": float(validation.get("evidence_corroboration", False)),
        },
    }
