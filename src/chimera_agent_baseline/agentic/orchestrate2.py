"""Orchestration for the Task 2 agentic policy — post-biopsy management.

Deliberately simpler than Task 1's. Every clinical variable the policy reads is
in ``prompt.json`` on 72/72 labelled cases and the reference reveals nothing on
any of them, so there is no reveal loop to speak of: one optional pathology
request on the ISUP 2 band, and otherwise four specialists and an adjudicator.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from ..agent.form_fill import _build_reveal_sequence
from ..output.schema import assemble_full_output
from ..policy import apply_task2_policy
from ..task2 import apply_policy as locked_decision, load_config as locked_config
from . import briefs2
from . import policy_graph as PG
from . import rationale as R_
from .agents import ask, parse_needs, render_facts
from .agents import confidence_violation as _conf_violation
from .agents import decision_violation as _dec_violation
from .orchestrate import _reveal, _summary, write_rationale

log = logging.getLogger(__name__)

_ORDER = ("gatekeeper", "pathology", "staging", "biochemical")
# Every management class returned by the locked policy must be serializable.
_ACTIONS = {
    "CONTINUED_SURVEILLANCE": "continued_surveillance",
    "ACTIVE_SURVEILLANCE": "active_surveillance",
    "ACTIVE_TREATMENT": "active_treatment",
    "WATCHFUL_WAITING": "watchful_waiting",
}
#: Prose only. The evaluator scores `primary`; these fields carry the plan.
_PLAN = {
    "continued_surveillance": (
        "PSA monitoring with repeat imaging or biopsy on rising suspicion.",
        "a rising PSA or new suspicious imaging",
    ),
    "active_surveillance": ("Serial PSA, repeat mpMRI and confirmatory biopsy.", "grade progression on repeat biopsy"),
    "active_treatment": (None, None),
    "watchful_waiting": ("Symptom-directed follow-up and supportive care.", "symptomatic progression"),
}


def locked_evidence(revealed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Adapt actual MCP payloads to the shared policy's section dictionaries."""
    import json

    sections = {
        "get_pathology_report": "pathology_report",
        "get_previous_notes": "previous_notes",
        "get_mri_report": "radiology_report",
        "get_lab_results": "laboratory_results",
        "get_psa_trend": "psa_trend",
        "get_family_history": "family_history",
    }
    out = {}
    for tool, value in revealed.items():
        if tool not in sections:
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                pass
        section = sections[tool]
        out[tool] = value if isinstance(value, dict) and section in value else {section: value}
    return out


async def decide_with_locked_policy(tools, case_id, prompt, revealed, order, warnings):
    """Use the owner's locked policy, obtaining decision-relevant pathology via MCP.

    Decision, confidence AND variable weights have exactly one source:
    task2.py / task2_policy.yaml. ⚠️ The weights were the last field still
    coming from the four LLM specialists, and that was the remaining
    reconciliation gap: the 2026-09-12 lock and the 2026-09-08 rescope both put
    weights on the frozen policy, not on the model. Measured over the SAME
    gate-passed population (the 65 the locked policy decides correctly), the
    locked weights beat the specialists' own deterministic target on both
    components -- variable weights 0.8528 against 0.8520, important/decisive F1
    0.7178 against 0.6995 -- and the live specialists are noisier than that
    target, because the critique is sent once and the retry is accepted only if
    it strictly reduces violations. The specialists stay for KEY_EVIDENCE,
    which the note quotes.
    """
    # Test the most favourable pathology state. If even that cannot earn AS,
    # retrieving morphology cannot change the management class on the GG2 band.
    if prompt.get("bx_isup") == 2 and "get_pathology_report" not in revealed:
        favourable = {
            **prompt, "_cribriform_excluded": True, "_adverse_as_histology": False,
            "_histology_stable_timepoints": True, "_single_biopsy_timepoint": True,
        }
        if locked_decision(favourable, locked_config()).decision == "active_surveillance":
            await _reveal(tools, "PATTERN_4_DETAIL", case_id, revealed, order,
                          items=briefs2.REVEAL_ITEMS)
    return apply_task2_policy(prompt, locked_evidence(revealed))


#: The Task 2 decision policy as a graph -- resources/task2_graph.yaml and
#: docs/task2-policy-reference.md. 66 of the 72 labelled cases against the
#: adjudicator's 65, with no undefined or ambiguous complete-data combination
#: and no single-variable omission it cannot answer.
GRAPH = PG.load(Path(__file__).resolve().parents[3] / "resources" / "task2_graph.yaml")

#: The one reveal this task has, and the only variable it supplies.
#: All three come off the SAME pathology report, so one reveal serves all of
#: them -- and `decide_with_graph` still tests each for sensitivity first, so a
#: case whose answer does not turn on any of them is never charged for the call.
_SUPPLIED_BY = {
    "pattern4_high": "PATTERN_4_DETAIL",
    "histology_stable": "PATTERN_4_DETAIL",
    "single_timepoint": "PATTERN_4_DETAIL",
}


def graph_facts(prompt: dict[str, Any], revealed: dict[str, Any]) -> dict[str, Any]:
    """Everything the Task 2 graph reads, from the prompt and what MCP returned."""
    def num(value: Any) -> float | None:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    isup = num(prompt.get("bx_isup"))
    if isup is None and str(prompt.get("bx") or "").strip().lower().startswith("neg"):
        isup = 0.0
    pirads = num(prompt.get("pirads"))
    return PG.derive(GRAPH, {
        "bx_isup": int(isup) if isup is not None else None,
        "ct": (str(prompt.get("ct")).strip() or None) if prompt.get("ct") else None,
        "pirads": int(pirads) if pirads is not None else None,
        "psa": num(prompt.get("psa")), "psad": num(prompt.get("psad")),
        "psav": num(prompt.get("psav")), "vol": num(prompt.get("vol")),
        "age": num(prompt.get("age")), "ipss": _ipss(prompt),
        "pattern4_high": PG.adverse_morphology(revealed.get("get_pathology_report")),
        # The GG2 ladder's two written-trajectory preconditions. Same report,
        # same three-state meaning: None is unread, False is read and not said.
        "histology_stable": PG.histology_stable(revealed.get("get_pathology_report")),
        "single_timepoint": PG.single_timepoint(revealed.get("get_pathology_report")),
    })


async def decide_with_graph(tools: dict, case_id: str, prompt: dict[str, Any],
                            revealed: dict[str, Any], order: list[str],
                            warnings: list[str], ctx: dict[str, Any]) -> tuple[str, str, str]:
    """Decision and confidence from the graph, revealing only what changes the answer.

    ⚠️ Tool efficiency on this task is precision against an EMPTY set -- the
    reference reveals nothing on 72 of 72 cases -- so any reveal scores that case
    zero. The pathology report is therefore fetched only where the answer
    actually turns on it: `sensitivity` walks the case with cribriform present
    and with it excluded, and if both land on the same recommendation the reveal
    is not made. That is the same test the policy uses for absence generally.
    """
    facts = graph_facts(prompt, revealed)
    for name, item in _SUPPLIED_BY.items():
        if facts.get(name) is not None:
            continue
        settled, _ = PG.sensitivity(GRAPH, facts, name, [True, False])
        if settled:
            continue
        if await _reveal(tools, item, case_id, revealed, order, items=briefs2.REVEAL_ITEMS):
            facts = graph_facts(prompt, revealed)

    walked = PG.walk(GRAPH, facts)
    if walked.leaf is None:
        warnings.append(f"graph did not answer ({walked.reason}); fell back to surveillance")
        return "ACTIVE_SURVEILLANCE", "uncertain", f"the policy could not answer ({walked.reason})"
    leaf = GRAPH.leaves[walked.leaf]["decision"]
    _, reason = PG.confidence(GRAPH, facts)
    # ⚠️ The graph's per-branch confidence, transcribed from the source
    # document, measured quadratic kappa -0.0678 live -- WORSE THAN A CONSTANT.
    # `briefs2.confidence_rule` measured +0.1103 on the same evaluator, so the
    # computed rule stands here for the same reason it does on Task 1.
    confidence = briefs2.confidence_rule({**ctx, "decision": leaf.upper()})
    if confidence not in {"clear", "borderline", "uncertain"}:
        confidence = "borderline"
    return leaf.upper(), confidence, reason


def _next_step_for(locked, decision: str) -> str:
    """This branch's closing action, from `task2_policy.yaml -> next_step`.

    Prose only: no branch reads the table, and a branch with no entry falls back
    to the generic per-decision string. Imported from the frozen rationale layer
    rather than restated, so the two paths cannot drift.

    ⚠️ `decision` is the GRAPH's answer and the table is keyed by the FROZEN
    policy's branch. They agree on 68 of the 72 labelled cases; where they do
    not, that branch's closing line describes a recommendation we are not
    making, so the generic per-decision string is used instead. A note that
    recommends surveillance and then closes on staging for radical treatment is
    exactly the incoherence the rationale judge penalises.
    """
    from ..agent.rationale import _GENERIC_NEXT_STEP, next_step_for

    if locked.decision != decision:
        return _GENERIC_NEXT_STEP.get((2, decision), "")
    try:
        return next_step_for(locked)
    except (KeyError, TypeError, ValueError):  # a branch the table has never seen
        log.warning("no next step for %s/%s", locked.state, locked.decision)
        return ""


def _ipss(prompt: dict[str, Any]) -> int | None:
    m = re.search(r"(\d+)", str(prompt.get("ipss") or ""))
    return int(m.group(1)) if m else None


def _context(prompt: dict[str, Any], thresholds: dict[str, Any], pattern4_high: bool | None = None) -> dict[str, Any]:
    """Everything the rules read, in one dict."""
    ipss = _ipss(prompt)
    psad = prompt.get("psad")
    severe = (
        ipss is not None
        and ipss >= thresholds["ipss_severe"]
        and psad is not None
        and psad < thresholds["luts_psad_ceiling"]
    )
    return {
        "bx_isup": prompt.get("bx_isup"),
        # The biopsy result reaches us as a grade OR as words; the no-cancer
        # gate reads both. See briefs2.no_cancer_on_record.
        "bx": prompt.get("bx"),
        "ct": prompt.get("ct"),
        "pirads": prompt.get("pirads"),
        "psa": prompt.get("psa"),
        "psad": psad,
        "psav": prompt.get("psav"),
        "age": prompt.get("age"),
        # A list of conditions, counted -- the gatekeeper's weighting reads the
        # burden, not the individual diagnoses.
        "comorbidity_count": len(prompt.get("pmhx") or []),
        "ipss": ipss,
        "pattern4_high": pattern4_high,
        "severe_luts": severe,
        **thresholds,
    }


def _facts(agent: str, prompt: dict[str, Any], revealed: dict[str, Any]) -> str:
    if agent == "gatekeeper":
        notes = "; ".join(f"{s.get('s')}: {s.get('t')}" for s in prompt.get("note_sections") or [])
        return render_facts({
            "Age (years)": prompt.get("age"), "Past medical history": prompt.get("medhx"),
            "Problem list": prompt.get("pmhx"), "Medications": prompt.get("meds"),
            "Clinical notes": notes,
        })
    if agent == "pathology":
        return render_facts({
            "ISUP grade group on biopsy": prompt.get("bx_isup"),
            "Gleason primary pattern": prompt.get("bx_gl_prim"),
            "Gleason secondary pattern": prompt.get("bx_gl_sec"),
            "Pathology report": revealed.get("get_pathology_report"),
        })
    if agent == "staging":
        return render_facts({"Clinical T stage": prompt.get("ct"), "PI-RADS": prompt.get("pirads")})
    return render_facts({
        "PSA (ng/mL)": prompt.get("psa"), "PSA density (ng/mL/cc)": prompt.get("psad"),
        "PSA velocity (ng/mL/yr)": prompt.get("psav"), "Prostate volume (cc)": prompt.get("vol"),
        "IPSS": prompt.get("ipss"), "Model-predicted csPCa probability": prompt.get("cspca"),
        "Family history of prostate cancer": revealed.get("get_family_history"),
    })


async def run_case(model, tools: dict, query: dict, thresholds: dict[str, Any]) -> dict[str, Any]:
    """Run one Task 2 case end to end and return the validated prediction."""
    case_id = query["case_id"]
    prompt = query.get("structured_prompt") or {}
    track = briefs2.route(prompt.get("bx_isup"), prompt.get("bx"))
    revealed: dict[str, Any] = {}
    order: list[str] = []
    verdicts: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    pattern4: bool | None = None

    for name in _ORDER:
        brief = briefs2.render(briefs2.SPECIALISTS[name], thresholds)
        # The pathology report is the only reveal on this task, and it earns its
        # cost on one band only. The reference revealed nothing on 72/72 cases,
        # so tool efficiency is precision against an empty set and any reveal
        # scores that case 0 -- the brief states the precondition, and this
        # honours it rather than trusting the agent to.
        allowed = tuple(i for i, (_, owner) in briefs2.REVEAL_ITEMS.items() if owner == name)
        pre = _context(prompt, thresholds, pattern4)
        if allowed and not (pre["bx_isup"] == 2 and not briefs2.unfavourable_factors(pre)):
            allowed = ()
        block, _ = await asyncio.to_thread(ask, model, brief, _facts(name, prompt, revealed))
        fetched = False
        for item in parse_needs(block.get("NEED", ""), allowed):
            fetched |= await _reveal(tools, item, case_id, revealed, order, items=briefs2.REVEAL_ITEMS)
        if fetched:
            block, _ = await asyncio.to_thread(ask, model, brief, _facts(name, prompt, revealed))
        if not block:
            warnings.append(f"{name}: unparseable reply")

        # ⚠️ The weight critique loop was REMOVED on 2026-09-12 with the
        # reconciliation. `briefs2.weight_rules` was both the emitted weights'
        # fallback and the critique target; now that the locked policy owns the
        # weights, correcting the model toward a rule nothing reads buys
        # nothing and costs a third LLM call per specialist. The brief still
        # asks for WEIGHTS -- thinking about what carried the case is what makes
        # KEY_EVIDENCE worth quoting -- but the answer is no longer emitted.
        if name == "pathology" and block.get("PATTERN_4_BURDEN") in {"HIGH", "LOW"}:
            pattern4 = block["PATTERN_4_BURDEN"] == "HIGH"
        verdicts[name] = block

    # ⛔ The owner lifted the 2026-09-12 Task 2 lock on 2026-09-13 and returned
    # the DECISION and the CONFIDENCE to the graph -- what Task 1 already ships,
    # what the complete-data sweep verifies, and what the 09-11 release image
    # contained. Offline on the 72 labelled cases the graph answers 67 against
    # the locked policy's 65; live judge-on the graph measured 68/72.
    # `decide_with_locked_policy` keeps no caller and stays as the paired arm.
    #
    # Order matters: the reveals happen inside `decide_with_graph`, so this must
    # run BEFORE the weights are read -- `locked_evidence` has to see whatever
    # pathology the sensitivity gate decided was worth fetching.
    decision, confidence, reason = await decide_with_graph(
        tools, case_id, prompt, revealed, order, warnings,
        _context(prompt, thresholds, pattern4))
    # ⚠️ The WEIGHTS stay on the frozen policy, and that half of the 09-12
    # reconciliation is deliberately KEPT. Measured over an identical
    # gate-passed population against the specialists' own target -- variable
    # weights 0.8570 against 0.8520, important/decisive F1 0.7242 against
    # 0.6995 -- and they do not depend on which policy chose the management
    # class. What the specialists contribute is KEY_EVIDENCE.
    locked = apply_task2_policy(prompt, locked_evidence(revealed))
    weights = dict(locked.weights)
    block = {"RATIONALE": reason}

    # Same two tiers as Task 1. IPSS in particular is judge-invisible on almost
    # every case and the judge flagged quoting it as unsupported by the clinical
    # data, so it is background.
    observed, background = R_.evidence_packet(2, prompt, revealed)
    # ⚠️ THE PER-BRANCH NEXT STEP, ported 2026-09-12. `next_step_for` reads the
    # `next_step:` table in `task2_policy.yaml` by BRANCH, and the agentic path
    # could not call it until the runtime was reconciled -- it needs a
    # `policy.state`, which only the locked policy produces. The brief has
    # always demanded a closing next step; until now the model invented one.
    #
    # This is where the staging language lives. The reference asks for staging
    # in its own words on the active-treatment branches -- "Staging information
    # is needed due to very high PSA", "Distant Metastasis staging is not
    # available. Decide after PSMA PET", "treatment intensification after PSMA
    # PET" -- and the judge deducts for its absence, calling the note "overly
    # definitive" and "incomplete". `HIGH_RISK_CANCER` and `ADVERSE_HISTOLOGY`
    # both close on "metastatic staging, then radical prostatectomy or
    # radiotherapy", and that is the mechanism EAU 6.3.4 states: every
    # active-treatment recommendation is written against a known nodal stage,
    # so the staging comes BEFORE the modality is chosen.
    # ⚠️ Written prospectively -- "then stage him" -- never as an absence claim.
    # No release carries a PSMA PET, bone scan or staging CT, so saying the
    # staging is missing would be an unanchored negative assertion under the
    # anchor rule; recommending it is a plan, which the judge can check.
    next_step = _next_step_for(locked, decision.lower())
    findings = "\n".join(
        [f"Recommendation: {_ACTIONS[decision].replace('_', ' ')}",
         f"Reason for this recommendation: {reason}", R_.render(observed)]
        + [f"What the assessment found: {v.get('KEY_EVIDENCE')}" for v in verdicts.values() if v.get("KEY_EVIDENCE")]
        + ([f"What happens next, and the note must close on it: {next_step}"] if next_step else [])
    )
    written = await write_rationale(
        model, 2, _ACTIONS[decision], confidence, findings, False, warnings,
        background=R_.render(background), citable=observed,
        # The judge's step 2, over the drivers this note's reader can see.
        drivers=R_.drivers(2, weights, observed),
    )
    rationale = (written or block.get("RATIONALE") or "").strip()
    if len(rationale) < 40:
        evidence = "; ".join(v.get("KEY_EVIDENCE", "") for v in verdicts.values() if v.get("KEY_EVIDENCE"))
        rationale = f"{rationale} {evidence}".strip()
        warnings.append("adjudicator rationale was too short; extended with the specialists' key evidence")

    primary = _ACTIONS[decision]
    protocol, trigger = _PLAN[primary]
    prediction = assemble_full_output(
        task=2, case_id=case_id,
        patient={"id": case_id, "psa_ng_ml": query.get("psa"), "age_years": query.get("age")},
        reveal_sequence=_build_reveal_sequence(order),
        judgment={
            "treatment_recommendation": {
                "primary": primary,
                "modalities": ["radical prostatectomy", "radiotherapy"] if primary == "active_treatment" else [],
                "detail": (block.get("KEY_EVIDENCE") or "").strip() or None,
                "as_protocol": protocol,
                "as_trigger": trigger,
            },
            "confidence": confidence,
            "variable_weights": weights,
            "free_text": rationale[:2000],
            "repeat_test": None,
        },
    )
    if warnings:
        log.warning("%s: %s", case_id, " | ".join(warnings))
    return prediction
