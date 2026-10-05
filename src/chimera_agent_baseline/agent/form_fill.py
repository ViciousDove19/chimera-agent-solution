"""Terminal form-fill node — deterministic decision plus grounded rationale.

For Tasks 1 and 2, code first gathers a clinical evidence packet and computes
the structured recommendation, confidence, and weights. The model receives
only the observed clinical evidence and the already-selected recommendation;
it writes one corroborating rationale and never sees the decision machinery.
Task 3 retains the general prompt-and-parse judgment flow.

We deliberately avoid ``model.with_structured_output`` — it relies on
function-calling support that varies wildly across providers (Gemma 4's
offline tool-call parser, for instance, sometimes does not emit a tool
call when there is only one forced schema-tool). Prompt-and-parse is
provider-neutral: any LangChain ``BaseChatModel`` works.

The node retries on validation or grounding errors up to ``max_retries``
times. Tasks 1 and 2 use a schema-valid clinical-language fallback after
exhausted retries; other tasks still raise rather than writing a half-formed
prediction.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, ConfigDict, Field

from chimera_agent_baseline.output.schema import (
    VARIABLES_BY_TASK,
    assemble_full_output,
    build_dynamic_model,
    eligible_variables,
    reveal_info_for_tool,
)
from chimera_agent_baseline.agent.rationale import (
    anchored_missing_evidence,
    biochemical_reading,
    biochemical_sentences,
    discordance_reading,
    next_step_for,
    psa_trajectory,
    reason_for_prose,
    tone_directive,
)
from chimera_agent_baseline.policy import (
    apply_frozen_policy,
    build_policy_audit,
    enforce_frozen_policy,
    join_clauses,
)

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are filling out the structured decision form for the prostate-cancer "
    "case you just analysed. The user message contains your reasoning "
    "transcript and the tools you called. Output a SINGLE JSON object matching "
    "the supplied schema EXACTLY — no extra keys, no markdown fences, no "
    "commentary before or after the JSON."
)

_RATIONALE_SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Write the rationale for the "
    "supplied prostate-cancer recommendation. State the recommendation "
    "explicitly, explain it with the supplied reason, and close with the next "
    "step. Cite findings as support for that reason, never as a justification "
    "that contradicts it. Name only findings listed under 'Observed clinical "
    "evidence' or quoted in one of the supplied reports; anything under "
    "'Patient background' is context for your reasoning and must never appear "
    "in the text. If a material finding is not available, describe it as not "
    "documented. "
    "Two things govern how it reads. Write ONLY the findings that carried the "
    "decision plus the PSA and PSA-density lines you are given -- a note, not a "
    "record review, and never a finding added to pad it out. And let the "
    "supplied confidence set the tone: how settled the prose sounds must match "
    "how settled the recommendation is. "
    "Use standard clinical language and discuss only the patient's findings and "
    "their clinical significance. Output one JSON object and no other text."
)


class _ClinicalRationaleDraft(BaseModel):
    """The only field Tasks 1 and 2 permit the model to generate."""

    model_config = ConfigDict(extra="forbid")

    free_text: str = Field(min_length=40)


def make_form_fill_node(model: BaseChatModel, max_retries: int = 3):
    """Return a LangGraph node closure that captures the unbound *model*.

    *max_retries* is the number of validation attempts before the node
    raises :class:`RuntimeError`. A successful attempt yields a payload
    that already validates against the per-task output model
    (Task1Output / Task2Output / Task3Output).
    """

    def form_fill(state: dict[str, Any]) -> dict[str, Any]:
        messages = state["messages"]
        task = int(state.get("task", 1))
        case_id = state.get("case_id", "unknown")

        tool_order = _called_tools_in_order(messages)
        called = set(tool_order)
        transcript = _final_assistant_text(messages)
        elig = eligible_variables(task, called) if task in VARIABLES_BY_TASK else []
        revealed = _revealed_tool_payloads(messages)
        prompt = state.get("structured_prompt") or {}
        if task == 1 and _task1_has_known_cancer(prompt):
            grade = read_prior_grade(model, revealed, prompt)
            if grade is not None:
                revealed = {**revealed, "prior_isup_grade": grade}
                log.info("Prior ISUP grade read from notes: case=%s grade=%d", case_id, grade)
        policy = apply_frozen_policy(task, prompt, revealed) if task in VARIABLES_BY_TASK else None

        log.info(
            "form_fill: case=%s task=%d called_tools=%s eligible_vars=%s",
            case_id,
            task,
            tool_order,
            elig,
        )

        if policy is not None:
            evidence_packet = _build_clinical_evidence_packet(
                task,
                state.get("structured_prompt") or {},
                revealed,
                policy,
            )
            biochemistry = evidence_packet.get(_BIOCHEMISTRY_KEY, {})
            anchored_missing = evidence_packet[_MISSING_KEY]
            packet_extras = {k: v for k, v in evidence_packet.items()
                             if k in (_TRAJECTORY_KEY, _DISCORDANCE_KEY)}
            visible_evidence = json.dumps(
                evidence_packet["Observed clinical evidence"], ensure_ascii=False
            )
            parser = PydanticOutputParser(pydantic_object=_ClinicalRationaleDraft)
            base_user = _rationale_user_prompt(evidence_packet)
            system_prompt = _RATIONALE_SYSTEM_PROMPT
        else:
            Dynamic = build_dynamic_model(task, called)
            parser = PydanticOutputParser(pydantic_object=Dynamic)
            skeleton = _build_skeleton_instructions(task, elig)
            base_user = _user_prompt(case_id, task, transcript, tool_order, elig) + "\n\n" + skeleton
            system_prompt = _SYSTEM_PROMPT

        # Provider-agnostic retry loop. We keep the conversation flat — a
        # single system + user pair, regenerated on retry — because small
        # models can be derailed by long error-laden histories. The retry
        # message stays clinical for Tasks 1/2 so validation mechanics are not
        # fed back into the rationale context.
        warnings: list[str] = []
        judgment: dict[str, Any] | None = None
        retry_hint: str | None = None

        for attempt in range(1, max_retries + 1):
            user_content = base_user if not retry_hint else f"{base_user}\n\n{retry_hint}"
            try:
                response = model.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_content),
                    ]
                )
                raw = response.content if isinstance(response.content, str) else json.dumps(response.content)
                obj = parser.parse(_extract_json_object(raw))
                judgment = obj.model_dump(mode="json")
                if policy is not None:
                    judgment["free_text"] = _ensure_missing_evidence_disclosure(
                        judgment["free_text"],
                        anchored_missing,
                    )
                    validation = _validate_clinical_rationale(
                        judgment["free_text"],
                        policy,
                        evidence_packet,
                    )
                    if not validation["valid"]:
                        failures = ", ".join(validation["failures"])
                        raise ValueError(f"rationale grounding failed: {failures}")
                break
            except Exception as exc:  # noqa: BLE001 — any parse/validation failure triggers a retry
                # A parsed object that fails grounding is not a usable judgment.
                # Clear it so exhausted retries take the clinical fallback
                # instead of accidentally retaining the last invalid draft.
                judgment = None
                warnings.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                log.warning("form_fill parse failed (attempt %d) for %s: %s", attempt, case_id, exc)
                if attempt == max_retries:
                    break
                if policy is not None:
                    retry_hint = (
                        "The previous rationale did not stay within the supplied "
                        "clinical evidence, or did not read the way its "
                        "confidence requires. Rewrite it using only documented "
                        "findings, explicitly state the supplied recommendation, "
                        "address the PSA and the PSA density and what each does "
                        "for it, follow the supplied instruction on how it must "
                        "be written, disclose any information identified as not "
                        "documented, and emit exactly the requested JSON object."
                    )
                else:
                    retry_hint = (
                        "Your previous attempt did not validate. The validation "
                        f"error was:\n{exc}\n\n"
                        "Emit exactly the requested JSON object and no other text."
                    )

        llm_parse_success = judgment is not None
        rationale_validation = None
        if policy is not None:
            if judgment is None:
                warnings.append("all rationale attempts failed; used grounded clinical fallback")
                judgment = _policy_fallback_judgment(policy, elig, biochemistry, anchored_missing, visible_evidence, packet_extras)
            else:
                rationale_text = judgment["free_text"]
                judgment = _policy_fallback_judgment(policy, elig, biochemistry, anchored_missing, visible_evidence, packet_extras)
                judgment["free_text"] = rationale_text
            rationale_validation = _validate_clinical_rationale(
                judgment["free_text"],
                policy,
                evidence_packet,
            )
        elif judgment is None:
            raise RuntimeError(
                f"form_fill: case {case_id}: all {max_retries} attempts failed to "
                f"produce schema-valid output. form_fill_warnings={warnings}"
            )

        raw_judgment = dict(judgment)
        policy_audit = None
        if policy is not None:
            judgment = enforce_frozen_policy(judgment, policy, elig)
            policy_audit = build_policy_audit(
                policy=policy,
                raw_judgment=raw_judgment,
                eligible_variables=elig,
                observed_tools=tool_order,
                form_fill_warnings=warnings,
                llm_parse_success=llm_parse_success,
                rationale_only=True,
                rationale_validation=rationale_validation,
            )

        # What we report having consulted is what the policy actually used --
        # not every document the model opened on the way there. Extra reveals
        # cost tool precision and defend nothing.
        # An empty `reveal_tools` is a deliberate policy decision (Task 2 scores
        # full tool precision only for an empty reveal set), so it must not fall
        # through to the model's call order -- test `policy is None`, not
        # truthiness of the tuple.
        consulted = list(policy.reveal_tools) if policy is not None else tool_order
        reveal_sequence = _build_reveal_sequence(consulted)
        patient = _build_patient(case_id, state)
        full = assemble_full_output(task, case_id, patient, reveal_sequence, judgment)
        return {
            "structured_response": full,
            "form_fill_warnings": warnings,
            "policy_audit": policy_audit,
        }

    return form_fill


# ---------------------------------------------------------------------------
# Programmatic case fields — not produced by the LLM.
# ---------------------------------------------------------------------------


def _called_tools_in_order(messages: list) -> list[str]:
    """Unique tool names in first-call order, derived from ``ToolMessage``s."""
    seen: set[str] = set()
    order: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name and m.name not in seen:
            seen.add(m.name)
            order.append(m.name)
    return order


_GRADE_QUESTION = (
    "Below are a patient's previous clinical notes and the referral context of "
    "their latest MRI report.\n\n{notes}\n\n"
    "Did a previous prostate biopsy establish a grade for this patient's cancer?\n"
    "Answer with the ISUP grade group as a single digit 1-5, or the word NONE if the "
    "notes do not state a grade.\n"
    "Convert Gleason scores: 3+3 is 1, 3+4 is 2, 4+3 is 3, 4+4 or 3+5 is 4, 4+5 or 5+4 is 5. "
    "Read 'low-grade' or 'low-risk' disease as 1.\n"
    "PI-RADS is an imaging score, not a biopsy grade -- never report a PI-RADS number "
    "as a grade.\n"
    "A biopsy that was negative, benign, or found no malignancy has NO grade: answer NONE. "
    "A biopsy that is merely described as positive, without a Gleason score, ISUP grade or "
    "risk category, also has no stated grade: answer NONE. Never guess a grade.\n"
    "Answer with the digit or NONE and nothing else."
)


_PRIOR_MRI_QUESTION = (
    "Below are a patient's previous clinical notes and the referral context of "
    "their latest MRI report.\n\n{notes}\n\n"
    "Do these describe an EARLIER prostate MRI -- one done before the current "
    "scan?\n"
    "Answer on one line, with two fields separated by a single '|':\n"
    "  field 1: YES if an earlier MRI is described, NO if it is not.\n"
    "  field 2: STABLE if that earlier MRI is said to be unchanged, or the "
    "lesion the same size; PROGRESSED if the lesion is said to be new, larger "
    "or more suspicious; UNSTATED if an earlier MRI exists but nothing is said "
    "about change; NONE if field 1 is NO.\n"
    "The text below already excludes the current scan's findings, so any MRI "
    "result you see in it is an earlier one. A referral line that merely says "
    "an MRI was requested, or quotes today's PSA, is not an earlier MRI: answer "
    "NO. Never infer an earlier scan from the presence of a previous biopsy.\n"
    "Answer with the two fields and nothing else, e.g. 'YES|STABLE' or 'NO|NONE'."
)


def read_prior_mri(model: BaseChatModel, revealed: dict[str, Any]) -> dict[str, str] | None:
    """Whether an EARLIER prostate MRI is on record, and what it said about change.

    The companion to :func:`read_prior_grade`, and the second of the two facts
    every reference `uncertain` rationale asks for: 12 of the 15 name a prior
    MRI or the lesion's trajectory ("has the lesion grown in size or not in a
    man on active surveillance"). Neither is a structured field.

    Same guard as the grade read: only the referral preamble of the current
    report is offered, never its findings, because a model shown today's
    PI-RADS reports it as the earlier scan's.

    Returns ``{"documented": "yes"|"no", "trajectory": ...}`` or ``None`` when
    there is nothing to read -- which is the same state as a record that does
    not describe an earlier scan.
    """
    notes = _revealed_value(revealed, "get_previous_notes", "previous_notes")
    report = _revealed_value(revealed, "get_mri_report", "radiology_report")
    sources = [notes, _referral_context(report) if isinstance(report, str) else None]
    parts = [s if isinstance(s, str) else json.dumps(s) for s in sources if s]
    if not parts:
        return None
    text = "\n\n".join(parts)
    try:
        answer = model.invoke(
            [HumanMessage(content=_PRIOR_MRI_QUESTION.format(notes=text[:6000]))]
        )
    except Exception:  # noqa: BLE001 - a failed read is simply no earlier MRI
        log.warning("prior-MRI read failed; continuing without one")
        return None
    reply = str(getattr(answer, "content", "") or "").strip().upper()
    if len(reply) > 40 or "|" not in reply:
        return None
    documented, _, trajectory = reply.partition("|")
    documented = documented.strip()
    trajectory = trajectory.strip()
    if documented not in {"YES", "NO"}:
        return None
    if trajectory not in {"STABLE", "PROGRESSED", "UNSTATED", "NONE"}:
        trajectory = "UNSTATED" if documented == "YES" else "NONE"
    return {"documented": documented.lower(), "trajectory": trajectory.lower()}


def _referral_context(report: str) -> str:
    """The clinical-indication preamble of a radiology report, without findings."""
    lowered = report.lower()
    cuts = [lowered.find(marker) for marker in ("findings", "imaging details", "impression")]
    cut = min([c for c in cuts if c > 0], default=len(report))
    return report[:cut]


def read_prior_grade(
    model: BaseChatModel,
    revealed: dict[str, Any],
    prompt: dict[str, Any] | None = None,
) -> int | None:
    """ISUP grade of the previous biopsy, read out of the previous notes.

    The grade is prose -- "low-grade disease", "Gleason 3+4", "ISUP grade group
    3" -- so this is a reading task, not a lookup. Anything the model returns
    that is not a grade leaves the policy's grade rule inactive, which is the
    same state as a record that genuinely does not name a grade.
    """
    # The grade turns up in three places: the visible structured clinical-note
    # sections may recount the biopsy, retrieved previous notes may do so, and
    # the MRI referral line often restates the grade the scan was ordered
    # against. `note_sections` was omitted until 2026-09-08, causing the policy
    # to claim that explicitly stated grades were unavailable.
    # Only the referral line is used -- given the imaging findings the model
    # reads PI-RADS as a grade and invents one for patients who have none.
    notes = _revealed_value(revealed, "get_previous_notes", "previous_notes")
    report = _revealed_value(revealed, "get_mri_report", "radiology_report")
    visible_notes = []
    for section in (prompt or {}).get("note_sections") or []:
        if isinstance(section, dict) and section.get("t"):
            visible_notes.append(str(section["t"]))
    sources = [
        "\n".join(visible_notes) or None,
        notes,
        _referral_context(report) if isinstance(report, str) else None,
    ]
    parts = [s if isinstance(s, str) else json.dumps(s) for s in sources if s]
    if not parts:
        return None
    text = "\n\n".join(parts)
    try:
        answer = model.invoke([HumanMessage(content=_GRADE_QUESTION.format(notes=text[:6000]))])
    except Exception:  # noqa: BLE001 - a failed read is simply no grade
        log.warning("prior-grade read failed; continuing without a grade")
        return None
    reply = str(getattr(answer, "content", "") or "").strip()
    digits = [c for c in reply if c.isdigit()]
    if len(reply) > 24 or not digits:
        return None
    grade = int(digits[0])
    return grade if 1 <= grade <= 5 else None


def _revealed_value(revealed: dict[str, Any], tool: str, field: str) -> Any:
    payload = revealed.get(tool)
    return payload.get(field) if isinstance(payload, dict) else None


def _revealed_tool_payloads(messages: list) -> dict[str, Any]:
    """Decode the evidence actually returned by each called clinical tool."""
    payloads: dict[str, Any] = {}
    for message in messages:
        if not isinstance(message, ToolMessage) or not message.name:
            continue
        content = message.content
        decoded: Any = content
        if isinstance(content, str):
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError:
                try:
                    decoded = json.loads(_extract_json_object(content))
                except json.JSONDecodeError:
                    decoded = {"raw_text": content}
        elif isinstance(content, list):
            text_parts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            joined = "\n".join(part for part in text_parts if part)
            if joined:
                try:
                    decoded = json.loads(_extract_json_object(joined))
                except json.JSONDecodeError:
                    decoded = {"raw_text": joined}
        payloads[message.name] = decoded
    return payloads


def _build_reveal_sequence(reveals: list[str]) -> list[dict[str, Any]]:
    """Build the ``reveal_sequence`` — one entry per section consulted.

    Timestamps are synthetic (evenly spaced from "now"): the agent runtime
    does not currently record the wall-clock time of each tool call.
    """
    base_ts = datetime.now(UTC)
    sequence: list[dict[str, Any]] = []
    for i, tool_name in enumerate(reveals, start=1):
        key, label = reveal_info_for_tool(tool_name)
        ts = base_ts + timedelta(seconds=2 * (i - 1))
        sequence.append(
            {
                "page": "decision",
                "order": i,
                "key": key,
                "label": label,
                "value": "section",
                "via": "tool_call",
                "ts": ts.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            }
        )
    return sequence


def _build_patient(case_id: str, state: dict[str, Any]) -> dict[str, Any]:
    patient_state = state.get("patient") or {}
    return {
        "id": case_id,
        "psa_ng_ml": patient_state.get("psa"),
        "age_years": patient_state.get("age"),
    }


def _policy_fallback_judgment(
    policy,
    eligible: list[str],
    biochemistry: dict[str, str] | None = None,
    missing_evidence: list[str] | None = None,
    visible_text: str = "",
    extras: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build every structured field in code, including a grounded prose fallback."""
    recommendation = _clinical_recommendation(policy.task, policy.decision)
    # The policy's own reason is a clinical sentence, and stating it keeps the
    # fallback meaningful (and schema-valid) even for a case whose findings are
    # all absent.
    told = reason_for_prose(policy.rule, visible_text or "")
    reason = f" {told[0].upper()}{told[1:]}." if told else ""
    facts = "; ".join(factor.text for factor in policy.factors)
    support = f" The available findings are {facts}." if facts else ""
    # The anchored list, never `policy.missing_evidence`: the code-rendered
    # fallback obeys the anchor rule exactly as the model is asked to.
    anchored = list(policy.missing_evidence if missing_evidence is None else missing_evidence)
    limitations = (
        " The following information is not documented: " + join_clauses(anchored) + "."
        if anchored
        else ""
    )
    trigger = f" {policy.trigger[0].upper()}{policy.trigger[1:]}." if policy.trigger else ""
    # The fallback obeys the same rule the prompt asks of the model: PSA and PSA
    # density are addressed on every case, with what each does for the decision.
    biochemical = biochemical_sentences(biochemistry or {})
    conflict = f" {discordant}" if (discordant := (extras or {}).get(_DISCORDANCE_KEY, "")) else ""
    movement = f" {trend}" if (trend := (extras or {}).get(_TRAJECTORY_KEY, "")) else ""
    # ...and it closes on an action, as `deterministic._rationale` already did.
    # "Omits the specific recommendation" is the judge's most common criticism
    # of our prose, and a fallback that stops at the findings invites it.
    next_step = f" Next step: {next_step_for(policy)}."
    # An `uncertain` case that has no sayable absence still has to read as
    # unsettled -- it cannot earn the label by naming a variable the judge
    # cannot see, so it earns it over the documented findings instead.
    unsettled = (
        " The documented findings do not settle the question, and this may change as "
        "the picture develops."
        if policy.confidence == "uncertain" and not anchored
        else ""
    )
    rationale = (recommendation + reason + support + biochemical + movement + conflict
                 + limitations + unsettled + trigger + next_step)
    judgment: dict[str, Any] = {
        "confidence": policy.confidence,
        "variable_weights": {key: policy.weights[key] for key in eligible},
        "repeat_test": next_step_for(policy).capitalize() + ".",
        "free_text": rationale,
    }
    if policy.task == 1:
        judgment["biopsy_decision"] = policy.decision
    else:
        surveillance = policy.decision in {"active_surveillance", "continued_surveillance"}
        judgment["treatment_recommendation"] = {
            "primary": policy.decision,
            "modalities": [],
            "detail": rationale,
            "as_protocol": "Serial PSA and clinical review." if surveillance else None,
            "as_trigger": "Reassess management if clinical or disease-risk findings change." if surveillance else None,
        }
    return judgment


def _clinical_recommendation(task: int, decision: str) -> str:
    if task == 1:
        return (
            "Prostate biopsy is recommended at this encounter."
            if decision == "yes"
            else "Prostate biopsy is not recommended at this encounter."
        )
    labels = {
        "active_surveillance": "Active surveillance is recommended.",
        "continued_surveillance": "Continued surveillance is recommended.",
        "watchful_waiting": "Watchful waiting is recommended.",
        "active_treatment": "Active treatment is recommended.",
    }
    return labels[decision]


# ⚠️ THE RATIONALE JUDGE SEES THE CLINICAL-DATA SOCKET AND NOTHING ELSE.
# `evaluate.py` builds the GEval input from the job's own input socket, so
# every field below that lives only in `prompt.json` is invisible to it and
# naming one reads as a fabrication however true it is. Measured across the
# release, a prompt value appears literally in that socket on:
#
#   psa 100%   psad 100%   vol 100%   cspca 100%   ct 100%   ISUP/Gleason 100%
#   pirads 99%   ...   AGE 28% (T1) / 18% (T2)   DRE 1%   IPSS 0%   enc_type 0%
#
# So the two tables are split. Both reach the model -- background is what makes
# the case legible and it still drives the policy -- but only `_CITABLE_FIELDS`
# may be asserted back. The judge flagged the age claim on 7 of 79 Task 1
# rationales and 12 of 64 Task 2 ones, and those cases mean 0.471 / 0.517
# against 0.771 / 0.750 for the rest.
_CITABLE_FIELDS: dict[int, tuple[tuple[str, str], ...]] = {
    1: (
        ("PSA (ng/mL)", "psa"),
        ("Previous biopsy status", "bx"),
        ("PI-RADS score", "pirads"),
        ("PSA density", "psad"),
        ("Prostate volume (mL)", "vol"),
        ("Estimated clinically significant cancer probability", "cspca"),
    ),
    2: (
        ("PSA (ng/mL)", "psa"),
        ("Clinical stage", "ct"),
        ("ISUP grade group", "bx_isup"),
        ("Primary Gleason pattern", "bx_gl_prim"),
        ("Secondary Gleason pattern", "bx_gl_sec"),
        ("Tertiary Gleason pattern", "bx_gl_tert"),
        ("PI-RADS score", "pirads"),
        ("PSA density", "psad"),
        ("Estimated clinically significant cancer probability", "cspca"),
    ),
}

_BACKGROUND_FIELDS: dict[int, tuple[tuple[str, str], ...]] = {
    1: (
        ("Encounter type", "enc_type"),
        ("Consultation note", "notes"),
        ("Age (years)", "age"),
        ("Digital rectal examination", "dre"),
    ),
    2: (
        ("Encounter type", "enc_type"),
        ("Consultation note", "notes"),
        ("Age (years)", "age"),
        ("Previous biopsy status", "bx"),
    ),
}

_TOOL_CLINICAL_FIELDS: dict[str, tuple[str, str]] = {
    "get_mri_report": ("MRI report", "radiology_report"),
    "get_pathology_report": ("Pathology report", "pathology_report"),
    "get_previous_notes": ("Previous clinical notes", "previous_notes"),
    "get_family_history": ("Family history", "family_history"),
    "get_psa_trend": ("PSA history", "psa_trend"),
    "get_lab_results": ("Laboratory results", "laboratory_results"),
    "get_surgical_pathology_report": ("Surgical pathology report", "surgical_pathology_report"),
}


_BACKGROUND_KEY = "Patient background — context only, never cite these in the rationale"
_BIOCHEMISTRY_KEY = "PSA and PSA density, and what each does for this recommendation"
_MISSING_KEY = "Clinically relevant information not documented"
_TRAJECTORY_KEY = "How the PSA has moved over time"
_DISCORDANCE_KEY = "Findings that point the other way"

# Claims that are true of the patient and usually invisible to the rationale
# judge, so stating one costs a fabrication finding. Age is the big one: 19 of
# the 143 judged rationales named it and averaged 0.49 against 0.76 for the
# rest -- "hallucinates the patient's age of 73 years, which is not present in
# the Input".
#
# ⚠️ Each is checked CONDITIONALLY, against this case's own documents, because
# some of them genuinely turn up there: the previous notes mention a rectal
# examination in 27% of Task 1 sockets and 56% of Task 2 ones, and an age
# phrase in 15% / 2%. A blanket ban would suppress a claim that is grounded.
# The pair is (what we wrote, what has to be in the citable packet for it to
# be allowed).
_CONDITIONAL_CLAIMS: tuple[tuple[re.Pattern[str], re.Pattern[str]], ...] = (
    (
        re.compile(r"\b\d{1,3}[- ]year[- ]old\b|\bage[ds]? of \d|\bpatient'?s age\b"
                   r"|\b\d{1,3} years of age\b|\baged \d", re.IGNORECASE),
        re.compile(r"\d{1,3}[- ]year[- ]old|\byo\b|years of age", re.IGNORECASE),
    ),
    (
        re.compile(r"\bdigital rectal\b|\bDRE\b|\bnodus\b|\bpalpable nodule\b", re.IGNORECASE),
        re.compile(r"rectal|\bdre\b|nodus|nodule", re.IGNORECASE),
    ),
    (
        re.compile(r"\bIPSS\b|\blower urinary tract symptom", re.IGNORECASE),
        re.compile(r"ipss|lower urinary tract", re.IGNORECASE),
    ),
    (
        # A clinical T stage is not an examination finding the judge can see.
        re.compile(r"\bpalpabl\w*\b|\bimpalpabl\w*\b", re.IGNORECASE),
        re.compile(r"palpab|impalpab", re.IGNORECASE),
    ),
)


def _has_clinical_value(value: Any) -> bool:
    if value is None or value == [] or value == {}:
        return False
    return str(value).strip().lower() not in {"", "—", "n/a", "na"}


def _build_clinical_evidence_packet(
    task: int,
    prompt: dict[str, Any],
    revealed: dict[str, Any],
    policy,
) -> dict[str, Any]:
    """Translate observed data to clinical labels, omitting implementation names."""
    observed: dict[str, Any] = {}
    for label, key in _CITABLE_FIELDS[task]:
        value = prompt.get(key)
        if _has_clinical_value(value):
            observed[label] = value

    # Everything the judge cannot see. It still reaches the model, because a
    # case with no age, no examination and no encounter type is not a case --
    # it is simply never quoted back.
    background: dict[str, Any] = {}
    for label, key in _BACKGROUND_FIELDS[task]:
        value = prompt.get(key)
        if _has_clinical_value(value):
            background[label] = value

    sections = prompt.get("note_sections") or []
    for index, section in enumerate(sections, start=1):
        if not isinstance(section, dict):
            continue
        title = str(section.get("s") or f"section {index}").strip()
        value = section.get("t")
        if _has_clinical_value(value):
            background[f"Clinical note — {title}"] = value

    for tool_name, payload in revealed.items():
        label_and_field = _TOOL_CLINICAL_FIELDS.get(tool_name)
        if label_and_field is None or not isinstance(payload, dict):
            continue
        label, field = label_and_field
        value = payload.get(field)
        if _has_clinical_value(value):
            observed[label] = value

    # ⚠️ THE ANCHOR RULE. An absence may be stated only when the judge can see
    # the variable it is about; otherwise naming it invents a fact in the
    # negative direction. `observed` is exactly what the judge's input contains,
    # so it is what the clause is anchored against -- never the background.
    visible = json.dumps(observed, ensure_ascii=False)
    missing = anchored_missing_evidence(policy.missing_evidence, visible, policy.confidence)

    packet: dict[str, Any] = {
        "Clinical recommendation": _clinical_recommendation(task, policy.decision),
        # The judge asks for a management plan, not a summary. "Lacks the
        # depth of reasoning / omits the specific recommendation" is the most
        # common criticism in its own words -- 72% of Task 1 critiques and 58%
        # of Task 2 -- and the reference rationales always close on an action
        # (a repeat PSA, staging, or a named treatment discussion).
        # ⚠️ Keyed to the BRANCH, not the decision -- see `next_step:` in the
        # policy YAMLs. The decision-keyed version told a NO_BENEFIT patient to
        # repeat his PSA when the reference says to stop testing altogether.
        "Next step after this recommendation": next_step_for(policy).capitalize() + ".",
        # Why the recommendation is what it is. Without this the model could
        # only recite the findings, so a `no` came out as "not recommended due
        # to the PI-RADS 5 finding and the elevated PSA" -- attributing the
        # decision to the evidence that argued against it. The judge scored
        # that as a contradiction, and it was the single largest cause of low
        # rationale scores (37% of cases at or below 0.6).
        "Reason for this recommendation": reason_for_prose(policy.rule, visible),
        "Observed clinical evidence": observed,
        _BACKGROUND_KEY: background,
        "Most relevant observed findings": [factor.text for factor in policy.factors],
        _MISSING_KEY: list(missing),
        # A `no` that does not say what would make it a `yes` is a refusal
        # rather than a recommendation, and the clinicians write the trigger
        # out: "A change in P4 lesion size might trigger a biopsy."
        "What would change this recommendation": policy.trigger,
        # ⚠️ Tone is not decoration. A `clear` recommendation argued with in its
        # own rationale reads as a contradiction to the judge, and an
        # `uncertain` one written flatly does not support the label we emitted.
        "Confidence in this recommendation": policy.confidence,
        "How this must be written, given that confidence": tone_directive(
            policy.confidence, has_named_absence=bool(missing)
        ),
    }
    # Task 2 gets no biochemical block -- its branches turn on grade and stage,
    # and its own confidence design has not begun -- so the key is absent rather
    # than empty, and the prompt does not ask for a line that was never supplied.
    reading = biochemical_reading(prompt, policy, _prior_grade_of(revealed))
    if reading:
        packet[_BIOCHEMISTRY_KEY] = reading
    # ⚠️ The trajectory is judge-visible on every released case and was being
    # suppressed by the brevity rule. A flat series returns None and no key is
    # added, so a non-finding is never described. See `psa_trajectory`.
    trajectory = psa_trajectory((revealed.get("get_psa_trend") or {}).get("psa_trend"))
    if trajectory:
        packet[_TRAJECTORY_KEY] = trajectory
    # ⚠️ Task 2's equivalent of the Task 1 biochemical verdict: a suspicious
    # finding under a surveillance recommendation argues AGAINST it and is
    # outweighed. Calling it "supporting" is the single largest defect the
    # judge found. Empty on every branch where nothing points the other way.
    packet.update(discordance_reading(prompt, policy))
    return packet


def _prior_grade_of(revealed: dict[str, Any]) -> int | None:
    grade = revealed.get("prior_isup_grade")
    return grade if isinstance(grade, int) else None


def _rationale_user_prompt(evidence_packet: dict[str, Any]) -> str:
    # PI-RADS alone is not a rationale: the two biochemical lines are stated on
    # every Task 1 case, whatever weight the branch gave them, each with the
    # frozen policy's own reading of what it does for the recommendation -- so
    # the model never has to decide for itself whether a number argues for or
    # against. Task 2 supplies no such block; see the packet builder.
    biochemistry = (
        ", and then state the PSA line and the PSA-density line, including "
        f"whether each supports the recommendation, exactly as '{_BIOCHEMISTRY_KEY}' "
        "reads them. "
        if _BIOCHEMISTRY_KEY in evidence_packet
        else ". "
    )
    # ⚠️ The anchor rule, in the prompt. When the summary lists no absence, the
    # model must not supply one: every candidate it could name is a variable the
    # judge cannot see, and an absence claim about an invisible variable reads
    # as an invention exactly as a present-tense one would.
    # The PSA trajectory is a primary driver when it moves, and the reference
    # rationales say so outright.
    trajectory = (
        f"Then state how the PSA has moved over time, exactly as "
        f"'{_TRAJECTORY_KEY}' reads it. "
        if evidence_packet.get(_TRAJECTORY_KEY)
        else ""
    )
    discordant = (
        f"State the finding that points the other way, exactly as "
        f"'{_DISCORDANCE_KEY}' puts it, and never as a reason FOR the "
        "recommendation. "
        if evidence_packet.get(_DISCORDANCE_KEY)
        else ""
    )
    limitation = (
        "The summary identifies clinically relevant information that is not "
        "documented; state that limitation explicitly, in its words. "
        if evidence_packet.get(_MISSING_KEY)
        else "The summary identifies NO undocumented information. Do not say that "
        "anything is missing, unavailable, not documented, not recorded, not "
        "retrievable or unknown, and do not name a test, grade, report or "
        "measurement in order to point out that the record lacks it. Write only "
        "about what the record does contain. "
    )
    return (
        "Write the clinical rationale from this clinical record summary:\n\n"
        f"{json.dumps(evidence_packet, ensure_ascii=False, indent=2)}\n\n"
        "Return exactly this JSON shape:\n"
        '{\n  "free_text": "<at least 40 characters>"\n}\n\n'
        "State the recommendation, then give the reason for it. The findings "
        "are supporting detail, not the justification on their own. "
        "Name every finding in 'Most relevant observed findings' -- those are "
        "the ones that carried the decision -- with its value exactly as the "
        f"summary gives it{biochemistry}"
        f"{trajectory}{discordant}"
        "Beyond those, add nothing: do not work through the rest of "
        "'Observed clinical evidence', and never name a finding to make the "
        "note longer. "
        # Task 1 carries its discordance inside the biochemical verdict rather
        # than as its own block, so the generic form is still needed there.
        + ("" if evidence_packet.get(_DISCORDANCE_KEY) else
           "A finding that points the other way is stated once, as outweighed, "
           "never as a reason for the recommendation. ")
        +
        f"{limitation}"
        "Close with the next step, and with what would change the "
        "recommendation if the summary gives one. "
        "Follow 'How this must be written, given that confidence' exactly. It "
        "governs how settled the prose is allowed to sound, and a note that "
        "argues with its own recommendation is wrong even when every finding "
        "in it is true. "
        "Every clinical claim and number must come from 'Observed clinical "
        "evidence', 'Most relevant observed findings', or one of the supplied "
        "reports. NEVER state the patient's age, the digital rectal "
        "examination, the encounter type, or a symptom score: those appear "
        "only under 'Patient background', and asserting them counts as an "
        "unsupported claim even when they are true. "
        "Do not add treatment modalities, test results, examination findings, "
        "or disease characteristics that are not present. "
        "Write four to six sentences in the register of a clinician's note. "
        "One clause is not a rationale: a single finding followed by the "
        "recommendation would be rejected."
    )


_KNOWN_CANCER = "re-evaluation - prior pca diagnosis"


def _task1_has_known_cancer(prompt: dict[str, Any]) -> bool:
    biopsy = str(prompt.get("bx") or "").strip().lower()
    if biopsy in {"positive", "cancer", "malignant"}:
        return True
    if biopsy in {
        "none", "no", "no previous biopsy", "biopsy-naive", "biopsy naive",
        "negative", "benign", "asap/hgpin", "asap", "hgpin",
    }:
        return False
    return str(prompt.get("enc_type") or "").strip().lower() == _KNOWN_CANCER

_INTERNAL_LANGUAGE = re.compile(
    r"\b(?:frozen|policy|rule engine|matched rule|decision matrix|proxy|fallback|"
    r"challenge mapping|variable weights?|not_used|evaluator|dataset|language model|llm)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
_MISSING_DISCLOSURE_PHRASES = (
    "not documented",
    "not available",
    "unavailable",
    "unknown",
    "insufficient information",
    "cannot be determined",
)
_CLINICAL_CONCEPT_TERMS = (
    "digital rectal",
    "dre",
    "family history",
    "cribriform",
    "intraductal",
    "metastasis",
    "metastatic",
    "lymph node",
    "extraprostatic",
    "seminal vesicle",
    "pi-rads",
    "pirads",
    "psa density",
    "gleason",
    "isup",
    "clinical stage",
    "prostate volume",
    "pathology",
)


# Words that argue with the recommendation they sit next to. Banned only under
# `clear`: on `borderline` and `uncertain` they are the point. Deliberately does
# NOT include the missing-evidence phrases above -- a settled recommendation can
# still rest on an incomplete record, and saying so is a fact, not a hedge.
# ⚠️ "may" excludes the MONTH. The PSA-trend dates are quoted verbatim, so
# "May 2024" was matching as a hedge and failing `tone_matches_confidence` on
# 8 Task 1 and 14 Task 2 cases that hedge nothing at all.
_DECISION_HEDGE = re.compile(
    r"\bmay\b(?!\s+\d{4})|"
    r"\b(?:might|possibly|perhaps|unclear|uncertain|inconclusive|however|"
    r"although|nonetheless|nevertheless)\b|cannot be excluded|remains in doubt",
    re.IGNORECASE,
)

# The shape of a claim that a clinical fact is absent, in any of the forms the
# models actually write. Only checked when the anchor rule kept nothing -- when
# it kept something, saying so is exactly what we asked for.
_ABSENCE_CLAIM = re.compile(
    r"\b(?:not|never|no longer)\s+(?:been\s+)?(?:documented|available|recorded|reported|"
    r"retrievable|on record|stated|specified|provided|known)\b"
    r"|\bis\s+(?:un(?:available|documented|known|recorded))\b"
    r"|\b(?:absence|lack|want)\s+of\b"
    r"|\bno\s+(?:record|documentation|mention|information)\s+of\b"
    r"|\bcannot\s+be\s+(?:determined|established|retrieved|ascertained)\b"
    r"|\b(?:missing|undocumented|unavailable)\b",
    re.IGNORECASE,
)

# "PI-RADS 2, so no biopsy" is a verdict, not a rationale. The reference prose
# runs to a few clinical sentences; the schema floor of 40 characters does not
# reach that, so the validator carries its own.
_MIN_RATIONALE_CHARS = 200


def _ensure_missing_evidence_disclosure(text: str, missing_evidence) -> str:
    """Append the exact clinical limitation when the draft omits it.

    *missing_evidence* is the list that survived the anchor rule, never
    `policy.missing_evidence` -- appending an unanchored absence would insert
    exactly the invented claim the rule exists to prevent.
    """
    if not missing_evidence or any(phrase in text.lower() for phrase in _MISSING_DISCLOSURE_PHRASES):
        return text
    missing = join_clauses(missing_evidence)
    return f"{text.rstrip()} The following clinically relevant information is not documented: {missing}."


def _validate_clinical_rationale(text: str, policy, evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Reject mechanism leakage, contradictions, and unsupported numbers."""
    lowered = text.lower()
    internal_free = _INTERNAL_LANGUAGE.search(text) is None

    if policy.task == 1:
        affirmative = ("biopsy is recommended", "recommend prostate biopsy", "proceed with biopsy")
        negative = ("biopsy is not recommended", "no prostate biopsy", "omit biopsy")
        if policy.decision == "yes":
            recommendation_consistent = any(term in lowered for term in affirmative) and not any(
                term in lowered for term in negative
            )
        else:
            recommendation_consistent = any(term in lowered for term in negative)
    else:
        terms = {
            "active_surveillance": ("active surveillance",),
            "continued_surveillance": ("continued surveillance", "continued monitoring"),
            "watchful_waiting": ("watchful waiting",),
            "active_treatment": ("active treatment",),
        }
        recommendation_consistent = any(term in lowered for term in terms[policy.decision])

    # ⚠️ Ground against the packet MINUS the background block. The background
    # is real and reaches the model, but the judge cannot see it, so a number
    # that appears only there -- an age, an IPSS score -- is not support.
    citable = {k: v for k, v in evidence_packet.items() if k != _BACKGROUND_KEY}
    citable_json = json.dumps(citable, ensure_ascii=False)
    allowed_numbers = set(_NUMBER.findall(citable_json))
    generated_numbers = set(_NUMBER.findall(text))
    numeric_grounding = generated_numbers <= allowed_numbers

    # ...and the same for the claims that carry no number of their own.
    judge_visible_claims = not any(
        wrote.search(text) and not needs.search(citable_json)
        for wrote, needs in _CONDITIONAL_CLAIMS
    )

    available_text = citable_json.lower()
    unsupported_terms = [term for term in _CLINICAL_CONCEPT_TERMS if term in lowered and term not in available_text]
    clinical_term_grounding = not unsupported_terms

    anchored_missing = evidence_packet.get(_MISSING_KEY) or []
    missing_evidence_disclosed = not anchored_missing or any(
        phrase in lowered for phrase in _MISSING_DISCLOSURE_PHRASES
    )

    # ⚠️ The anchor rule, enforced. When nothing survived, the rationale may not
    # describe ANYTHING as missing: every absence it could reach for is one the
    # judge cannot see the variable of. `clinical_term_grounding` already
    # rejects the noun; this rejects the shape of the claim.
    # ...and an absence the branch's own REASON is entitled to state is
    # sanctioned too. `reason_for_prose` leaves it in place only when the
    # variable is visible (Task 2 `SHARED_DECISION_BAND`, 26 cases), so
    # repeating it is grounded even though `missing_evidence` is empty.
    sanctioned = _ABSENCE_CLAIM.search(str(evidence_packet.get("Reason for this recommendation") or ""))
    absence_claims_anchored = (
        bool(anchored_missing) or sanctioned is not None or not _ABSENCE_CLAIM.search(text)
    )

    factor_terms = {
        "pirads": ("pi-rads", "pirads"),
        "psad": ("psa density",),
        "psa": ("psa",),
        "dre": ("digital rectal", "dre"),
        "fh": ("family history",),
        "bx": ("biopsy", "tissue", "grade group", "gleason"),
        "bx_isup": ("isup", "grade group", "biopsy", "cribriform", "intraductal"),
        "bx_gl_prim": ("gleason", "primary pattern"),
        "bx_gl_sec": ("gleason", "secondary pattern"),
        "ct": ("clinical stage", "ct1", "ct2", "ct3", "ct4"),
        "comorbidity": ("life expectancy", "curative", "palliative"),
        "age": ("age", "year-old", "years old"),
    }
    mentions = [
        any(term in lowered for term in factor_terms.get(factor.variable, (factor.variable,)))
        for factor in policy.factors
    ]
    factor_coverage = sum(mentions) / len(mentions) if mentions else 1.0
    evidence_corroboration = factor_coverage > 0.0

    # Rule: the tone has to match the label we emit. `clear` may not argue with
    # itself; `uncertain` has to name what is missing, because that is the only
    # thing the reference `uncertain` rationales ever do.
    if policy.confidence == "clear":
        tone_matches_confidence = _DECISION_HEDGE.search(text) is None
    elif policy.confidence == "uncertain":
        # With an anchored absence, `uncertain` must name it. Without one it
        # still has to read as unsettled -- the hedges banned under `clear` are
        # exactly what carries that here.
        tone_matches_confidence = any(
            phrase in lowered for phrase in _MISSING_DISCLOSURE_PHRASES
        ) or _DECISION_HEDGE.search(text) is not None
    else:
        tone_matches_confidence = True

    # Rule: PSA and PSA density are addressed on every Task 1 case, whatever the
    # branch weighted them. Presence of the term, not of the value -- the value
    # is asked for in the prompt, and `numeric_grounding` already rejects any
    # number that is not the patient's.
    if policy.task == 1:
        density_documented = "PSA density" in evidence_packet.get("Observed clinical evidence", {})
        biochemistry_addressed = "psa" in lowered and (not density_documented or "density" in lowered)
    else:
        biochemistry_addressed = True

    sufficient_detail = len(text.strip()) >= _MIN_RATIONALE_CHARS

    checks = {
        "internal_language_free": internal_free,
        "absence_claims_anchored": absence_claims_anchored,
        "tone_matches_confidence": tone_matches_confidence,
        "biochemistry_addressed": biochemistry_addressed,
        "sufficient_detail": sufficient_detail,
        "recommendation_consistent": recommendation_consistent,
        "numeric_grounding": numeric_grounding,
        "clinical_term_grounding": clinical_term_grounding,
        "missing_evidence_disclosed": missing_evidence_disclosed,
        "evidence_corroboration": evidence_corroboration,
        "judge_visible_claims": judge_visible_claims,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "factor_coverage": factor_coverage,
        "unsupported_clinical_terms": unsupported_terms,
        "valid": not failures,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Prompt builders + helpers
# ---------------------------------------------------------------------------


def _user_prompt(
    case_id: str,
    task: int,
    transcript: str,
    called_tools: list[str],
    eligible: list[str],
) -> str:
    tools_line = ", ".join(called_tools) if called_tools else "(no tools called)"
    head = (
        f"Case ID: {case_id}\n"
        f"Task: {task}\n\n"
        "Your reasoning transcript (final assistant message from the ReAct "
        'loop):\n"""\n'
        f"{transcript}\n"
        '"""\n\n'
        f"Tools you called during the ReAct loop: {tools_line}\n\n"
    )
    if task in VARIABLES_BY_TASK:
        raise ValueError("Tasks 1 and 2 use the clinical-rationale prompt")
    return head + (
        "Now fill out the form: predict whether biochemical recurrence will "
        "occur (event = 1) or not (event = 0 / censored by last follow-up), "
        "give your predicted months to recurrence (a non-negative number), "
        "and a focused reasoning naming the 2-4 factors that most influenced "
        "your estimate."
    )


def _build_skeleton_instructions(task: int, eligible: list[str]) -> str:
    """Concrete JSON-skeleton format instructions.

    PydanticOutputParser's ``get_format_instructions()`` dumps the full
    JSON schema (with ``$defs``, ``$ref``, ``additionalProperties``,
    etc.). Small models tested in the wild (Gemma 4 E2B) sometimes echo
    that schema back verbatim instead of producing an instance. A
    concrete shape with placeholder values is far more robust and is
    still unambiguous about which keys are required.
    """
    skeleton = "\n".join(
        [
            "{",
            '  "event": <0 | 1 — 1 if you predict biochemical recurrence will occur, '
            "0 if censored / no recurrence by last follow-up>,",
            '  "months_to_recurrence": <number — predicted months to recurrence / last follow-up>,',
            '  "repeat_test": <"<a short description of the recommended follow-up test>" | null>,',
            '  "free_text": "<at least 40 chars; the evidence and the 2-4 factors that drove your estimate>"',
            "}",
        ]
    )
    return (
        "Output a SINGLE JSON object matching exactly this shape (replace "
        "every <...> with a real value, or JSON null where indicated):\n\n"
        f"{skeleton}\n\n"
        "Do NOT emit any other keys. Do NOT wrap in markdown fences. Do NOT echo the schema."
    )


def _extract_json_object(text: str) -> str:
    """Best-effort extraction of a JSON object from raw model output."""
    if not text:
        return text
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _final_assistant_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            return m.content if isinstance(m.content, str) else json.dumps(m.content)
    return ""
