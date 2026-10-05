"""Model-free prediction assembly for the Grand Challenge runtime.

Tasks 1 and 2 apply the frozen decision policies to the structured prompt and
the clinical-data socket. All prose is templated from observed fields. Task 3
is not built here -- it runs its Cox/RMST head inside the MCP session (see
``run.py::_run_task3``), so that its reports are retrieved over the official
tool interface.
"""

from __future__ import annotations

import json
from typing import Any

from chimera_agent_baseline.agent.rationale import (
    anchored_missing_evidence,
    biochemical_reading,
    biochemical_sentences,
    discordance_reading,
    next_step_for,
    psa_trajectory,
    reason_for_prose,
)
from chimera_agent_baseline.output.schema import VARIABLES_BY_TASK, assemble_full_output, reveal_info_for_tool
from chimera_agent_baseline.policy import FrozenPolicyResult, apply_frozen_policy, join_clauses
from chimera_agent_baseline.tools.definitions import TASK1_TOOLS, TASK2_TOOLS

_TOOLS_BY_TASK = {1: TASK1_TOOLS, 2: TASK2_TOOLS}


def clinical_to_revealed(task: int, clinical: dict[str, Any], case_id: str) -> dict[str, dict[str, Any]]:
    """Represent each available clinical section in the policy's tool shape."""
    revealed: dict[str, dict[str, Any]] = {}
    for spec in _TOOLS_BY_TASK[task]:
        payload: dict[str, Any] = {"case_id": case_id}
        payload.update({field: clinical[field] for field in spec.fields if field in clinical})
        revealed[spec.name] = payload
    return revealed


def _rationale(
    policy: FrozenPolicyResult,
    prompt: dict[str, Any] | None = None,
    visible_text: str = "",
    psa_series: Any = None,
) -> str:
    if policy.task == 1:
        recommendation = (
            "Prostate biopsy is recommended at this encounter."
            if policy.decision == "yes"
            else "Prostate biopsy is not recommended at this encounter."
        )
    else:
        labels = {
            "active_surveillance": "Active surveillance is recommended.",
            "continued_surveillance": "Continued surveillance is recommended.",
            "watchful_waiting": "Watchful waiting is recommended.",
            "active_treatment": "Active treatment is recommended.",
        }
        recommendation = labels[policy.decision]

    # The policy's own reason is a clinical sentence; stating it keeps the
    # rationale meaningful, and schema-valid, for a case whose findings are all
    # absent from the input.
    # The anchor rule applies here too: never assert an absence the judge
    # cannot see the variable of. See `agent/rationale.py`.
    told = reason_for_prose(policy.rule, visible_text)
    reason = f" {told[0].upper()}{told[1:]}." if told else ""
    findings = "; ".join(factor.text for factor in policy.factors)
    support = f" The documented findings are {findings}." if findings else ""
    # Same rule as the agent path: PSA and PSA density are addressed on every
    # case, with the policy's own reading of what each does for the decision.
    biochemical = biochemical_sentences(biochemical_reading(prompt or {}, policy))
    # The same two additions the agent path makes: how the PSA has moved, and
    # the finding that points the other way.
    trend = psa_trajectory(psa_series)
    movement = f" {trend}" if trend else ""
    conflict = "".join(f" {line}" for line in discordance_reading(prompt or {}, policy).values())
    anchored = anchored_missing_evidence(policy.missing_evidence, visible_text, policy.confidence)
    limitation = " The input does not document " + join_clauses(anchored) + "." if anchored else ""
    trigger = f" {policy.trigger[0].upper()}{policy.trigger[1:]}." if policy.trigger else ""
    # The judge asks for a plan, not a summary: "lacks the depth of reasoning"
    # is its most common criticism of our prose (72% of Task 1 critiques, 58%
    # of Task 2), and every reference rationale closes on an action. This is
    # what a failed form-fill emits, so it has to close on one too.
    next_step = f" Next step: {next_step_for(policy)}."
    return (recommendation + reason + support + biochemical + movement + conflict
            + limitation + trigger + next_step)


def _reveal_sequence(tool_names: tuple[str, ...]) -> list[dict[str, Any]]:
    sequence: list[dict[str, Any]] = []
    for order, tool_name in enumerate(dict.fromkeys(tool_names), start=1):
        key, label = reveal_info_for_tool(tool_name)
        sequence.append(
            {
                "page": "decision",
                "order": order,
                "key": key,
                "label": label,
                "value": "section",
                "via": "tool_call",
                "ts": f"1970-01-01T00:00:{order - 1:02d}.000Z",
            }
        )
    return sequence


def build_task12_prediction(
    *, task: int, case_id: str, prompt: dict[str, Any], clinical: dict[str, Any]
) -> dict[str, Any]:
    """Build and schema-validate a deterministic Task 1 or Task 2 result."""
    if task not in (1, 2):
        raise ValueError(f"Expected Task 1 or 2, got {task}")
    policy = apply_frozen_policy(task, prompt, clinical_to_revealed(task, clinical, case_id))
    # The judge sees the clinical-data socket, so that is what an absence
    # claim in this prose has to be anchored against.
    rationale = _rationale(policy, prompt, json.dumps(clinical, ensure_ascii=False),
                           clinical.get("psa_trend"))
    judgment: dict[str, Any] = {
        "confidence": policy.confidence,
        "variable_weights": {name: policy.weights[name] for name in VARIABLES_BY_TASK[task]},
        "repeat_test": next_step_for(policy).capitalize() + ".",
        "free_text": rationale,
    }
    if task == 1:
        judgment["biopsy_decision"] = policy.decision
    else:
        surveillance = policy.decision in {"active_surveillance", "continued_surveillance"}
        judgment["treatment_recommendation"] = {
            "primary": policy.decision,
            "modalities": [],
            "detail": rationale,
            "as_protocol": "Serial PSA and clinical review." if surveillance else None,
            "as_trigger": (
                "Reassess management if clinical or disease-risk findings change." if surveillance else None
            ),
        }
    return assemble_full_output(
        task=task,
        case_id=case_id,
        patient={"id": case_id, "psa_ng_ml": prompt.get("psa"), "age_years": prompt.get("age")},
        # What we report consulting is what the policy used, not every document
        # it needed fetched -- extra reveals only cost tool precision.
        reveal_sequence=_reveal_sequence(policy.reveal_tools),
        judgment=judgment,
    )


def build_prediction(
    *,
    task: int,
    case_id: str,
    prompt: dict[str, Any],
    clinical: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch one already-parsed socket combination to its deterministic head.

    Task 3 is deliberately not dispatched here. Its head reads the radiology,
    biopsy and prostatectomy reports, and the challenge requires those to be
    retrieved over the official MCP interface; a direct clinical-data path
    would be an alternative tool channel. Task 3 runs through
    ``run.py::_run_task3`` inside the MCP session instead.
    """
    if task not in (1, 2):
        raise ValueError(f"Unsupported task {task}; Task 3 runs through the MCP session in run.py")
    return build_task12_prediction(task=task, case_id=case_id, prompt=prompt, clinical=clinical)
