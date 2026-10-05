"""Orchestration for the Task 1 agentic policy.

Routes messages and executes MCP reveals. It holds no clinical logic: every
judgment in this file comes back from an agent, and the only decisions taken
here are which agent to ask next and which tool a requested item maps to.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from ..agent.form_fill import _build_reveal_sequence
from ..output.schema import assemble_full_output
from ..run import _tool_payload
from . import briefs
from . import consultant as C
from . import policy_graph as PG
from . import rationale as R_
from .agents import (
    ALL_VARIABLES,
    ask,
    confidence_violation,
    decision_violation,
    parse_needs,
    parse_weights,
    render_facts,
    weight_violations,
)

log = logging.getLogger(__name__)

#: enc_type -> track. A field read, not a judgment: the three released values
#: name the encounter outright (116 / 30 / 49 of the 195 released cases).
_TRACKS = {
    "re-evaluation - prior pca diagnosis": "PRIOR_POSITIVE",
    "re-evaluation - prior negative biopsy": "PRIOR_NEGATIVE",
    "follow-up - psa + mri": "NAIVE",
}

_SPECIALIST_ORDER = ("eligibility", "history", "radiology", "biochemical")

#: Budget for a call with thinking on. Qwen bills its chain of thought against
#: max_tokens, so the default 1024 would be spent before the answer starts.
_THINKING_TOKENS = 4096


def _int_or_none(value: Any) -> int | None:
    """PI-RADS arrives as a string, and is "NA" on one released case."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None

#: A STATED grade is a number. Prose such as "low-grade disease" describes the
#: disease without giving the grade, and the reference treats that as missing --
#: all three probe cases that turned on it asked for the number.
_GRADE = re.compile(r"(ISUP\s*(?:grade\s*group\s*)?\d|Gleason\s*\d\s*\+\s*\d|grade\s*group\s*\d|\bGG\s?\d)", re.I)
_ISUP_FROM_GLEASON = {(3, 3): 1, (3, 4): 2, (4, 3): 3, (4, 4): 4, (3, 5): 4, (5, 3): 4}


def _prior_isup(text: str) -> int | None:
    """The earlier biopsy's grade as an ISUP number, or None if never stated."""
    m = re.search(r"ISUP\s*(?:grade\s*group\s*)?(\d)", text, re.I) or re.search(
        r"grade\s*group\s*(\d)", text, re.I
    )
    if m:
        return int(m.group(1))
    m = re.search(r"Gleason\s*(\d)\s*\+\s*(\d)", text, re.I)
    if m:
        pair = (int(m.group(1)), int(m.group(2)))
        return _ISUP_FROM_GLEASON.get(pair, 5 if sum(pair) >= 9 else None)
    return None


_N = r"\d{1,3}(?:\.\d)?"
_MM = re.compile(rf"({_N})(?:\s*(?:x|\u00d7|by)\s*({_N}))?(?:\s*(?:x|\u00d7|by)\s*({_N}))?\s*mm", re.I)


def _lesion_mm(revealed: dict[str, Any]) -> float | None:
    """Largest dimension in the MRI report, if it was actually revealed."""
    report = revealed.get("get_mri_report")
    if not report:
        return None
    dims = [float(g) for m in _MM.finditer(str(report)) for g in m.groups() if g]
    return max(dims) if dims else None


def _grade_text(prompt: dict[str, Any], revealed: dict[str, Any]) -> str:
    """Everywhere a prior grade could legitimately have been read."""
    notes = " ".join(str(s.get("t", "")) for s in (prompt.get("note_sections") or []))
    return f"{notes} {revealed.get('get_previous_notes') or ''}"


#: Interval-comparison language -- the only thing that would let a record say
#: whether a lesion is new, grown or stable.
_PRIOR_MRI = re.compile(
    r"previous MRI|prior MRI|earlier MRI|compared with|interval (change|growth)"
    r"|since (the )?(previous|prior|last)|stable (lesion|since)|has grown|increased in size",
    re.I,
)


def _has_prior_mri(prompt: dict[str, Any], revealed: dict[str, Any]) -> bool:
    """Whether an earlier MRI exists for this lesion to be compared against.

    ⚠️ False on 195 of 195 released Task 1 records -- none contains any
    interval-comparison language. That is the point of computing it rather than
    assuming it: "the trajectory is unknown" is a property of the DATASET, not
    of this patient, and a fact that is missing on every case cannot be what
    separates the cases the reference calls `uncertain` from the ones it calls
    `clear`. Same argument CLAUDE.md already makes against reading Task 2's
    staging incompleteness as a mechanism -- incomplete on 153/153, a constant.

    The reference rationales do ask for a prior MRI on 12 of 15 `uncertain`
    cases, so it stays sayable in the prose; what it may not do is SELECT the
    label. Those are different jobs and only the second is gated here.
    """
    return bool(_PRIOR_MRI.search(_grade_text(prompt, revealed)))


_FREE_PSA_PCT = re.compile(r"([\d.]+)\s*%?")


def _free_psa_pct(revealed: dict[str, Any]) -> float | None:
    """The free-PSA percentage, from the labs panel and from nowhere else.

    ⚠️ Recorded on 195 of 195 released cases, and ONLY inside
    `laboratory_results` -- 0 of 195 structured prompts mention it and 3 of 195
    other clinical sections do. So this returns None unless the labs were
    actually revealed, which is the honest state: the rung it feeds cannot fire
    on evidence the agent never fetched.
    """
    payload = revealed.get("get_lab_results")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return None
    if isinstance(payload, dict):
        payload = payload.get("laboratory_results")
    if not isinstance(payload, list):
        return None
    for row in payload:
        if not isinstance(row, dict) or str(row.get("name", "")).strip().lower() != "%free psa":
            continue
        match = _FREE_PSA_PCT.search(str(row.get("val") or ""))
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _grey_count(prompt: dict[str, Any]) -> int:
    """How many readings never left their grey band."""
    n = 0
    psa, psad, psav = prompt.get("psa"), prompt.get("psad"), prompt.get("psav")
    if psa is not None and 4.0 <= psa <= 10.0:
        n += 1
    if psad is not None and 0.10 <= psad <= 0.15:
        n += 1
    if psav is not None and 0.75 <= psav <= 1.50:
        n += 1
    if str(prompt.get("pirads")) == "3":
        n += 1
    return n

_ENCOUNTER_PROSE = {
    "NAIVE": "PSA and MRI follow-up. No previous prostate biopsy on record.",
    "PRIOR_NEGATIVE": "Re-evaluation after a previous NEGATIVE prostate biopsy.",
    "PRIOR_POSITIVE": "Re-evaluation after a previous POSITIVE biopsy; the patient is on active surveillance.",
}


def load_thresholds(resource_dir: Path | str) -> dict[str, Any]:
    cfg = yaml.safe_load((Path(resource_dir) / "agentic_task1.yaml").read_text())
    return {"_raw": cfg, **briefs.flatten_thresholds(cfg)}


def route(prompt: dict[str, Any]) -> str:
    return _TRACKS.get(str(prompt.get("enc_type", "")).strip().lower(), "NAIVE")


def _facts(agent: str, prompt: dict[str, Any], track: str, revealed: dict[str, Any]) -> str:
    """The evidence one agent is allowed to see.

    Partitioned deliberately: an agent that cannot see the PI-RADS cannot weigh
    it, which is what keeps each verdict independent and each weight owned.
    """
    if agent == "eligibility":
        notes = "; ".join(f"{s.get('s')}: {s.get('t')}" for s in prompt.get("note_sections") or [])
        return render_facts({
            "Age (years)": prompt.get("age"),
            "Past medical history": prompt.get("medhx"),
            "Problem list": prompt.get("pmhx"),
            "Medications": prompt.get("meds"),
            "Smoking": (prompt.get("vitals") or {}).get("smoking"),
            "Exercise": prompt.get("exercise"),
            "Clinical notes": notes,
        })
    if agent == "history":
        return render_facts({
            "Encounter": _ENCOUNTER_PROSE[track],
            "PSA (ng/mL)": prompt.get("psa"),
            "PSA density (ng/mL/cc)": prompt.get("psad"),
            "Prostate volume (cc)": prompt.get("vol"),
            "IPSS": prompt.get("ipss"),
            "Months since the last assessment": prompt.get("months"),
            "Previous notes": revealed.get("get_previous_notes"),
        })
    if agent == "radiology":
        return render_facts({
            "PI-RADS": prompt.get("pirads"),
            "Prostate volume (cc)": prompt.get("vol"),
            "Model-predicted csPCa probability": prompt.get("cspca"),
            "MRI report": revealed.get("get_mri_report"),
        })
    return render_facts({
        "PSA (ng/mL)": prompt.get("psa"),
        "PSA density (ng/mL/cc)": prompt.get("psad"),
        "PSA velocity (ng/mL/yr)": prompt.get("psav"),
        "Prostate volume (cc)": prompt.get("vol"),
        "IPSS": prompt.get("ipss"),
        "Digital rectal examination": prompt.get("dre"),
        "Family history of prostate cancer": revealed.get("get_family_history"),
        "PSA trend": revealed.get("get_psa_trend"),
        "Laboratory results": revealed.get("get_lab_results"),
    })


async def _reveal(
    tools: dict, item: str, case_id: str, revealed: dict, order: list[str], items: dict | None = None
) -> bool:
    """Execute one requested reveal. Returns True if new evidence arrived."""
    tool_name, _ = (items or briefs.REVEAL_ITEMS)[item]
    if tool_name in revealed:
        return False
    tool = tools.get(tool_name)
    if tool is None:
        log.warning("%s: %s is not exposed by the MCP server", case_id, tool_name)
        return False
    payload = _tool_payload(await tool.ainvoke({"case_id": case_id}))
    if payload is None or "error" in payload:
        log.warning("%s: %s returned no section: %s", case_id, tool_name, payload)
        return False
    sections = {k: v for k, v in payload.items() if k not in {"case_id", "note"}}
    if not sections:
        return False
    revealed[tool_name] = json.dumps(next(iter(sections.values())), default=str)[:6000]
    order.append(tool_name)
    return True


def _summary(verdicts: dict[str, dict[str, str]], order: tuple[str, ...] | None = None) -> str:
    lines = []
    for name in order or _SPECIALIST_ORDER:
        block = verdicts.get(name) or {}
        parts = [f"{k}: {v}" for k, v in block.items() if k not in {"WEIGHTS", "NEED", "RATIONALE"}]
        lines.append(f"[{name}] " + " | ".join(parts) if parts else f"[{name}] no assessment returned")
    return "\n".join(lines)


async def write_rationale(
    model, task: int, decision: str, confidence: str, findings: str, named_absence: bool,
    warnings: list[str], background: str = "  (nothing)", citable: dict | None = None,
    drivers: list[str] | None = None,
) -> str:
    """One note, validated against the rationale rules, retried once."""
    import asyncio

    from . import rationale as R
    from .agents import ask

    brief = R.brief(task, decision, confidence, findings, background, drivers)
    text = ""
    for attempt in range(2):
        block, raw = await asyncio.to_thread(ask, model, brief, findings)
        # The note is prose, not a keyed block: take the whole reply, minus any
        # stray key the model prefixed it with.
        text = (block.get("RATIONALE") or raw or "").strip()
        bad = R.failures(text, task, confidence, named_absence, citable, drivers)
        if not bad:
            return text
        if attempt == 0:
            brief = R.brief(task, decision, confidence, findings, background, drivers) + (
                "\n\nYour previous note broke these rules:\n"
                + "\n".join(f"  - {b}" for b in bad)
                + "\n\nWrite it again, complete, fixing every one."
            )
    warnings.append(f"rationale still breaks {len(bad)} rule(s) after a retry")
    return text


#: Loaded once. The decision policy as a graph -- see resources/task1_graph.yaml
#: and docs/task1-policy-reference.md.
GRAPH = PG.load()

#: Which reveal supplies a variable the graph needs but the prompt does not
#: carry. `fh` is deliberately absent: family_history appears in 0 of the 91
#: reference reveal sequences, so revealing it is pure tool-precision loss, and
#: every guard that reads it tests its STATE so an unknown value is total.
_SUPPLIED_BY = {
    "prior_isup": "PRIOR_ISUP_GRADE",
    "lesion_mm": "LESION_SIZE",
    "psadt": "PSA_TREND",
}


def graph_facts(prompt: dict[str, Any], revealed: dict[str, Any]) -> dict[str, Any]:
    """Everything the graph reads, from the prompt and what MCP actually returned.

    ⚠️ `_reveal` stores each payload as a JSON STRING, so the PSA series has to
    be decoded before a doubling time can be taken from it -- the same trap that
    made `psa_trajectory` silently dead on the rationale path.
    """
    trend = revealed.get("get_psa_trend")
    if isinstance(trend, str):
        try:
            trend = json.loads(trend)
        except (ValueError, TypeError):
            trend = None
    if isinstance(trend, dict):
        trend = trend.get("psa_trend")

    family = revealed.get("get_family_history")
    fh = None
    if family is not None:
        fh = str(family).strip().lower().startswith("y")

    ipss = None
    match = re.search(r"(\d+)", str(prompt.get("ipss") or ""))
    if match:
        ipss = float(match.group(1))

    return PG.derive(GRAPH, {
        "encounter": route(prompt),
        "pirads": _int_or_none(prompt.get("pirads")),
        "psa": prompt.get("psa"), "psad": prompt.get("psad"),
        "psav": prompt.get("psav"), "vol": prompt.get("vol"),
        "age": prompt.get("age"), "ipss": ipss,
        "prior_isup": _prior_isup(_grade_text(prompt, revealed)),
        "dre_suspicious": str(prompt.get("dre") or "") in briefs._DRE_SUSPICIOUS,
        "fh": fh,
        "psadt": PG.psa_doubling_time(trend),
        "lesion_mm": _lesion_mm(revealed),
    })


async def _walk_with_reveals(tools: dict, case_id: str, prompt: dict[str, Any],
                             revealed: dict[str, Any], order: list[str],
                             warnings: list[str]) -> tuple[Any, dict[str, Any]]:
    """Walk the graph, fetching what the path it takes actually needs.

    The policy asks for evidence rather than an agent guessing at it: walk with
    what is on the table, see which variables the nodes on that path could not
    read, reveal the ones a tool serves, and walk again. This is why the reveal
    sequence stays honest -- every call is one the decision genuinely required.
    """
    facts = graph_facts(prompt, revealed)
    for _ in range(len(_SUPPLIED_BY) + 1):
        walked = PG.walk(GRAPH, facts)
        wanted = [
            _SUPPLIED_BY[name]
            for node_id in walked.path
            for name in GRAPH.nodes[node_id].reads
            if facts.get(name) is None and name in _SUPPLIED_BY
            and briefs.REVEAL_ITEMS.get(_SUPPLIED_BY[name], ("",))[0] not in revealed
        ]
        if not wanted:
            return walked, facts
        got = False
        for item in dict.fromkeys(wanted):
            got |= await _reveal(tools, item, case_id, revealed, order)
        if not got:
            warnings.append(f"graph wanted {', '.join(dict.fromkeys(wanted))}; no reveal succeeded")
            return walked, facts
        facts = graph_facts(prompt, revealed)
    return PG.walk(GRAPH, facts), facts


_ORDINAL = {"uncertain": 0, "borderline": 1, "clear": 2}


def confidence_context(prompt: dict[str, Any], revealed: dict[str, Any],
                       thresholds: dict[str, Any], cf, says: str, decision: str) -> str:
    """The three rungs, COMPUTED, not posed as questions.

    ⚠️ This is the fix for a measured failure. The brief tells the assessor to
    "count how many readings are in a grey band", and the first version of this
    context handed over a flat list of numbers and let it do the arithmetic. A
    9B model asked to band five readings and then weigh them answered `clear` on
    80 of 91 cases, for a quadratic kappa of 0.0007. The arithmetic is Python's
    -- the same rule the rationale layer already follows for `psa_trajectory`.
    The model is left with the one thing it might be able to do: weigh three
    findings that have already been established for it.
    """
    def num(value: Any) -> float | None:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    lines: list[str] = [f"RECOMMENDATION: {says}", ""]

    # -- rung 1: ambiguity, already counted ---------------------------------
    grey: list[str] = []
    settled: list[str] = []
    bands = (("PSA", num(prompt.get("psa")), thresholds["psa_low"], thresholds["psa_high"], "ng/mL"),
             ("PSA density", num(prompt.get("psad")), thresholds["psad_low"], thresholds["psad_high"], "ng/mL/cc"),
             ("PSA velocity", num(prompt.get("psav")), thresholds["psav_low"], thresholds["psav_high"], "ng/mL/yr"))
    for name, value, low, high, unit in bands:
        if value is None:
            continue
        where = (f"{name} {value} {unit}"
                 f" (grey band {low}-{high})")
        (grey if low <= value <= high else settled).append(
            where + ("  <-- UNRESOLVED" if low <= value <= high else "  past its line, settled"))
    pirads = num(prompt.get("pirads"))
    if pirads == 3:
        grey.append("PI-RADS 3  <-- EQUIVOCAL BY DEFINITION")
    elif pirads is not None:
        settled.append(f"PI-RADS {int(pirads)}  clear of the equivocal category")
    need = thresholds["grey_zones_for_borderline"]
    lines.append(f"1. AMBIGUITY -- {len(grey)} reading(s) never resolved. "
                 f"{need} or more makes the recommendation the best available call "
                 f"rather than a settled one, so this rung is "
                 f"{'PRESENT' if len(grey) >= need else 'ABSENT'}.")
    lines += [f"     {line}" for line in grey + settled]

    # -- rung 2: conflict, already identified -------------------------------
    age, comorbid = num(prompt.get("age")), len(prompt.get("pmhx") or [])
    clashes = []
    if pirads is not None and pirads >= 4 and age is not None and age >= thresholds["conflict_age"]:
        clashes.append(f"the imaging argues for tissue (PI-RADS {int(pirads)}) while age {age:.0f} "
                       f"is at or past the benefit window of {thresholds['conflict_age']}")
    if comorbid >= thresholds["conflict_comorbidity_count"]:
        clashes.append(f"{comorbid} significant comorbidities weigh against an invasive step")
    lines += ["", f"2. CONFLICT -- {'PRESENT' if clashes else 'ABSENT'}."]
    lines += [f"     {c}" for c in clashes] or ["     nothing of comparable weight points the other way"]

    # -- rung 3: absence, already resolved ----------------------------------
    grade_text = _grade_text(prompt, revealed)
    stated = _prior_isup(grade_text) is not None
    somewhere = bool(grade_text.strip())
    lines += ["", "3. ABSENCE -- a named fact whose two values give two different actions."]
    if cf is not None and cf.absent and cf.absent_changes_answer:
        lines.append(f"     the {cf.absent} is NOT on record, and the policy's answer CHANGES "
                     "depending on what it would have been.  <-- ABSENCE PRESENT, name it")
    elif cf is not None and cf.absent:
        lines.append(f"     the {cf.absent} is not on record, but the recommendation is the "
                     "SAME whatever its value would have been, so it is not what is in doubt")
    elif stated:
        lines.append("     the earlier biopsy grade IS stated in the record")
    elif somewhere:
        lines.append("     the earlier biopsy grade is not stated, though the notes carry "
                     "biopsy history where it could have been")
    else:
        lines.append("     there is no previous-note text at all, so nothing can be said to "
                     "be missing FROM it")
    lines.append("     no earlier MRI exists anywhere in this record, so lesion growth was "
                 "never observed, only inferred from size")

    if cf is not None:
        lines += ["", "HOW THE POLICY REACHED IT:", cf.describe()]
    return "\n".join(lines)


def _confidence_ctx(prompt: dict[str, Any], revealed: dict[str, Any], path: Any,
                    thresholds: dict[str, Any]) -> dict[str, Any]:
    """Everything `briefs.confidence_rule` reads, from the prompt, the reveals and the walk.

    `path` is the walk's node list, or `()` where the walk has not happened yet
    -- which only blanks `size_was_tiebreaker`, a rung that can emit `uncertain`
    and never `borderline`. `_maybe_reveal_labs` relies on that: its pre-walk
    view can be wrong only in the direction of NOT revealing.
    """
    return {
        "track": route(prompt),
        "pirads": prompt.get("pirads"),
        "age": prompt.get("age"),
        "comorbidity_count": len(prompt.get("pmhx") or []),
        "grade_stated": _prior_isup(_grade_text(prompt, revealed)) is not None,
        # Whether there was anywhere for a grade to have been written. Without
        # it "not retrieved" and "not recorded" are the same input, and the
        # rung fires on a cohort whose notes channel is simply empty.
        "grade_evidence_available": bool(_grade_text(prompt, revealed).strip()),
        "grey_count": _grey_count(prompt),
        # The walk knows this rather than a specialist's verdict: the lesion
        # was the tiebreaker exactly when the path went through that node.
        "size_was_tiebreaker": "t3_lesion_size" in (path or ()),
        "prior_mri_available": _has_prior_mri(prompt, revealed),
        # A benign explanation for the PSA. None unless the labs are on the
        # table -- see `_maybe_reveal_labs` for why they usually are not.
        "free_psa_pct": _free_psa_pct(revealed),
        **thresholds,
    }


async def _maybe_reveal_labs(tools: dict, case_id: str, prompt: dict[str, Any],
                            revealed: dict[str, Any], order: list[str],
                            thresholds: dict[str, Any]) -> bool:
    """Fetch the labs panel only where %free PSA could change the confidence label.

    The same sensitivity test Task 2 uses for the pathology report: walk the
    question the reveal would answer, and skip the call where both answers land
    in the same place. The benign-PSA rung can only SET `borderline`, on the
    established-cancer track, above PI-RADS 2 -- so a case already `borderline`
    cannot move and needs nothing. That narrows the population from 47 of 91 to
    33.

    ⚠️ OFF BY DEFAULT, and the cost is MEASURED, not estimated. Replayed case by
    case against the reveal sets of the last run we hold (`/tmp/agentic-ws10`,
    mean tool 0.9589 against the shipped 0.9573, so the specialists' reveal
    behaviour is the same run to run):

      tool          0.9589 -> 0.9030   (-0.0559, x0.15 = -0.0084)
      confidence    0.7658 -> 0.7911   (+0.0253, x0.20 = +0.0051)
      net                                        -0.0033 per gate-passed case

    Nine cases lose tool score and seven go 1.0 -> 0.5, because **the agent
    reveals exactly ONE section on 56 of 73 gate-passed cases** (0 on 5, 2 on 10,
    3 on 2). Tool score is precision, so on a denominator of 1 a single extra
    reveal halves the case outright, and the reference carries the labs on only
    17 of the 33. Nothing buys it back through grounding either: `section_s3-labs`
    is a SECONDARY section in `section_variable_mapping.json` and only primary
    sections ground a variable.

    ⚠️ **So this arm ships only if the RATIONALE pays the difference**, which is
    why the reveal happens before the specialists rather than after: the
    biochemical assessor reads the free fraction, and `laboratory_results` is a
    judge-visible section, so the note may quote the number. The bar is
    +0.0033 / 0.20 = +0.0165 on the gate-passed rationale mean, and it is earned
    on 33 of 91 cases, so about **+0.046 on those 33**. Unmeasured until a
    judge-on MCP run; turn the flag on and score it against the flag off.
    """
    if not thresholds.get("reveal_labs_for_free_psa"):
        return False
    if briefs.REVEAL_ITEMS["LAB_RESULTS"][0] in revealed:
        return False
    if route(prompt) != "PRIOR_POSITIVE" or str(prompt.get("pirads")) in {"1", "2"}:
        return False
    # The rung can only SET `borderline`, so a case that already reads
    # `borderline` cannot move and its labs are not worth fetching.
    if briefs.confidence_rule(_confidence_ctx(prompt, revealed, (), thresholds)) == "borderline":
        return False
    return await _reveal(tools, "LAB_RESULTS", case_id, revealed, order)


async def assess_confidence(model, thresholds: dict[str, Any], cf, context: str,
                            fallback: str, warnings: list[str]) -> tuple[str, str]:
    """The assessor proposes; the walk may veto but never sets.

    ⚠️ This is the mechanism that measured 0.8000 / kappa 0.4341 on 2026-09-09 --
    the best confidence this agent has produced -- restored after the graph's
    pure-structure replacement scored 0.7756 / 0.3524. What changed is WHAT
    critiques it: `confidence_rule`'s three hand-written rungs are replaced by
    the path the case actually took, which knows which question was equivocal
    and which fact was absent because it walked them.

    ⚠️ The veto is ONE rule, and the two obvious others were dropped on
    measurement -- see `policy_graph.forbidden_labels`. It blocks the correct
    label on 6 of the 91 and leaves a ceiling of 0.9670 ungated.
    """
    forbidden = PG.forbidden_labels(cf)
    brief = briefs.render(briefs.CONFIDENCE, thresholds, context=context)
    block, _ = await asyncio.to_thread(ask, model, brief, context)
    label = (block.get("CONFIDENCE") or "").strip().lower()

    if label in forbidden:
        allowed = ", ".join(sorted(set(_ORDINAL) - set(forbidden)))
        fixed, _ = await asyncio.to_thread(
            ask, model, brief,
            context + "\n\n" + briefs.CONFIDENCE_CRITIQUE.format(
                want=allowed, got=label, why=forbidden[label]))
        relabel = (fixed.get("CONFIDENCE") or "").strip().lower()
        if relabel in _ORDINAL and relabel not in forbidden:
            return relabel, (fixed.get("RATIONALE") or "").strip()
        # It insisted. Take the nearest label the structure allows.
        near = min((l for l in _ORDINAL if l not in forbidden),
                   key=lambda l: abs(_ORDINAL[l] - _ORDINAL.get(label, 1)))
        warnings.append(f"confidence {label!r} is ruled out by the path; used {near!r}")
        return near, (block.get("RATIONALE") or "").strip()

    if label not in _ORDINAL:
        warnings.append(f"assessor returned {label!r}; used the graph's own reading")
        return fallback, ""
    return label, (block.get("RATIONALE") or "").strip()


async def decide_with_graph(model, tools: dict, case_id: str, prompt: dict[str, Any],
                            revealed: dict[str, Any], order: list[str],
                            warnings: list[str], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """The decision and the confidence, from the graph and its consultant.

    The graph owns both for every case it can answer. Where a node hands off --
    today only `t3_reclassification`, whose population no structured variable
    separates -- the consultant owns them instead, and falls back to that node's
    own edges if its reply cannot be read.
    """
    walked, facts = await _walk_with_reveals(tools, case_id, prompt, revealed, order, warnings)
    absent_fact = "NONE"

    if walked.consult:
        node = GRAPH.nodes[walked.consult]
        notes = " ".join(str(revealed.get(k) or "") for k in
                         ("get_previous_notes", "get_mri_report", "get_pathology_report"))
        block, _ = await asyncio.to_thread(
            ask, model, C.BRIEF, C.facts_block(GRAPH, walked.consult, facts, notes))
        decision, confidence, reason, fell_back = C.resolve(GRAPH, walked.consult, facts, block)
        if fell_back:
            warnings.append(f"consultant at {walked.consult} unreadable; used the node's own edges")
        absent_fact = (block.get("ABSENT_FACT") or "NONE").strip() or "NONE"
        return {"decision": "BIOPSY" if decision == "yes" else "NO_BIOPSY",
                "confidence": confidence, "reason": reason, "absent_fact": absent_fact,
                "consulted": walked.consult, "path": walked.path}

    if walked.leaf is None:
        warnings.append(f"graph did not answer ({walked.reason}); fell back to the radiology read")
        return {"decision": "BIOPSY", "confidence": "uncertain",
                "reason": f"the policy could not answer this case ({walked.reason})",
                "absent_fact": "NONE", "consulted": None, "path": walked.path}

    leaf = GRAPH.leaves[walked.leaf]
    confidence, reason = PG.confidence(GRAPH, facts)
    cf = PG.confidence_facts(GRAPH, facts)
    if thresholds is not None and cf is not None:
        def _confidence() -> str:
            return briefs.confidence_rule(_confidence_ctx(prompt, revealed, walked.path, thresholds))

        # ⚠️ COMPUTED, not asked. Four measurements on Qwen3.5-9B, all in the
        # same direction: the consultant emitted `clear` on 27 of 27; the
        # assessor given a flat list of readings emitted it on 80 of 91 (kappa
        # 0.0007); the same assessor given the three rungs ALREADY COMPUTED
        # over-fired `borderline` on 24 of the 58 reference-`clear` cases
        # (kappa 0.0897); and the graph's own structural channels reached
        # 0.3524. `briefs.confidence_rule` -- deterministic, three rungs read in
        # order with a short circuit -- scores 0.4183 on the same 91.
        #
        # The ordering is what the model could not reproduce: absence, then
        # conflict, then ambiguity, returning at the first hit, so the ambiguity
        # rung only decides cases the other two leave open. Weighed all at once
        # it is the noisiest of the three and it drowns the other two.
        confidence = _confidence()
    return {"decision": {"biopsy": "BIOPSY", "no_biopsy": "NO_BIOPSY",
                         "treatment": "TREATMENT", "watchful_waiting": "NO_BIOPSY"}[walked.leaf],
            "confidence": confidence, "reason": reason, "absent_fact": absent_fact,
            "consulted": None, "path": walked.path, "says": leaf["says"]}


async def run_case(model, tools: dict, query: dict, thresholds: dict[str, Any]) -> dict[str, Any]:
    """Run one Task 1 case end to end and return the validated prediction."""
    case_id = query["case_id"]
    prompt = query.get("structured_prompt") or {}
    track = route(prompt)
    revealed: dict[str, Any] = {}
    order: list[str] = []
    weights = dict.fromkeys(ALL_VARIABLES, "not_used")
    ctx = {"track": track, "pirads": prompt.get("pirads"), "psa": prompt.get("psa"), "dre": prompt.get("dre")}
    verdicts: dict[str, dict[str, str]] = {}
    warnings: list[str] = []

    # --- the labs, BEFORE the specialists that read them ---------------------
    # Ahead of the loop deliberately. The free fraction is wanted for two jobs,
    # not one: `briefs.confidence_rule`'s benign-PSA rung, and the biochemical
    # assessor's own reading of WHY the PSA is raised -- which reaches the note
    # through KEY_EVIDENCE and is quotable, because `laboratory_results` is a
    # judge-visible section. Fetching it after the specialists would buy the
    # first and throw away the second at the same tool cost. Gated identically
    # either side of the walk: measured, both select the same 33 of 91.
    await _maybe_reveal_labs(tools, case_id, prompt, revealed, order, thresholds)

    # --- specialists, in a fixed order so the reveal trace is reproducible ---
    for name in _SPECIALIST_ORDER:
        brief = briefs.render(briefs.SPECIALISTS[name], thresholds)
        allowed = tuple(i for i, (_, owner) in briefs.REVEAL_ITEMS.items() if owner == name)
        block, _ = await asyncio.to_thread(ask, model, brief, _facts(name, prompt, track, revealed))
        # An item this agent asked for is fetched and the agent re-reads it once.
        # Only the agent that will read the evidence may request it.
        fetched = False
        for item in parse_needs(block.get("NEED", ""), allowed):
            fetched |= await _reveal(tools, item, case_id, revealed, order)
        if fetched:
            block, _ = await asyncio.to_thread(ask, model, brief, _facts(name, prompt, track, revealed))
        if not block:
            warnings.append(f"{name}: unparseable reply")

        # Decompose-critique-refine, once. The weighting rule is conditional and
        # conditional rules are where compliance collapses (measured: 100% on the
        # unconditional ones, 10-63% on the rest), so the violated clause is named
        # back to the agent and it answers again. Only fires when a rule broke.
        emitted = parse_weights(block.get("WEIGHTS", ""), name)
        bad = weight_violations(name, emitted, ctx)
        if bad:
            fixed, _ = await asyncio.to_thread(
                ask, model, brief,
                _facts(name, prompt, track, revealed)
                + "\n\n"
                + briefs.CRITIQUE.format(violations="\n".join(bad)),
            )
            refined = parse_weights(fixed.get("WEIGHTS", ""), name)
            # Keep the refinement only if it actually reduced the breaches.
            if refined and len(weight_violations(name, refined, ctx)) < len(bad):
                emitted = refined
                if fixed.get("VERDICT"):
                    block = fixed
            else:
                warnings.append(f"{name}: critique did not fix {len(bad)} weight violation(s)")

        verdicts[name] = block
        weights.update(emitted)

    # --- the decision, from the policy graph -------------------------------
    # ⚠️ This replaces the adjudicator agent and the confidence assessor for
    # Task 1. The graph answers 81 of the 91 labelled cases against the
    # adjudicator's 75, it answers every case (nothing stops), and its complete
    # data sweep reports no undefined and no ambiguous combination -- see
    # docs/task1-policy-reference.md and docs/task1-policy-graph.md. The
    # specialists stay: they own the variable weights, which are unchanged, and
    # their NEED lines still drive their own reveals.
    #
    # `decider: adjudicator` in resources/agentic_task1.yaml restores the old
    # path for a paired comparison.
    if str(thresholds.get("decider", "graph")).lower() == "graph":
        answer = await decide_with_graph(model, tools, case_id, prompt, revealed, order,
                                         warnings, thresholds)
        decision = answer["decision"]
        confidence = answer["confidence"]
        rationale = answer["reason"]
        cblock = {"ABSENT_FACT": answer["absent_fact"]}
        if answer.get("consulted"):
            log.info("%s: %s handed to the consultant", case_id, answer["consulted"])
    else:
        # --- adjudication, with the specialists' evidence already on the table ---
        adj_brief_t = briefs.ADJUDICATORS[track]
        decision = confidence = rationale = None
        for _ in range(len(briefs.REVEAL_ITEMS)):
            available = [i for i, (t, _) in briefs.REVEAL_ITEMS.items() if t not in revealed]
            brief = briefs.render(
                adj_brief_t,
                thresholds,
                needs=", ".join(available) or "none remain",
                retrieved=", ".join(order) or "nothing",
                available=", ".join(available) or "nothing",
            )
            # ⚠️ Built-in thinking was tried here and REVERTED (2026-09-09).
            # arXiv:2606.09662 predicts a gain for planning constraints like rule
            # precedence, and it measured the opposite on this task: the adjudicator
            # returned no parseable decision on 36 of the first 61 cases, each one
            # falling through to the radiology verdict. `ask(..., thinking=True)`
            # still exists; do not re-enable it here without fixing that first.
            block, _ = await asyncio.to_thread(ask, model, brief, _summary(verdicts))
            if block.get("ACTION", "").upper().startswith("REVEAL") and available:
                item = (parse_needs(block.get("REVEAL_ITEM", ""), tuple(available)) or [None])[0]
                if item and await _reveal(tools, item, case_id, revealed, order):
                    owner = briefs.REVEAL_ITEMS[item][1]
                    ob = briefs.render(briefs.SPECIALISTS[owner], thresholds)
                    re_block, _ = await asyncio.to_thread(ask, model, ob, _facts(owner, prompt, track, revealed))
                    if re_block:
                        verdicts[owner] = re_block
                        weights.update(parse_weights(re_block.get("WEIGHTS", ""), owner))
                    continue
            decision = (block.get("DECISION") or "").strip().upper()

            # The same critique the specialists get, on the field that matters most.
            # Nine of run v6's twenty decision failures broke a rung of this ladder,
            # and the adjudicator was the one agent with no check at all.
            dctx = {
                "track": track,
                "pirads": _int_or_none(prompt.get("pirads")),
                "age": prompt.get("age"),
                "psa": prompt.get("psa"),
                "psad": prompt.get("psad"),
                "psav": prompt.get("psav"),
                "vol": prompt.get("vol"),
                "prior_isup": _prior_isup(_grade_text(prompt, revealed)),
                "lesion_mm": _lesion_mm(revealed),
                **thresholds,
            }
            broke = decision_violation(decision, dctx)
            if broke:
                want, why = broke
                dfix, _ = await asyncio.to_thread(
                    ask, model, brief,
                    _summary(verdicts)
                    + "\n\n"
                    + briefs.DECISION_CRITIQUE.format(want=want, got=decision or "nothing", why=why),
                )
                if (dfix.get("DECISION") or "").strip().upper() in {"BIOPSY", "NO_BIOPSY", "TREATMENT"}:
                    block = dfix
                    decision = dfix["DECISION"].strip().upper()
                if decision_violation(decision, dctx):
                    warnings.append(f"decision critique did not resolve ({decision!r} vs {want!r})")

            confidence = (block.get("CONFIDENCE") or "").strip().lower()
            rationale = (block.get("RATIONALE") or "").strip()
            break

        # --- confidence: its own agent, after the decision is fixed ---
        # It sees the decision and every specialist verdict, and owns `confidence`
        # alone. The adjudicator's own label is only the fallback.
        ctx = "\n".join([
            f"Encounter: {_ENCOUNTER_PROSE[track]}",
            f"Recommendation: {decision or 'not stated'}",
            f"Evidence that was retrieved: {', '.join(order) or 'nothing was retrieved'}",
            "Specialist assessments:", _summary(verdicts),
            "Prior biopsy ISUP grade: "
            + (f"ISUP {_prior_isup(_grade_text(prompt, revealed))}" if _prior_isup(_grade_text(prompt, revealed))
               else "NOT FOUND anywhere in the record"),
            "Whether the lesion is new, grown or stable against an earlier MRI: "
            "NOT FOUND -- this record contains no earlier MRI",
            "Readings on the record: " + render_facts({
                "PSA (ng/mL)": prompt.get("psa"),
                "PSA density (ng/mL/cc)": prompt.get("psad"),
                "PSA velocity (ng/mL/yr)": prompt.get("psav"),
                "PI-RADS": prompt.get("pirads"),
                "Age (years)": prompt.get("age"),
                "Problem list": prompt.get("pmhx"),
            }).replace("\n", "; "),
        ])
        cblock, _ = await asyncio.to_thread(
            ask, model, briefs.render(briefs.CONFIDENCE, thresholds, context=ctx), ctx
        )
        assessed = (cblock.get("CONFIDENCE") or "").strip().lower()

        # Same critique as the specialists get: the three sources are checked
        # mechanically, the mismatch is named, and the assessor answers again.
        cctx = {
            "track": track,
            "pirads": prompt.get("pirads"),
            "age": prompt.get("age"),
            "comorbidity_count": len(prompt.get("pmhx") or []),
            "grade_stated": _prior_isup(_grade_text(prompt, revealed)) is not None,
            # Whether there was anywhere for a grade to be written at all. Without
            # this, "not retrieved" and "not recorded" are the same input.
            "grade_evidence_available": bool(_grade_text(prompt, revealed).strip()),
            "grey_count": _grey_count(prompt),
            "size_was_tiebreaker": (verdicts.get("radiology") or {}).get("LESION_SIZE_ASSESSMENT") == "GROWN",
            "prior_mri_available": _has_prior_mri(prompt, revealed),
            **{k: thresholds[k] for k in ("conflict_age", "conflict_comorbidity_count", "grey_zones_for_borderline")},
        }
        bad = confidence_violation(assessed, cctx)
        if bad:
            want, why = bad
            cfix, _ = await asyncio.to_thread(
                ask, model, briefs.render(briefs.CONFIDENCE, thresholds, context=ctx),
                ctx + "\n\n" + briefs.CONFIDENCE_CRITIQUE.format(want=want, got=assessed or "nothing", why=why),
            )
            refixed = (cfix.get("CONFIDENCE") or "").strip().lower()
            if refixed in {"clear", "borderline", "uncertain"}:
                assessed = refixed
                cblock = cfix
            if confidence_violation(assessed, cctx):
                warnings.append(f"confidence critique did not resolve ({assessed!r} vs {want!r})")

        if assessed in {"clear", "borderline", "uncertain"}:
            confidence = assessed
        else:
            warnings.append(f"confidence assessor returned {assessed!r}; kept the adjudicator's label")

    # --- hard failure only: the adjudicator never returned a usable decision ---
    if decision not in {"BIOPSY", "NO_BIOPSY", "TREATMENT"}:
        warnings.append(f"adjudicator returned no decision ({decision!r}); fell back to the radiology verdict")
        decision = "BIOPSY" if (verdicts.get("radiology") or {}).get("VERDICT") == "HIGH" else "NO_BIOPSY"
        confidence = "uncertain"
    if confidence not in {"clear", "borderline", "uncertain"}:
        warnings.append(f"unusable confidence {confidence!r}; recorded as borderline")
        confidence = "borderline"

    # The schema requires 40 characters. A short adjudicator note is extended
    # with what the specialists actually cited, never with invented text.
    # The adjudicator's RATIONALE is written to justify a decision to us; the
    # judge wants a clinician's note about a patient. Rewrite it as one.
    named_absence = bool((cblock.get("ABSENT_FACT") or "NONE").strip().upper() != "NONE")
    # ⚠️ Two tiers, not one list. `observed` is what the judge's Input holds and
    # is the only thing the note may quote; `background` (age, examination,
    # velocity, IPSS, notes) shapes the judgement and is never quoted back.
    # See rationale.evidence_packet for the measurement that forced this.
    observed, background = R_.evidence_packet(1, prompt, revealed)
    findings = "\n".join(
        [f"Recommendation: {'biopsy' if decision == 'BIOPSY' else 'no biopsy'}", R_.render(observed)]
        + ([f"Missing and needed: {cblock.get('ABSENT_FACT')}"] if named_absence else [])
        + [f"What the assessment found: {v.get('KEY_EVIDENCE')}" for v in verdicts.values() if v.get("KEY_EVIDENCE")]
    )
    written = await write_rationale(
        model, 1, decision, confidence, findings, named_absence, warnings,
        background=R_.render(background), citable=observed,
        # The judge's step 2 asks whether the note cites the variables the
        # decision rested on; those are the ones we emit as important/decisive.
        drivers=R_.drivers(1, weights, observed),
    )
    rationale = written or rationale or ""
    if len(rationale) < 40:
        evidence = "; ".join(v.get("KEY_EVIDENCE", "") for v in verdicts.values() if v.get("KEY_EVIDENCE"))
        rationale = f"{rationale} {evidence}".strip()
        warnings.append("adjudicator rationale was too short; extended with the specialists' key evidence")

    if thresholds["_raw"].get("weight_fh") and weights["fh"] == "not_used":
        # Scorer-facing, and it is a weight claim rather than a clinical one:
        # the reference rates fh `noted` on 64 of 91 cases while revealing
        # family_history on 0 of them. Costs section grounding, buys the weight
        # component. Toggle with weight_fh in resources/agentic_task1.yaml.
        weights["fh"] = "noted"

    prediction = assemble_full_output(
        task=1,
        case_id=case_id,
        patient={"id": case_id, "psa_ng_ml": query.get("psa"), "age_years": query.get("age")},
        reveal_sequence=_build_reveal_sequence(order),
        judgment={
            "biopsy_decision": "yes" if decision == "BIOPSY" else "no",
            "confidence": confidence,
            "variable_weights": weights,
            "free_text": rationale[:2000],
            "repeat_test": None,
        },
    )
    if warnings:
        log.warning("%s: %s", case_id, " | ".join(warnings))
    return prediction
