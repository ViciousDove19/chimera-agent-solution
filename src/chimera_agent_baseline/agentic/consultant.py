"""The consultant a graph node hands a case to when the structure runs out.

⚠️ WHY THIS EXISTS, and why it is not more branches. `t3_reclassification`'s
population -- prior cancer, PI-RADS >= 4, PSA 4-20, the earlier ISUP grade not
retrievable -- is the residue `docs/task1-confidence.md` records as having no
structured variable that separates its confidence labels: n=18, splitting
7 `clear` / 5 `borderline` / 6 `uncertain`, with its own worked pair sitting in
the same cell of every table. Four structural designs were measured flat across
it (the stability radius, derived dissenters, path shape, and a blanket
`uncertain` that scored 5 right of 15 on a stratum splitting 5/6/4). The
separating information is in the prose, so the prose is what reads it.

The consultant owns the DECISION and the CONFIDENCE for the cases it takes, and
nothing else. Variable weights are untouched, and every other case is decided by
the graph.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from . import policy_graph as P

log = logging.getLogger(__name__)

DECISIONS = {"BIOPSY": "yes", "NO_BIOPSY": "no"}
CONFIDENCES = ("clear", "borderline", "uncertain")

#: The clinical reasoning the consultant is given. Every line is the frozen
#: policy's own, quoted from `task1.py::tissue` and `signal`, where it was
#: earned case by case -- this brief does not invent a rule.
BRIEF = """
You are the urologist this case has been referred to. A colleague has taken it
as far as the record allows and cannot take it further.

The situation: this man has prostate cancer on an earlier biopsy, and **the
grade of that biopsy cannot be retrieved from anything in front of us**. Nothing
cheaper can be done to obtain it -- the old pathology is either in the notes or
it is not. So the question is whether new tissue is what this encounter needs.

What decides it, and why:

1. **Corroboration routes the question; it does not answer it.** If the imaging
   is suspicious AND at least two of {PSA at or above 15, PSA density above
   0.10, a suspicious examination} are past their lines, the lesion is real --
   and then the only open question is what grade he is NOW, which only tissue
   closes. With fewer than two, the case is still diagnostic rather than a
   grading question, and the imaging is what answers it.
2. **The density can be the reassurance, but only when nothing corroborates the
   lesion.** A man known to carry cancer, whose grade is not on record, whose
   lesion nothing else corroborates and whose density is genuinely low, is a
   surveillance question and not a repeat-biopsy one.
3. **A diluted density must not veto the case.** A gland over 70 mL with a PSA
   over 8 makes the density uninterpretable -- you cannot say whether the PSA
   came from the tumour or from the bulk -- so it does not get to suppress the
   grade question. With a low PSA there is nothing to attribute and a low
   density really is reassuring.
4. **Age overrides the value of a REPEAT biopsy, never the need for a first
   one.** Past 75 he has been sampled once already and a change of grade can no
   longer change what is done for him.

Now read the notes for what the structured fields cannot tell you, and say what
you find:

* does the record actually STATE the earlier grade anywhere, in any vocabulary?
* is an earlier MRI described, and does it say this lesion is new, grown or
  stable?
* how many previous biopsy sessions were there, and how long ago?
* is his fitness, life expectancy or his own preference what is really in play?

CONFIDENCE means this, and nothing looser:
  clear       the evidence on the table settles this recommendation.
  borderline  the recommendation stands, but a reasonable clinician could take
              the other path -- a genuinely shared choice.
  uncertain   a SPECIFIC named fact is absent and its two possible values would
              give two different actions. Name that fact. "The record is
              incomplete" is not a reason; the missing grade only makes this
              uncertain if knowing it would change what you do.

Answer in EXACTLY this format. Flat lines, one per key, nothing before or after.

DECISION: BIOPSY | NO_BIOPSY
CONFIDENCE: clear | borderline | uncertain
GRADE_ON_RECORD: <the grade the notes state, in the record's own words, or NONE>
PRIOR_MRI: <what an earlier MRI said about this lesion, or NONE>
ABSENT_FACT: <the fact whose absence you called uncertain, or NONE>
RATIONALE: <two sentences: what settled it, and what would change it>
""".strip()

_LINE = re.compile(r"^\s*([A-Z][A-Z_]{2,})\s*:\s*(.*)$")


def facts_block(graph: P.Graph, node_id: str, facts: dict[str, Any],
                notes: str = "") -> str:
    """The readings this node had, plus the prose the consultant is here to read."""
    node = graph.nodes[node_id]
    lines = []
    for name in node.reads:
        value = facts.get(name)
        lines.append(f"{name}: {'not recorded' if value is None else value}")
    if notes.strip():
        lines.append("\nClinical notes and retrieved documents:\n" + notes.strip())
    return "\n".join(lines)


def parse(reply: str) -> dict[str, str]:
    out: dict[str, str] = {}
    last: str | None = None
    for raw in (reply or "").splitlines():
        m = _LINE.match(raw)
        if m:
            last = m.group(1)
            out[last] = m.group(2).strip()
        elif last and raw.strip():
            out[last] = f"{out[last]} {raw.strip()}".strip()
    return out


def resolve(graph: P.Graph, node_id: str, facts: dict[str, Any],
            block: dict[str, str] | None) -> tuple[str, str, str, bool]:
    """The consultant's answer, or the deterministic fallback.

    Returns ``(decision, confidence, reason, fell_back)``. The fallback is the
    node's own edges -- what the offline audit scores -- so a model that returns
    nothing usable costs the case its consultant reading and nothing else.
    """
    block = block or {}
    decision = DECISIONS.get((block.get("DECISION") or "").strip().upper())
    conf = (block.get("CONFIDENCE") or "").strip().lower()
    if decision and conf in CONFIDENCES:
        return decision, conf, (block.get("RATIONALE") or "").strip(), False

    walked = P.walk(graph, P.derive(graph, facts))
    fallback = graph.leaves[walked.leaf]["decision"] if walked.leaf else "yes"
    log.warning("%s: consultant returned %r/%r; fell back to the node's own edges",
                node_id, block.get("DECISION"), block.get("CONFIDENCE"))
    return (fallback, "uncertain",
            "the consultant could not be read; this is the deterministic branch", True)
