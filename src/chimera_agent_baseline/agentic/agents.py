"""Calling an agent, and parsing what it says back.

Flat ``KEY: value`` blocks, not JSON: a mid-size local model produces them far
more reliably, and a malformed line costs one field instead of the whole reply.
Nothing here raises on bad model output -- an unparseable field falls back and
is reported in ``warnings`` for the caller to log.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .briefs import WEIGHT_OWNERS, confidence_rule, decision_rule, weight_rules

log = logging.getLogger(__name__)

WEIGHT_LEVELS = frozenset({"not_used", "noted", "important", "decisive"})
ALL_VARIABLES: tuple[str, ...] = tuple(v for vals in WEIGHT_OWNERS.values() for v in vals)

_LINE = re.compile(r"^\s*([A-Z][A-Z_]{2,})\s*:\s*(.*)$")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_block(text: str) -> dict[str, str]:
    """Parse a flat ``KEY: value`` block.

    Later keys win over earlier ones, so a model that restates its answer after
    reasoning aloud is read correctly. Continuation lines are appended to the
    key they follow, which is what keeps a wrapped RATIONALE intact.
    """
    fields: dict[str, str] = {}
    last: str | None = None
    for raw in _THINK.sub("", text or "").splitlines():
        match = _LINE.match(raw)
        if match:
            last = match.group(1)
            fields[last] = match.group(2).strip()
        elif last and raw.strip():
            fields[last] = f"{fields[last]} {raw.strip()}".strip()
    return fields


def parse_weights(value: str, owner: str, owners: dict[str, tuple[str, ...]] | None = None) -> dict[str, str]:
    """Read ``pirads=decisive; vol=noted`` into a dict, owned variables only.

    A variable the agent does not own, or a level outside the ordinal scale, is
    dropped rather than guessed at: the caller's default of ``not_used`` is the
    honest reading of an agent that failed to rate its own variable.
    """
    owned = set((owners or WEIGHT_OWNERS)[owner])
    out: dict[str, str] = {}
    for part in re.split(r"[;,\n]", value or ""):
        if "=" not in part:
            continue
        name, _, level = part.partition("=")
        name, level = name.strip().lower(), level.strip().lower()
        if name in owned and level in WEIGHT_LEVELS:
            out[name] = level
    return out


def parse_needs(value: str, allowed: tuple[str, ...]) -> list[str]:
    """Read the NEED line. Anything not on the allowed list is ignored."""
    if not value:
        return []
    tokens = [t.strip().upper().replace(" ", "_") for t in re.split(r"[;,]", value)]
    return [t for t in dict.fromkeys(tokens) if t in allowed]


def ask(model, brief: str, facts: str, *, thinking: bool = False, max_tokens: int | None = None):
    """One agent turn. Returns the parsed block and the raw reply.

    ``thinking`` toggles Qwen's built-in chain of thought for this call only.
    Enable it for constraints about ordering and precedence, which measurably
    improve under it, and leave it off for constraints about exact output form,
    which measurably worsen (arXiv:2606.09662, tested on Qwen3 1.7B-32B). The
    budget must rise with it: the thoughts are billed against ``max_tokens``, so
    a reply can otherwise come back empty.
    """
    client = model
    if thinking or max_tokens:
        kwargs: dict[str, Any] = {}
        if thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        try:
            client = model.bind(**kwargs)
        except Exception:  # a backend that cannot take per-call kwargs
            log.warning("per-call kwargs unsupported; falling back to the default model settings")
            client = model
    reply = client.invoke([SystemMessage(content=brief), HumanMessage(content=facts)])
    raw = reply.content if isinstance(reply.content, str) else str(reply.content)
    if not raw.strip():
        # Some servers return the chain of thought in its own field and leave
        # `content` empty when the budget ran out mid-thought.
        raw = str(getattr(reply, "additional_kwargs", {}).get("reasoning_content") or "")
    return parse_block(raw), raw


def render_facts(fields: dict[str, Any]) -> str:
    """Render the case facts an agent is allowed to see.

    A ``None`` value is written as 'not recorded' rather than omitted: an agent
    that cannot see a variable must know it is absent, not infer it is normal.
    """
    lines = []
    for label, value in fields.items():
        if value is None or value == "" or value == []:
            lines.append(f"{label}: not recorded")
        else:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def weight_violations(agent: str, emitted: dict[str, str], ctx: dict[str, Any]) -> list[str]:
    """Which of this agent's weights break the rule it was given.

    A rule-based critique, in DeCRIM's sense: the check is mechanical, but the
    correction is asked of the agent rather than written for it. An absent
    variable counts as a violation -- the parser dropped it, so the agent did
    not actually answer for it.
    """
    out = []
    for var, want in weight_rules(agent, ctx).items():
        got = emitted.get(var)
        if got != want:
            out.append(f"  - `{var}` should be `{want}` on this case; you gave `{got or 'nothing'}`.")
    return out


_WHY = {
    "uncertain": "A named fact is absent and its two values give two different actions.",
    "borderline": "Either two forces of comparable weight disagree, or the readings never "
    "left their grey bands, so the recommendation stands as a shared choice.",
    "clear": "Nothing is absent that would change the direction, nothing of comparable "
    "weight points the other way, and the deciding reading is past a line.",
}


def confidence_violation(got: str, ctx: dict[str, Any], rule=None) -> tuple[str, str] | None:
    """The label the rules give, when it differs from the one emitted."""
    want = (rule or confidence_rule)(ctx)
    return None if got == want else (want, _WHY[want])


def decision_violation(got: str, ctx: dict[str, Any], rule=None) -> tuple[str, str] | None:
    """The recommendation the track's rules give, when it differs from the one made.

    Silent in two cases: where the rules do not settle it at all, and where the
    rung that settles it is one the agent reads better than the rules do. The
    agent outscores these rules overall, so the critique is a nudge on the rungs
    it demonstrably gets wrong, not a general override -- see `enforced_rungs`.
    """
    rule = (rule or decision_rule)(ctx)
    if rule is None:
        return None
    want, why, rung = rule
    if rung not in (ctx.get("enforced_rungs") or ()):
        return None
    return None if got == want else (want, why)
