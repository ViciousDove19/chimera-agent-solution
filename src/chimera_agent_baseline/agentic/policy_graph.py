"""The Task 1 decision policy as a walkable graph.

`resources/task1_graph.yaml` is a transcription of `docs/updated_policy.md`:
one node per clinical question, edges as guard expressions over named
variables, leaves as recommendations. The point of having it as an object
rather than as prose is that it can be enumerated -- so "the policy answers
every situation" becomes something a script checks instead of something a
document claims.

⚠️ COMPLETE DATA ONLY, by design. No node here carries a missing-evidence
edge, so `walk` on a record with an absent variable stops at the node that
needed it and says so. That is the current finding surface, not a bug.

`walk` never raises on a gap. A node that cannot be answered is returned as
data, because the audit's whole job is to count those.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "resources" / "task1_graph.yaml"

#: Guard expressions are policy text, not user input, but they are still eval'd.
#: Only these names are reachable from one.
def _count(*flags: Any) -> int:
    """How many of these readings are concerning. The one helper a guard gets.

    It exists so a node can say "two of these four" without the graph having to
    carry a derived variable that the sweep would then vary independently of
    the readings it is derived from.
    """
    return sum(1 for f in flags if f)


def _excess(*pairs: tuple[Any, float]) -> float:
    """How far past its own line each reading sits, summed.

    The measure of a disagreement's STRENGTH rather than its existence: a PSA
    of 21 against a line of 10 contributes 1.1, a density that has just crossed
    0.15 contributes almost nothing.
    """
    return round(sum(v / line - 1 for v, line in pairs if v is not None and v > line), 3)


_SAFE_BUILTINS: dict[str, Any] = {"True": True, "False": False, "None": None}
#: Callable in a guard or a derivation, and not variables.
def _usable(psad: Any, psa: Any, vol: Any, tolerance: float = 0.5) -> Any:
    """The density, or None where the record's own numbers contradict it.

    Input validation, not a clinical rule, ported from `task1.py::usable_density`:
    the record states the density twice, directly and implicitly as PSA over
    volume, and where the two disagree past the tolerance the quotient is not a
    measurement of this patient. No branch gets to read it -- neither to raise
    suspicion nor to lower it.
    """
    if psad is None or psa is None or not vol:
        return psad
    implied = psa / vol
    return psad if abs(implied - psad) <= tolerance * implied else None


_FUNCTIONS: dict[str, Any] = {"count": _count, "excess": _excess, "usable": _usable}

_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_STRING = re.compile(r"'[^']*'|\"[^\"]*\"")
_KEYWORDS = frozenset({"and", "or", "not", "in", "is", "if", "else"})


def _names(expr: str) -> set[str]:
    """The variables a guard reads. String literals are not variables."""
    bare = _STRING.sub(" ", expr)
    return {n for n in _IDENT.findall(bare)
            if n not in _KEYWORDS and n not in _SAFE_BUILTINS and n not in _FUNCTIONS}


@dataclass(frozen=True)
class Edge:
    when: str
    to: str


@dataclass(frozen=True)
class Node:
    id: str
    track: str
    asks: str
    reads: tuple[str, ...]
    source: str
    edges: tuple[Edge, ...]
    conflict: str | None = None
    un_evaluable: str | None = None
    #: What this node's guideline input would have been, where that input
    #: cannot exist in this release and an obtainable reading stands in for it.
    substitutes_for: str | None = None
    #: Why the node is written the way it is -- the amendment note, kept beside
    #: the rule rather than only in the document it came from.
    resolved: str | None = None
    #: Set when a node can only be entered because a variable is ABSENT. The
    #: complete-data sweep never produces that, so it would otherwise report the
    #: node as unreachable when it is merely out of that sweep's scope.
    reachable_only_when: str | None = None
    reads_note: str | None = None
    dead_edge_note: str | None = None
    #: The EAU rule ids (docs/eau_biopsy_policy_2026.md) that license this node.
    #: Empty means nothing in the guideline licenses it -- a finding, not a gap
    #: in the annotation.
    eau: tuple[str, ...] = ()
    #: How this node contributes to confidence: what it decides on (never a
    #: source of doubt), which of its edges are equivocal by construction, the
    #: absent fact it turns on, and the conflicts that make its answer a shared
    #: choice. See `confidence`.
    conf: dict[str, Any] = field(default_factory=dict)
    #: Set where this node hands off to a consultant rather than deciding from
    #: its own edges. The edges remain, as the deterministic fallback.
    consultant: dict[str, Any] = field(default_factory=dict)
    also_stated: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Graph:
    entry: str
    nodes: dict[str, Node]
    leaves: dict[str, dict[str, str]]
    variables: dict[str, dict[str, Any]]
    #: What to do when a variable is absent, declared once per VARIABLE rather
    #: than per node -- see `missing_policy` in the YAML and `resolve_missing`.
    missing: dict[str, dict[str, Any]]
    source: str

    def is_leaf(self, name: str) -> bool:
        return name in self.leaves


def load(path: Path | str | None = None) -> Graph:
    raw = yaml.safe_load(Path(path or DEFAULT_PATH).read_text())
    nodes = {}
    for entry in raw["nodes"]:
        nodes[entry["id"]] = Node(
            id=entry["id"],
            track=entry.get("track", "ALL"),
            asks=entry.get("asks", ""),
            reads=tuple(entry.get("reads") or ()),
            source=entry.get("source", ""),
            edges=tuple(Edge(e["when"], e["to"]) for e in entry.get("edges") or ()),
            conflict=entry.get("conflict"),
            un_evaluable=entry.get("un_evaluable"),
            substitutes_for=entry.get("substitutes_for"),
            resolved=entry.get("resolved"),
            reachable_only_when=entry.get("reachable_only_when"),
            reads_note=entry.get("reads_note"),
            dead_edge_note=entry.get("dead_edge_note"),
            eau=tuple(entry.get("eau") or ()),
            conf=dict(entry.get("confidence") or {}),
            consultant=dict(entry.get("consultant") or {}),
            also_stated=dict(entry.get("also_stated") or {}),
        )
    graph = Graph(
        entry=raw["entry"],
        nodes=nodes,
        leaves=dict(raw["leaves"]),
        variables=dict(raw["variables"]),
        missing=dict(raw.get("missing_policy") or {}),
        source=raw.get("source", ""),
    )
    _validate(graph)
    _ASKS.update({n.id: n.asks for n in nodes.values()})
    return graph


def derive(graph: Graph, facts: dict[str, Any]) -> dict[str, Any]:
    """Fill in every derived variable from the primitives it is computed from.

    ⚠️ This is what keeps the audit sound. A derived quantity enumerated as if
    it were independent invents combinations that cannot occur -- two
    biochemical channels past their lines while both the density and the
    velocity sit below them -- and those phantoms show up as undefined or
    ambiguous findings against a policy that is neither.
    """
    out = dict(facts)
    for name, spec in graph.variables.items():
        if spec.get("kind") != "derived" or out.get(name) is not None:
            continue
        if any(out.get(n) is None for n in spec.get("needs", ())):
            continue
        try:
            out[name] = eval(spec["derived"], {"__builtins__": {}}, {**_FUNCTIONS, **out})  # noqa: S307
        except Exception:
            out[name] = None
    return out


def _validate(graph: Graph) -> None:
    """Every edge points somewhere, and every guard reads a declared variable."""
    known = set(graph.nodes) | set(graph.leaves)
    for node in graph.nodes.values():
        for edge in node.edges:
            if edge.to not in known:
                raise ValueError(f"{node.id}: edge points at unknown target {edge.to!r}")
        for name in _names(" ".join(e.when for e in node.edges)):
            if name not in graph.variables:
                raise ValueError(f"{node.id}: guard reads undeclared variable {name!r}")
    if graph.entry not in graph.nodes:
        raise ValueError(f"entry {graph.entry!r} is not a node")
    # ⚠️ THE SCALABILITY GUARANTEE. A variable with no declared answer for its
    # own absence is a hole waiting for a record that omits it, so the graph
    # refuses to load rather than discovering it at inference time.
    undeclared = sorted(
        name for name, spec in graph.variables.items()
        if spec.get("kind") != "derived" and name not in graph.missing
    )
    if undeclared:
        raise ValueError(f"no missing_policy for {', '.join(undeclared)}")


def resolve_missing(graph: Graph, name: str, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the policy does when `name` is absent. Declared once, per variable.

    Three classes, and the difference between them is clinical rather than
    technical -- it is about whether the fact can still be obtained, and at what
    cost relative to the action being contemplated:

    ``retrieve``     a tool serves it, so fetch it and walk again.
    ``order_first``  it is a cheaper and safer test than the action under
                     consideration, so the answer is to obtain it rather than to
                     act without it -- an MRI before a biopsy.
    ``historical``   it existed once and is gone, so nothing can produce it. The
                     policy sweeps the values the record could plausibly have
                     carried; where they agree the absence changes nothing, and
                     where they diverge the case takes the declared branch and
                     is `uncertain`, naming the fact.
    """
    spec = graph.variables.get(name) or {}
    if spec.get("kind") == "derived":
        # ⚠️ A derived reading is absent only because a primitive it is computed
        # from is absent, and it is that primitive's absence the policy has an
        # answer for. `suspicion_likely` has no declaration and needs none;
        # the gland volume it could not read does. Without this the three
        # derived readings on Task 1 swallowed 58 single-omission walks.
        for need in spec.get("needs") or ():
            if facts is None or facts.get(need) is None:
                found = resolve_missing(graph, need, facts)
                if found:
                    return {**found, "via": name}
        return {}
    return dict(graph.missing.get(name) or {})


class Unanswerable(Exception):
    """A guard could not be evaluated -- a variable it reads is absent."""


#: A guard may ASK whether a variable is absent -- `pirads is None`. That is a
#: question about the record, not a reading of a value, so the absence must not
#: make the guard unanswerable. Any name tested this way is exempt.
#: A guard may ask about a variable's STATE -- `pirads is None`, `fh is False`,
#: `fh is not False`. That is a question about the record, not a reading of a
#: value, so an absent variable must not make the guard unanswerable.
_ASKS_ABSENT = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\s+is\s+(?:not\s+)?(?:None|True|False)\b")
#: Only the positive absence test -- `x is None`, never `x is not None`.
_ASKS_ABSENT_ONLY = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\s+is\s+None")


def _test(expr: str, facts: dict[str, Any]) -> bool:
    exempt = set(_ASKS_ABSENT.findall(expr))
    for name in _names(expr) - exempt:
        if facts.get(name) is None:
            raise Unanswerable(name)
    # A name the caller never supplied is ABSENT, not an error. Only a state
    # test can reach here for such a name -- everything else was refused above.
    scope = {n: None for n in _names(expr)}
    return bool(eval(expr, {"__builtins__": {}}, {**scope, **_FUNCTIONS, **facts}))  # noqa: S307


@dataclass(frozen=True)
class Walk:
    leaf: str | None
    path: tuple[str, ...]
    #: Why it stopped, when it did not reach a leaf.
    stopped_at: str | None = None
    reason: str | None = None
    #: Nodes where more than one guard was true. The first still won.
    ambiguous: tuple[str, ...] = ()
    #: A node on this path that hands off to a consultant. `leaf` still carries
    #: the deterministic fallback, so an offline walk stays scoreable.
    consult: str | None = None
    #: Variables whose absence the walk had to resolve through `missing_policy`
    #: rather than read. An answer that rests on one of these is never `clear`.
    resolved_absent: tuple[str, ...] = ()


def walk(graph: Graph, facts: dict[str, Any], *, max_steps: int = 64) -> Walk:
    """Follow the graph from its entry. Never raises on an unanswered node."""
    path: list[str] = []
    ambiguous: list[str] = []
    consult: str | None = None
    absent_seen: list[str] = []
    current = graph.entry
    for _ in range(max_steps):
        if graph.is_leaf(current):
            return Walk(current, tuple(path), ambiguous=tuple(ambiguous), consult=consult,
                        resolved_absent=tuple(dict.fromkeys(absent_seen)))
        node = graph.nodes[current]
        path.append(node.id)
        if node.consultant:
            consult = consult or node.id
        # ⚠️ First match wins, and an unanswerable LATER edge does not discard
        # an earlier one that already matched. `_test` refuses any guard naming
        # an absent variable, so evaluating the whole list first would report a
        # case as "missing" when the policy had in fact already answered it --
        # which it did on a labelled case whose density settles the node before
        # the edge reading an unrecorded volume stability is ever reached.
        matched = []
        absent = None
        for edge in node.edges:
            try:
                if _test(edge.when, facts):
                    matched.append(edge)
                    if len(node.edges) == 1:
                        break
            except Unanswerable as missing:
                if matched:
                    break
                absent = absent or missing.args[0]
        if not matched:
            if absent:
                # ⚠️ The declared answer for THIS variable's absence, from
                # `missing_policy` -- per variable, never per node, so a new
                # variable cannot be added without one and `load` refuses a
                # graph that leaves one undeclared.
                policy = resolve_missing(graph, absent, facts)
                target = policy.get("then")
                if target in graph.leaves or target in graph.nodes:
                    absent_seen.append(absent)
                    current = target
                    continue
            reason = f"missing:{absent}" if absent else "undefined"
            return Walk(None, tuple(path), node.id, reason, tuple(ambiguous), consult,
                        tuple(absent_seen))
        if len({e.to for e in matched}) > 1:
            ambiguous.append(node.id)
        current = matched[0].to
    return Walk(None, tuple(path), current, "cycle", tuple(ambiguous), consult,
                tuple(absent_seen))


# ---------------------------------------------------------------------------
# Confidence: two channels, because the reference's two non-`clear` labels mean
# different things.
#
# `docs/task1-confidence.md` records that all 15 reference `uncertain`
# rationales name a SPECIFIC ABSENT FACT -- the prior grade on 13, a prior MRI
# on 12 -- and that `decision_boundary_fragile` was falsified as a single cap
# because "for an ordinal driver, an adjacent-value decision change is the
# guideline working, not an unstable record". Both findings are honoured here:
# absence and margin are measured separately, and margin is measured in units of
# each variable's own MEASUREMENT UNCERTAINTY, under which an ordinal step is
# cheap and a five-unit PSA move is not.
# ---------------------------------------------------------------------------


#: Adverse morphology, and the language that positively EXCLUDES it. EAU
#: `P2_exclude_cribriform` is a Strong recommendation to keep such patients off
#: surveillance, so the surveillance rung needs it shown absent -- which means
#: telling "not mentioned" apart from "explicitly absent". It is stated on 147
#: of the 153 released Task 2 pathology reports.
_ADVERSE_MORPHOLOGY = re.compile(r"cribriform|intraductal", re.I)
_MORPHOLOGY_ABSENT = re.compile(
    r"(?:no|without|absent|negative for|not identified|not seen|none)\b[^.]{0,60}"
    r"(?:cribriform|intraductal)"
    r"|(?:cribriform|intraductal)[^.]{0,60}"
    r"(?:are|is|were|was)?\s*(?:not identified|not seen|absent|negative)",
    re.I,
)


def adverse_morphology(report: Any) -> bool | None:
    """True if cribriform or intraductal is reported, False if explicitly absent.

    None means the report does not mention it at all, which is NOT the same as
    absent and must not satisfy a rung that requires it excluded.
    """
    text = str(report or "")
    if not _ADVERSE_MORPHOLOGY.search(text):
        return None
    return not _MORPHOLOGY_ABSENT.search(text)


def histology_stable(report: Any) -> bool | None:
    """Does the pathology state stability across MORE THAN ONE biopsy?

    The cT2a-b rung of the GG2 ladder is earned by a written trajectory, not by
    a number, so this has to tell "stable across two timepoints" from the
    corpus's vacuous "stable across the single available timepoint" -- one
    sample cannot demonstrate a trajectory -- and from any sentence reporting an
    upgrade, which vetoes it outright.

    ⚠️ IMPORTED from the frozen `policy.py`, never re-implemented, for the same
    reason `_CONDITIONAL_CLAIMS` is: the frozen Task 2 policy scores this rung at
    +2 on the labelled set and two readers of the same clinical claim would
    drift. `None` means there was no report to read; `False` means the report
    was read and did not say it, which does not admit.
    """
    from ..policy import _histology_stable_across_timepoints
    text = str(report or "")
    if not text.strip():
        return None
    return _histology_stable_across_timepoints({"get_pathology_report": {"pathology_report": text}})


def single_timepoint(report: Any) -> bool | None:
    """Does the pathology state that only ONE biopsy timepoint exists?

    Positively stated, like the stability read -- a report that simply describes
    one specimen has not told us the sampling is complete. Same import rule and
    the same three-state meaning as `histology_stable`.
    """
    from ..policy import _single_biopsy_timepoint
    text = str(report or "")
    if not text.strip():
        return None
    return _single_biopsy_timepoint({"get_pathology_report": {"pathology_report": text}})


def psa_doubling_time(series: Any) -> float | None:
    """Years for the PSA to double, from the first and last recorded values.

    None when the series is too short or is not rising: a falling PSA has no
    doubling time, and inventing one would be a claim about the patient.
    """
    import math
    if not isinstance(series, list) or len(series) < 2:
        return None
    vals = [p.get("val") for p in series
            if isinstance(p, dict) and isinstance(p.get("val"), (int, float))]
    if len(vals) < 2 or vals[0] <= 0 or vals[-1] <= vals[0]:
        return None
    years = len(vals) - 1        # one reading a year, which is what these carry
    return round(years * math.log(2) / math.log(vals[-1] / vals[0]), 3)


def sensitivity(graph: Graph, facts: dict[str, Any], variable: str,
                domain: list[Any]) -> tuple[bool, dict[Any, str | None]]:
    """CHANNEL 1. Would knowing this absent variable change the answer?

    Walks the case once per value the record could plausibly carry. Returns
    whether every value lands on the same leaf, and what each one gives. That
    is the reference's own test for `uncertain`: not "the record is
    incomplete", but a named absent fact whose values give different actions.
    """
    out: dict[Any, str | None] = {}
    for value in domain:
        trial = derive(graph, {**facts, variable: value,
                               **{n: None for n, spec in graph.variables.items()
                                  if spec.get("kind") == "derived"
                                  and variable in spec.get("needs", ())}})
        walked = walk(graph, trial)
        out[value] = walked.leaf
    settled = len({v for v in out.values() if v is not None}) <= 1
    return settled, out


def _unit(graph: Graph, name: str, value: Any) -> float | None:
    """One unit of this variable's own measurement uncertainty, in its units."""
    spec = (graph.variables.get(name) or {}).get("uncertainty")
    if not spec:
        return None
    if spec.get("kind") == "exact":
        return None
    if "fraction" in spec:
        try:
            return abs(float(value)) * float(spec["fraction"]) or None
        except (TypeError, ValueError):
            return None
    return float(spec.get("step", 1))


def radius(graph: Graph, facts: dict[str, Any], *, cap: float = 4.0,
           steps: int = 40) -> tuple[float, str | None]:
    """CHANNEL 2. How many measurement-uncertainty units from a different answer.

    The robustness radius of this case: the smallest perturbation of a variable
    the realised path actually READ that lands the case on a different leaf,
    measured in units of that variable's own uncertainty.

    Two things the graph gives for free that a flat model would have to learn:
    a variable the path never read cannot matter however far it moves (a
    biopsy-naive PI-RADS 5 case is stable in PSA, correctly), and crossing a
    threshold into the SAME leaf costs nothing, so this measures distance to a
    different answer rather than to the nearest line.

    Returns the radius and the variable that carries it. `cap` is returned when
    nothing within four of its own reference change values moves the answer --
    at that distance the record is no longer describing the same patient.
    """
    base = walk(graph, derive(graph, facts))
    if base.leaf is None:
        return cap, None
    best, carrier = cap, None
    for name in dict.fromkeys(v for node in base.path for v in graph.nodes[node].reads):
        value = facts.get(name)
        unit = _unit(graph, name, value)
        if unit is None or value is None:
            continue
        for i in range(1, steps + 1):
            distance = cap * i / steps
            for signed in (distance, -distance):
                trial = {**facts, name: value + signed * unit}
                trial = derive(graph, {**trial, **{n: None for n, spec in graph.variables.items()
                                                   if spec.get("kind") == "derived"}})
                moved = walk(graph, trial)
                if moved.leaf is not None and moved.leaf != base.leaf and distance < best:
                    best, carrier = round(distance, 3), name
            if carrier == name:
                break
    return best, carrier


def _fired(graph: Graph, node: Node, facts: dict[str, Any]) -> Edge | None:
    """The edge this node actually took."""
    for edge in node.edges:
        try:
            if _test(edge.when, facts):
                return edge
        except Unanswerable:
            continue
    return None


def confidence(graph: Graph, facts: dict[str, Any]) -> tuple[str, str]:
    """The confidence label for this case, and the reason in one clause.

    THREE NAMED CHANNELS, each measured on the 91 and each written next to the
    node it belongs to. Deliberately not a global perturbation: three broader
    designs were built and measured, and all three scored below the all-`clear`
    baseline of 0.7363 ungated --

    * the measurement-stability radius over every variable the path read, 0.7253;
    * dissenters derived as "every variable the node did not decide on", graded
      by distance past their lines, 0.61-0.71 -- the totals do not separate the
      labels at all (median 3.38 `clear`, 4.72 `borderline`, 3.86 `uncertain`);
    * path shape alone -- depth, grey traverse, absent grade -- 0.7802, but
      emitting zero `uncertain`, so it carries no kappa.

    What survived is what is local and named. `radius` is kept in this module,
    unused here, for the record.

    ⚠️ The absence test here is `settled_by` -- is there a present finding
    decisive enough to say what the missing one would have said -- and NOT the
    `sensitivity` sweep, even though the sweep is the reference's own literal
    definition. Both were measured over the 91:

    | absence test | ungated ordinal | quadratic kappa | `uncertain` emitted |
    |---|---:|---:|---:|
    | `settled_by` alone (this one) | 0.7582 | **0.3384** | 15 |
    | the sweep alone, else `borderline` | 0.7637 | 0.2279 | 4 |
    | either one (union) | 0.7363 | 0.3119 | 19 |

    The sweep is too narrow -- it fires on 4 of 27 grade-missing cases while the
    reference calls 15 `uncertain` -- so on its own it buys ordinal and loses
    the kappa that makes a confidence chain worth having. `sensitivity` stays in
    this module for the record and for the consultant to use.

    ⚠️ Score every candidate on ordinal agreement AND quadratic kappa. Ordinal
    rewards saying `clear` -- the all-`clear` policy scores 0.7363 for emitting
    one label -- so a chain that never says `uncertain` can look respectable and
    be worthless.
    """
    derived = derive(graph, facts)
    base = walk(graph, derived)
    if base.leaf is None:
        return "uncertain", f"the policy does not answer this case ({base.reason})"

    if base.consult:
        node = graph.nodes[base.consult]
        return "consult", (f"{base.consult} hands this to a consultant: "
                           f"{node.consultant.get('asks','')}")

    if base.resolved_absent:
        name = base.resolved_absent[0]
        policy = resolve_missing(graph, name)
        if policy.get("when_missing") == "historical":
            return "uncertain", f"the {name} is not on record and cannot be obtained"
        return "borderline", (f"the {name} is not on record; this recommendation is to "
                              "obtain it rather than to act without it")

    grey: list[str] = []
    for i, node_id in enumerate(base.path):
        node = graph.nodes[node_id]
        conf, edge = node.conf, _fired(graph, node, derived)

        # Channel 3 -- a named conflict: the decision stands, but a variable this
        # node did NOT decide on points the other way.
        for clash in conf.get("conflicts") or ():
            try:
                if _test(clash["when"], derived):
                    return clash.get("gives", "borderline"), clash["because"].split(".")[0].strip()
            except Unanswerable:
                continue

        # Channel 1 -- an absent fact this node turns on, unless something
        # present settles what the absence would have told us.
        absent = conf.get("absent")
        if absent and derived.get(absent) is None:
            present = conf.get("settled_by")
            try:
                answered = bool(present) and _test(present, derived)
            except Unanswerable:
                answered = False
            if not answered:
                return "uncertain", (f"the {absent} is not on record and nothing "
                                     "present settles what it would have told us")

        # Channel 2 -- an edge that is equivocal by construction. The decider
        # could not settle it and handed the case on.
        if edge and edge.to in (conf.get("equivocal_edges") or ()):
            grey.append(node_id)

    if grey:
        return "borderline", f"the {graph.nodes[grey[0]].reads[0]} could not settle it on its own"
    return "clear", "nothing on the path was equivocal and nothing argues the other way"


@dataclass(frozen=True)
class ConfidenceFacts:
    """What the WALK knows about how settled a case was. Not a label -- inputs.

    The assessor agent proposes the label; these are the facts it may not
    contradict. Every one is a property of the path this case actually took, so
    none of them can be invented and none had to be written by hand.
    """
    decided_at: str
    decides_on: tuple[str, ...]
    equivocal: tuple[str, ...]
    absent: str | None
    absent_changes_answer: bool
    rests_on_absence: str | None

    def describe(self) -> str:
        """The block the assessor is given, in its own words."""
        lines = [f"The recommendation was settled at the question: "
                 f"{_ASKS.get(self.decided_at, self.decided_at)}"]
        if self.decides_on:
            lines.append(f"It turned on: {', '.join(self.decides_on)}.")
        if self.equivocal:
            lines.append("A question on the way could not settle the case on its own and "
                         "handed it to the next one: " + ", ".join(self.equivocal) + ".")
        else:
            lines.append("No question on the way was equivocal; each one settled cleanly.")
        if self.rests_on_absence:
            lines.append(f"The recommendation is to OBTAIN the {self.rests_on_absence}, "
                         "which is not on record, rather than to act without it.")
        elif self.absent and self.absent_changes_answer:
            lines.append(f"The {self.absent} is not on record, and its possible values "
                         "give DIFFERENT recommendations.")
        elif self.absent:
            lines.append(f"The {self.absent} is not on record, but the recommendation is "
                         "the same whatever its value would have been.")
        else:
            lines.append("Every fact the recommendation needed was on the record.")
        return "\n".join(lines)


#: Filled by `load` so ConfidenceFacts can quote a node's own question.
_ASKS: dict[str, str] = {}


def confidence_facts(graph: Graph, facts: dict[str, Any]) -> ConfidenceFacts | None:
    """Read the walk for everything that bears on how settled the case is."""
    derived = derive(graph, facts)
    base = walk(graph, derived)
    if base.leaf is None:
        return None
    equivocal = []
    for node_id in base.path:
        node = graph.nodes[node_id]
        edge = _fired(graph, node, derived)
        if edge and edge.to in (node.conf.get("equivocal_edges") or ()):
            equivocal.append(node_id)

    absent = changes = None
    for node_id in base.path:
        name = graph.nodes[node_id].conf.get("absent")
        if name and derived.get(name) is None:
            absent = name
            domain = (graph.variables.get(name) or {}).get("absent_domain") or []
            settled, _ = sensitivity(graph, facts, name, domain) if domain else (True, {})
            changes = not settled
            break

    rests = None
    for name in base.resolved_absent:
        if resolve_missing(graph, name).get("when_missing") == "order_first":
            rests = name
            break

    decided = graph.nodes[base.path[-1]]
    return ConfidenceFacts(
        decided_at=base.path[-1],
        decides_on=tuple(decided.conf.get("decides_on") or decided.reads),
        equivocal=tuple(equivocal),
        absent=absent,
        absent_changes_answer=bool(changes),
        rests_on_absence=rests,
    )


def forbidden_labels(cf: ConfidenceFacts) -> dict[str, str]:
    """Which labels the STRUCTURE rules out, and why. It may veto, never set.

    ⚠️ Three rules, each of which the model demonstrably gets wrong on its own:
    the consultant emitted `clear` on all 27 of its cases including the 8 the
    reference called `uncertain`, and the old hand-written `confidence_rule`
    could not see which question had actually been equivocal because it never
    walked the case.
    """
    out: dict[str, str] = {}
    if not cf.equivocal and cf.absent is None and cf.rests_on_absence is None:
        out["uncertain"] = ("every fact this recommendation needed was on the record and no "
                            "question on the way was equivocal, so there is no absent fact to name")
    # ⚠️ "an absent fact whose values change the leaf forbids `clear`" was the
    # obvious third rule and it is REMOVED on measurement: it fires on 6 of the
    # 91 and FOUR of those are labelled `clear`. It is the same finding the
    # confidence doc already records -- "a decisive present finding outranks a
    # missing history", 14 of the 27 established-cancer cases with no
    # retrievable grade are `clear`. The sweep is a good description of what is
    # absent and a bad prohibition on being sure anyway. It is still told to the
    # assessor in `describe()`; it just may not overrule it.
    if cf.rests_on_absence:
        out["clear"] = (f"the recommendation is to obtain the {cf.rests_on_absence} rather "
                        "than to act without it, which is not a settled recommendation")
    return out


def sample_values(graph: Graph, name: str) -> list[Any]:
    """The values the sweep walks for one variable, all of them present.

    A continuous quantity is sampled below, at and above every threshold the
    policy names, so a boundary written `<` where `<=` was meant is exercised
    rather than assumed correct.
    """
    spec = graph.variables[name]
    if spec["kind"] in {"categorical", "ordinal"}:
        return list(spec["values"])
    out: list[float] = []
    for line in spec["thresholds"]:
        step = max(abs(line) * 0.1, 0.01)
        out += [round(line - step, 4), float(line), round(line + step, 4)]
    return sorted(set(out))


def combinations(graph: Graph, names: list[str]) -> list[dict[str, Any]]:
    """The complete-data cross-product over `names`."""
    grids = [sample_values(graph, n) for n in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*grids)]


def paths(graph: Graph) -> list[tuple[str, ...]]:
    """Every structural route from the entry to a leaf, ignoring guards."""
    out: list[tuple[str, ...]] = []

    def go(node_id: str, seen: tuple[str, ...]) -> None:
        if graph.is_leaf(node_id):
            out.append(seen + (node_id,))
            return
        if node_id in seen:
            return
        for edge in graph.nodes[node_id].edges:
            go(edge.to, seen + (node_id,))

    go(graph.entry, ())
    return out
