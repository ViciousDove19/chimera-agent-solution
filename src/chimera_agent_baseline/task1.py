"""Task 1 biopsy policy: three questions, asked in order.

    1. f        does this person have cancer?
    2. gate     how sure are we?
    3. net      if the answer is already secure, is the tissue worth taking?

Stage 1 reads a primary channel and three corroborating ones: PI-RADS says
which way the case points, and PSA, PSA density and DRE each vote on whether
they see the same thing. The vote count makes the grade/reclassification
question reachable; agreement does not make tissue redundant by itself.

A variable's reported weight is its role in the branch structure this case
reaches, so the weight we report describes how the answer was actually reached.

Pure function of the structured prompt plus family history -- no LLM, no tool
calls, no text parsing.  All thresholds live in ``resources/task1_policy.yaml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CONFIG_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "resources" / "task1_policy.yaml",
    Path("/opt/app/resources/task1_policy.yaml"),
    Path("resources/task1_policy.yaml"),
)

# Clinical names for the corroborating channels, used in the stated reason.
_CHANNEL_NAMES = {"psa": "the PSA", "psad": "the PSA density", "dre": "the examination"}

VARIABLES = ("pirads", "psa", "psad", "vol", "dre", "age", "comorbidity", "fh", "cspca", "bx")


def _default_config_path() -> Path:
    override = os.getenv("CHIMERA_TASK1_POLICY")
    if override:
        return Path(override)
    for candidate in _CONFIG_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("task1_policy.yaml not found; tried " + ", ".join(str(c) for c in _CONFIG_CANDIDATES))


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or _default_config_path()).read_text())


@dataclass(frozen=True)
class Verdict:
    decision: str
    confidence: str
    weights: dict[str, str]
    reveals: list[str]
    stage: str
    reasons: tuple[str, ...] = field(default=())
    roles: dict[str, str] = field(default_factory=dict)
    votes: int = 0


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is(value: Any, options: list[str]) -> bool:
    return str(value).strip().lower() in {str(o).strip().lower() for o in options}


def _biopsy_state(prompt: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Return structured biopsy status, with a release-era encounter fallback."""
    if "bx" in prompt and prompt.get("bx") is not None:
        value = str(prompt.get("bx")).strip().lower()
        if value in {"none", "no", "no previous biopsy", "biopsy-naive", "biopsy naive"}:
            return "naive"
        if value in {"negative", "benign", "asap/hgpin", "asap", "hgpin"}:
            return "previous_negative"
        if value in {"positive", "cancer", "malignant"}:
            return "established"

    encounter = prompt.get("enc_type")
    if _is(encounter, cfg["live"]["no_biopsy_history"]):
        return "naive"
    if _is(encounter, cfg["certainty"]["uncertain_encounters"]):
        return "established"
    return "previous_negative"


@dataclass(frozen=True)
class Signal:
    """Stage 1, computed rather than asked.

    Everything here is a comparison against a configured line: the PI-RADS
    direction table, the density contradiction and its trustworthiness guard,
    the three corroborating votes, and whether the gland is large.  It lives in
    one function so that `apply_policy` and the prompt chain answer stage 1
    identically, and so that no prompt ever has to evaluate a number.

    ⚠️ This used to be an LLM link, and the model did not execute it.  Measured
    2026-08-31 on the two cached runs: the vote list matched this function on
    45% of cases (Gemma 2B) and 33% (Qwen 35B), and the `vol > 70` rule --
    written as an explicit table with worked examples at 45, 68 and 90 mL --
    put 68 of 68 normal glands in `weighed` on the 2B model.  Substituting this
    function for the link and keeping the rest of the chain took
    important/decisive F1 from 0.7000 to 0.7446.  If a rule is arithmetic,
    Python owns it.
    """

    likely: bool
    driver: str
    contradicted: bool
    votes: int
    voted: dict[str, bool]
    leaned_on: tuple[str, ...]
    vol_large: bool
    confounded: bool = False


def usable_density(prompt: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Drop a stated PSA density that the record itself contradicts.

    This is input validation, not a clinical rule: it carries no view about
    biopsy and, applied on its own, changes no decision on either cohort.  The
    record states the density twice -- once directly, once implicitly as PSA
    over volume -- and the two statements can disagree.  On one labelled case
    the two are 72% apart; the next worst disagreement in the 91 is 17%, so any
    tolerance from roughly 20% to 70% selects the same record.  A quotient its
    own inputs contradict is not a measurement of this patient, so no branch
    gets to read it -- neither to raise suspicion nor to lower it.

    Fires on 1 of the 91 labelled cases and 4 of the 50 development cases in
    `external_data/`.  (Some of the latter may be reconstruction error in that
    CSV rather than genuine record inconsistency; do not quote the rate.)
    """
    psad, psa, vol = _num(prompt.get("psad")), _num(prompt.get("psa")), _num(prompt.get("vol"))
    if psad is None or psa is None or not vol:
        return prompt
    if abs(psa / vol - psad) <= cfg["suspicion"]["psad_max_disagreement"] * (psa / vol):
        return prompt
    return {**prompt, "psad": None}


def signal(prompt: dict[str, Any], cfg: dict[str, Any], prior_grade: int | None = None) -> Signal:
    """Does the current non-histologic evidence point to cancer, and on what.

    ``prior_grade`` is read only by the confounded-density clause below, which
    needs to know whether the grade question is still open.  Everything else
    here is a function of the structured prompt alone.
    """
    pirads, psad = _num(prompt.get("pirads")), _num(prompt.get("psad"))
    psa, vol = _num(prompt.get("psa")), _num(prompt.get("vol"))
    dre = prompt.get("dre")
    biopsy_state = _biopsy_state(prompt, cfg)
    s, co, live = cfg["suspicion"], cfg["corroboration"], cfg["live"]

    # -- stage 1: does this person have cancer? ---------------------------
    # PI-RADS is the primary channel and always carries the direction.
    if pirads is not None and pirads >= s["pirads_likely"]:
        likely, driver = True, "pirads"
    elif pirads is not None and pirads == s["pirads_equivocal"]:
        # Equivocal imaging: the density decides, read against its volume.
        likely, driver = (psad or 0) > s["psad_likely"], "psad"
    else:
        likely, driver = False, "pirads"

    # A clearly low density can *contradict* the imaging, but contradiction is
    # an uncertainty signal, not permission to reverse PI-RADS 4-5.  The EAU
    # PI-RADS x PSA-density matrix recommends biopsy in every density cell for
    # PI-RADS 4-5.  In a previously sampled gland, discordance creates exactly
    # the question a repeat biopsy can answer: whether the lesion represents
    # clinically significant disease despite reassuring biochemistry.
    # A density we cannot verify does not get to overrule the imaging -- but
    # that test no longer lives here.  `usable_density` has already discarded a
    # figure the record contradicts, so any density still present is one the
    # policy is entitled to read.
    if (
        likely
        and psad is not None
        and psad < s["psad_contradicts"]
        and biopsy_state != "naive"
    ):
        # Keep the imaging direction.  ``contradicted`` is carried forward so
        # confidence and the rationale can disclose the conflicting evidence.
        contradicted = True
    else:
        contradicted = False

    # A density is PSA divided by volume, so a gland large enough drives the
    # quotient down whatever the numerator is doing.  `density_trustworthy`
    # above catches a density the record contradicts *arithmetically*; this
    # catches one the record contradicts *clinically* -- the arithmetic is
    # fine, and that is exactly the problem.  A gland of ~150 mL can turn a PSA
    # that quintupled in under two years into a density of 0.08, and that
    # number then vetoes the case twice: it holds PI-RADS 3
    # below `psad_likely`, so stage 1 answers "no cancer indicated", and it
    # sits below `psad_high`, so it casts no corroborating vote either.
    # Between them they stop the case ever reaching stage 3 -- the one
    # question a clinician actually asks of a man who already has prostate
    # cancer: is his grade still what we think it is?
    #
    # So this is not a new decision rule.  `RECLASSIFICATION` already holds
    # that an unretrievable grade is a reason to take tissue; this only stops
    # an uninterpretable number from suppressing the case before the question
    # is put, which is why it fires only where stage 1 currently says `no`.
    #
    # ⚠️ The confound is not the gland on its own -- it is being unable to say
    # whether an ELEVATED PSA came from the tumour or from the bulk.  With a
    # low PSA there is nothing to attribute and a diluted density really is
    # reassuring, which is why a large gland with a PSA under 5 must not fire
    # and does not.  See `confounded_density` in the YAML.
    conf = cfg["confounded_density"]
    confounded = (
        not likely
        and prior_grade is None
        and vol is not None
        and vol > conf["vol_confounds_ml"]
        and psa is not None
        and psa >= conf["psa_needs_explaining"]
        and biopsy_state == "established"
    )
    if confounded:
        # The gland, not the density, is what carries the case now.
        likely, driver, contradicted = True, "vol", False

    # The three corroborating channels vote when their own reading is
    # suspicious.  Voting and being leaned on are not the same thing: PSA is
    # measured for every patient and read on every case, so it is always
    # consulted; PSA density only becomes something we lean on once it is high
    # enough to settle a case by itself; a normal DRE argues nothing at all.
    votes = 0
    voted_by: dict[str, bool] = {}
    leaned: list[str] = []
    for name, voted, leaned_on in (
        ("psa", psa is not None and psa >= co["psa_high"], True),
        # ⚠️ The density is leaned on only at the HIGH end. "Both tails do work"
        # was tested and does not survive: the reference names the density in 4
        # of 9 cases below `psad_contradicts`, but in only 2 of the 7 where it
        # actually contradicts a positive MRI -- the mechanism's own population.
        # The apparent signal was two low-PI-RADS cases. Do not re-derive.
        ("psad", (psad or 0) > co["psad_high"], (psad or 0) > s["psad_likely"]),
        ("dre", _is(dre, co["dre_suspicious"]), _is(dre, co["dre_suspicious"])),
    ):
        votes += voted
        voted_by[name] = voted
        if leaned_on:
            leaned.append(name)
    # A large gland is what makes the density reading interpretable -- and that
    # is the ONLY thing that makes it so. Two branches used to mark the volume
    # consulted merely because they had read a density, whatever the gland
    # measured; on the 7 cases that produced the reference agreed just once,
    # against 9 of 16 for the size test here. Read the size test as the single
    # gate, which is what `live.vol_large` already claims to be.
    vol_large = (vol or 0) > live["vol_large"]

    return Signal(
        likely=likely,
        driver=driver,
        contradicted=contradicted,
        votes=votes,
        voted=voted_by,
        leaned_on=tuple(leaned),
        vol_large=vol_large,
        confounded=confounded,
    )


# The branches. Stage 2 asks whether tissue has already settled the diagnosis;
# stage 3, reached only when it has, asks whether more tissue could still change
# the grade -- because grade is what chooses between surveillance and treatment.
NO_BENEFIT = "NO_BENEFIT"
DISEASE_ALREADY_SIGNIFICANT = "DISEASE_ALREADY_SIGNIFICANT"
BIOCHEMICAL_OVERRIDE = "BIOCHEMICAL_OVERRIDE"
CERTAIN = "CERTAIN"
DENSITY_CONTRADICTS_IMAGING = "DENSITY_CONTRADICTS_IMAGING"
MANAGEMENT_INVARIANT = "MANAGEMENT_INVARIANT"
DENSITY_FOLLOW_UP = "DENSITY_FOLLOW_UP"
RECLASSIFICATION = "RECLASSIFICATION"
GRADE_ON_RECORD = "GRADE_ON_RECORD"
TISSUE_INFORMATIVE = "TISSUE_INFORMATIVE"

# Branches whose answer is fixed; the rest answer from stage 1.  Discordant
# PI-RADS 4-5 / low-density evidence favours obtaining tissue, not omitting it.
_DECISION = {NO_BENEFIT: "no", DISEASE_ALREADY_SIGNIFICANT: "no",
             BIOCHEMICAL_OVERRIDE: "yes",
             MANAGEMENT_INVARIANT: "no", DENSITY_CONTRADICTS_IMAGING: "yes",
             RECLASSIFICATION: "yes", GRADE_ON_RECORD: "no",
             DENSITY_FOLLOW_UP: "no"}

# Branches that have left the diagnostic question behind, where the density can
# no longer bear on the answer -- the reference marks it important in 2 of those
# 12 cases against our 10.
_PSAD_INERT = (DISEASE_ALREADY_SIGNIFICANT, MANAGEMENT_INVARIANT)


def diagnosis(prompt: dict[str, Any], cfg: dict[str, Any], sig: Signal,
              prior_grade: int | None = None) -> str | None:
    """Question 2 -- has tissue already settled this, and does that end the case?

    Returns the branch, or ``None`` when the diagnosis is established but the
    grade question is still open, which is the one route to question 3.

    ⚠️ The order is load-bearing: `NO_BENEFIT` is tested BEFORE the encounter
    split, so it reaches a man with a prior negative biopsy too. Age overrides
    the value of a REPEAT biopsy, not the need for a first one -- you cannot
    treat what you have not diagnosed.
    """
    age = _num(prompt.get("age"))
    biopsy_state = _biopsy_state(prompt, cfg)
    established = biopsy_state == "established"
    naive = biopsy_state == "naive"
    override = cfg["biochemical_override"]
    pirads, psa = _num(prompt.get("pirads")), _num(prompt.get("psa"))
    psad, vol = _num(prompt.get("psad")), _num(prompt.get("vol"))
    if (
        pirads == override["pirads"]
        and psa is not None and psa > override["psa_above"]
        and psad is not None and psad > override["psad_above"]
        and vol is not None and vol <= override["volume_at_or_below"]
    ):
        # The owner treats concordant marked biochemistry as sufficient to
        # override a PI-RADS 2 result. This precedes the repeat-biopsy age gate:
        # age/risk changes confidence, not the requested biopsy decision.
        return BIOCHEMICAL_OVERRIDE
    if age is not None and age > cfg["net_benefit"]["age_no_benefit"] and not naive:
        return NO_BENEFIT
    # Start with the prior diagnosis.  A man with documented prostate cancer
    # does not become cancer-free because PI-RADS is missing or because PSA
    # density is low.  His open question is grade/reclassification.
    if established and prior_grade is not None and prior_grade >= cfg["prior_grade"]["significant_at"]:
        # The record already names significant disease; the next step is
        # treatment, and no new core changes that.
        return DISEASE_ALREADY_SIGNIFICANT
    if not established:
        return DENSITY_CONTRADICTS_IMAGING if sig.contradicted else CERTAIN
    return None


def tissue(prompt: dict[str, Any], cfg: dict[str, Any], sig: Signal,
           prior_grade: int | None = None) -> str:
    """Question 3 -- would new histology change what happens next?

    ⚠️ Corroboration ROUTES and does not answer. Asking the grade question of
    every established-diagnosis case scores 0.7912; gating it on >= 2
    corroborating channels scores 0.9121. With 0-1 channels agreeing the case is
    still diagnostic -- is this lesion real? -- and PI-RADS answers that.
    """
    pirads, psad = _num(prompt.get("pirads")), _num(prompt.get("psad"))
    # If PI-RADS is absent, do not turn the absent category into a negative
    # finding. The documented cancer plus unavailable grade carries the case.
    if prior_grade is None and pirads is None:
        return RECLASSIFICATION

    # In ordinary follow-up, retain the evidence-strength gate. This avoids
    # treating every unavailable historical grade as an automatic indication.
    if not (sig.likely and sig.votes >= cfg["corroboration"]["secure_at"]):
        # ...unless the density itself is the reassurance.  A man already known
        # to carry prostate cancer, whose grade is not on record, whose lesion
        # nothing else corroborates, and whose PSA density is genuinely low, is
        # a surveillance question rather than a repeat-biopsy one, pending the
        # result of the earlier biopsy.
        #
        # ⚠️ The reassurance is the density and nothing else. A PI-RADS gate
        # cannot stand in for it: all four cases this fires on are PI-RADS 4-5,
        # where EAU recommends biopsy in every density cell, and the one case
        # that must stay `yes` is PI-RADS 4 like three of them. That case is
        # excluded upstream instead, by `usable_density`, because its stated
        # density is not a number its own record supports.
        if prior_grade is None and psad is not None and psad < cfg["suspicion"]["psad_contradicts"]:
            return DENSITY_FOLLOW_UP
        return TISSUE_INFORMATIVE

    # Missing grade is the open management question once the current disease
    # signal is secure. Neither a high PSA nor concordant imaging can substitute
    # for histology; PSA may trigger staging, but cannot prove grade irrelevant.
    if prior_grade is None:
        return RECLASSIFICATION

    return GRADE_ON_RECORD


def roles(prompt: dict[str, Any], cfg: dict[str, Any], sig: Signal, stage: str) -> dict[str, str]:
    """What each variable did on the branch this case reached.

    ⚠️ This is not stage 1's provenance. Stage 1 answers "what carried the
    suspicion"; a role answers "what carried THIS BRANCH", and they differ
    wherever a later question took over -- the biopsy history settles an
    established-diagnosis case that PI-RADS pointed at.
    """
    out = dict.fromkeys(VARIABLES, "available")
    out["cspca"] = out["fh"] = "unused"
    out["age"] = "consulted"
    for name in sig.leaned_on:
        out[name] = "consulted"
    if sig.vol_large:
        out["vol"] = "consulted"
    if sig.confounded:
        # Here the gland is not context for the density, it IS the finding.
        out["vol"] = "settled"
    if _biopsy_state(prompt, cfg) != "naive":
        out["bx"] = "consulted"

    if stage == NO_BENEFIT:
        out["age"] = "settled"
    elif stage == DISEASE_ALREADY_SIGNIFICANT:
        out["bx"], out["pirads"] = "settled", "consulted"
    elif stage == BIOCHEMICAL_OVERRIDE:
        out["pirads"] = "consulted"
        out["psa"] = out["psad"] = out["vol"] = "settled"
    elif stage in (CERTAIN, DENSITY_CONTRADICTS_IMAGING):
        out[sig.driver] = "settled"
    else:
        # Stage 3. The biopsy history routes every case here but settles one
        # only where the earlier result is actually in hand.
        out["bx"], out["pirads"] = "consulted", "consulted"
        if stage in (MANAGEMENT_INVARIANT, GRADE_ON_RECORD):
            out["bx"] = "settled"
        if stage == DENSITY_FOLLOW_UP:
            # The density is what held this case back, so it is live here even
            # though it is too low to be leaned on for suspicion. The reference
            # names it in 2 of the 4 cases and never calls it decisive, which is
            # why it is consulted rather than settled.
            out["psad"] = "consulted"

    # Reporting corrections, applied to every branch alike. PI-RADS carries the
    # direction and is always at least important -- the reference calls it so in
    # 99% of cases -- and the density goes inert once the diagnosis is settled.
    if out["pirads"] in ("available", "unused"):
        out["pirads"] = "consulted"
    if stage in _PSAD_INERT and out["psad"] == "consulted":
        out["psad"] = "available"
    return out

def procedural_risk(prompt: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, ...]:
    """What it costs THIS man to be biopsied, from what the record documents.

    Bleeding, sepsis and anaesthetic reserve -- the three things that make a
    diagnostic biopsy a decision rather than a formality. Read from the
    medication list and the past history, never inferred: an undocumented
    comorbidity is not a risk factor, it is silence.
    """
    risk = cfg["procedural_risk"]
    found: list[str] = []
    meds = {m.strip().lower() for m in str(prompt.get("meds") or "").split(",")}
    history = str(prompt.get("medhx") or "")
    anticoagulated = any(
        any(drug.lower() in med for med in meds if med)
        for drug in risk["anticoagulants"]
    )
    if anticoagulated:
        found.append("bleeding")
    elif any(h in history for h in risk["bleeding_history"]):
        found.append("bleeding")
    if any(h in history for h in risk["infection_history"]):
        found.append("infection")
    if any(h in history for h in risk["anaesthetic_history"]):
        found.append("anaesthetic")
    age = _num(prompt.get("age"))
    if age is not None and age >= cfg["net_benefit"]["age_no_benefit"]:
        found.append("life_expectancy")
    return tuple(found)


def risk_is_outranked(prompt: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Is there a present finding decisive enough to settle the conversation?

    PI-RADS 5 in a man young enough to be treated. EAU's evidence summary puts
    csPCa detection at 83% for PI-RADS 5 against 55% at PI-RADS 4, so at 5 the
    indication is not what is in doubt. Same precedence as a decisive present
    finding beating a missing prior grade.
    """
    risk = cfg["procedural_risk"]
    pirads, age = _num(prompt.get("pirads")), _num(prompt.get("age"))
    return (
        pirads is not None
        and pirads >= risk["settled_pirads"]
        and age is not None
        and age < risk["settled_below_age"]
    )


def confidence(prompt: dict[str, Any], cfg: dict[str, Any], stage: str,
               prior_grade: int | None = None, decision: str | None = None) -> str:
    """A property of the evidence that carried the branch, never of the branch.

    ⚠️ Worth more than its 0.20 weight: the rationale judge is handed the
    expected value and marks the prose down when ours differs, so it is scored
    twice.
    """
    psa, pirads = _num(prompt.get("psa")), _num(prompt.get("pirads"))
    if stage == BIOCHEMICAL_OVERRIDE:
        return "borderline" if procedural_risk(prompt, cfg) else "clear"
    if stage == DISEASE_ALREADY_SIGNIFICANT:
        return "clear" if (prior_grade or 0) >= cfg["prior_grade"]["unambiguous_at"] else "borderline"
    if stage == GRADE_ON_RECORD:
        return "clear" if psa is not None and psa >= cfg["net_benefit"]["psa_legible"] else "borderline"
    if stage == DENSITY_FOLLOW_UP:
        # Established-cancer pathway: both the earlier grade and the meaning of
        # the low density against positive MRI remain unresolved. Either can
        # modify the repeat-biopsy action, so a forced answer is uncertain.
        return "uncertain"
    if stage == RECLASSIFICATION:
        # Established-cancer pathway with a missing earlier grade. PI-RADS 5
        # has a materially higher csPCa detection rate than PI-RADS 4 in EAU's
        # evidence summary (83% vs 55%), so the present indication for tissue
        # remains clear at 5. At 4 (or with MRI unavailable), the missing grade
        # can realistically reverse whether repeat tissue is useful.
        return "clear" if pirads is not None and pirads >= 5 else "uncertain"
    if stage == DENSITY_CONTRADICTS_IMAGING:
        # Resolving this uncertainty is the reason to biopsy.
        return "borderline"
    if stage == CERTAIN and decision == "yes" and procedural_risk(prompt, cfg) \
            and not risk_is_outranked(prompt, cfg):
        # No established diagnosis, so the biopsy is the diagnostic step and it
        # stands -- but this man carries a documented reason it is not free, and
        # nothing present is decisive enough to settle that for him. The
        # recommendation is made; the conversation is shared.
        return "borderline"
    if stage != TISSUE_INFORMATIVE:
        return "clear"
    # Nothing settles the case, so our confidence is the MRI's own: PI-RADS is a
    # certainty scale, and its extremes are its confident readings.
    mri = cfg["mri_unambiguous"]
    return "clear" if pirads is not None and (
        pirads <= mri["at_or_below"] or pirads >= mri["at_or_above"]) else "borderline"


def reason(prompt: dict[str, Any], cfg: dict[str, Any], sig: Signal, stage: str,
           prior_grade: int | None = None) -> str:
    """The `why`, rendered from the branch rather than by a model.

    ⚠️ Every string here is audited by `scripts/check_rationale_grounding.py`
    against the clinical-data socket the judge can actually see. That audit
    found 17 ungrounded assertions the first time it ran -- an age, a
    comorbidity count, a symptom score, none of which is in any document -- so
    reword nothing without re-running it.
    """
    psa, psad = _num(prompt.get("psa")), _num(prompt.get("psad"))
    if stage == NO_BENEFIT:
        # No number: age decides this branch and the judge cannot see it.
        return "he has outlived the benefit of re-sampling tissue"
    if stage == BIOCHEMICAL_OVERRIDE:
        return (f"despite PI-RADS 2, PSA {psa:g} ng/mL and PSA density {psad:g} "
                f"in a {prompt.get('vol')} mL gland provide a concordant biochemical "
                "indication for tissue")
    if stage == DISEASE_ALREADY_SIGNIFICANT:
        return f"the record documents ISUP {prior_grade} disease; the next step is treatment"
    if stage == DENSITY_CONTRADICTS_IMAGING:
        # Conflicting evidence is why tissue is useful here, not why it should
        # be withheld.
        if psa is not None:
            return (f"PI-RADS-positive imaging conflicts with PSA {psa:g} ng/mL and PSA density "
                    f"{psad:g}; the discordance cannot safely exclude significant cancer, so "
                    "targeted tissue is informative")
        return (f"PI-RADS-positive imaging conflicts with a PSA density of {psad:g}; the "
                "discordance cannot safely exclude significant cancer, so targeted tissue is informative")
    if stage == DENSITY_FOLLOW_UP:
        return (f"a PSA density of {psad:g} is reassuring against the imaging and nothing else "
                "corroborates the lesion, so this is a follow-up rather than a repeat biopsy; "
                "the grade of the earlier biopsy is not on record and would settle it")
    if stage == CERTAIN:
        return f"{sig.driver} settles it and there is no established diagnosis"
    if stage == MANAGEMENT_INVARIANT:  # reserved until staging/fitness evidence is available
        return (f"at PSA {psa:g} the disease burden already sets the management: the next "
                "step is staging and systemic treatment, which no grade would change")
    if stage == RECLASSIFICATION:
        return ("the grade of the earlier biopsy is not retrievable from the record, and "
                "it is what separates surveillance from curative treatment, so tissue can "
                "still change management")
    if stage == GRADE_ON_RECORD:
        agreeing = ["the imaging"] + [_CHANNEL_NAMES[k] for k, voted in sig.voted.items() if voted]
        named = " and ".join(filter(None, [", ".join(agreeing[:-1]), agreeing[-1]]))
        return (f"the record already grades this disease as ISUP {prior_grade}, and {named} "
                "agree with it, so re-sampling would not reclassify him")
    if sig.confounded:
        return (f"the gland measures {prompt.get('vol')} mL, so the PSA density "
                f"cannot say whether a PSA of {psa:g} reflects the tumour or the bulk "
                "of the gland; the grade of the earlier biopsy is what would settle the "
                "management, and it is not on record")
    return "an established diagnosis, but the picture is not yet secure"


def compose(prompt: dict[str, Any], cfg: dict[str, Any], sig: Signal, stage: str,
            prior_grade: int | None = None) -> Verdict:
    """Branch to emitted fields. A lookup and a handful of ifs -- no judgement."""
    weights_by_role = roles(prompt, cfg, sig, stage)
    weights = {v: cfg["roles"][weights_by_role[v]] for v in VARIABLES}
    decision = _DECISION.get(stage, "yes" if sig.likely else "no")
    return Verdict(
        decision=decision,
        confidence=confidence(prompt, cfg, stage, prior_grade, decision),
        weights=weights,
        reveals=sorted({cfg["sections"][v] for v, w in weights.items()
                        if w != "not_used" and cfg["sections"].get(v)}),
        stage=stage,
        reasons=(reason(prompt, cfg, sig, stage, prior_grade),),
        roles=weights_by_role,
        votes=sig.votes,
    )


def apply_policy(prompt: dict[str, Any], cfg: dict[str, Any], family_history: Any = None,
                 prior_grade: int | None = None) -> Verdict:
    """The three questions, asked in order, then composed.

    ``family_history`` is accepted for call-site compatibility and read by no
    rule. Structured ``bx`` supplies prior-biopsy status in validation; the
    prior ISUP grade is still not in the prompt and has to be retrieved from
    previous notes, so it is passed in rather than looked up. Released prompts
    without ``bx`` retain the encounter-type fallback.
    """
    prompt = usable_density(prompt, cfg)
    sig = signal(prompt, cfg, prior_grade)
    stage = diagnosis(prompt, cfg, sig, prior_grade) or tissue(prompt, cfg, sig, prior_grade)
    return compose(prompt, cfg, sig, stage, prior_grade)
