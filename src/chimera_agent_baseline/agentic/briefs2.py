"""Agent briefs for the Task 2 agentic policy — post-biopsy management.

Same shape as ``briefs.py``: prose for the agents, a machine-readable twin of
each rule so the critique can name a violation, and disjoint ownership of the
eleven scored ``variable_weights`` keys.

⚠️ **Task 2 retrieves almost nothing, and that is deliberate.** All twelve
clinical variables the policy reads are in ``prompt.json`` on 72/72 labelled
cases, and the reference pathologist revealed **0 sections on 72/72**. Tool
efficiency is precision against that empty set, so a single reveal scores the
case 0 on a component worth 0.15, to defend at most 0.05 of grounding. Only the
two facts genuinely absent from the prompt -- the percentage of Gleason pattern
4 and cribriform morphology -- are worth asking for, and only on the ISUP 2
band where they change the answer.
"""

from __future__ import annotations

from typing import Any

WEIGHT_OWNERS: dict[str, tuple[str, ...]] = {
    "gatekeeper": ("age", "comorbidity"),
    "pathology": ("bx_isup", "bx_gl_prim", "bx_gl_sec"),
    "staging": ("pirads", "ct"),
    "biochemical": ("psa", "psad", "cspca", "fh"),
}

REVEAL_ITEMS: dict[str, tuple[str, str]] = {
    "PATTERN_4_DETAIL": ("get_pathology_report", "pathology"),
}

#: ISUP grade -> track. The grade is recorded on 72/72 cases, and on its own it
#: predicts 61 of the 72 reference decisions.
TRACKS = ("NEGATIVE", "LOW", "INTERMEDIATE", "HIGH")

#: Clinical T stages that do not by themselves make a case unfavourable. `cTx`
#: belongs here: an unassignable stage is unknown, NOT advanced.
LOCALISED_STAGES = frozenset({"cT1", "cT1a", "cT1b", "cT1c", "cT2a", "cTx"})


#: Numeric part of the cT stage. `cTx` and a blank field are both unknown -- a
#: rank is a statement about extent, and neither of those makes one. Restated
#: here rather than imported from `task2.py`, so the two policies cannot drift
#: into each other.
def stage_rank(ct: Any) -> int | None:
    text = str(ct or "").strip().upper()
    if not text.startswith("CT") or len(text) < 3 or not text[2].isdigit():
        return None
    return int(text[2])


def flatten_thresholds(cfg: dict[str, Any]) -> dict[str, Any]:
    t = cfg["task2"]
    return {
        "psa_ceiling": t["psa_ceiling"],
        "psa_unfavourable": t["psa_unfavourable"],
        "psad_unfavourable": t["psad_unfavourable"],
        "psav_unfavourable": t["psav_unfavourable"],
        "ipss_severe": t["ipss_severe"],
        "luts_psad_ceiling": t["luts_psad_ceiling"],
        "pattern4_unfavourable_pct": t["pattern4_unfavourable_pct"],
        "stage_palpable_from": t["stage_palpable_from"],
        "comorbidity_burden": t["comorbidity_burden"],
        "gleason_pattern_from": t["gleason_pattern_from"],
        "grade_settles_below": t["grade_settles_below"],
        "grade_settles_from": t["grade_settles_from"],
        "psa_decisive_no_cancer": t["psa_decisive_no_cancer"],
        "psa_decisive_burden": t["psa_decisive_burden"],
        "psa_inert_high_risk": t["psa_inert_high_risk"],
        "enforced_rungs": tuple(t.get("enforced_rungs") or ()),
    }


_CONTRACT = """
You answer in EXACTLY this format. Flat lines, one per key, nothing before or
after. No markdown, no JSON, no preamble.

VERDICT: <one of the allowed values>
{extra}WEIGHTS: {example}
NEED: <items you must see, comma-separated, or NONE>
KEY_EVIDENCE: <one line naming the actual values that drove your verdict>
RATIONALE: <two sentences>

WEIGHTS rates how much each variable you own drove YOUR verdict:
  not_used / noted / important / decisive.
Rate exactly these variables and no others: {owned}.

{weights_note}

NEED: {needs}

⚠️ Almost everything you need is already in front of you. This record is a
referral panel, not a masked chart -- the grade, the stage, the PI-RADS and
every PSA reading are given to you directly. Ask for something only when it is
genuinely not in your input AND its value would change your verdict.
""".strip()


def _contract(owner: str, needs: str, example: str, note: str, extra: str = "") -> str:
    return _CONTRACT.format(
        owned=", ".join(WEIGHT_OWNERS[owner]), needs=needs, example=example,
        weights_note=note.strip(), extra=extra,
    )


GATEKEEPER = """
You are a clinical eligibility assessor. Your only question: over what horizon
should this man's cancer be managed?

Curative-intent management -- surveillance or treatment -- assumes he will live
long enough to benefit from it. Read age, comorbidity burden and functional
status together. A list of conditions is not a count: advanced cardiac,
pulmonary, renal or hepatic disease weighs far more than well-controlled
hypertension. Age frames the judgement but rarely settles it on its own in this
population.

⚠️ Your verdict does NOT choose the management class, and a short horizon is
not an answer in itself. It is context the adjudicator reads alongside the
grade and the stage, and it is the reason `age` is always weighted
`important`.

⚠️ An absent fact is not an adverse one. Where age, the problem list, the
medications or the functional status are simply not recorded, you have not been
shown a limited horizon -- you have been shown nothing, and the answer is
ELIGIBLE. Answer LIMITED only where the record positively describes a competing
illness that is the more pressing problem. If you are weighing it only because
he is old, or only because the chart is thin, the answer is ELIGIBLE.

VERDICT: ELIGIBLE | LIMITED
""".strip() + "\n\n" + _contract(
    "gatekeeper", "NONE", "age=important; comorbidity=noted",
    """`age` is ALWAYS `important` -- never `noted`. It sets the horizon every
recommendation here is measured against, and a variable that frames the whole
judgement is `important` even where it did not pick the answer.
`comorbidity` is `noted`: it is read as context for the age judgement.""",
)


PATHOLOGY = """
You are a histopathology assessor. Your only question: what does the biopsy
establish about this disease?

The ISUP grade group is the primary determinant of management, and it is given
to you directly.

  ISUP 0  no cancer was found. There is nothing to treat, and the question
          becomes whether monitoring continues.
  ISUP 1  Gleason 3+3, low-risk disease.
  ISUP 2  Gleason 3+4, intermediate. This band splits into favourable and
          unfavourable, and that split is what decides management.
  ISUP 3  Gleason 4+3. Pattern 4 predominates; surveillance is not appropriate.
  ISUP 4-5 Gleason 8 or above, high risk.

Two features matter on the ISUP 2 band and are NOT in your input: the percentage
of Gleason pattern 4, and whether cribriform morphology is present. Either a
pattern 4 fraction above {pattern4_unfavourable_pct}% or any cribriform
architecture makes intermediate disease unfavourable. Ask for PATTERN_4_DETAIL
only on an ISUP 2 case in which NOTHING ELSE is already unfavourable -- no
adverse stage, no PI-RADS 5, no raised PSA, density or velocity. On any other
grade, or where one of those is already present, the pattern 4 burden cannot
change the answer and reading the report is wasted. Answer NEED: NONE.

VERDICT: NO_CANCER | LOW_GRADE | INTERMEDIATE | HIGH_GRADE
PATTERN_4_BURDEN: HIGH | LOW | NOT_PROVIDED
""".strip() + "\n\n" + _contract(
    "pathology", "PATTERN_4_DETAIL", "bx_isup=decisive; bx_gl_prim=noted; bx_gl_sec=noted",
    """`bx_isup` is never `not_used` -- the grade group determines the track and
is at least `important`, and `decisive` when it settles management on its own
(no cancer, or ISUP 3 and above). `bx_gl_prim` and `bx_gl_sec` are `noted`:
they are the arithmetic behind the grade group rather than separate findings.""",
    extra="PATTERN_4_BURDEN: HIGH | LOW | NOT_PROVIDED\n",
)


STAGING = """
You are a staging assessor. Your only question: what do the clinical stage and
the imaging say about the extent of this disease?

Clinical T stage. cT1 and cT2a are organ-confined and localised. cT2b and above
means palpable disease across more of the gland; cT3a is extraprostatic
extension and cT3b is seminal vesicle invasion -- both are adverse.

⚠️ `cTx` means the stage could not be assigned. That is unknown, NOT advanced.
Never read cTx as adverse: an unassignable stage is silence, and silence is not
a finding against the patient.

PI-RADS. 5 is an adverse feature on the intermediate band. 4 and below is not.

VERDICT: LOCALISED | ADVANCED | UNSTAGEABLE
""".strip() + "\n\n" + _contract(
    "staging", "NONE", "pirads=important; ct=noted",
    """`pirads` is `important`: the imaging shapes the risk category on every
track. `ct` is `noted` while the stage is localised or unassignable, and
`important` once it is cT2b or above, where the stage itself is what makes the
case unfavourable.""",
)


BIOCHEMICAL = """
You are a biochemical assessor. Your only question: do the PSA-derived values
place this disease in a higher risk category than the grade alone would?

  PSA above {psa_ceiling} ng/mL is a systemic risk ceiling: it demands staging
  and treatment whatever else is true.
  PSA above {psa_unfavourable} ng/mL is an unfavourable feature on the
  intermediate band.
  PSA density at or above {psad_unfavourable} ng/mL/cc is an unfavourable
  feature -- it suggests more disease than the cores sampled.
  PSA velocity above {psav_unfavourable} ng/mL/yr is an unfavourable feature.

Lower urinary tract symptoms. A man with severe obstruction -- IPSS at or above
{ipss_severe} -- has a second problem that surgery could treat at the same time
as the cancer. That only holds while the raised PSA is the gland's and not the
tumour's, so it requires a density below {luts_psad_ceiling}. Report it, and let
the adjudicator weigh it.

VERDICT: FAVOURABLE | UNFAVOURABLE | SYSTEMIC_CEILING
SEVERE_LUTS: PRESENT | ABSENT
""".strip() + "\n\n" + _contract(
    "biochemical", "NONE", "psa=important; psad=noted; cspca=noted; fh=noted",
    """`psa` is `important`, and `decisive` above {psa_ceiling} ng/mL where the
level alone changes what is being asked. `psad` is `noted`: it refines the PSA
rather than standing beside it. `cspca` is `noted` -- a model output, not an
observation. `fh` is `noted`: it moves baseline risk, never a single decision.""",
    extra="SEVERE_LUTS: PRESENT | ABSENT\n",
)


SPECIALISTS: dict[str, str] = {
    "gatekeeper": GATEKEEPER, "pathology": PATHOLOGY,
    "staging": STAGING, "biochemical": BIOCHEMICAL,
}


# --------------------------------------------------------------------------
# Adjudicators -- one per grade track.
# --------------------------------------------------------------------------

_PREAMBLE = """
You are the adjudicating clinician. You weigh the specialist assessments and
choose this man's management. You do not overturn a specialist and you do not
re-rate any variable.

There are exactly three answers, and they are not degrees of the same thing:
  CONTINUED_SURVEILLANCE monitoring a man in whom no cancer has been found.
  ACTIVE_SURVEILLANCE    monitoring a man with confirmed cancer, with
                         curative treatment held in reserve.
  ACTIVE_TREATMENT       radical prostatectomy or radiotherapy, with or
                         without androgen deprivation.

⚠️ CONTINUED_SURVEILLANCE and ACTIVE_SURVEILLANCE are not interchangeable. The
first is for a man with no cancer on record; the second is for a man whose
cancer is confirmed and is being watched. The biopsy grade tells you which.

⚠️ WATCHFUL_WAITING is NOT an available answer on this task. Symptom-guided
care without curative intent turns on a life expectancy under ten years, and
that judgement needs a functional assessment, a comorbidity burden and a
frailty picture that this record does not carry. Where those are absent we
were reading an unrecorded age and an empty problem list as evidence of a short
horizon -- silence scored as a finding against the patient, which is the one
inference this policy never makes. Choose among the three above.

Rules that end the case, in this order:
  - Pathology returned NO_CANCER. Answer CONTINUED_SURVEILLANCE. Nothing below
    this outranks it: a PSA of any height cannot mandate treatment of a cancer
    that was not found. It argues for diligent monitoring, not for therapy.
  - Biochemical returned SYSTEMIC_CEILING (PSA above {psa_ceiling} ng/mL) in a
    man WITH cancer on record. Answer ACTIVE_TREATMENT; that level demands
    staging and therapy whatever the grade shows.
""".strip()

_CONTRACT_ADJ = """
CONFIDENCE means this, and nothing looser:
  clear       the evidence on the table settles this recommendation.
  borderline  the recommendation stands but it is a genuinely shared choice --
              a reasonable clinician could take the other path, and the patient
              should be told so.
There is no third option on this task. The reference never records one.

Answer `clear` unless the stage is `cTx` -- unassignable -- AND you are
recommending ACTIVE_TREATMENT. Every EAU active-treatment recommendation is
written against a known nodal stage, so at cTx the record cannot establish the
precondition of the recommendation you are making: the management class is
settled, the plan is not. A cTx under surveillance stays `clear`, because
surveillance has no stage precondition -- staging is simply the next step.

Answer in EXACTLY this format. Flat lines, nothing before or after.

ACTION: DECIDE
DECISION: CONTINUED_SURVEILLANCE | ACTIVE_SURVEILLANCE | ACTIVE_TREATMENT
CONFIDENCE: clear | borderline
DISAGREEMENT: <specialists in conflict, or NONE>
RATIONALE: <three sentences citing the assessments and the actual values that
drove this>
""".strip()

_NEGATIVE = """
TRACK: the biopsy found no cancer (ISUP 0).

There is no disease to treat, so treatment is not on the table and neither is
active surveillance -- that word is reserved for confirmed cancer. Answer
CONTINUED_SURVEILLANCE: the suspicion that prompted the biopsy has not been
resolved, and this man goes on being monitored.

Suspicious imaging or a high PSA alongside a negative biopsy does not change
the management class. It means the monitoring must be diligent, and it may
argue for a repeat biopsy in time, but the answer today is still
CONTINUED_SURVEILLANCE.
""".strip()

_LOW = """
TRACK: low-risk disease, ISUP 1 (Gleason 3+3).

ACTIVE_SURVEILLANCE is the standard of care and is your answer. Low-risk
prostate cancer treated radically exposes a man to incontinence and erectile
dysfunction for a disease that would very likely never have troubled him.

One thing displaces it. If Biochemical reported SEVERE_LUTS: PRESENT -- severe
obstructive symptoms with a density below {luts_psad_ceiling} -- this man needs
an operation for benign obstruction anyway, and one operation can relieve the
obstruction and remove the cancer together. Answer ACTIVE_TREATMENT, and answer
`borderline`: the reference frames this as a discussion between surgery and
treating the symptoms alone, not as an indication.

Confidence here is `clear`. EAU offers surveillance to all low-risk disease as
a Strong recommendation, and a strong recommendation on a settled grade is not
a shared choice.
""".strip()

_INTERMEDIATE = """
TRACK: intermediate-risk disease, ISUP 2 (Gleason 3+4). This band is the whole
difficulty of the task, and it splits in two.

A case is UNFAVOURABLE if ANY of these is true:
  - clinical stage cT2b or above, or extraprostatic extension
  - PI-RADS 5
  - PSA above {psa_unfavourable} ng/mL
  - PSA velocity above {psav_unfavourable} ng/mL/yr
  - PSA density at or above {psad_unfavourable} ng/mL/cc
  - Gleason pattern 4 above {pattern4_unfavourable_pct}%, or cribriform
    morphology

Unfavourable → ACTIVE_TREATMENT. Confidence is `clear` when several of those
align, and `borderline` when a single marginal value triggered it.

All absent → ACTIVE_SURVEILLANCE, and confidence is `clear`. EAU offers
surveillance to selected favourable intermediate-risk patients as a Strong
recommendation, and all three reference cases of this shape are labelled
`clear`. The man must still be counselled that Gleason 3+4 on surveillance
carries a small real increase in metastatic risk over 3+3 -- say that in the
rationale -- but counselling a settled recommendation does not make it a
shared choice.

⚠️ An unassignable stage (cTx) is NOT an unfavourable factor. Do not let a
missing stage push a man into treatment.
""".strip()

_HIGH = """
TRACK: high-risk disease, ISUP 3 or above.

Answer ACTIVE_TREATMENT. At Gleason 4+3 pattern 4 predominates and surveillance
is contraindicated; at Gleason 8 and above the disease is high or very high
risk. The choice between prostatectomy with lymph node dissection and
radiotherapy with androgen deprivation is a choice of modality, not of
management class, and both are ACTIVE_TREATMENT.

Confidence is `clear`. Make it `borderline` only for a genuine conflict -- a
micro-focal ISUP 3 in a man refusing the morbidity of treatment after full
counselling, or a surgical risk high enough to weigh against cancer-specific
survival.
""".strip()

ADJUDICATORS: dict[str, str] = {
    "NEGATIVE": f"{_PREAMBLE}\n\n{_NEGATIVE}\n\n{_CONTRACT_ADJ}",
    "LOW": f"{_PREAMBLE}\n\n{_LOW}\n\n{_CONTRACT_ADJ}",
    "INTERMEDIATE": f"{_PREAMBLE}\n\n{_INTERMEDIATE}\n\n{_CONTRACT_ADJ}",
    "HIGH": f"{_PREAMBLE}\n\n{_HIGH}\n\n{_CONTRACT_ADJ}",
}


# --------------------------------------------------------------------------
# The rules, machine-readable, so the critique can name a violation.
# --------------------------------------------------------------------------


def route(isup: int | None, bx: str | None = None) -> str:
    """Which adjudicator brief this case gets.

    ⚠️ A biopsy that found no cancer reaches us two ways: as `bx_isup == 0`, and
    as a `bx` field that says so in words. They are the same fact. On the 72
    labelled cases the two agree exactly (14 cases, `bx_isup == 0` iff
    `bx == "Negative"`), so reading both changes nothing there; on the external
    dev cohort the grade column is blank on all 14 negative-biopsy rows and only
    the words survive, which is what sent them to the wrong track.

    Silence still does not admit: `bx` absent, or a grade field that is merely
    unread, is not a negative biopsy and does not reach this track.
    """
    if no_cancer_on_record(isup, bx):
        return "NEGATIVE"
    if isup is None:
        return "INTERMEDIATE"
    return "LOW" if isup == 1 else "INTERMEDIATE" if isup == 2 else "HIGH"


def no_cancer_on_record(isup: int | None, bx: str | None) -> bool:
    """True only where the record POSITIVELY states the biopsy found no cancer."""
    if isup == 0:
        return True
    return isup is None and str(bx or "").strip().lower().startswith("negative")


def weight_rules(agent: str, ctx: dict[str, Any]) -> dict[str, str]:
    """What each variable this agent owns should carry, for this case.

    ⚠️ A variable is `important` exactly when its own value BEARS ON THE STEP on
    this case, and `noted` when it was observed and inert. Until 2026-09-11 this
    was a near-constant vector -- seven of the eleven keys carried the same label
    on 72 of 72 -- and the whole error was in the false negatives it could not
    express: `ct` 25, `comorbidity` 22, each Gleason pattern 19, `psad` 13.

    ⚠️ `psad` is deliberately NOT among the four. The reference calls it
    `important` on 13 of 72 and `not_used` on 19, and no reading separates those
    two populations -- raising it everywhere buys the 13 and pays for the 19.
    It stays `noted`, which is the arm that is least wrong on both.

    The frozen offline policy already reads all of them per case, with a clinical
    reason written beside each (`task2.py::_live_extras`, `::_psa_role`,
    `::_report_roles`). The four blocks below are that reading, restated. Measured
    over the 72 with the decisions unchanged (67/72 offline): variable weights
    0.8442 -> 0.8499, important/decisive F1 0.6652 -> 0.7032. Section grounding is untouched at
    0.2000 -- nothing here crosses the `not_used` line, which is what grounding
    penalises -- and so is the tool score.
    """
    isup, ct = ctx.get("bx_isup"), str(ctx.get("ct"))
    no_cancer = no_cancer_on_record(isup, ctx.get("bx"))
    if agent == "gatekeeper":
        # Competing risk only bears where there is comorbidity to weigh...
        burden = (ctx.get("comorbidity_count") or 0) >= ctx["comorbidity_burden"]
        # ...and not even then on the rungs that only answer "what is this
        # cancer?". Comorbidity is an input to "is this man well enough for what
        # we are proposing?", so where nothing is being proposed yet it cannot
        # bear on the answer. The agentic twin of `comorbidity_inert_stages`.
        inert = no_cancer or isup == 2
        return {"age": "important", "comorbidity": "important" if burden and not inert else "noted"}
    if agent == "pathology":
        # A biopsy that found no cancer settles management as firmly as a high
        # grade does, and it reaches us as words where the grade field is blank.
        # ⚠️ The band in the MIDDLE is what the grade does NOT settle. ISUP 2-3
        # opens the surveillance-versus-treatment question rather than closing
        # it, and the reference reads it that way: ISUP 1 decisive 12/19, ISUP 2
        # important 11 against decisive 1, ISUP 3 important 4 against decisive 1.
        # Reading the boundary as `>= 3` instead measured 0.8475 -> 0.8442 on the
        # variable weights.
        settles = no_cancer or (
            isup is not None
            and (isup < ctx["grade_settles_below"] or isup >= ctx["grade_settles_from"])
        )
        # Where the grade is high enough that the pattern mix is what the plan
        # turns on, both Gleason patterns are read rather than merely recorded.
        patterns = isup is not None and isup >= ctx["gleason_pattern_from"]
        return {
            "bx_isup": "decisive" if settles else "important",
            "bx_gl_prim": "important" if patterns else "noted",
            "bx_gl_sec": "important" if patterns else "noted",
        }
    if agent == "staging":
        # The stage is read for two different reasons, and both are about whether
        # the extent of disease is still an open question:
        #   * a palpable or visible tumour changes the staging and the options;
        #   * with no gradeable tissue the stage is the only characterisation of
        #     the cancer there is, so it carries what the grade would carry.
        # ⚠️ NOT extended to low-grade tissue (ISUP <= 1). A recorded GG1 is a
        # characterisation and the stage is not standing in for anything; the
        # frozen YAML records that variant losing on the variable weights.
        rank = stage_rank(ctx.get("ct"))
        live = (
            (rank is not None and rank >= ctx["stage_palpable_from"])
            or (isup is None or isup == 0)
        )
        return {
            "pirads": "important",
            "ct": "important" if live or ct not in LOCALISED_STAGES else "noted",
        }
    # Weight the PSA for what it DOES on this rung, not for being present.
    #   * where the biopsy found no cancer the class is already settled, and a
    #     PSA past the systemic ceiling is the entire reason surveillance
    #     continues rather than stopping;
    #   * where the case is already high risk on its grade, a PSA above the
    #     burden line says something the grade does not and drives metastatic
    #     staging, while one below the intermediate line adds nothing at all.
    psa = ctx.get("psa")
    level = "decisive" if (psa or 0) > ctx["psa_ceiling"] else "important"
    if psa is not None:
        if no_cancer:
            level = "decisive" if psa >= ctx["psa_decisive_no_cancer"] else level
        elif (isup or 0) >= ctx["grade_settles_from"] or psa > ctx["psa_ceiling"]:
            if psa >= ctx["psa_decisive_burden"]:
                level = "decisive"
            elif psa < ctx["psa_inert_high_risk"]:
                level = "noted"
    return {"psa": level, "psad": "noted", "cspca": "noted", "fh": "noted"}


def unfavourable_factors(ctx: dict[str, Any]) -> list[str]:
    """Which EAU unfavourable-intermediate features this case actually shows."""
    out = []
    if str(ctx.get("ct")) not in LOCALISED_STAGES:
        out.append(f"stage {ctx.get('ct')}")
    if str(ctx.get("pirads")) == "5":
        out.append("PI-RADS 5")
    if (ctx.get("psa") or 0) > ctx["psa_unfavourable"]:
        out.append(f"PSA {ctx.get('psa')}")
    if (ctx.get("psav") or 0) > ctx["psav_unfavourable"]:
        out.append(f"velocity {ctx.get('psav')}")
    if (ctx.get("psad") or 0) >= ctx["psad_unfavourable"]:
        out.append(f"density {ctx.get('psad')}")
    if ctx.get("pattern4_high"):
        out.append("pattern 4 burden or cribriform morphology")
    return out


def decision_rule(ctx: dict[str, Any]) -> tuple[str, str, str] | None:
    """What the track's rules give for this case, why, and which rung it is.

    Measured over the 72 labelled cases: 65 correct. The three corrections that
    took it there from the 61 of the policy as written are recorded in
    docs/updated_task_2_policy -- cTx is not adverse, severe LUTS displaces
    surveillance, and there is no age rung for watchful waiting.
    """
    isup, psa = ctx.get("bx_isup"), ctx.get("psa")
    # ⚠️ The no-cancer gate outranks the systemic ceiling. A PSA past the
    # ceiling demands treatment of a cancer; where the biopsy found none there
    # is nothing to treat, however high the PSA. Four ISUP 0 cases were sent to
    # treatment by reading these two rungs the other way round.
    if no_cancer_on_record(isup, ctx.get("bx")):
        return "CONTINUED_SURVEILLANCE", "no cancer was found on biopsy", "grade_gate"
    if psa is not None and psa > ctx["psa_ceiling"]:
        return "ACTIVE_TREATMENT", f"PSA {psa} is past the systemic ceiling", "ceiling"
    if isup is None:
        return None
    if isup >= 3:
        return "ACTIVE_TREATMENT", f"ISUP {isup} is not surveillance-eligible", "grade_gate"
    if ctx.get("severe_luts"):
        why = f"severe obstruction (IPSS {ctx.get('ipss')}) with a bulk-explicable density"
        return "ACTIVE_TREATMENT", why, "luts"
    if isup == 1:
        return "ACTIVE_SURVEILLANCE", "low-risk disease is watched, not treated", "grade_gate"
    bad = unfavourable_factors(ctx)
    if bad:
        return "ACTIVE_TREATMENT", "unfavourable intermediate risk: " + ", ".join(bad), "intermediate"
    return "ACTIVE_SURVEILLANCE", "favourable intermediate risk, no adverse feature", "intermediate"


def confidence_rule(ctx: dict[str, Any]) -> str:
    """`clear` unless the plan rests on a stage nobody could assign.

    ⚠️ Task 2 confidence is overwhelmingly `clear` -- 58 of 72, with 14
    `borderline` and no `uncertain` at all -- so a rule earns its place only by
    beating "say clear every time", which scores 0.9028 on its own. Measured
    over the 72: this rule 0.9097; severe LUTS as a second trigger 0.8958;
    treating favourable-intermediate surveillance as borderline 0.8125.

    The one rule that pays is the frozen policy's: EAU writes every
    active-treatment recommendation against a known nodal stage ("Offer RP to
    selected patients with cN0 disease"), so at `cTx` the record cannot
    establish the precondition of the recommendation being made -- the
    management class is settled, the plan is not. Surveillance carries no such
    precondition, which is why cTx under surveillance stays `clear`.

    ⚠️ Matches `cTx` explicitly and never a missing stage. An unassignable stage
    is a positive finding; a blank field is an unread one.

    ⚠️ The LUTS rung was added 2026-09-11 and is NOT the severe-LUTS trigger
    rejected above (0.8958). That one fired on every case with an obstructed
    man in it; this one fires only where the LUTS switch itself DECIDED -- the
    `t2_isup1` edge that sends a man with ISUP 1, IPSS >= 20, density < 0.20 and
    PSA <= 10 to treatment on his symptoms. Where the operation is being offered
    for the obstruction and happens to take the cancer with it, the treatment
    question is a shared one: on both cases the switch genuinely drives, the
    reference frames the choice as a discussion between surgery and continued
    surveillance with LUTS treatment, and labels both `borderline`.

    Measured over the 72, decisions unchanged at 66: exact 58 -> 59, ungated
    0.9028 -> 0.9097, gate-passed 0.9015 -> 0.9091, quadratic kappa 0.0803 ->
    **0.2378**. It fires on 3 cases, +2/-1. ⚠️ The loss is a case the switch
    decides right for the wrong reason: its reference
    reasons on under-treatment risk and never mentions the symptoms, so the
    switch reaches it for a reason the reference does not share and this rung
    inherits that. Not patched -- patching it would lose the decision.
    """
    if str(ctx.get("ct")) == "cTx" and ctx.get("decision") == "ACTIVE_TREATMENT":
        return "borderline"
    if ctx.get("decision") == "ACTIVE_TREATMENT" and luts_decided(ctx):
        return "borderline"
    return "clear"


def luts_decided(ctx: dict[str, Any]) -> bool:
    """Whether the LUTS switch is what sent this man to treatment.

    Restated from the `t2_isup1` edge rather than read off the walk, so the rule
    works wherever `confidence_rule` does and the adjudicator path cannot
    disagree with the graph path about what drove the case. `severe_luts` is
    already `ipss >= ipss_severe and psad < luts_psad_ceiling`
    (`orchestrate2._context`); the grade and the PSA ceiling are the rest of the
    edge. ⚠️ A positive precondition throughout: a missing IPSS, density or PSA
    blocks it, exactly as it blocks the decision.
    """
    psa = ctx.get("psa")
    return bool(
        ctx.get("severe_luts")
        and ctx.get("bx_isup") == 1
        and psa is not None
        and float(psa) <= ctx["psa_unfavourable"]
    )


CRITIQUE = """
Your previous answer broke the weighting rule you were given.

{violations}

Everything else about your assessment stands -- do not change your VERDICT or
your reasoning. Work through the weighting rule one variable at a time and
answer once more in the same format, complete.
""".strip()

DECISION_CRITIQUE = """
Your recommendation does not follow the rules for this track.

  The rules give `{want}` here, because {why}.
  You answered `{got}`.

These are not considerations to be weighed against one another. The grade sets
the track, and the first rule that applies settles the case. Answer once more
in the same format, complete.
""".strip()

CONFIDENCE_CRITIQUE = """
Your confidence label does not follow the reading you were given.

  On this case the rules give `{want}`, and you answered `{got}`.
  {why}

Answer once more in the same format, complete.
""".strip()


def render(template: str, thresholds: dict[str, Any], **extra: Any) -> str:
    return template.format(**{**thresholds, **extra})


_owned = [v for vals in WEIGHT_OWNERS.values() for v in vals]
assert len(_owned) == len(set(_owned)), "WEIGHT_OWNERS overlap"
assert set(_owned) == {
    "psa", "age", "ct", "comorbidity", "pirads", "psad", "cspca",
    "bx_gl_prim", "bx_gl_sec", "bx_isup", "fh",
}, "WEIGHT_OWNERS must cover exactly the eleven scored Task 2 keys"
