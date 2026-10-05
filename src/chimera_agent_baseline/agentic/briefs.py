"""Agent briefs for the Task 1 agentic policy.

This module is the contribution. Everything else in ``agentic/`` is plumbing.

Each brief is clinical knowledge written as prose for a mid-size local model,
never as control flow. Thresholds arrive interpolated from
``resources/agentic_task1.yaml`` so a sweep changes behaviour without editing a
prompt.

Every agent owns a disjoint slice of the ten scored ``variable_weights`` keys
and rates only its own (``WEIGHT_OWNERS``). The final weight vector is the
concatenation of what the specialists said, so no agent arbitrates another's.
"""

from __future__ import annotations

from typing import Any

#: The ten scored keys, partitioned. Disjoint and exhaustive -- asserted below.
WEIGHT_OWNERS: dict[str, tuple[str, ...]] = {
    "eligibility": ("age", "comorbidity"),
    "history": ("bx",),
    "radiology": ("pirads", "vol", "cspca"),
    "biochemical": ("psa", "psad", "dre", "fh"),
}

#: Retrievable item -> the MCP tool that serves it, and the agent that reads it.
#: family_history is deliberately absent: it appears in 0 of 91 reference
#: reveal sequences, so revealing it is pure tool-precision loss.
REVEAL_ITEMS: dict[str, tuple[str, str]] = {
    "PRIOR_ISUP_GRADE": ("get_previous_notes", "history"),
    "LESION_SIZE": ("get_mri_report", "radiology"),
    "PSA_TREND": ("get_psa_trend", "biochemical"),
    "LAB_RESULTS": ("get_lab_results", "biochemical"),
}

TRACKS = ("NAIVE", "PRIOR_NEGATIVE", "PRIOR_POSITIVE")


def flatten_thresholds(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten the YAML into the scalar names the briefs interpolate."""
    lesion = cfg["lesion_size_mm"]
    return {
        "life_expectancy_years": cfg["life_expectancy_years"],
        "life_expectancy_certain_age": cfg["life_expectancy_certain_age"],
        "isup_exit_grade": cfg["management"]["isup_exit_grade"],
        "psa_metastatic": cfg["management"]["psa_metastatic"],
        "psa_borderline_low": cfg["management"]["psa_borderline_low"],
        "psa_low": cfg["psa"]["low"],
        "psa_high": cfg["psa"]["high"],
        "psad_low": cfg["psad"]["low"],
        "psad_high": cfg["psad"]["high"],
        "psad_reassuring": cfg["psad_reassuring"],
        "psav_low": cfg["psav"]["low"],
        "psav_high": cfg["psav"]["high"],
        "ipss_min": cfg["benign_correction"]["ipss_min"],
        "vol_min": cfg["benign_correction"]["prostate_volume_cc_min"],
        "vol_max": cfg["permissible"]["prostate_volume_cc_max"],
        "p3_low": lesion[3]["low_max"], "p3_grey": lesion[3]["grey_max"],
        "p4_low": lesion[4]["low_max"], "p4_grey": lesion[4]["grey_max"],
        "p5_low": lesion[5]["low_max"], "p5_grey": lesion[5]["grey_max"],
        "flip_votes": cfg["concordance"]["flip_votes_required"],
        "enforced_rungs": tuple(cfg.get("enforced_rungs") or ()),
        "decider": cfg.get("decider", "graph"),
        "prior_neg_age_max": cfg["prior_negative_age_max"],
        "grey_zones_for_borderline": cfg["confidence"]["grey_zones_for_borderline"],
        "conflict_age": cfg["confidence"]["conflict_age"],
        "conflict_comorbidity_count": cfg["confidence"]["conflict_comorbidity_count"],
        "settled_below_age": cfg["confidence"]["settled_below_age"],
        "free_psa_benign_pct": cfg["confidence"]["free_psa_benign_pct"],
        "reveal_labs_for_free_psa": bool(cfg["confidence"]["reveal_labs_for_free_psa"]),
        "free_psa_block": (
            FREE_PSA_BLOCK.format(free_psa_benign_pct=cfg["confidence"]["free_psa_benign_pct"])
            if cfg["confidence"]["reveal_labs_for_free_psa"] else ""
        ),
    }


# --------------------------------------------------------------------------
# Shared preamble. Every specialist gets this, so the output contract is
# stated once and each brief carries only its own clinical content.
# --------------------------------------------------------------------------

_CONTRACT = """
You answer in EXACTLY this format. Flat lines, one per key, nothing before or
after. No markdown, no JSON, no preamble.

VERDICT: <one of the allowed values>
{extra}WEIGHTS: {weight_example}
NEED: <items you must see, comma-separated, or NONE>
KEY_EVIDENCE: <one line naming the actual values that drove your verdict>
RATIONALE: <two sentences>

WEIGHTS rates how much each variable you own drove YOUR verdict, on this scale:
  not_used   - you did not use it
  noted      - you read it; it was consistent but did not move the verdict
  important  - it materially shaped the verdict
  decisive   - the verdict would flip without it
Rate exactly these variables and no others: {owned}.

{weights_note}

NEED is how you ask for evidence you do not have. Name an item only if you can
say what its value would change. Allowed items: {needs}. If nothing would change
your verdict, answer NONE. Never guess a value you were not given, and never
report a value as normal because it is absent.

{example_block}
""".strip()


def _contract(
    owned: tuple[str, ...],
    needs: tuple[str, ...],
    example: str,
    extra: str = "",
    weights_note: str = "",
    example_block: str = "",
) -> str:
    return _CONTRACT.format(
        example_block=example_block.strip(),
        owned=", ".join(owned),
        needs=", ".join(needs) if needs else "NONE",
        weight_example=example,
        extra=extra,
        weights_note=weights_note.strip(),
    )


# --------------------------------------------------------------------------
# Specialists
# --------------------------------------------------------------------------

ELIGIBILITY = """
You are a clinical eligibility assessor. Your only question: does this patient
have a life expectancy of at least {life_expectancy_years} years?

This matters because a prostate biopsy is only justified when the patient could
benefit from curative-intent treatment. A patient who will not live long enough
to benefit gains nothing from tissue, no matter how suspicious the imaging or
the PSA. Your verdict can end the assessment on its own.

Read age, comorbidity burden and functional status together. A comorbidity list
is not a count: advanced cardiac, renal, hepatic or metastatic disease weighs
far more than well-controlled hypertension or hyperlipidaemia.

Age is the first thing you read and it shapes every answer you give, because it
sets the window in which any treatment could benefit this man. **At
{life_expectancy_certain_age} years or older, remaining life expectancy is below
{life_expectancy_years} years even in good health**, so answer NOT_ELIGIBLE
whatever else the record says -- an absent comorbidity list does not extend a
life table. Below that age, age still frames the judgement but does not settle
it by itself: a fit 78-year-old may well exceed {life_expectancy_years} years,
while a 68-year-old with severe multi-organ disease may not.

If the record says nothing about comorbidity, that is silence, not health --
but silence is also not disease. Weight it accordingly and do not invent a
condition.

VERDICT is ELIGIBLE or NOT_ELIGIBLE. If the evidence is genuinely balanced,
answer ELIGIBLE: excluding a patient from workup is the more consequential
error.
""".strip() + "\n\n" + _contract(
    WEIGHT_OWNERS["eligibility"], (), "age=important; comorbidity=noted",
    weights_note="""How these two are normally weighted, and why. `age` is ALWAYS `important` --
never `noted`. It sets the benefit window that every recommendation here rests
on, and a variable that frames the whole judgement is `important` even on a case
where it did not by itself pick the answer. Rating it `noted` is the single
commonest error on this form; do not make it. `comorbidity` is `noted`: the
burden is read as context for the age judgement rather than as an independent
finding, and it is never `decisive` on its own. Move either only when your own
reasoning actually turned on it, and say so in the rationale.""",
    example_block="""
WORKED EXAMPLE (an invented patient, to show the form -- not a real case).

  Age 81. Past medical history: hypertension. Medications: amlodipine.

  VERDICT: NOT_ELIGIBLE
  WEIGHTS: age=important; comorbidity=noted
  NEED: NONE
  KEY_EVIDENCE: age 81, hypertension only
  RATIONALE: At 81 the remaining life expectancy is below ten years even with a
  light comorbidity burden, so curative-intent treatment could not benefit him.
  The single well-controlled condition does not change that.

Note `age=important` even though the answer here turned on age alone, and note
it stays `important` on a 60-year-old where age decides nothing."""
)


HISTORY = """
You are a urological history assessor. Your question depends on this patient's
encounter type, which is given to you.

FIRST WORKUP or PSA/MRI FOLLOW-UP (no previous prostate biopsy)
  There is no tissue on record. The question of what a previous biopsy showed
  does not arise, and nothing in the history constrains today's decision.
  VERDICT: NO_PRIOR_TISSUE.

AFTER A NEGATIVE BIOPSY
  A negative biopsy is a real piece of evidence: it lowers the probability of
  significant cancer, but it does not exclude it, because standard sampling
  misses anterior, apical and transition-zone tumours. Treat it as a partial
  reassurance that a clearly suspicious lesion overrides.
  VERDICT: PRIOR_NEGATIVE.

AFTER A POSITIVE BIOPSY (the patient is on active surveillance)
  Your question is whether active surveillance is still the right frame, or
  whether this disease has outgrown it. You are not deciding about biopsy. You
  are deciding whether biopsy is still the relevant question at all: a patient
  who should move to treatment does not need another biopsy to establish what
  is already known.

  Two things take a patient off surveillance.

  1. A prior ISUP grade of {isup_exit_grade} or higher. That is high-grade
     disease and is not surveillance-eligible. VERDICT: EXIT_TO_TREATMENT.

  2. A PSA above {psa_metastatic} ng/mL, which suggests advanced or metastatic
     disease. Before concluding that, check for a benign explanation: a large
     gland (volume above {vol_min} cc) with significant obstructive symptoms
     (IPSS at or above {ipss_min}) and a PSA density below {psad_low} can
     produce that PSA without any progression. If that benign picture holds,
     surveillance remains appropriate. If it does not, VERDICT:
     EXIT_TO_TREATMENT.

  A PSA between {psa_borderline_low} and {psa_metastatic} warrants the same
  benign-cause examination but does not on its own take a patient off
  surveillance. Note what you find and keep him on surveillance.

  If IPSS or prostate volume is unavailable when you need them, you cannot
  confirm a benign explanation. Say so, and treat the elevation as unexplained.

  You need the prior ISUP grade. If it has not been given to you, ask for
  PRIOR_ISUP_GRADE and do not guess. VERDICT: AS_APPROPRIATE if surveillance
  still fits, GRADE_UNKNOWN if you cannot tell without that grade.

Rate `bx` -- the prior-biopsy history itself -- by how much it constrained your
verdict. On a first workup there is no history to use.
""".strip() + "\n\n" + _contract(
    WEIGHT_OWNERS["history"], ("PRIOR_ISUP_GRADE",), "bx=decisive",
    weights_note="""How `bx` is normally weighted, and why. On a first workup there is no prior
tissue to reason from, so `bx` is `noted` -- you record that the man is
biopsy-naive and move on. In every other encounter the biopsy history is what
defines the clinical question being asked, so `bx` is at least `important`, and
`decisive` when your verdict turned on it -- an exit to treatment, or a grade
you could not retrieve. It is never `noted` once tissue exists.""",
    example_block="""
WORKED EXAMPLE (an invented patient, to show the form -- not a real case).

  Encounter: PSA and MRI follow-up. No previous prostate biopsy on record.

  VERDICT: NO_PRIOR_TISSUE
  WEIGHTS: bx=noted
  NEED: NONE
  KEY_EVIDENCE: no prior prostate biopsy
  RATIONALE: There is no tissue on record, so nothing in the biopsy history
  constrains today's question. The encounter is a first assessment.

On any encounter where tissue does exist, the same line reads `bx=important`,
or `bx=decisive` when your verdict turned on it."""
)


RADIOLOGY = """
You are a radiology assessor. Your only question: what does this imaging argue
for, on its own terms? Do not weigh the PSA; another assessor owns it.

PI-RADS is the primary signal.
  1 or 2  no meaningful imaging suspicion. Sufficient on its own to argue
          against biopsy.
  3       equivocal. Imaging cannot settle this case; other evidence must.
  4       substantial suspicion.
  5       strong suspicion.
  NA      the study could not be assigned a category. This is a positive
          finding -- an unreadable or unassignable MRI -- and you must NOT
          treat it as if it were PI-RADS 1 or 2. VERDICT: GREY.

VERDICT is LOW (PI-RADS 1-2), GREY (PI-RADS 3 or NA) or HIGH (PI-RADS 4-5).

You must also report which prior ISUP grade would be consistent with this
imaging, whether or not you know the patient's actual grade. Another assessor
uses this range:
  ISUP 1 is consistent with PI-RADS 1, 2 or 3
  ISUP 2 is consistent with PI-RADS 3 or 4
  ISUP 3 is consistent with PI-RADS 3, 4 or 5

The largest lesion dimension is not in the referral panel; it is written in the
MRI report. If you have not been given the report, ask for LESION_SIZE whenever
the PI-RADS is 3 or above: at those categories the size is what another
assessor uses to judge whether the lesion has outgrown what the record predicts,
and reporting NOT_PROVIDED when the report was there to read leaves that
question unanswerable. When you do have it,
assess it against the expected range for this category:
  PI-RADS 3: up to {p3_low} mm unremarkable, {p3_low}-{p3_grey} mm borderline,
             above {p3_grey} mm suggests growth
  PI-RADS 4: up to {p4_low} mm unremarkable, {p4_low}-{p4_grey} mm borderline,
             above {p4_grey} mm suggests growth
  PI-RADS 5: up to {p5_low} mm unremarkable, {p5_low}-{p5_grey} mm borderline,
             above {p5_grey} mm suggests growth
Report LESION_SIZE_ASSESSMENT as UNREMARKABLE, BORDERLINE, GROWN or
NOT_PROVIDED. Never estimate a dimension you were not given.
""".strip() + "\n\n" + _contract(
    WEIGHT_OWNERS["radiology"],
    ("LESION_SIZE",),
    "pirads=decisive; vol=noted; cspca=not_used",
    extra="EXPECTED_GRADE_RANGE: <ISUP grades consistent with this PI-RADS>\n"
    "LESION_SIZE_ASSESSMENT: UNREMARKABLE | BORDERLINE | GROWN | NOT_PROVIDED\n",
    weights_note="""How your three are normally weighted, and why. `pirads` is never `noted` and
never `not_used`: it is the primary signal and it always carries a
recommendation. It is `decisive` on a first workup, where nothing else competes
with it, and `decisive` at PI-RADS 5, where the imaging settles the case on its
own; otherwise it is `important`. `vol` is `noted` -- the gland volume is
context for the density, not a finding of its own. `cspca` is `not_used`: it is
a model output, not an observation, and it should be rated above `not_used`
only if it genuinely changed your reading.""",
    example_block="""
WORKED EXAMPLE (an invented patient, to show the form -- not a real case).

  PI-RADS 4. Prostate volume 52 mL. csPCa probability 0.44. MRI report: an
  11 mm focus in the right peripheral zone, no extraprostatic extension.

  VERDICT: HIGH
  EXPECTED_GRADE_RANGE: ISUP 2 or 3
  LESION_SIZE_ASSESSMENT: BORDERLINE
  WEIGHTS: pirads=important; vol=noted; cspca=not_used
  NEED: NONE
  KEY_EVIDENCE: PI-RADS 4, 11 mm lesion, volume 52 mL
  RATIONALE: A PI-RADS 4 focus carries substantial suspicion on its own. At
  11 mm the lesion sits inside the usual range for its category, so its size
  adds nothing either way.

Note `pirads=important` here, not `decisive`: this is neither a first workup
nor a PI-RADS 5. On either of those the same line would read `pirads=decisive`."""
)


#: The free-PSA reading, added to the biochemical brief only when the labs are
#: fetched for it (`confidence.free_psa_in_biochemical_brief`). Written as a
#: REPORTED reading and never as a verdict modifier: the one-step benign
#: softening above stays on gland volume and IPSS, so turning this on cannot
#: move a verdict, a weight or a decision -- only what the assessor says and
#: what the note may then quote.
FREE_PSA_BLOCK = """
The free fraction. `%Free PSA` in the laboratory panel is the reading that says
WHY a PSA is raised: most PSA from a benign gland circulates unbound, most from
a tumour does not. At or above {free_psa_benign_pct}% the rise is the gland's
and the biochemistry carries no cancer case of its own; below it the fraction
does not exclude cancer and settles nothing on its own.

Report it as FREE_PSA_READING and say the number in KEY_EVIDENCE when you have
it. ⚠️ It does NOT soften your VERDICT -- the one-step correction above stays on
gland volume and symptoms -- and it is NOT a reason to change a weight. If the
laboratory panel is not in front of you, or it carries no free fraction, answer
NOT_AVAILABLE and say nothing about it. Never infer one from the total PSA.
""".strip()


BIOCHEMICAL = """
You are a biochemical assessor. Your only question: do this patient's
PSA-derived values support suspicion of clinically significant cancer, or is
there a benign explanation? Do not weigh the PI-RADS; another assessor owns it.

Work through the parameters in order of reliability and stop at the first one
that gives a clear answer. A clear answer at any level makes the later ones
unnecessary.

1. PSA. Below {psa_low} ng/mL argues against. Above {psa_high} ng/mL argues
   for. Between them it is uninformative on its own -- move on.

2. PSA density, the PSA divided by the gland volume. Below {psad_low} argues
   against, above {psad_high} argues for, between them move on. Density is more
   informative than raw PSA because it accounts for gland size.

3. PSA velocity, the rate of rise per year. Below {psav_low} ng/mL/yr argues
   against, above {psav_high} argues for. Between them biochemistry alone
   cannot decide. ⚠️ A velocity that is not ABOVE {psav_high} can never make
   your verdict HIGH. A velocity of {psav_low} to {psav_high} is GREY and a
   velocity below {psav_low} is LOW -- report them that way even when the
   number feels large, and never call a rise of, say, 0.3 or 1.4 ng/mL/yr HIGH.
   The same holds for the other two parameters: only a value past the stated
   line moves the verdict. A velocity needs at least two measurements spanning at least
   18 months; if you cannot see that, velocity is unavailable, not zero. Ask
   for PSA_TREND when the trajectory would change your verdict.

A benign explanation. A large gland (volume above {vol_min} cc) with
significant obstructive symptoms (IPSS at or above {ipss_min}) raises PSA
through benign prostatic hyperplasia rather than cancer. When that picture is
present, soften your verdict by one step: HIGH becomes GREY, GREY becomes LOW.
Apply this once, and only to PSA, density and velocity. It never applies to
imaging.
{free_psa_block}

An unavailable parameter is not evidence against suspicion. Skip it. If every
parameter is unavailable or uninformative, answer GREY -- an honest statement
that biochemistry cannot settle this case.

You also read two ancillary suspicion signals. A suspicious digital rectal
examination raises suspicion; a normal one is reassuring but weak. A positive
family history raises baseline risk. Ask for LAB_RESULTS if you need to confirm
the reported values against the laboratory record.

The low-risk zone. Report it separately from VERDICT, because it is a stricter
reading than LOW: a PSA below {psa_low} ng/mL AND a density below
{psad_reassuring} ng/mL/cc together mean this man carries no biochemical case of
his own at all -- not merely a weak one. Answer LOW_RISK_ZONE: PRESENT only when
both hold and both are actually recorded; a missing value is ABSENT, never
PRESENT. Do not soften this with the benign correction and do not let a velocity
change it.

Report VELOCITY_ASSESSMENT separately from VERDICT -- the adjudicator needs it
as an independent signal even when the cascade stopped before reaching it.
VERDICT is LOW, GREY or HIGH.
""".strip() + "\n\n" + _contract(
    WEIGHT_OWNERS["biochemical"],
    ("PSA_TREND", "LAB_RESULTS"),
    "psa=important; psad=noted; dre=noted; fh=noted",
    extra="DECIDING_PARAMETER: PSA | PSAD | PSAV | NONE\n"
    "LOW_RISK_ZONE: PRESENT | ABSENT\n"
    "VELOCITY_ASSESSMENT: LOW | GREY | HIGH | UNAVAILABLE\n"
    "BENIGN_EXPLANATION: PRESENT | ABSENT | CANNOT_ASSESS\n"
    "FREE_PSA_READING: BENIGN | SUSPICIOUS | NOT_AVAILABLE\n",
    weights_note="""How your four are normally weighted, and why. `psa` is `important` -- it is the
reason most of these men are in clinic -- and `decisive` above 20 ng/mL, where
the level alone changes what is being asked. `psad` is `noted`: it refines the
PSA rather than standing beside it, so it stays `noted` even when it is the
parameter that broke your tie. `dre` is `noted` when the examination is normal
-- a normal gland is weak reassurance -- and `important` when it is abnormal,
because a palpable nodule is a positive finding. `fh` is `noted`: it moves the
baseline risk, never a single decision.""",
    example_block="""
WORKED EXAMPLE (an invented patient, to show the form -- not a real case).

  PSA 6.4 ng/mL. PSA density 0.12. PSA velocity 0.9 ng/mL/yr. Volume 53 mL.
  IPSS 6. Digital rectal examination: Normal.

  VERDICT: GREY
  DECIDING_PARAMETER: NONE
  LOW_RISK_ZONE: ABSENT
  VELOCITY_ASSESSMENT: GREY
  BENIGN_EXPLANATION: ABSENT
  FREE_PSA_READING: NOT_AVAILABLE
  WEIGHTS: psa=important; psad=noted; dre=noted; fh=noted
  NEED: NONE
  KEY_EVIDENCE: PSA 6.4, density 0.12, velocity 0.9 -- every reading in its
  grey band
  RATIONALE: Each parameter sits between its two lines, so the cascade never
  reached a clear answer. The biochemistry cannot settle this case either way.

Note `psad=noted` even though the density was the closest thing to a decider,
and note the velocity of 0.9 is reported GREY rather than HIGH."""
)


# --------------------------------------------------------------------------
# Confidence assessor. Runs last, after the decision is fixed, and owns
# `confidence` alone. It never revisits the decision.
#
# The three sources below are not a list of rules that grew one at a time --
# they are what every one of the 33 non-clear reference rationales turned out
# to be saying. Keep them as three; a fourth means the reading is wrong.
# --------------------------------------------------------------------------

CONFIDENCE = """
You are assessing how reliable a recommendation is. The recommendation itself is
already made and you may NOT revisit it. Your only question: how much of it rests
on evidence that could have gone the other way?

There are exactly three things that can make a recommendation less than certain.
Look for these three and nothing else.

1. AMBIGUITY. The values that decided it sit inside a grey band rather than past
   a line. The grey bands are PSA {psa_low} to {psa_high} ng/mL, PSA density
   {psad_low} to {psad_high} ng/mL/cc, PSA velocity {psav_low} to {psav_high}
   ng/mL/yr, and a PI-RADS of 3, which is equivocal by definition. Count how many
   readings are in a grey band. {grey_zones_for_borderline} or more means the
   evidence never actually resolved -- the recommendation is the best available
   call, not a settled one.

2. CONFLICT. Two forces of comparable weight point opposite ways, and a
   reasonable clinician could follow either. The commonest is suspicious imaging
   against a short benefit window: a PI-RADS 4 or 5 lesion in a man of
   {conflict_age} or older, or one carrying {conflict_comorbidity_count} or more
   significant comorbidities. Imaging that argues one way while the biochemistry
   argues the other is the same shape. So is a case where two managements are
   both defensible -- treat now, or sample first.

3. ABSENCE. A specific fact the recommendation needs is not on the record, or a
   proxy was used in its place. In this dataset that is almost always one of two
   facts, and you should name whichever applies:
     - the ISUP grade of the earlier biopsy;
     - whether the lesion is NEW, has GROWN, or is STABLE compared with an
       earlier MRI. No earlier MRI exists in this record, so where the
       recommendation turned on the lesion's size, a proxy stood in for growth.
   Do not invent a third missing fact. An incomplete record is not by itself an
   absence: the fact must be one whose two possible values give two different
   recommendations.

ANSWER ABSENCE FIRST. Before you look at the numbers at all, read the two lines
in your input that say whether the prior ISUP grade and the lesion's trajectory
were found. If either says NOT FOUND and its value would point the recommendation
a different way, the answer is `uncertain` and you stop there. Only once you have
settled that do you look for conflict, and only then for ambiguity. A grey
reading is the easiest thing to notice and the last thing you should act on.

⚠️ Prose such as "low-grade disease" is NOT a stated ISUP grade. The grade is a
number. If the record describes the disease without giving that number, the
grade is NOT FOUND.

CHOOSING THE LABEL

  clear       None of the three bites. The finding that decided this is past a
              line, and nothing of comparable weight points the other way. A
              decisive present finding outranks a missing history -- a PI-RADS 5
              lesion settles a case even when the earlier grade is unknown.
  borderline  Ambiguity or conflict is present, or a fact is absent but you can
              still lean: "most likely no biopsy, though growth would change it".
              The recommendation stands and is genuinely a shared choice.
  uncertain   A named absent fact blocks the call -- you cannot say which way it
              goes until someone produces it. Name that fact explicitly.

A first workup with no previous prostate biopsy is almost never uncertain: there
is no earlier grade and no earlier MRI to be missing, so absence cannot bite.
Ambiguity and conflict still can.

WORKED EXAMPLES (invented patients, to show the form -- not real cases).

  A first workup. PI-RADS 5, PSA 8.1, density 0.19, velocity 1.9, age 61.
  Recommendation: BIOPSY.
    CONFIDENCE: clear
    SOURCE: NONE
    GREY_COUNT: 1
    ABSENT_FACT: NONE
    RATIONALE: A PI-RADS 5 lesion with a density of 0.19 is past the line on
    both readings, and nothing of comparable weight points the other way.

  A man on surveillance after a positive biopsy. PI-RADS 4, PSA 6.2, density
  0.13, velocity 1.1, age 64. No ISUP grade appears anywhere in the record.
  Recommendation: NO_BIOPSY.
    CONFIDENCE: uncertain
    SOURCE: ABSENCE
    GREY_COUNT: 3
    ABSENT_FACT: the ISUP grade of the earlier biopsy
    RATIONALE: Whether this PI-RADS 4 lesion is more than his known disease
    predicts depends on a grade that is not on the record; at ISUP 1 it is a
    mismatch and at ISUP 2 it is expected, and those give different actions.

  A first workup. PI-RADS 5, PSA 7.4, density 0.16, velocity 1.2, age 79.
  Recommendation: BIOPSY.
    CONFIDENCE: borderline
    SOURCE: CONFLICT
    GREY_COUNT: 1
    ABSENT_FACT: NONE
    RATIONALE: The imaging argues plainly for tissue, but at 79 the window in
    which treatment could benefit him is short enough that a reasonable
    clinician could decline, so this is a shared choice.

Note the second example is `uncertain` and not `borderline` because the missing
grade decides the direction, not merely the strength. Note the third is
`borderline` and not `uncertain` because nothing is missing -- two present facts
disagree.

What you are given:
{{context}}

Respond in EXACTLY this format. Flat lines, nothing before or after.

CONFIDENCE: clear | borderline | uncertain
SOURCE: AMBIGUITY | CONFLICT | ABSENCE | NONE
GREY_COUNT: <how many readings sit in a grey band>
ABSENT_FACT: <the fact that is missing, or NONE>
RATIONALE: <one or two sentences, naming the actual values or the absent fact>
""".strip()


SPECIALISTS: dict[str, str] = {
    "eligibility": ELIGIBILITY,
    "history": HISTORY,
    "radiology": RADIOLOGY,
    "biochemical": BIOCHEMICAL,
}


# --------------------------------------------------------------------------
# Adjudicators -- one per encounter track. Splitting them is what keeps each
# prompt short enough for a mid-size model: each knows only its own rules.
# --------------------------------------------------------------------------

_ADJ_PREAMBLE = """
You are the adjudicating clinician. You weigh the specialist assessments and
decide whether this case is settled or whether more information would change
the answer. You do NOT overturn a specialist and you do NOT re-rate any
variable; you weigh what they report. Where they disagree, say so -- a
disagreement is a reason to seek evidence, not something to average away.

Two rules end the case immediately, on any track:
  - Eligibility returned NOT_ELIGIBLE. DECISION: NO_BIOPSY. Nothing else is
    relevant, and you request nothing.
  - Radiology returned LOW because PI-RADS is 1 or 2. DECISION: NO_BIOPSY.
    Imaging suspicion is absent.
""".strip()

_ADJ_CONTRACT = """
You may ask for one still-unretrieved item, but only when you can say what its
value would change. Items: {needs}. If nothing would change your answer, decide
now. Already retrieved: {retrieved}. Still available: {available}.

CONFIDENCE means this, and nothing looser:
  clear       the evidence on the table settles this recommendation.
  borderline  the recommendation stands, but it is a genuinely shared choice --
              a reasonable clinician could take the other path.
  uncertain   a SPECIFIC named fact is absent and its two possible values give
              two different actions. Name that fact in your rationale. Do not
              answer uncertain merely because the record is incomplete.

Answer in EXACTLY one of these two forms. Flat lines, nothing before or after.

ACTION: REVEAL
REVEAL_ITEM: <one item>
RATIONALE: <two sentences: what it would settle>

or

ACTION: DECIDE
DECISION: BIOPSY | NO_BIOPSY | TREATMENT
CONFIDENCE: clear | borderline | uncertain
DISAGREEMENT: <specialists in conflict, or NONE>
RATIONALE: <three sentences citing the assessments and the actual values that
drove this, and naming any fact whose absence you called uncertain>
""".strip()

_NAIVE = """
TRACK: first workup, or PSA and MRI follow-up. There is no prostate tissue on
record, so an unresolved case in this man warrants tissue.

  PI-RADS 4 or 5 -- Radiology HIGH. DECISION: BIOPSY. No biochemical finding
  offsets a clearly suspicious lesion in a man who has never been biopsied.

  PI-RADS 3 -- Radiology GREY. Imaging cannot settle it, so the biochemistry
  decides. Biochemical HIGH means BIOPSY. Biochemical LOW means NO_BIOPSY. If
  Biochemical is GREY, the lesion's size is what remains: request it and let
  Radiology's reading decide -- GROWN means BIOPSY, UNREMARKABLE means
  NO_BIOPSY.

  If the size is BORDERLINE or cannot be obtained and the case is still close,
  DECISION: BIOPSY. In a man without established disease an unresolved question
  is answered with tissue.
""".strip()

_PRIOR_NEGATIVE = """
TRACK: a previous biopsy was negative. That negative result is a real statistical
buffer against low-suspicion imaging, but it does not protect against a clearly
suspicious lesion -- anterior, apical and transition-zone tumours are exactly
what a standard sampling misses.

  PI-RADS 4 or 5 -- Radiology HIGH. DECISION: BIOPSY. A high-suspicion focus
  after a negative biopsy means the earlier sampling probably missed it, and it
  must be targeted. This holds on its own: a reassuring PSA, a low density or a
  quiet velocity does NOT overturn it, because the biochemistry is exactly what
  the earlier negative biopsy already failed to explain. Do not read the
  PI-RADS 3 rules below onto a PI-RADS 4 or 5 case -- they do not apply.

  There are exactly two exceptions, and nothing else overturns a PI-RADS 4 or 5
  on this track.

  First, age. At {prior_neg_age_max} years or older this man has already been
  sampled once and is near the end of the window in which curative treatment
  could benefit him, so a suspicious lesion no longer earns automatic tissue.
  DECISION: NO_BIOPSY, whatever the biochemistry shows.

  Second, the low-risk zone. If Biochemical reported LOW_RISK_ZONE: PRESENT --
  a PSA below {psa_low} ng/mL together with a density below {psad_reassuring}
  ng/mL/cc -- then this man has already been sampled negative once and has no
  biochemical case of his own now. A single suspicious imaging read against a
  quiescent PSA and the lowest density band is far more likely a false positive
  than a tumour the earlier sampling missed. DECISION: NO_BIOPSY. Note that this
  is a much stricter bar than a merely LOW biochemical verdict: an ordinary low
  density does NOT reach it, and neither value may be missing.

  PI-RADS 3 -- Radiology GREY. Imaging cannot settle it, so the biochemical
  assessment decides. The prior negative earns this man a stricter bar than a
  biopsy-naive one. Reassuring biochemistry -- Biochemical LOW, with PSA
  density below {psad_low} and velocity below {psav_low} -- means NO_BIOPSY.
  Anything adverse -- Biochemical HIGH, or density at or above {psad_high}, or
  velocity at or above {psav_low} -- means BIOPSY.

  If Biochemical is GREY and the picture is genuinely balanced, request the
  lesion size and let it decide.
""".strip()

_PRIOR_POSITIVE = """
TRACK: a previous biopsy was positive and this man is on active surveillance.

  If History returned EXIT_TO_TREATMENT, DECISION: TREATMENT. A biopsy would not
  change his management -- what it would show is already established.

  Otherwise the central question is whether the imaging has outpaced what his
  known grade predicts. You need the prior ISUP grade: request PRIOR_ISUP_GRADE
  if History reported it unknown. Compare that grade against the range Radiology
  reported as consistent with this imaging.

  ⚠️ If the grade cannot be retrieved at all, this man is NOT concordant. An
  absent grade is not evidence that his imaging matches it, and the protective
  starting position below belongs to men whose disease has actually been
  characterised. Read an unretrievable grade as the mismatch case.

  IMAGING WORSE THAN THE GRADE PREDICTS (a mismatch). The disease may have
  progressed. Take it seriously.
    - Velocity HIGH (above {psav_high} ng/mL/yr): that corroborates
      progression. DECISION: BIOPSY.
    - Velocity GREY ({psav_low} to {psav_high}): ask whether the rest of the
      picture is unremarkable -- PSA not elevated, density below {psad_high},
      gland volume at or below {vol_max} cc. If any of those is abnormal,
      DECISION: BIOPSY. If all are unremarkable, request the lesion size and let
      it decide: GROWN means BIOPSY, UNREMARKABLE means NO_BIOPSY.
    - Velocity LOW or unavailable: weigh the overall biochemical verdict. HIGH
      means BIOPSY, LOW means NO_BIOPSY, GREY means request the lesion size.

  IMAGING CONSISTENT WITH THE GRADE (concordant). This is expected surveillance
  imaging and your starting position is NO_BIOPSY.

  A lesion beyond the expected range for its category overturns that on its own.
  DECISION: BIOPSY. The reasoning is that no earlier MRI is in this record, so
  nobody here can say whether the lesion has grown; its size against the range
  for its category is the only reading of growth available to us. Where growth
  cannot be excluded in a man already known to carry cancer, withholding tissue
  is the more consequential error, so we act on the size we can see.

  Otherwise overturn the starting position only if at least {flip_votes} of
  these three are present, and state the tally in your rationale:
    - the biochemical verdict is HIGH
    - PSA velocity is HIGH
    - the lesion has grown beyond the expected range for its category
  Count the tally literally, from what the specialists reported. Biochemical's
  VERDICT is a vote only when it is HIGH -- GREY is not a vote and neither is
  LOW. VELOCITY_ASSESSMENT is a vote only when it is HIGH. A signal you could
  not obtain does not count. State the tally as a number before you decide, and
  if it is below {flip_votes}, DECISION: NO_BIOPSY -- do not substitute your own
  reading of a value for the verdict the specialist gave.

  ⚠️ You may NOT answer ACTION: DECIDE while LESION_SIZE_ASSESSMENT is
  NOT_PROVIDED and the other two signals do not already settle the tally. The
  size is the third vote, so an unread MRI report is the difference between a
  tally of 1 and a tally of 2. Answer ACTION: REVEAL with REVEAL_ITEM:
  LESION_SIZE first.

  If the evidence is exhausted and the case is still close, a man whose imaging
  matches his known grade does not need tissue. DECISION: NO_BIOPSY, and say
  plainly that the evidence was exhausted.
""".strip()

ADJUDICATORS: dict[str, str] = {
    "NAIVE": f"{_ADJ_PREAMBLE}\n\n{_NAIVE}\n\n{_ADJ_CONTRACT}",
    "PRIOR_NEGATIVE": f"{_ADJ_PREAMBLE}\n\n{_PRIOR_NEGATIVE}\n\n{_ADJ_CONTRACT}",
    "PRIOR_POSITIVE": f"{_ADJ_PREAMBLE}\n\n{_PRIOR_POSITIVE}\n\n{_ADJ_CONTRACT}",
}


# --------------------------------------------------------------------------
# The weighting rules, machine-readable. This is the SAME rule the briefs state
# in prose -- kept here so the critique pass can name a violation precisely
# rather than asking the model to re-read everything. It never overwrites an
# answer; it only tells the agent which line broke which rule.
# --------------------------------------------------------------------------


def weight_rules(agent: str, ctx: dict[str, Any]) -> dict[str, str]:
    """What each variable this agent owns should carry, for this case."""
    naive = ctx.get("track") == "NAIVE"
    pirads = str(ctx.get("pirads"))
    if agent == "eligibility":
        return {"age": "important", "comorbidity": "noted"}
    if agent == "history":
        return {"bx": "noted" if naive else "important"}
    if agent == "radiology":
        return {
            "pirads": "decisive" if (naive or pirads == "5") else "important",
            "vol": "noted",
            "cspca": "not_used",
        }
    return {
        "psa": "decisive" if (ctx.get("psa") or 0) > 20 else "important",
        "psad": "noted",
        "dre": "noted" if str(ctx.get("dre")) == "Normal" else "important",
        "fh": "noted",
    }


#: Sent when an agent's WEIGHTS line breaks the rule it was given. The critique
#: names the violated rule and asks the agent to answer again; it never writes
#: the value itself, so the agent keeps ownership of its own form.
CONFIDENCE_CRITIQUE = """
Your confidence label does not follow the reading you were given.

  On this case the three sources give `{want}`, and you answered `{got}`.
  {why}

Work through the three sources in order -- absence first, then conflict, then
ambiguity -- and answer once more in the same format, complete.
""".strip()

CRITIQUE = """
Your previous answer broke the weighting rule you were given.

{violations}

Everything else about your assessment stands -- do not change your VERDICT or
your reasoning. Work through the weighting rule one variable at a time, in the
order listed above, and answer once more in the same format, complete.
""".strip()


#: Which PI-RADS categories an earlier ISUP grade predicts, and the size beyond
#: which a lesion has outgrown its category. Both restated from the track briefs.
_EXPECTED_PIRADS = {1: {1, 2, 3}, 2: {3, 4}, 3: {3, 4, 5}}
_SIZE_CEILING = {3: 12, 4: 14, 5: 20}


DECISION_CRITIQUE = """
Your recommendation does not follow the rules for this track.

  The track's ladder gives `{want}` here, because {why}.
  You answered `{got}`.

The rules on this track are not considerations to be weighed against one
another. They are a ladder: the first rung that applies settles the case, and a
finding further down it does not overturn one further up. Find the first rung
that applies to this patient and answer once more, in the same format, complete.
""".strip()


#: What the frozen policy counts as a suspicious examination
#: (task1_policy.yaml -> corroboration.dre_suspicious). Restated rather than
#: imported so retuning one task cannot move the other silently.
_DRE_SUSPICIOUS = {"Nodus", "Suspicious", "Abnormal"}


def decision_rule(ctx: dict[str, Any]) -> tuple[str, str, str] | None:
    """What the track's own rules give for this case, and why.

    The same ladder the three adjudicator briefs state in prose. Returns None
    where the rules genuinely do not settle it, so the critique stays silent
    rather than inventing a verdict.

    The third element names the rung, because the rungs are not equally worth
    enforcing. Measured on run v6, over the cases where the agent and this
    function disagree: the hard gates are right 2/2 and the concordant tally
    4/6, but the lesion rung is right only 2/6 and the discretionary rungs 1/3.
    Enforcing everything scores 72/91 and enforcing only the first two scores
    75, against 71 for leaving the agent alone -- so `enforced_rungs` in
    resources/agentic_task1.yaml keeps the critique off the judgement calls.
    """
    pirads, track, age = ctx.get("pirads"), ctx.get("track"), ctx.get("age") or 0
    psa, psad, psav = ctx.get("psa"), ctx.get("psad"), ctx.get("psav")
    if age >= ctx["life_expectancy_certain_age"]:
        return "NO_BIOPSY", f"age {age} is at or past the benefit window", "hard_gate"
    if pirads is None:
        return None
    if pirads <= 2:
        return "NO_BIOPSY", f"PI-RADS {pirads} is an immediate stop on every track", "hard_gate"
    if track == "NAIVE":
        if pirads >= 4:
            return "BIOPSY", f"a first workup at PI-RADS {pirads}", "pirads_high"
        if psa is None:
            return None
        if psa < ctx["psa_low"]:
            return "NO_BIOPSY", f"PI-RADS 3 with PSA {psa} below {ctx['psa_low']}", "discretionary"
        if psa > ctx["psa_high"]:
            return "BIOPSY", f"PI-RADS 3 with PSA {psa} above {ctx['psa_high']}", "discretionary"
        if psad is None:
            return None
        why = f"PI-RADS 3 with density {psad}"
        return ("BIOPSY" if psad >= ctx["psad_high"] else "NO_BIOPSY"), why, "discretionary"
    if track == "PRIOR_NEGATIVE":
        if age >= ctx["prior_neg_age_max"]:
            return "NO_BIOPSY", f"a previously sampled man at age {age}", "hard_gate"
        if pirads >= 4:
            if psa is not None and psad is not None and psa < ctx["psa_low"] and psad < ctx["psad_reassuring"]:
                return "NO_BIOPSY", f"the low-risk zone: PSA {psa} with density {psad}", "prior_neg_gate"
            return "BIOPSY", f"a prior negative biopsy at PI-RADS {pirads}", "pirads_high"
        if psad is None or psav is None:
            return None
        quiet = psad < ctx["psad_low"] and psav < ctx["psav_low"]
        why = f"PI-RADS 3 after a negative biopsy, density {psad}, velocity {psav}"
        return ("NO_BIOPSY" if quiet else "BIOPSY"), why, "discretionary"
    grade, size = ctx.get("prior_isup"), ctx.get("lesion_mm")
    # ⚠️ Input validation, not a clinical rule, ported from the frozen policy's
    # `usable_density`: the record states the density twice -- once directly,
    # once implicitly as PSA over volume -- and where the two disagree by more
    # than the tolerance the quotient is not a measurement of this patient, so
    # no rung may read it. Fires on 1 of the 91 labelled cases.
    # A lesion past its category's ceiling settles the case on either branch.
    # No earlier MRI exists in this release, so size is the only reading of
    # growth available, and it does not become unavailable because the grade is.
    if size is not None and size > _SIZE_CEILING.get(pirads, 20):
        return "BIOPSY", f"the lesion is {size} mm, past the range for PI-RADS {pirads}", "lesion"
    # ⚠️ An unretrievable grade is NOT evidence that the imaging matches it.
    # Reading silence as concordance would hand the protective surveillance
    # default to exactly the cases nobody has characterised -- the same error
    # class as treating a missing PI-RADS as reassuring.
    #
    if psav is None or psad is None or psa is None:
        return None
    concordant = False if grade is None else pirads in _EXPECTED_PIRADS.get(grade, {3, 4, 5})
    if not concordant:
        if psav > ctx["psav_high"]:
            return ("BIOPSY",
                    f"imaging beyond what ISUP {grade} predicts, with velocity {psav}",
                    "discretionary")
        if psav < ctx["psav_low"]:
            why = f"a mismatch on quiet kinetics, density {psad}"
            return ("BIOPSY" if psad >= ctx["psad_high"] else "NO_BIOPSY"), why, "discretionary"
        if psa <= ctx["psa_high"] and psad < ctx["psad_high"] and (ctx.get("vol") or 0) <= ctx["vol_max"]:
            if size is None:
                return None
            grown = size > _SIZE_CEILING.get(pirads, 20)
            why = f"a mismatch on a permissible triad, lesion {size} mm"
            return ("BIOPSY" if grown else "NO_BIOPSY"), why, "lesion"
        return "BIOPSY", "a mismatch with an abnormal triad", "discretionary"
    votes = int(psa > ctx["psa_high"] or psad >= ctx["psad_high"]) + int(psav > ctx["psav_high"])
    if votes >= ctx["flip_votes"]:
        return "BIOPSY", f"concordant imaging with a tally of {votes} of 3", "tally"
    return "NO_BIOPSY", f"concordant imaging with a tally of only {votes} of 3", "tally"


def confidence_rule(ctx: dict[str, Any]) -> str:
    """The label the three sources give for this case.

    Order matters -- absence is checked before conflict, conflict before
    ambiguity.

    ⚠️ The track names here were `PRIOR_POS` and `PRIOR_NEG` until 2026-09-09,
    and `route` has always returned `PRIOR_POSITIVE` / `PRIOR_NEGATIVE`
    (`TRACKS`, top of this file). Neither string ever matched, so the absence
    rung was dead code and this function could not emit `uncertain` at all: it
    answered `clear` or `borderline` on 91 of 91 labelled cases. The unit test
    did not catch it because its own fixture carried the same truncated
    constant, while the `route` test asserted the full one -- the two halves
    were never joined. Measured over the 91 with the names joined up: exact
    55 -> 59, ordinal 0.7582 -> 0.7747, quadratic kappa 0.2142 -> 0.4386, and
    the count of `uncertain` goes 0 -> 15 against a reference 15.

    ⚠️ The benign-PSA rung was added 2026-09-11 and runs FIRST, before absence.
    `%Free PSA` is recorded on 195 of 195 released cases, is in the judge-visible
    `laboratory_results` section, and until now nothing in this repo read it.
    Above 25% -- Catalona's benign line -- the fraction is raised by BPH bulk
    rather than by tumour, so the biochemistry carries no case of its own.
    Measured over the 91, decisions unchanged at 79: exact 58 -> 62, ungated
    0.7692 -> 0.7967, gate-passed 0.7658 -> 0.7911, quadratic kappa 0.4183 ->
    0.4320, on 6 wins and 1 loss. It
    wins on the dev and holdout halves independently (0.7308 -> 0.7692 and
    0.7981 -> 0.8173 ungated), which is why it is not read as a fit to 9 cases.
    ⚠️ Inert on the external dev cohort: `free_psa_context` is populated on 2 of
    its 50 Task 1 rows, and both say "low", the other direction. That is
    EXTRACTION coverage, not clinical absence -- it may not be read as evidence
    against the rung.
    ⚠️ `20.0` scored better on kappa (0.4499) and worse on exact matches (58);
    it is not a clinical line and was not taken. Do not sweep this threshold.

    ⚠️ The outranking rung was added 2026-09-11 and runs ahead of both the
    conflict and the ambiguity rungs, on the non-established tracks only. A
    PI-RADS 5 lesion in a man under `settled_below_age` is maximal suspicion in a
    man young enough to be treated, and the benefit window may not unsettle it --
    the same asymmetry the age-gate guard applies on the DECISION side that day,
    and the frozen policy's `task1.risk_is_outranked`. Measured over the 91,
    decisions unchanged at 79: exact 58 -> 61, ungated 0.7692 -> 0.7857,
    gate-passed 0.7532 -> 0.7727, quadratic kappa 0.4183 -> 0.4483, distribution
    53/24/14 -> 56/21/14 against a reference 58/18/15, on 3 wins and 0 losses.
    ⚠️ All 3 are in the dev half; holdout is unchanged. It rests on the
    mechanism, not on the split.
    ⚠️ The rule was over-emitting `borderline` (24 against 18), which is why the
    only frozen-policy principle that PAYS here is one that removes a borderline
    source. Three that ADD one were measured on 2026-09-11 and all lost:
    procedural risk (exact 57, kappa 0.4083), `RECLASSIFICATION` broadened below
    PI-RADS 5 (57, 0.3907), and the MRI certainty scale reading PI-RADS 3/4 as
    borderline (47, 0.3654). Do not retry them.

    ⚠️ `grade_evidence_available` is a POSITIVE precondition on the absence
    rung, and it is not optional. "The grade is not on record" is a claim about
    the patient; "we had nowhere to look" is a claim about the extraction, and
    only the first licenses `uncertain`. On the external dev cohort the two are
    indistinguishable without it -- that CSV has no grade column at all (0 of 50
    rows mention ISUP, Gleason or a grade group anywhere), so the rung fired on
    6 cases whose reference label is `clear` on all 6 and cost ordinal
    0.8696 -> 0.7391. Gated on the notes channel actually carrying text, it is
    inert there (dev returns to 0.8696) and unchanged on the labelled 91, where
    `note_sections` carry 981 characters or more on 91 of 91.
    """
    track, pirads = ctx.get("track"), str(ctx.get("pirads"))
    if pirads in {"1", "2"}:
        return "clear"
    # Confounded biochemistry: the raised PSA has a BENIGN EXPLANATION, so it
    # corroborates nothing and the encounter turns on the imaging alone.
    # Ahead of the absence rung deliberately -- this is the one reading that
    # says an absent prior grade is not what is holding the case up. Sits in the
    # established-cancer track only, where the question is whether this
    # encounter raises a NEW tissue question; on the naive and prior-negative
    # tracks a high free fraction does not stop a suspicious lesion earning
    # tissue, and applied there it measured ungated 0.7692 -> 0.7582.
    free_psa = ctx.get("free_psa_pct")
    if track == "PRIOR_POSITIVE" and free_psa is not None and free_psa >= ctx["free_psa_benign_pct"]:
        return "borderline"
    # Absence: a named fact whose two values give two different actions -- and
    # only where the record HAS a place that fact could have been written.
    if track == "PRIOR_POSITIVE" and ctx.get("grade_evidence_available"):
        if pirads == "4" and not ctx.get("grade_stated"):
            return "uncertain"
        # ⚠️ Only where an earlier MRI actually exists to compare against.
        # `size_was_tiebreaker` is the radiology agent reading the CURRENT
        # lesion against a size ceiling -- a proxy that was computed, not a
        # fact that is missing. Ungated it fired on ~21 of 91 labelled cases
        # and pushed the chain to 36 `uncertain` against a reference 15.
        if ctx.get("size_was_tiebreaker") and ctx.get("prior_mri_available"):
            return "uncertain"
    # Outranked: where there is NO established diagnosis the question is whether
    # to take tissue, and maximal suspicion in a man young enough to be treated
    # settles it -- neither the benefit window below nor a grey biochemistry has
    # anything left to make shared. EAU's evidence summary puts csPCa detection
    # at 83% for PI-RADS 5 against 55% at PI-RADS 4, so at 5 the indication is
    # not what is in doubt. The same asymmetry the 2026-09-11 age-gate guard
    # applies on the DECISION side ("the age gates may not overrule maximal
    # suspicion"), and the same line the frozen policy reads in
    # `task1.risk_is_outranked`.
    # ⚠️ Scoped OFF the established-cancer track, exactly as the frozen rule is.
    # There the question is re-grading, and a PI-RADS 5 does not settle what the
    # earlier biopsy found. Applied there too it gives one more exact match (63)
    # and a WORSE kappa (0.4114 against 0.4483) -- do not widen it.
    if track in {"NAIVE", "PRIOR_NEGATIVE"} and pirads == "5" \
            and (ctx.get("age") or 0) < ctx["settled_below_age"]:
        return "clear"
    # Conflict: suspicion against the benefit window.
    if track in {"NAIVE", "PRIOR_NEGATIVE"} and pirads in {"4", "5"}:
        if (ctx.get("age") or 0) >= ctx["conflict_age"] or (ctx.get("comorbidity_count") or 0) >= ctx[
            "conflict_comorbidity_count"
        ]:
            return "borderline"
    # Ambiguity: nothing was ever past a line.
    if track != "NAIVE" and (ctx.get("grey_count") or 0) >= ctx["grey_zones_for_borderline"]:
        return "borderline"
    return "clear"


def render(template: str, thresholds: dict[str, Any], **extra: Any) -> str:
    """Interpolate thresholds into a brief. Unknown braces are left alone."""
    return template.format(**{**thresholds, **extra})


_owned = [v for vals in WEIGHT_OWNERS.values() for v in vals]
assert len(_owned) == len(set(_owned)), "WEIGHT_OWNERS overlap"
assert set(_owned) == {
    "psa", "age", "dre", "comorbidity", "bx", "pirads", "psad", "vol", "cspca", "fh"
}, "WEIGHT_OWNERS must cover exactly the ten scored Task 1 keys"
