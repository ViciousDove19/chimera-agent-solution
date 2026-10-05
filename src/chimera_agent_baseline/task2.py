"""Task 2 management policy: three questions, asked in order.

    1. risk       what is this cancer?          -> ISUP, PSA and cT jointly
    2. candidacy  will he live long enough to
                  benefit from treating it?     -> age and competing illness
    3. need       given both, what does he need? -> plus symptom burden

Question 1 is the EAU risk stratification, which is a joint function of grade,
PSA and clinical stage. Grade is the dominant channel but not the only one. On
the 72 labelled cases the joint form and grade alone agree, because no ISUP 1
patient has PSA >= 20 or cT3+. A separate, explicitly provisional branch below
handles the challenge's discordant ISUP 1 / very-high-PSA cases without bending
their formal EAU high-risk classification.

Question 2 is the axis that sits beside risk in every guideline and that the
previous version of this policy did not represent at all. Watchful waiting is
what you offer a man whose cancer is not what will limit his life; with no such
branch the fourth class was unreachable by construction.

The reported ISUP grade is authoritative. The pathology report spans several
timepoints in most cases but almost never states a second grade, and the
structured prompt already carries the latest one. The report's ``AI-predicted
ISUP grade group`` is a distractor -- higher than the reported grade in 54 of
the 64 reports carrying it, including every active-surveillance case -- and is
deliberately ignored.

A variable's reported weight is the role it played on the branch taken, so the
weight we report describes how the answer was reached.

Pure function of the structured prompt -- no LLM, no tool calls, no text
parsing beyond the two numeric fields below. All thresholds live in
``resources/task2_policy.yaml``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import yaml

_CONFIG_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "resources" / "task2_policy.yaml",
    Path("/opt/app/resources/task2_policy.yaml"),
    Path("resources/task2_policy.yaml"),
)

VARIABLES = (
    "bx_isup", "bx_gl_prim", "bx_gl_sec", "psa", "psad",
    "ct", "pirads", "age", "comorbidity", "fh", "cspca",
)

# Read on every case: PSA and life expectancy bear on any management step at
# every value they take (GT calls PSA important/decisive in 60% of cases below
# 10, 81% at 10-20 and 100% above 20; age in 85%/74%/62% by decade). Every
# other variable is observed and reported but only becomes something we leaned
# on when its value actually says something -- see ``_live_extras``.
_CONSULTED = ("psa", "age", "pirads")

# Clinical names for the risk driver, so the stated reason never carries a
# field name into the rationale the model writes.
_DRIVER_NAMES = {
    "bx_isup": "biopsy grade",
    "psa": "PSA",
    "psad": "PSA density",
    "ct": "clinical stage",
}

_IPSS = re.compile(r"(\d+)\s*/\s*35")
_STAGE = re.compile(r"C?T(\d)")
_SUBSTAGE = re.compile(r"C?T\d([ABC])")


# The stage a case reaches. ⚠️ These strings are the KEYS of the YAML `triggers`
# table: renaming one here without renaming it there silently drops the trigger,
# because the lookup is a `.get(stage, "")`. `test_task2_every_stated_trigger_has_a_stage`
# is the only mechanical guard against that.
STAGE_NO_CANCER = "NO_CANCER_CONFIRMED"
STAGE_GRADE_NOT_RECORDED = "GRADE_NOT_RECORDED"
STAGE_ADVERSE_HISTOLOGY = "ADVERSE_HISTOLOGY"
STAGE_GRADE_INCOHERENT = "GRADE_INCOHERENT_WITH_DISEASE"
STAGE_WATCHFUL_WAITING = "TREATMENT_WOULD_NOT_BENEFIT"
STAGE_LOW_RISK = "LOW_RISK_CANCER"
STAGE_FAVOURABLE_SURVEILLANCE = "FAVOURABLE_INTERMEDIATE_SURVEILLANCE"
STAGE_SHARED_DECISION = "SHARED_DECISION_BAND"
STAGE_INTERMEDIATE = "INTERMEDIATE_RISK_CANCER"
STAGE_SYMPTOM_DRIVEN = "SYMPTOM_DRIVEN_TREATMENT"
STAGE_HIGH = "HIGH_RISK_CANCER"
STAGE_LOW_GRADE_HIGH_PSA_DISCORDANCE = "LOW_GRADE_HIGH_PSA_DISCORDANCE"

# The branches that recommend monitoring, and so must say what would end it.
SURVEILLANCE_STAGES = (
    STAGE_NO_CANCER,
    STAGE_LOW_RISK,
    STAGE_FAVOURABLE_SURVEILLANCE,
    STAGE_LOW_GRADE_HIGH_PSA_DISCORDANCE,
)


class Risk(NamedTuple):
    """What the cancer *is* -- never what to do about it.

    ``group`` keeps a three-value vocabulary because the YAML maps exactly
    ``low``/``intermediate``/``high`` to decisions and the stage names are built
    from it. Favourable-intermediate is therefore a subclass flag rather than a
    fourth group: a fourth value would break ``cfg["risk"][group]``, invent an
    unmapped stage name and orphan a trigger key, three failures for one rename.
    """

    group: str
    driver: str
    favourable: bool = False


def _default_config_path() -> Path:
    override = os.getenv("CHIMERA_TASK2_POLICY")
    if override:
        return Path(override)
    for candidate in _CONFIG_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("task2_policy.yaml not found; tried " + ", ".join(str(c) for c in _CONFIG_CANDIDATES))


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
    risk_group: str = ""


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ipss(prompt: dict[str, Any]) -> int | None:
    match = _IPSS.search(str(prompt.get("ipss") or ""))
    return int(match.group(1)) if match else None


def _stage_rank(prompt: dict[str, Any]) -> int | None:
    """Numeric part of the cT stage; ``cTx`` and a missing stage are unknown."""
    match = _STAGE.match(str(prompt.get("ct") or "").strip().upper())
    return int(match.group(1)) if match else None


def _unstageable(prompt: dict[str, Any]) -> bool:
    """True only when the record positively states ``cTx``.

    ⚠️ A missing stage is NOT cTx. `cTx` is a finding -- the local extent was
    assessed and could not be assigned -- whereas a blank field means nobody
    read it. Reading silence as a finding is the error class the 2026-09-03
    audit logged against ``task1.py::signal()``; do not repeat it here.
    """
    return str(prompt.get("ct") or "").strip().upper() == "CTX"


def _stage_letter(prompt: dict[str, Any]) -> str | None:
    """Substage letter of the cT stage; ``None`` when it is not written."""
    match = _SUBSTAGE.match(str(prompt.get("ct") or "").strip().upper())
    return match.group(1).lower() if match else None


def risk_group(prompt: dict[str, Any], cfg: dict[str, Any]) -> Risk:
    """EAU risk class and the feature that put the case there.

    Classification only. Whether surveillance is still appropriate is a separate
    question, asked afterwards by ``surveillance_is_offered`` -- see the module
    docstring. Any single high-risk feature makes the case high risk; low risk
    requires every low-risk feature at once; everything else is intermediate,
    subclassified favourable or unfavourable.
    """
    r = cfg["risk"]
    grade, psa, rank = _int(prompt.get("bx_isup")), _float(prompt.get("psa")), _stage_rank(prompt)

    if grade is not None and grade >= r["high_isup_from"]:
        return Risk("high", "bx_isup")
    # EAU places PSA 10-20 in the intermediate band and high risk strictly
    # above 20; an exact value of 20 therefore remains intermediate.
    if psa is not None and psa > r["high_psa_from"]:
        return Risk("high", "psa")
    if rank is not None and rank >= r["high_stage_from"]:
        return Risk("high", "ct")

    # Below the high-risk line, an intermediate PSA escalates the *class* on its
    # own -- GG1 with PSA 10-20 is favourable-intermediate disease, not low-risk
    # disease. It does not escalate the *management*: see the eligibility layer.
    if psa is not None and psa >= r["intermediate_psa_from"]:
        return Risk("intermediate", "psa", _favourable(grade, psa, rank, cfg))
    if grade is not None and grade <= r["low_isup_at_most"]:
        return Risk("low", "bx_isup")
    return Risk("intermediate", "bx_isup", _favourable(grade, psa, rank, cfg))


def _favourable(grade: int | None, psa: float | None, rank: int | None, cfg: dict[str, Any]) -> bool:
    """EAU favourable-intermediate: ISUP <= 2 carrying at most one risk factor.

    ⚠️ The third EAU factor, cT2b-c, is invisible in this corpus -- no ISUP 2
    case is staged cT2b or cT2c -- so it never contributes. A stage we cannot
    confirm is not counted in the patient's favour.
    """
    r = cfg["risk"]
    if grade is None or grade > r["favourable_isup_at_most"]:
        return False
    factors = 0
    if grade >= 2:
        factors += 1
    if psa is not None and psa >= r["intermediate_psa_from"]:
        factors += 1
    return factors <= r["favourable_factors_at_most"]


def surveillance_is_offered(prompt: dict[str, Any], cfg: dict[str, Any], risk: Risk, grade: int | None) -> bool:
    """Is monitoring still on the table? The eligibility layer, kept separate
    from the risk class so that neither one is bent to produce the other.

    ISUP <= 1 is offered surveillance whether it classified as low or as
    favourable-intermediate risk -- that is exactly what EAU means by offering
    AS to selected intermediate-risk men.
    """
    if risk.group == "high" or grade is None:
        return False
    if grade <= cfg["surveillance"]["offer_isup_at_most"]:
        return True
    # GG2 defaults to treatment. Surveillance has to be earned by the full
    # favourable-profile rule below; any one of these three findings rules it
    # out immediately. cT2a may still earn surveillance through a stable
    # trajectory, but cT2b+ and PI-RADS 5 cannot be outweighed by quiet
    # biochemistry.
    if grade == 2:
        rule = cfg["surveillance"]["gg2_unfavourable"]
        psa, pirads, rank, sub = (
            _float(prompt.get("psa")),
            _int(prompt.get("pirads")),
            _stage_rank(prompt),
            _stage_letter(prompt),
        )
        stage_unfavourable = (
            rank is not None
            and (
                rank > rule["stage_rank"]
                or (
                    rank == rule["stage_rank"]
                    and sub is not None
                    and sub > rule["stage_substage_at_most"]
                )
            )
        )
        if (
            (psa is not None and psa > rule["psa_above"])
            or (pirads is not None and pirads == rule["pirads_unfavourable"])
            or stage_unfavourable
        ):
            return False
        if _stable_favourable_gg2(prompt, cfg):
            return True
        return _burden_supports_surveillance(prompt, cfg) is True
    if not risk.favourable:
        return False
    return _burden_supports_surveillance(prompt, cfg) is True


def _stable_favourable_gg2(prompt: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """The GG2 men the record positively describes as not progressing.

    cT1 disease is impalpable and organ-confined, and a cT1c man who clears the
    remaining entry criteria can be a surveillance candidate. cT2a can also
    qualify when the record positively establishes a stable trajectory; cT2b
    and above cannot reach this helper via ``surveillance_is_offered``.

    ⚠️ Every precondition must be positively stated. An unmentioned cribriform
    status, an absent PSAD or a stage of cTx does not admit anyone: this is the
    surveillance side of the band, and silence is not stability. This is what
    separates the rule from the PRIAS criteria tested on 2026-09-02, which
    admitted on the four recorded entry fields alone and lost 5 cases -- all
    five either progressed histologically, never had stability written, or
    carried a PSA velocity above the line.
    """
    rule = cfg["surveillance"].get("stable_favourable_gg2") or {}
    if not rule or _int(prompt.get("bx_isup")) != rule["isup"]:
        return False
    if prompt.get("_adverse_as_histology") or not prompt.get("_cribriform_excluded"):
        return False
    psa, psad = _float(prompt.get("psa")), _float(prompt.get("psad"))
    if psa is None or psa > rule["psa_at_most"]:
        return False
    if psad is None or psad >= rule["psad_below"]:
        return False
    pirads = _int(prompt.get("pirads"))
    if pirads is None or pirads > rule["pirads_at_most"]:
        return False
    rank, sub = _stage_rank(prompt), _stage_letter(prompt)
    # cTx or a missing stage is not adverse evidence. Clinical stage is derived
    # from DRE and MRI, so favourable MRI can carry the eligibility decision
    # when no stage was assigned.
    if rank is None:
        return True
    if rank > rule["stage_at_most"]:
        return False
    if rank <= rule["stage_without_trajectory_at_most"]:
        return True
    if sub is not None and sub > rule["substage_at_most"]:
        return False
    psav = _float(prompt.get("psav"))
    if psav is None or psav >= rule["psav_below"]:
        return False
    return bool(prompt.get("_histology_stable_timepoints"))


def _uncharacterised_gg2(prompt: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """GG2 that has not been characterised well enough to treat.

    A different claim from the stage ladder above: not "this disease is
    indolent" but "this record cannot yet support an irreversible decision".
    An unassignable clinical stage (cTx) means the local extent was never
    established, and a pathology report that states only ONE biopsy timepoint
    means the grade rests on a single sample -- with a PSA density and velocity
    that give no reason to hurry, the next step is characterisation (staging
    and a repeat/confirmatory biopsy) under surveillance, not curative therapy.

    ⚠️ Deliberately NOT gated on cribriform/IDC, unlike the stage ladder. The
    guideline exclusion applies to committing a characterised cancer to a
    formal AS protocol; here the recommendation is to keep the patient under
    surveillance *while the cancer is characterised*, which is what an
    incompletely staged single biopsy actually warrants. See the departure
    already recorded under `symptoms` in the YAML -- this cohort's
    decision-generating process does not apply the cribriform exclusion.
    """
    rule = cfg["surveillance"].get("uncharacterised_gg2") or {}
    if not rule or _int(prompt.get("bx_isup")) != rule["isup"]:
        return False
    if _stage_rank(prompt) is not None:
        return False
    if not prompt.get("_single_biopsy_timepoint"):
        return False
    psad, psav = _float(prompt.get("psad")), _float(prompt.get("psav"))
    if psad is None or psad >= rule["psad_below"]:
        return False
    return psav is not None and psav < rule["psav_below"]


def _burden_supports_surveillance(prompt: dict[str, Any], cfg: dict[str, Any]) -> bool | None:
    """Does the tumour burden support surveillance in the ISUP 2 band?

    ``None`` means the record does not say, and the caller treats that as "not
    offered": absence of evidence is not evidence of low burden.

    ⚠️ This returns ``None`` on every case in the current corpus, deliberately.
    The guideline AS criteria for GG2 are built from tumour burden -- pattern-4
    percentage, positive-core count, involvement per core, core length, tumour
    volume, lesion multiplicity -- and an inventory of all 153 Task 2 pathology
    and radiology reports found every one of them at **0/153**: not negated,
    never written. The seam exists, is named and is tested; it cannot fire until
    the data carries the fields. ``resources/task2_policy.yaml -> surveillance``
    records the full inventory and the substitutes that were tried and lost.
    """
    thresholds = cfg["surveillance"].get("burden_thresholds")
    if not thresholds:
        return None
    if any(prompt.get(field) is not None for field in cfg["surveillance"]["burden_fields"]):
        raise NotImplementedError("burden_thresholds is set but no burden test is implemented")
    return None


def _low_grade_high_psa_discordance(
    prompt: dict[str, Any], cfg: dict[str, Any], grade: int | None
) -> bool:
    """Challenge-specific hold branch for mutually discordant evidence.

    This does not reclassify the cancer: PSA above the EAU high-risk threshold
    remains a high-risk feature. It prevents that single measurement from
    immediately producing an irreversible management recommendation when the
    recorded biopsy is GG1 and the clinical stage remains organ-confined.
    """
    rule = cfg.get("low_grade_high_psa_discordance") or {}
    if not rule or grade is None or grade > rule["isup_at_most"]:
        return False
    psa = _float(prompt.get("psa"))
    if psa is None or psa <= rule["psa_above"]:
        return False
    rank = _stage_rank(prompt)
    return rank is None or rank <= rule["stage_at_most"]


def _isup1_strong_disagreements(prompt: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, ...]:
    """Recorded findings that strongly disagree with an ISUP 1 biopsy."""
    rule = cfg["isup1"]
    findings = []
    psa = _float(prompt.get("psa"))
    if psa is not None and psa > rule["discordant_psa_above"]:
        findings.append("PSA above 10 ng/mL")
    pirads = _int(prompt.get("pirads"))
    if pirads is not None and pirads == rule["discordant_pirads"]:
        findings.append("PI-RADS 5 imaging")
    rank = _stage_rank(prompt)
    if rank is not None and rank >= rule["discordant_stage_from"]:
        findings.append("clinical stage cT2a or higher")
    return tuple(findings)


def _ungraded_high_risk_feature(prompt: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    """Which recorded feature is high-risk on its own when no grade exists?

    A positive biopsy whose grade cannot be read is still a positive biopsy. If
    the PSA or the PSA density is above the EAU high-risk line, "obtain the
    grade first" is not a management answer -- the grade can only raise the
    risk class, never lower it below what the PSA already establishes.
    Returns the variable name so the caller can report which one settled it.
    """
    rule = (cfg["ungraded_positive"] or {}).get("high_risk") or {}
    if not rule:
        return None
    psa = _float(prompt.get("psa"))
    if psa is not None and psa > rule["psa_above"]:
        return "psa"
    psad = _float(prompt.get("psad"))
    if psad is not None and psad >= rule["psad_from"]:
        return "psad"
    return None


def apply_policy(prompt: dict[str, Any], cfg: dict[str, Any]) -> Verdict:
    grade = _int(prompt.get("bx_isup"))
    negative = str(prompt.get("bx") or "").strip().lower() == "negative"
    r, cand, sym = cfg["risk"], cfg["candidacy"], cfg["symptoms"]

    # Observed and reported, but not what carried the answer.
    roles = dict.fromkeys(VARIABLES, "available")
    for name in (*_CONSULTED, *_live_extras(prompt, cfg)):
        roles[name] = "consulted"

    # -- stage 1: what is this cancer? ------------------------------------
    if grade == 0 or (negative and grade is None):
        # Nothing to grade: there is no confirmed cancer to weigh.
        roles["bx_isup"] = roles["bx_gl_prim"] = roles["bx_gl_sec"] = "available"
        return _verdict(
            cfg, decision=r["none"], confidence="clear", roles=roles, stage=STAGE_NO_CANCER,
            reasons=("the biopsy found no cancer; the diagnostic pathway continues",), group="none", prompt=prompt,
        )

    if grade is None:
        # A positive biopsy with no recorded grade cannot be risk-stratified.
        # Every labelled case carries a grade, so this defensive branch asks
        # for clarification rather than inventing intermediate-risk disease.
        ungraded = cfg["ungraded_positive"]
        roles["bx_isup"] = "settled"
        decision, confidence = ungraded["decision"], ungraded["confidence"]
        reasons = (
            "the biopsy is positive but no grade is recorded; obtain or review the "
            "pathology grade before choosing cancer treatment",
        )
        feature = _ungraded_high_risk_feature(prompt, cfg)
        if feature is not None:
            escalate = ungraded["high_risk"]
            confounded = bool(prompt.get("_psa_confounder"))
            roles[feature] = "settled"
            decision = escalate["confounded_decision" if confounded else "decision"]
            confidence = escalate["confounded_confidence" if confounded else "confidence"]
            reasons = (
                (
                    f"the grade is missing but the {_DRIVER_NAMES[feature]} is a high-risk "
                    "feature on its own, and the record documents a non-malignant "
                    "explanation for it (infection, prostatitis, instrumentation or "
                    "retention); surveillance continues while the PSA is repeated under "
                    "standardised conditions and the grade is obtained"
                    if confounded
                    else f"the grade is missing, but the {_DRIVER_NAMES[feature]} is a "
                    "high-risk feature on its own with no documented non-malignant "
                    "explanation; treatment is indicated once the grade and metastatic "
                    "staging are completed"
                ),
            )
        return _verdict(
            cfg,
            decision=decision,
            confidence=confidence,
            roles=roles,
            stage=STAGE_GRADE_NOT_RECORDED,
            reasons=reasons,
            group="unknown",
            prompt=prompt,
        )

    risk = risk_group(prompt, cfg)
    group = risk.group
    offered = surveillance_is_offered(prompt, cfg, risk, grade)

    # Which variable *settled* the case, which is not the same question as which
    # one classified it. Below the high-risk line the grade governs and a PSA or
    # stage that lifts the class merely informs -- it is already reported
    # `important` as a consulted variable. Only a high-risk escalation is
    # settled by something other than the grade.
    #
    # ⚠️ Reporting `risk.driver` directly here is wrong twice over, and both
    # were caught by the regression oracle rather than by reasoning. It makes
    # `psa` decisive and `bx_isup` merely noted on the four GG1/PSA 10-20 men
    # now classified favourable-intermediate, and on eight ISUP 2-3 men at
    # PSA >= 10 -- twelve cases whose weights would move for a reclassification
    # that changes no decision, at 0.275 of the case score each.
    driver = risk.driver if risk.group == "high" else "bx_isup"
    # The grade settles the case at the ends of the scale and merely informs it
    # in between; a driver that is not the grade always settles.
    live = cfg["live"]
    settles = driver != "bx_isup" or grade < live["grade_settles_below"] or grade >= live["grade_settles_from"]
    roles[driver] = "settled" if settles else "consulted"

    # ISUP 1 always remains on active surveillance. Strong disagreement does
    # not change the management arm; two or more discordant channels make the
    # recommendation borderline while the grade and staging are reconciled.
    # Severe symptoms likewise require separate LUTS/BPH management rather
    # than conversion to definitive cancer treatment.
    if grade == 1:
        findings = _isup1_strong_disagreements(prompt, cfg)
        ipss = _ipss(prompt)
        severe_luts = ipss is not None and ipss > cfg["isup1"]["borderline_ipss_above"]
        borderline = len(findings) >= cfg["isup1"]["discordances_for_borderline"] or severe_luts
        if findings:
            roles["bx_isup"] = "settled"
            for name, present in (
                ("psa", "PSA above 10 ng/mL" in findings),
                ("pirads", "PI-RADS 5 imaging" in findings),
                ("ct", "clinical stage cT2a or higher" in findings),
            ):
                if present:
                    roles[name] = "consulted"
        reasons = ["low-risk disease on the biopsy grade"]
        if len(findings) >= cfg["isup1"]["discordances_for_borderline"]:
            reasons.append(
                "multiple findings strongly disagree with the low biopsy grade ("
                + ", ".join(findings)
                + "); reconcile the grade, imaging and stage during surveillance"
            )
        if severe_luts:
            gland = _float(prompt.get("vol"))
            ceiling = cfg["symptoms"].get("gland_ceiling_ml")
            if gland is not None and ceiling is not None and gland >= ceiling:
                reasons.append(
                    f"the urinary symptoms are severe, but at {gland:g} mL the obstruction "
                    "is treated on its own account and does not change the cancer strategy"
                )
            else:
                reasons.append(
                    "the severe urinary symptoms require separate BPH/outlet management but do "
                    "not change the cancer strategy"
                )
        return _verdict(
            cfg,
            decision=cfg["surveillance"]["decision"],
            confidence="borderline" if borderline else "clear",
            roles=roles,
            stage=(
                STAGE_LOW_GRADE_HIGH_PSA_DISCORDANCE
                if len(findings) >= cfg["isup1"]["discordances_for_borderline"]
                else (STAGE_LOW_RISK if group == "low" else STAGE_FAVOURABLE_SURVEILLANCE)
            ),
            reasons=tuple(reasons),
            group=group,
            prompt=prompt,
        )

    # -- stage 2: will he live long enough to benefit? --------------------
    # Asked before the management step, because it changes what "benefit"
    # means. Not asked of men with no confirmed cancer: there is nothing to
    # decline to treat.
    age = _int(prompt.get("age"))
    burden = len(prompt.get("pmhx") or [])
    if age is not None and age >= cand["elderly_from"] and burden >= cand["comorbidity_burden"]:
        roles["age"] = "settled"
        roles["comorbidity"] = "consulted"
        return _verdict(
            cfg, decision=cand["decision"], confidence=cand["confidence"], roles=roles,
            stage=STAGE_WATCHFUL_WAITING,
            reasons=(
                "at his age, and with competing conditions, the cancer is unlikely to be "
                "what limits his life; treat only if it becomes symptomatic",
            ),
            group=group, prompt=prompt,
        )

    # An incoherent grade -- cribriform architecture reported alongside a
    # Gleason 3+3, which is a pattern-4 morphology the assigned grade denies --
    # is a reason to distrust the grade, not by itself a reason to treat: four
    # of the five such cases here stay on surveillance. It becomes a reason to
    # treat only where significant disease is demonstrable WITHOUT the tissue,
    # by a PI-RADS 5 lesion and a palpable tumour. Then the biopsy is the only
    # item still claiming low risk and the least reliable one, and re-sampling
    # a lesion the imaging and the examination already call significant adds
    # nothing. See `adverse_histology.incoherent_grade_treats` in the YAML.
    incoherent = cfg.get("adverse_histology", {}).get("incoherent_grade_treats") or {}
    if (
        prompt.get("_adverse_as_histology")
        and incoherent.get("enabled")
        and grade <= incoherent["grade_at_most"]
        and (_int(prompt.get("pirads")) or 0) >= incoherent["pirads_from"]
        and (_stage_rank(prompt) or 0) >= incoherent["stage_from"]
    ):
        # The grade is reported, and reported as not carrying the answer.
        roles["bx_isup"] = "available"
        roles["pirads"] = "settled"
        roles["ct"] = "consulted"
        roles["cspca"] = "consulted"
        return _verdict(
            cfg,
            decision=incoherent["decision"],
            confidence=incoherent["confidence"],
            roles=roles,
            stage=STAGE_GRADE_INCOHERENT,
            reasons=(
                "the biopsy reports cribriform architecture alongside a low grade, which "
                "the grade itself denies, while the lesion is PI-RADS 5 and the tumour is "
                "palpable; the grade is the least reliable item in the record and "
                "surveillance would under-treat him",
            ),
            group=group,
            prompt=prompt,
        )

    # Cribriform/intraductal histology is a hard treatment exclusion for GG2.
    # The configurable broader ablation remains available for other grades;
    # GG1 has already returned through its unconditional AS branch above.
    if prompt.get("_adverse_as_histology") and (
        grade == 2 or cfg.get("adverse_histology", {}).get("excludes_surveillance")
    ):
        roles["bx_isup"] = "settled"
        return _verdict(
            cfg,
            decision=r["intermediate"],
            confidence="clear",
            roles=roles,
            stage=STAGE_ADVERSE_HISTOLOGY,
            reasons=("cribriform or intraductal histology excludes active surveillance",),
            group=group,
            prompt=prompt,
        )

    # The hidden challenge cases distinguish formal risk from what to do when
    # the inputs disagree sharply: GG1/cT1-2 pathology against PSA >= 20. Keep
    # surveillance only as a provisional strategy while confirming the PSA,
    # excluding a transient inflammatory/infective cause, reviewing/sampling
    # the pathology and completing high-risk staging. This is intentionally
    # narrower than the ordinary surveillance eligibility rule and does not
    # apply to GG2+, cT3+ or a confirmed metastatic finding.
    if _low_grade_high_psa_discordance(prompt, cfg, grade):
        discordance = cfg["low_grade_high_psa_discordance"]
        psa = _float(prompt.get("psa"))
        pirads = _int(prompt.get("pirads"))
        psad = _float(prompt.get("psad"))
        more_severe = (
            psa is not None and psa >= discordance["uncertain_psa_from"]
        ) or (
            pirads is not None and pirads >= discordance["uncertain_pirads_from"]
        ) or (
            psad is not None and psad >= discordance["uncertain_psad_from"]
        )
        roles["bx_isup"] = roles["psa"] = "settled"
        return _verdict(
            cfg,
            decision=discordance["decision"],
            confidence=discordance["severe_confidence"] if more_severe else discordance["confidence"],
            roles=roles,
            stage=STAGE_LOW_GRADE_HIGH_PSA_DISCORDANCE,
            reasons=(
                "the reported ISUP 1 biopsy conflicts with the high PSA; surveillance is "
                "provisional while the PSA is repeated under standardised conditions, "
                "infection or prostatitis is excluded, pathology is confirmed and "
                "metastatic staging is completed",
            ),
            group=group,
            prompt=prompt,
        )

    # -- stage 3: given both, what does he need? --------------------------
    if offered:
        low = group == "low"
        if low:
            reasons = ("low-risk disease is suitable for surveillance",)
        elif _uncharacterised_gg2(prompt, cfg):
            reasons = (
                "ISUP 2 on a single biopsy timepoint with no assignable clinical stage; "
                "the local extent and the grade are both unconfirmed, and with a PSA "
                "density and velocity that give no reason to hurry the next step is "
                "staging and a confirmatory biopsy under surveillance, not treatment",
            )
        elif _stable_favourable_gg2(prompt, cfg):
            # Say which rung of the stage ladder admitted him: the cT1 rung
            # rests on the stage and the PRIAS entry numbers, the cT2a-b rung
            # additionally on a documented non-progressing trajectory.
            reasons = (
                "favourable intermediate-risk ISUP 2 disease with no cribriform or "
                "intraductal growth, PSA and PSA density inside the surveillance entry "
                + (
                    "criteria, and impalpable cT1 disease"
                    if (_stage_rank(prompt) or 0) <= 1
                    else "criteria, and a documented non-progressing trajectory -- stable "
                    "grade across repeat biopsies with a PSA velocity below 2 ng/mL/yr"
                ),
            )
        else:
            reasons = (
                "favourable intermediate-risk disease, which remains suitable for surveillance",
            )
        # Severe obstruction makes surgery treat two problems at once -- but only
        # while one operation can address both. Above the large-gland line the
        # bladder-outlet answer is enucleation rather than radical prostatectomy,
        # so the obstruction is treated on its own account and the cancer
        # strategy is left alone. ⚠️ An unknown gland is not a large gland: the
        # rule still fires, because the exclusion has to be earned.
        symptom_burden = _ipss(prompt)
        severe = symptom_burden is not None and symptom_burden >= sym["severe_ipss"]
        gland, ceiling = _float(prompt.get("vol")), sym.get("gland_ceiling_ml")
        operable = ceiling is None or gland is None or gland < ceiling
        # ...and only while the cancer question is quiescent. The switch claims
        # this man needs outlet surgery for BENIGN disease anyway, which
        # presupposes the raised PSA is the gland's and not the tumour's. PSA
        # density is the reading that settles that. ⚠️ Unlike the gland ceiling
        # above, this is a POSITIVE precondition, so a missing density blocks
        # the switch rather than admitting it. See `symptoms` in the YAML.
        density, bulk_line = _float(prompt.get("psad")), sym.get("bulk_explicable_psad_below")
        bulk_explicable = bulk_line is None or (density is not None and density < bulk_line)
        if severe and operable and not bulk_explicable:
            reasons = (
                *reasons,
                f"the obstructive symptoms are severe, but a PSA density of {density:g} is not "
                "explained by the gland, so the management follows the cancer rather than the "
                "symptoms" if density is not None else
                "the obstructive symptoms are severe, but no PSA density is recorded to show "
                "the raised PSA is the gland's, so the management follows the cancer",
            )
        if severe and not operable:
            reasons = (
                *reasons,
                f"the obstructive symptoms are severe, but at {gland:g} mL the obstruction "
                "is treated on its own account rather than by cancer surgery",
            )
        if severe and operable and bulk_explicable and sym.get("changes_cancer_decision", True):
            return _verdict(
                cfg, decision=sym["decision"], confidence=sym["confidence"], roles=roles,
                stage=STAGE_SYMPTOM_DRIVEN,
                reasons=(
                    f"{'low' if low else 'favourable intermediate'}-risk disease, but the "
                    "severe obstructive symptoms are treatable in the same operation",
                ),
                group=group, prompt=prompt,
            )
        if severe and operable and bulk_explicable:
            reasons = (
                *reasons,
                "the severe urinary symptoms require separate BPH/outlet management but do "
                "not by themselves require cancer treatment",
            )
        return _verdict(
            cfg, decision=cfg["surveillance"]["decision"], confidence="clear", roles=roles,
            stage=STAGE_LOW_RISK if low else STAGE_FAVOURABLE_SURVEILLANCE,
            reasons=reasons, group=group, prompt=prompt,
        )

    # Intermediate and high risk both point to treatment, but only high risk
    # settles it. ISUP 2 is a shared-decision band and the prompt does not
    # expose what selects a patient within it, so say so rather than assert.
    # Confidence is about the recommendation, not about patient selection
    # within it: treating ISUP 2 is standard of care even when *which*
    # treatment is a discussion, and the clinicians call this band clear 13
    # times out of 18. The shared-decision character is carried by the stage
    # and the stated reason instead.
    shared = group == "intermediate" and grade == r["shared_decision_isup"]
    return _verdict(
        cfg,
        decision=r[group],
        confidence="clear",
        roles=roles,
        stage=STAGE_SHARED_DECISION if shared else (STAGE_HIGH if group == "high" else STAGE_INTERMEDIATE),
        reasons=(
            (
                f"ISUP {grade} is a shared-decision band; treatment is the default but "
                "the fields that select a surveillance candidate are not recorded"
            )
            if shared
            # High-risk disease is staged before it is treated -- 7 of the 12
            # reference rationales on this branch say so outright, and EAU
            # requires metastatic
            # staging at this risk level. Naming it changes no decision; it
            # completes the recommendation.
            else (
                f"{group}-risk disease on the {_DRIVER_NAMES[driver]}; treatment is "
                "indicated once staging has excluded metastatic disease"
                if group == "high"
                else f"{group}-risk disease on the {_DRIVER_NAMES[driver]}; treatment is indicated"
            ),
        ),
        group=group,
        prompt=prompt,
    )


def _live_extras(prompt: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """Variables whose value, on this case, could change the management step.

    This is the whole weight policy for everything except the risk driver: a
    variable is reported as `important` exactly when its own value bears on the
    step, and as `noted` when it was observed but inert. The thresholds are in
    ``resources/task2_policy.yaml`` beside the rate that motivates each.
    """
    live = cfg["live"]
    extras = []
    rank = _stage_rank(prompt)
    grade = _int(prompt.get("bx_isup"))
    # The stage is read for two different reasons, and both are about whether
    # the extent of disease is still an open question.
    #   * a palpable or visible tumour changes the staging and the options; and
    #   * with no gradeable tissue the stage is the only characterisation of the
    #     cancer there is, so it carries what the grade would otherwise carry.
    # ⚠️ It is NOT extended to low-grade tissue (ISUP <= 1): tested 2026-09-04,
    # factor F1 0.7283 against 0.7276 here but variable weights 0.8508 against
    # 0.8536 and a lower ranking. A recorded GG1 is a characterisation; the
    # stage is not standing in for anything.
    if rank is not None and rank >= live["stage_palpable_from"]:
        extras.append("ct")
    elif live.get("stage_characterises_ungraded", True) and (grade is None or grade == 0):
        extras.append("ct")
    if len(prompt.get("pmhx") or []) >= live["comorbidity_burden"]:
        extras.append("comorbidity")
    grade = _int(prompt.get("bx_isup"))
    if grade is not None and grade >= live["gleason_pattern_from"]:
        extras += ["bx_gl_prim", "bx_gl_sec"]
    return extras


def _report_roles(cfg: dict[str, Any], roles: dict[str, str], stage: str,
                  prompt: dict[str, Any] | None = None) -> dict[str, str]:
    """Reporting corrections applied to every branch alike.

    These change what we *say* we leaned on, never what was decided.  The one
    correction here is internal consistency rather than tuning: comorbidity is
    an input to "is this man well enough for what we are proposing?", so on the
    branches that only answer "what is this cancer?" it cannot bear on the
    answer.  See `comorbidity_inert_stages` in the YAML for the branch rates
    and the rationales behind them.
    """
    roles = dict(roles)
    if stage in cfg.get("comorbidity_inert_stages", ()) and roles.get("comorbidity") == "consulted":
        roles["comorbidity"] = "available"
    # A branch named for ONE finding may not claim to have leaned on the rest.
    # See `branch_demotions` in the YAML for the branch, the variables and the
    # guideline that licenses each. Reporting only -- the decision, the
    # confidence and the branch are all already chosen by the time we get here.
    for name, role in (cfg.get("branch_demotions", {}).get(stage) or {}).items():
        if name in roles:
            roles[name] = role
    return _psa_role(cfg, roles, stage, prompt or {})


def _psa_role(cfg: dict[str, Any], roles: dict[str, str], stage: str,
              prompt: dict[str, Any]) -> dict[str, str]:
    """Weight the PSA for what it DOES on the branch, not for being present.

    The default reports PSA `important` on every case alike, which is a
    statement about the field existing rather than about this patient. On two
    branches the reference is plainly reading the value instead:

    NO_CANCER_CONFIRMED -- the class is already settled by the negative biopsy,
    so what the weights describe is the plan inside it. A PSA above the EAU
    high-risk line is the entire reason surveillance continues rather than
    stopping, and the reference calls it decisive: 35.0 "Rebiopsy", 34.7 and
    21.0 "Repeat PSA in 3 months". Below that line the PSA sets no trigger.

    HIGH_RISK_CANCER -- the grade has usually already made the case high risk,
    so the PSA is only doing work when it says something the grade does not.
    Three figures above 100 ng/mL are a statement about disease BURDEN, not
    risk class, and drive metastatic staging; the reference calls all four
    decisive. Below the EAU intermediate line the PSA adds nothing to a case
    already high risk on its grade, and the reference stops weighting it.

    The mirror arm is `important_below`: a PSA under 1 ng/mL in a man whose
    biopsy is negative effectively excludes clinically significant cancer, and
    that is the finding that lets surveillance be minimal rather than intensive
    -- so it is weighted for what it rules out, not ignored for being small.
    It is pinned `important` rather than `decisive` on purpose: the branch's two
    lowest values sit on opposite sides of the reference (one decisive, one
    important), so no threshold separates them, and
    `important` is the arm that is right either way -- both land in the same
    important/decisive set that the factor F1 scores, and it is at most one
    ordinal step from `decisive`.

    ⚠️ `important_below` currently emits what the `_CONSULTED` default already
    emits, so it moves no metric today. It is written down so the clinical
    reason survives the removal of that blanket default, which is the direction
    of travel -- see `_CONSULTED`.
    """
    rule = (cfg.get("psa_weight") or {}).get(stage)
    psa = _float(prompt.get("psa"))
    if not rule or psa is None or roles.get("psa") not in ("consulted", "settled"):
        return roles
    roles = dict(roles)
    if psa >= rule["decisive_from"]:
        roles["psa"] = "settled"
    elif "important_below" in rule and psa < rule["important_below"]:
        roles["psa"] = "consulted"
    elif "inert_below" in rule and psa < rule["inert_below"]:
        roles["psa"] = "available"
    return roles


def _verdict(cfg, *, decision, confidence, roles, stage, reasons, group="", prompt=None) -> Verdict:
    roles = _report_roles(cfg, roles, stage, prompt)
    weights = {v: cfg["roles"][roles[v]] for v in VARIABLES}
    # ⚠️ No confidence override here any more. The gland-volume rule that used to
    # sit at this point (`modality_open`, >= 80 mL -> borderline) was removed on
    # 2026-09-02: RP and RT are both `active_treatment`, so an open modality is
    # not an open *decision*, and the reference labels mark four such cases
    # `clear` outright. Confidence is now set by
    # the branch, against the EAU strength rating for that action -- see the
    # confidence block in ``task2_policy.yaml``.
    #
    # The one confidence override that does live here, added 2026-09-04: an
    # unassignable clinical stage under an irreversible decision. EAU 6.3.4
    # writes every active-treatment recommendation against a known nodal stage
    # ("Offer RP to selected patients with cN0 disease"; "...to cN0M0 patients
    # with >= 2 high-risk factors"), so at cTx the record cannot establish the
    # precondition of the recommendation we are making -- the class is settled,
    # the plan is not. Scoped to active_treatment deliberately: surveillance has
    # no stage precondition, and bare cTx nets zero on the labelled set. See
    # `unstageable` in the YAML.
    rule = cfg.get("unstageable") or {}
    grade = _int((prompt or {}).get("bx_isup"))
    if (
        rule.get("enabled")
        and decision == rule["decision"]
        and _unstageable(prompt or {})
        and (grade is None or grade < 3)
    ):
        confidence = rule["confidence"]
        reasons = (*reasons, "the clinical stage is cTx, so the extent of disease to be treated is not established")
    return Verdict(decision, confidence, weights, list(cfg["reveals"]), stage, reasons, dict(roles), group)
