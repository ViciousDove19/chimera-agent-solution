"""Task-3 recurrence head: frozen slide embeddings -> predicted months.

Task 3's leaderboard metric is Harrell's C-index alone, and the evaluator
computes it over the submitted ``months_to_recurrence`` with *shorter =
higher risk* (``evaluate.py::concordance_index``). The LLM has no way to
produce a well-ordered continuous risk from a slide embedding, so the number
is set here deterministically and the agent's free-text rationale is kept.

The artifact is a linear Cox proportional-hazards model over the
prostatectomy-slide embedding, fitted by ``scripts/train_task3_head.py`` and
baked into the image at ``resources/task3_head.json`` (~44 KB). Inference is
pure Python: standardise, dot with the weights, convert the risk score to a
restricted mean survival time. See the training script for why RMST rather
than median survival.

The production runner handles missing artifacts or unusable embeddings with a
schema-valid deterministic fallback. ``apply_head`` remains as a compatibility
helper for older callers that first assembled an agent prediction.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from chimera_agent_baseline.features import FeatureStore, Vector

log = logging.getLogger(__name__)

HEAD_FILENAME = "task3_head.json"


class ClinicalStage:
    """Stage B: linear Cox over the clinical features plus the stage-A risk score."""

    def __init__(self, payload: dict[str, Any]):
        self.names: list[str] = list(payload["features"])
        self.median: list[float] = [float(x) for x in payload["median"]]
        self.mean: list[float] = [float(x) for x in payload["mean"]]
        self.scale: list[float] = [float(x) for x in payload["scale"]]
        self.weights: list[float] = [float(x) for x in payload["weights"]]
        self.embedding_risk_mean: float = float(payload["embedding_risk_mean"])
        self.coefficient_constraint: str | None = payload.get("coefficient_constraint")
        if self.coefficient_constraint == "nonnegative" and any(weight < 0 for weight in self.weights):
            raise ValueError("clinical artifact violates its nonnegative coefficient constraint")

    def risk(self, features: list[float], embedding_risk: float | None) -> float | None:
        """Combined risk score, or None if the feature vector is the wrong width."""
        if len(features) != len(self.names):
            log.warning("Clinical stage expects %d features, got %d — skipping", len(self.names), len(features))
            return None
        filled = [m if x is None or math.isnan(x) else float(x) for x, m in zip(features, self.median)]
        filled.append(self.embedding_risk_mean if embedding_risk is None else embedding_risk)
        return sum(((x - m) / s if s else 0.0) * w for x, m, s, w in zip(filled, self.mean, self.scale, self.weights))


class Task3Head:
    """Two-stage Cox + RMST head loaded from a JSON artifact.

    Stage A is a linear Cox on the 960-d prostatectomy-slide embedding; its
    risk score is a single covariate in stage B, alongside the clinical
    features parsed from the case's own report prose. A stage-A-only artifact
    (``clinical`` absent) still loads and behaves exactly as before.
    """

    def __init__(self, payload: dict[str, Any]):
        self.origin: str = payload["origin"]
        self.dim: int = int(payload["dim"])
        self.mean: list[float] = payload["mean"]
        self.scale: list[float] = payload["scale"]
        self.weights: list[float] = payload["weights"]
        self.horizon: float = float(payload["horizon_months"])
        self.event_threshold: float = float(payload["event_threshold_months"])
        # Ascending (time, cumulative baseline hazard) steps.
        self.baseline: list[tuple[float, float]] = [(float(t), float(h)) for t, h in payload["baseline_hazard"]]
        # How the risk score is turned back into months. Both maps are strictly
        # decreasing in risk, so both induce the same ordering and the C-index
        # cannot tell them apart; they differ only in the (unranked) per-case
        # time-closeness term. See `_to_months`.
        self.time_map: str = str(payload.get("time_map", "rmst"))

        clinical = payload.get("clinical")
        self.clinical: ClinicalStage | None = ClinicalStage(clinical) if clinical else None

    def _risk(self, vectors: list[Vector]) -> float | None:
        """Mean-pool a case's vectors, standardise, and dot with the Cox weights."""
        usable = [v for v in vectors if len(v) == self.dim]
        if not usable:
            return None
        pooled = [sum(col) / len(usable) for col in zip(*usable)]
        return sum(((x - m) / s if s else 0.0) * w for x, m, s, w in zip(pooled, self.mean, self.scale, self.weights))

    def predict(self, vectors: list[Vector], features: list[float] | None = None) -> tuple[float, int] | None:
        """Return ``(months, event)``, or None if the case cannot be scored at all.

        With a two-stage artifact and *features* supplied, a case missing its
        embedding is still scorable: the stage-A risk falls back to the
        training-set mean, so the clinical covariates carry the prediction.
        """
        risk = self._risk(vectors)
        if self.clinical is not None:
            # Always go through stage B so every case is scored on one risk
            # scale, whichever of the two inputs happens to be missing.
            if features is None:
                features = [math.nan] * len(self.clinical.names)
            risk = self.clinical.risk(features, risk)
        if risk is None:
            return None
        return self._to_months(risk)

    def _to_months(self, risk: float) -> tuple[float, int]:
        """Compose the risk score with the fitted monotone map."""
        if self.time_map.startswith("quantile:"):
            return self._survival_quantile(risk, float(self.time_map.split(":", 1)[1]))
        return self._rmst(risk)

    def _survival_quantile(self, risk: float, q: float) -> tuple[float, int]:
        """inf{t : S(t|r) < q}, extrapolating H0 past its last observed step.

        ⚠️ The extrapolation is the point, not a detail. Clamping at the horizon
        instead -- which is what a plain median-survival read does -- collapses
        every patient who never reaches S=q onto one value, and those ties score
        0.5 each in the C-index. Measured out-of-fold on the 75 released cases,
        a clamped median costs C-index 0.827 -> 0.711. Extrapolating keeps the
        map strictly decreasing, so the ordering is preserved exactly.
        """
        exp_r = math.exp(risk)
        need = -math.log(q) / exp_r          # required cumulative baseline hazard
        prev_t, prev_h = 0.0, 0.0
        for t, h in self.baseline:
            if h >= need:
                if h > prev_h:
                    months = prev_t + (t - prev_t) * ((need - prev_h) / (h - prev_h))
                else:
                    months = t
                return months, int(months <= self.event_threshold)
            prev_t, prev_h = t, h
        # Past the last event time: extend H0 at its average slope so that two
        # low-risk patients still separate rather than sharing the horizon.
        slope = (prev_h / prev_t) if prev_t > 0 else 1e-9
        months = need / slope if slope > 0 else self.horizon
        return months, int(months <= self.event_threshold)

    def _rmst(self, risk: float) -> tuple[float, int]:
        exp_r = math.exp(risk)

        # RMST = integral of S(t) = exp(-H0(t) * e^r) from 0 to the horizon,
        # over the piecewise-constant baseline hazard.
        months = 0.0
        prev_t, prev_h = 0.0, 0.0
        for t, h in self.baseline:
            if t > self.horizon:
                break
            months += math.exp(-prev_h * exp_r) * (t - prev_t)
            prev_t, prev_h = t, h
        months += math.exp(-prev_h * exp_r) * (self.horizon - prev_t)

        return months, int(months <= self.event_threshold)


def load_head(resource_dir: str | Path) -> Task3Head | None:
    """Load the head artifact from *resource_dir*, or None if absent/unreadable."""
    path = Path(resource_dir) / HEAD_FILENAME
    if not path.is_file():
        log.info("No task-3 head at %s — leaving the agent's prediction as-is", path)
        return None
    try:
        return Task3Head(json.loads(path.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        log.warning("Ignoring unusable task-3 head at %s: %s", path, exc)
        return None


def apply_head(
    structured: dict[str, Any],
    head: Task3Head | None,
    feature_store: FeatureStore | None,
    case_id: str,
) -> dict[str, Any]:
    """Overwrite ``months_to_recurrence``/``event`` in *structured* from the head.

    Returns *structured* unchanged when the head cannot score this case, so a
    missing artifact or a case without the prostatectomy embedding degrades to
    the baseline behaviour rather than failing the run.
    """
    if head is None or feature_store is None:
        return structured
    vectors = (feature_store.get(case_id) or {}).get(head.origin) or []
    if not vectors:
        log.info("Case %s has no %r embedding — keeping the agent's prediction", case_id, head.origin)
        return structured
    result = head.predict(vectors)
    if result is None:
        log.warning("Case %s: %r embedding has unexpected width — keeping the agent's prediction", case_id, head.origin)
        return structured

    months, event = result
    log.info(
        "Case %s: head predicts %.1f months (event=%d), replacing agent's %r",
        case_id,
        months,
        event,
        structured.get("months_to_recurrence"),
    )
    structured["months_to_recurrence"] = round(months, 2)
    structured["event"] = event
    return structured
