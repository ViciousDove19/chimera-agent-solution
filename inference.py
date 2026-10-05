"""Grand Challenge single-case, single-interface entry point.

The clinical-data socket identifies the active Task 1/2/3 interface. Exactly
the matching prediction and reasoning sockets are written for that invocation.
The runtime is offline. Tasks 1 and 2 use a local Gemma agent with frozen
decision policies; Task 3 is deterministic and loads no model. All three run
through the same agent runner, so every clinical section is read over the
official MCP tool server; the neural representations are consumed from the
platform-supplied socket rather than raw images.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from chimera_agent_baseline.features import FEATURES_FILENAME
from chimera_agent_baseline.output.schema import TOOL_REVEAL_INFO
from chimera_agent_baseline.run import run_agent
from chimera_agent_baseline.tools.base import CASE_DATA_FILENAMES_BY_TASK

log = logging.getLogger(__name__)

INPUT_PATH = Path(os.environ.get("CHIMERA_INPUT_PATH", "/input"))
OUTPUT_PATH = Path(os.environ.get("CHIMERA_OUTPUT_PATH", "/output"))
RESOURCE_PATH = Path(os.environ.get("CHIMERA_RESOURCE_PATH", "/opt/app/resources"))
CONFIG_PATH = Path(os.environ.get("CHIMERA_CONFIG_PATH", "/opt/app/configs/config.yaml"))
MODEL_PATH = Path(os.environ.get("CHIMERA_MODEL_PATH", "/opt/ml/model"))
LLAMA_SERVER_PATH = Path(os.environ.get("CHIMERA_LLAMA_SERVER", "/opt/app/bin/llama-server"))
LLAMA_PORT = int(os.environ.get("CHIMERA_LLAMA_PORT", "8765"))

STRUCTURED_PROMPT_SLUG = "structured-prompt"
NEURAL_REP_SLUG = "prostate-modality-level-neural-representations"

# Copied from the platform interface export in test/input. The Task 3 slug is
# truncated by Grand Challenge, while its relative path remains full.
CLINICAL_SLUG_TO_TASK: dict[str, int] = {
    "prostate-biopsy-decision-clinical-data": 1,
    "prostate-treatment-decision-clinical-data": 2,
    "prostate-time-to-recurrence-or-last-follow-up-clin": 3,
}
CLINICAL_FILES: dict[str, str] = {
    "prostate-biopsy-decision-clinical-data": "prostate-biopsy-decision-clinical-data.json",
    "prostate-treatment-decision-clinical-data": "prostate-treatment-decision-clinical-data.json",
    "prostate-time-to-recurrence-or-last-follow-up-clin": (
        "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json"
    ),
}

OUTPUT_SOCKETS: dict[int, dict[str, str]] = {
    1: {
        "decision": "prostate-biopsy-decision.json",
        "reasoning": "prostate-biopsy-decision-reasoning.json",
    },
    2: {
        "decision": "prostate-treatment-decision.json",
        "reasoning": "prostate-treatment-decision-reasoning.json",
    },
    3: {
        "decision": "prostate-time-to-recurrence-or-last-follow-up.json",
        "reasoning": "prostate-time-to-recurrence-or-last-follow-up-reasoning.json",
    },
}

SECTION_KEY_TO_REVEAL_NAME = {
    key: {
        "section_s3-mri": "radiology_report",
        "section_s3-labs": "laboratory_results",
        "section_s3-psa": "psa_trend",
        "section_s3-path": "pathology_report",
        "section_s3-prev": "previous_notes",
        "section_s3-fh": "family_history",
    }.get(key, "")
    for key, _label in TOOL_REVEAL_INFO.values()
}

# Exact reasoning-socket contract published by Grand Challenge. Keep this
# boundary explicit: the research agent may carry additional internal fields or
# call tools that are not rateable for a particular task, but those must never
# leak into the submitted JSON.
REASONING_WEIGHT_KEYS: dict[int, tuple[str, ...]] = {
    1: ("age", "fh", "cspca", "pirads", "vol", "psa", "comorbidity", "psad", "dre", "bx"),
    2: (
        "bx_gl_prim",
        "pirads",
        "bx_isup",
        "ct",
        "fh",
        "comorbidity",
        "psa",
        "bx_gl_sec",
        "age",
        "psad",
        "cspca",
    ),
}
REASONING_REVEALS: dict[int, tuple[str, ...]] = {
    1: ("family_history", "previous_notes", "laboratory_results", "psa_trend", "radiology_report"),
    2: (
        "family_history",
        "previous_notes",
        "laboratory_results",
        "psa_trend",
        "radiology_report",
        "pathology_report",
    ),
}


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2)
        stream.write("\n")


def _safe_socket_path(relative_path: str) -> Path:
    root = INPUT_PATH.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Input socket escapes /input: {relative_path!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Input socket file not found: {candidate}")
    return candidate


def _socket_paths() -> dict[str, Path]:
    """Resolve socket paths from metadata, with direct-file discovery fallback."""
    metadata_path = INPUT_PATH / "inputs.json"
    if metadata_path.is_file():
        inputs = _load_json(metadata_path)
        if not isinstance(inputs, list):
            raise TypeError("/input/inputs.json must contain a list")
        resolved: dict[str, Path] = {}
        for item in inputs:
            socket = item.get("socket") if isinstance(item, dict) else None
            if not isinstance(socket, dict) or not socket.get("slug") or not socket.get("relative_path"):
                raise ValueError("Each inputs.json item must contain socket.slug and socket.relative_path")
            resolved[str(socket["slug"])] = _safe_socket_path(str(socket["relative_path"]))
        return resolved

    candidates = {
        STRUCTURED_PROMPT_SLUG: f"{STRUCTURED_PROMPT_SLUG}.json",
        NEURAL_REP_SLUG: f"{NEURAL_REP_SLUG}.json",
        **CLINICAL_FILES,
    }
    return {slug: INPUT_PATH / filename for slug, filename in candidates.items() if (INPUT_PATH / filename).is_file()}


def _detect_task(slug_to_path: dict[str, Path]) -> int:
    present = [(slug, task) for slug, task in CLINICAL_SLUG_TO_TASK.items() if slug in slug_to_path]
    if len(present) != 1:
        raise ValueError(
            "Expected exactly one clinical-data socket; "
            f"found {[slug for slug, _task in present]} among {sorted(slug_to_path)}"
        )
    missing = {STRUCTURED_PROMPT_SLUG, NEURAL_REP_SLUG} - slug_to_path.keys()
    if missing:
        raise ValueError(f"Missing required shared input socket(s): {sorted(missing)}")
    return present[0][1]


def _shape(value: Any, depth: int = 0) -> Any:
    """Describe parsed JSON without logging patient values or embedding data."""
    if isinstance(value, dict):
        if depth >= 2:
            return {"type": "object", "keys": sorted(map(str, value))}
        return {str(key): _shape(item, depth + 1) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return {"type": "array", "length": len(value), "item": _shape(value[0], depth + 1) if value else None}
    return {"type": type(value).__name__}


def _log_input_contract(slug_to_path: dict[str, Path]) -> None:
    files = sorted(path for path in INPUT_PATH.rglob("*") if path.is_file())
    log.info("/input listing: %s", [(str(path.relative_to(INPUT_PATH)), path.stat().st_size) for path in files])
    for slug, path in sorted(slug_to_path.items()):
        log.info(
            "input socket %s -> %s shape=%s", slug, path.name, json.dumps(_shape(_load_json(path)), sort_keys=True)
        )


def _case_id(prompt: dict[str, Any], clinical: dict[str, Any]) -> str:
    for payload in (prompt, clinical):
        for key in ("case_id", "patient_id", "id"):
            value = payload.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return "gc-case"


def _materialise_case(
    task: int,
    case_id: str,
    prompt: dict[str, Any],
    clinical: dict[str, Any],
    features: dict[str, Any],
    root: Path,
) -> None:
    """Create the one-case tree consumed by the research agent runner."""
    case_dir = root / f"task{task}" / "agent_input" / case_id
    prompt = {**prompt, "case_id": case_id, "task": task}
    clinical = {**clinical, "case_id": case_id}
    features = {**features, "case_id": case_id}
    _write_json(case_dir / "prompt.json", prompt)
    _write_json(case_dir / CASE_DATA_FILENAMES_BY_TASK[task], clinical)
    _write_json(case_dir / FEATURES_FILENAME, features)


def _load_agent_config(data_root: Path, output_dir: Path, task: int):
    cfg = OmegaConf.load(CONFIG_PATH)
    OmegaConf.update(cfg, "paths.data_root", str(data_root))
    OmegaConf.update(cfg, "paths.output_dir", str(output_dir))
    OmegaConf.update(cfg, "paths.resource_dir", str(RESOURCE_PATH))
    OmegaConf.update(cfg, "paths.model_dir", str(MODEL_PATH))
    OmegaConf.update(cfg, "agent.tasks", [task])
    OmegaConf.update(cfg, "agent.concurrency", 1)
    OmegaConf.update(cfg, "model.provider", "openai")
    # The name the llama.cpp server reports; it is the model file's own name.
    OmegaConf.update(cfg, "model.model_id", _model_file().name)
    # Tasks 1 and 2 run the agentic policy. The frozen policies are not in this
    # path any more -- see docs/agentic_plan.md.
    OmegaConf.update(cfg, "agent.mode", "agentic")
    # Qwen bills its chain of thought against max_tokens, which would leave the
    # flat KEY: value blocks the agents emit truncated to nothing.
    OmegaConf.update(cfg, "model.extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
    OmegaConf.update(cfg, "generation.temperature", 0.0)
    OmegaConf.update(cfg, "generation.max_new_tokens", 1024)
    OmegaConf.update(cfg, "model.base_url", f"http://127.0.0.1:{LLAMA_PORT}/v1")
    OmegaConf.update(cfg, "model.request_timeout", 60)
    return cfg


def _model_file() -> Path:
    configured = os.environ.get("CHIMERA_GGUF_MODEL")
    if configured:
        candidate = MODEL_PATH / configured
        if not candidate.is_file():
            raise FileNotFoundError(f"Configured GGUF model is missing: {candidate}")
        return candidate
    candidates = sorted(MODEL_PATH.glob("*.gguf"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one GGUF model under {MODEL_PATH}; found {candidates}")
    return candidates[0]


def _wait_for_server(process: subprocess.Popen, timeout_seconds: float = 180.0) -> None:
    health_url = f"http://127.0.0.1:{LLAMA_PORT}/health"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"llama-server exited during startup with code {exit_code}")
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:  # noqa: S310 - loopback only
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"llama-server did not become healthy within {timeout_seconds:.0f} seconds")


def _start_llama_server() -> subprocess.Popen:
    model_file = _model_file()
    command = [
        str(LLAMA_SERVER_PATH),
        "--model",
        str(model_file),
        "--host",
        "127.0.0.1",
        "--port",
        str(LLAMA_PORT),
        "--ctx-size",
        os.environ.get("CHIMERA_CONTEXT_SIZE", "16384"),
        "--parallel",
        "1",
        "--n-gpu-layers",
        os.environ.get("CHIMERA_N_GPU_LAYERS", "99"),
        "--jinja",
    ]
    log.info("Starting the offline model server from %s", model_file)
    process = subprocess.Popen(command)
    try:
        _wait_for_server(process)
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        raise
    return process


def _stop_llama_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _decision_value(task: int, prediction: dict[str, Any]) -> Any:
    if task == 1:
        return prediction["biopsy_decision"]
    if task == 2:
        return prediction["treatment_recommendation"]["primary"]
    return {
        "event": int(prediction["event"]),
        "months_to_recurrence": float(prediction["months_to_recurrence"]),
    }


def _reasoning_value(task: int, prediction: dict[str, Any]) -> Any:
    if task == 3:
        return prediction["free_text"]
    allowed_reveals = set(REASONING_REVEALS[task])
    reveals: list[str] = []
    for entry in prediction.get("reveal_sequence", []):
        name = SECTION_KEY_TO_REVEAL_NAME.get(entry.get("key", ""), "") if isinstance(entry, dict) else ""
        if name in allowed_reveals and name not in reveals:
            reveals.append(name)
    weights = prediction.get("variable_weights", {})
    return {
        "free_text": prediction["free_text"],
        "confidence": prediction["confidence"],
        "variable_weights": {key: weights[key] for key in REASONING_WEIGHT_KEYS[task]},
        "reveal_sequence": reveals,
    }


def _assert_empty_output() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    existing = [str(path.relative_to(OUTPUT_PATH)) for path in OUTPUT_PATH.rglob("*") if path.is_file()]
    if existing:
        raise RuntimeError(f"/output must be empty for a single-case invocation; found {existing}")


def run() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _assert_empty_output()
    slug_to_path = _socket_paths()
    _log_input_contract(slug_to_path)
    task = _detect_task(slug_to_path)
    clinical_slug = next(slug for slug in CLINICAL_SLUG_TO_TASK if slug in slug_to_path)

    prompt = _load_json(slug_to_path[STRUCTURED_PROMPT_SLUG])
    clinical = _load_json(slug_to_path[clinical_slug])
    features = _load_json(slug_to_path[NEURAL_REP_SLUG])
    if not all(isinstance(payload, dict) for payload in (prompt, clinical, features)):
        raise TypeError("All three input sockets must contain JSON objects")

    case_id = _case_id(prompt, clinical)
    # Every task runs through the same agent runner, so every task reaches its
    # clinical sections over the official MCP tool server. Task 3 differs only
    # in needing no language model: its Cox/RMST head and templated rationale
    # are deterministic, so no llama-server is started for it.
    with tempfile.TemporaryDirectory(prefix="chimera-gc-", dir="/tmp") as temp_dir:
        temp_root = Path(temp_dir)
        data_root = temp_root / "input"
        agent_output = temp_root / "output"
        _materialise_case(task, case_id, prompt, clinical, features, data_root)
        cfg = _load_agent_config(data_root, agent_output, task)
        server: subprocess.Popen | None = None
        try:
            if task != 3:
                server = _start_llama_server()
            asyncio.run(run_agent(cfg))
        finally:
            _stop_llama_server(server)
        prediction = _load_json(agent_output / f"task{task}" / case_id / "prediction.json")
    sockets = OUTPUT_SOCKETS[task]
    _write_json(OUTPUT_PATH / sockets["decision"], _decision_value(task, prediction))
    _write_json(OUTPUT_PATH / sockets["reasoning"], _reasoning_value(task, prediction))
    log.info("Task %d complete; wrote exactly %s", task, sorted(sockets.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
