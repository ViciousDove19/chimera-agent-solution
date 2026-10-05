"""Agent entry-point.

Hydra-driven runner. Walks the hierarchical input tree
``<data_root>/task<N>/agent_input/<case>/`` and runs either the LangGraph ReAct
+ form-fill graph (Tasks 1/2) or the deterministic Cox/rationale path (Task 3);
both read their clinical evidence through the official MCP tool server,
one task at a time, writing
``<output_dir>/task<N>/<case>/prediction.json``.

By default every task present under ``data_root`` is run (``agent.tasks``);
missing task dirs are skipped. The model is loaded once and reused across
tasks. The Grand Challenge container uses the same layout, rooted at
``/input`` / ``/output``.

Usage::

    make run                                       # all tasks under data/
    make run RUN_ARGS="agent.tasks=[2]"            # just task 2
    make run RUN_ARGS="+experiment=qwen_local"     # swap to Qwen
    make run RUN_ARGS="agent.limit=5"              # first 5 cases per task
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from omegaconf import DictConfig

from chimera_agent_baseline.case_loader import load_cases
from chimera_agent_baseline.features import FeatureStore
from chimera_agent_baseline.models import load_model
from chimera_agent_baseline.tools.task3_head import load_head
from chimera_agent_baseline.tools.task3_recurrence import build_task3_prediction
from chimera_agent_baseline.utils import setup_logging

load_dotenv()
log = logging.getLogger(__name__)


_VALID_TASKS = (1, 2, 3)


def _task_input_dir(cfg: DictConfig, task: int) -> Path:
    return Path(cfg.paths.data_root) / f"task{task}" / "agent_input"


def _run_plan(cfg: DictConfig) -> list[tuple[int, Path]]:
    """Resolve ``agent.tasks`` to ``(task_int, input_dir)`` pairs that exist."""
    plan: list[tuple[int, Path]] = []
    for raw in cfg.agent.tasks:
        task = int(raw)
        if task not in _VALID_TASKS:
            raise ValueError(f"Unknown task {task!r} in agent.tasks; expected one of {list(_VALID_TASKS)}")
        input_dir = _task_input_dir(cfg, task)
        if input_dir.is_dir():
            plan.append((task, input_dir))
        else:
            log.warning("Skipping task %d: %s not found", task, input_dir)
    if not plan:
        raise FileNotFoundError(f"No task data found under {cfg.paths.data_root} for tasks {list(cfg.agent.tasks)}")
    return plan


def _filter_queries(queries: list[dict], cfg: DictConfig) -> list[dict]:
    """Apply optional ``cfg.agent.pids`` / ``cfg.agent.limit`` subset filters."""
    pids = cfg.agent.get("pids")
    if pids:
        wanted = set(pids)
        out = [q for q in queries if q["case_id"] in wanted]
        missing = wanted - {q["case_id"] for q in out}
        if missing:
            log.warning("Requested pids not found in input dir: %s", sorted(missing))
        log.info("Filtered to %d hand-picked cases: %s", len(out), [q["case_id"] for q in out])
        return out
    limit = cfg.agent.get("limit")
    if limit:
        out = queries[: int(limit)]
        log.info("Limiting to first %d cases", len(out))
        return out
    return queries


def _mcp_args(cfg: DictConfig, input_dir: Path, registry: str) -> list[str]:
    args = [
        "-m",
        "chimera_agent_baseline.mcp_server",
        "--data-dir",
        str(input_dir),
        "--resource-dir",
        str(cfg.paths.resource_dir),
        "--tool-registry",
        registry,
    ]
    # Optional image-embedding predictor tool (off by default).
    predictor = cfg.agent.get("predictor")
    if predictor and predictor.get("enabled"):
        args += ["--enable-predictor"]
    return args


#: The Task 3 sections the Cox head and the rationale actually read, in the
#: order they are revealed. ``get_previous_notes`` and ``get_family_history``
#: are exposed by the registry but never called: their univariate C-indices
#: were 0.49-0.51 and they ground nothing in the rationale.
TASK3_REVEAL_TOOLS: tuple[str, ...] = (
    "get_mri_report",
    "get_pathology_report",
    "get_surgical_pathology_report",
)


def _tool_payload(raw: Any) -> dict[str, Any] | None:
    """Decode one MCP tool result into its JSON object.

    ``load_mcp_tools`` returns either the raw JSON string or a list of content
    blocks (``[{"type": "text", "text": "..."}]``) depending on the adapter
    version, so both shapes are unwrapped here.
    """
    if isinstance(raw, list):
        texts = [block.get("text", "") for block in raw if isinstance(block, dict) and block.get("type") == "text"]
        raw = "".join(texts)
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


async def _reveal_task3_sections(tools: list, case_id: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Fetch Task 3's clinical sections over MCP.

    Returns the reassembled clinical record and the tools that actually
    returned a section, in call order. A tool that is missing, errors, or has
    no data for the case is skipped: the feature extractor already treats an
    absent section as a missing value, so a biopsy-naive case is not a failure.
    """
    by_name = {tool.name: tool for tool in tools}
    clinical: dict[str, Any] = {"case_id": case_id}
    revealed: list[str] = []
    for tool_name in TASK3_REVEAL_TOOLS:
        tool = by_name.get(tool_name)
        if tool is None:
            log.warning("Task 3: %s is not exposed by the MCP server; skipping", tool_name)
            continue
        payload = _tool_payload(await tool.ainvoke({"case_id": case_id}))
        if payload is None or "error" in payload:
            log.warning("Task 3: %s returned no section for %s: %s", tool_name, case_id, payload)
            continue
        sections = {key: value for key, value in payload.items() if key not in {"case_id", "note"}}
        if not sections:
            continue
        clinical.update(sections)
        revealed.append(tool_name)
    return clinical, tuple(revealed)


async def _run_task3(cfg: DictConfig, queries: list[dict], tools: list, input_dir: Path, task_dir: Path) -> int:
    """Run Task 3 over the MCP session; returns the number of predictions written."""
    head = load_head(cfg.paths.resource_dir)
    # The modality-level neural representations are a platform-supplied input
    # socket, like prompt.json -- not a masked EHR section behind a tool.
    feature_store = FeatureStore(input_dir)
    n_done = 0
    for query in queries:
        case_id = query["case_id"]
        try:
            clinical, revealed = await _reveal_task3_sections(tools, case_id)
            structured = build_task3_prediction(
                case_id=case_id,
                prompt=query.get("structured_prompt") or {},
                clinical=clinical,
                head=head,
                prostatectomy_vectors=feature_store.get_origin(case_id, "Prostatectomy slide"),
                revealed_tools=revealed,
            )
            case_dir = task_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "prediction.json").write_text(json.dumps(structured, indent=2))
            n_done += 1
        except Exception:
            log.exception("Task 3: deterministic prediction failed for %s", case_id)
    if n_done != len(queries):
        log.warning("Task 3: %d of %d deterministic cases failed", len(queries) - n_done, len(queries))
    log.info("Task 3: wrote %d deterministic predictions under %s", n_done, task_dir)
    return n_done


async def _run_agentic(
    cfg: DictConfig, task_int: int, queries: list[dict], tools: list, task_dir: Path, sem, model
) -> int:
    """Run Task 1 or 2 through the agentic policy; returns predictions written."""
    # Imported here, not at module scope: orchestrate reads `_tool_payload`
    # back out of this module.
    from .agentic import briefs2
    from .agentic.orchestrate import load_thresholds
    from .agentic.orchestrate import run_case as run_case1
    from .agentic.orchestrate2 import run_case as run_case2

    raw = load_thresholds(cfg.paths.resource_dir)
    if task_int == 1:
        thresholds, run_case = raw, run_case1
    else:
        thresholds, run_case = briefs2.flatten_thresholds(raw["_raw"]), run_case2
    by_name = {tool.name: tool for tool in tools}

    async def one(query: dict) -> bool:
        async with sem:
            case_id = query["case_id"]
            try:
                prediction = await run_case(model, by_name, query, thresholds)
            except Exception:
                log.exception("Task %d (agentic): case %s failed; no prediction written", task_int, case_id)
                return False
        case_dir = task_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "prediction.json").write_text(json.dumps(prediction, indent=2))
        return True

    results = await asyncio.gather(*(one(q) for q in queries))
    n_done = sum(1 for ok in results if ok)
    log.info("Task %d (agentic): wrote %d of %d predictions under %s", task_int, n_done, len(queries), task_dir)
    return n_done


async def _run_task(cfg: DictConfig, task_int: int, input_dir: Path, model) -> int:
    """Run every case for one task; returns the number of predictions written."""
    registry = f"task{task_int}"
    queries = _filter_queries(load_cases(input_dir, task=task_int), cfg)

    log.info("Task %d: starting MCP server (data_dir=%s)", task_int, input_dir)
    client = MultiServerMCPClient(
        {
            "chimera": {
                "command": sys.executable,
                "args": _mcp_args(cfg, input_dir, registry),
                "env": dict(os.environ),
                "transport": "stdio",
            },
        }
    )

    # Output mirrors the agent-input hierarchy: <output_dir>/task<N>/<case_id>/prediction.json
    task_dir = Path(cfg.paths.output_dir) / f"task{task_int}"

    # ``cfg.agent.concurrency`` cases are in flight at once. Keep it at 1 for
    # provider=vllm: that backend is the in-process offline ``vllm.LLM``, which
    # serialises generate() calls, so concurrency buys nothing and only
    # interleaves the logs. It pays off with provider=openai against a vLLM
    # OpenAI server, where the server's scheduler batches the concurrent
    # requests -- which is the point of running one locally for iteration.
    concurrency = max(1, int(cfg.agent.get("concurrency", 1) or 1))
    sem = asyncio.Semaphore(concurrency)

    # ONE stdio session for the whole task. ``client.get_tools()`` returns tools
    # that open a fresh session per call, which respawns the MCP server process
    # and re-reads the case index and the guidelines DB on every single tool
    # call (~1.5 s each, ~5-7 calls per case). Binding the tools to a session
    # held open here loads that state once.
    async with client.session("chimera") as session:
        tools = await load_mcp_tools(session)
        log.info(
            "Task %d: loaded %d tools from MCP server (concurrency=%d)",
            task_int,
            len(tools),
            concurrency,
        )

        # Task 3 takes no LLM: the prostatectomy-slide Cox head owns the
        # numeric outcome and a parser/template grounds the rationale. It still
        # goes through this MCP session -- the challenge requires the official
        # interface for every task, and reading the clinical-data file directly
        # would be an alternative tool channel.
        if task_int == 3:
            return await _run_task3(cfg, queries, tools, input_dir, task_dir)

        # Tasks 1/2: the agentic policy, five agent roles over the same MCP
        # session. The ReAct `graph` mode was retired on 2026-09-09 -- it had
        # not been the submission path since the agentic policy landed, and it
        # was the only caller of `search_guidelines` and so of the RAG stack.
        # `_run_agentic` does its own gather and its own completion logging.
        return await _run_agentic(cfg, task_int, queries, tools, task_dir, sem, model)


async def run_agent(cfg: DictConfig) -> None:
    """Run every task present under ``data_root`` (model loaded once, reused)."""
    plan = _run_plan(cfg)
    log.info("Run plan: tasks %s", [t for t, _ in plan])

    # Task 3 has a deterministic head + rationale path and needs no language
    # model.  Mixed-task runs still load one model for Tasks 1/2 and reuse it.
    model = load_model(cfg) if any(task_int != 3 for task_int, _ in plan) else None

    total = 0
    for task_int, input_dir in plan:
        total += await _run_task(cfg, task_int, input_dir, model)
    log.info("Done. Wrote %d predictions across %d task(s).", total, len(plan))


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)

    # No embedding service: `search_guidelines` and the RAG stack behind it
    # were removed on 2026-09-09 with the `graph` mode that was their only
    # caller. The agentic policy reaches evidence through the clinical MCP
    # tools alone (briefs.REVEAL_ITEMS / briefs2.REVEAL_ITEMS).
    asyncio.run(run_agent(cfg))


if __name__ == "__main__":
    main()
