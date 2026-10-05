# CHIMERA-Agent: a verifiable policy-graph agent for prostate cancer decisions

Our submission to the [CHIMERA-Agent challenge](https://chimera-agent.grand-challenge.org/chimera-agent/)
(MICCAI 2026). The challenge asks for an agent that makes three sequential
prostate-cancer decisions from **incomplete, multimodal clinical records**, and
scores the reasoning trace as well as the prediction. Each trace records the
agent's confidence, how much weight it gave each variable, which record sections
it opened, and a free-text rationale.

| Task | Question | Output |
|---|---|---|
| 1 | Should this patient be biopsied? | `yes` / `no` |
| 2 | Which management strategy? | `active_surveillance`, `continued_surveillance`, `watchful_waiting` or `active_treatment` |
| 3 | When will biochemical recurrence follow prostatectomy? | months to recurrence + event flag |

This repository contains **exactly the code that runs inside our final
Docker image** (built 2026-09-14), and nothing else. There are no experiments, notebooks or
data here.

---

## The method in one paragraph

Decisions are made by **guideline-derived policy graphs** rather than by the
language model. Each graph is a YAML file: one node per clinical question,
edges as guard expressions over named variables, and leaves as
recommendations. Because the graph is data, it can be swept exhaustively, and
every complete combination of inputs reaches exactly one answer. There are no
undefined, ambiguous or unreachable paths, and every variable declares what the
agent does when that variable is missing. Confidence is **computed** by an
ordered rule, because the small local models we tested could not rank
competing reasons reliably. The **language model** (Gemma 4 E2B, 4-bit, served
by `llama.cpp` inside the container) does what it is good at. Specialist
prompts read the record and decide which sections to retrieve, and a writer
turns the decision and its drivers into a short rationale. A validator checks
that rationale against the text the evaluator can actually see. Every clinical
section is read through the challenge's official **MCP tool server**, and the
reported `reveal_sequence` is the list of calls that actually succeeded, in
call order.

```
            structured prompt (always visible)
                         │
        ┌────────────────┴────────────────┐
        │  specialist agents (LLM)        │──► MCP tools ──► clinical sections
        │  read, request evidence,        │◄──────────────── (radiology, labs,
        │  summarise key findings         │                   pathology, notes…)
        └────────────────┬────────────────┘
                         ▼
        policy graph (YAML)  ──►  decision
        confidence rule      ──►  clear | borderline | uncertain
        weight rules         ──►  per-variable importance
                         │
                         ▼
        rationale writer (LLM) ──► grounding validator ──► prediction + trace
```

---

## How each task is solved

### Task 1: biopsy decision

* **Routing.** The encounter type (biopsy-naive, prior negative biopsy, or
  known prostate cancer) selects one of three tracks in
  [`resources/task1_graph.yaml`](resources/task1_graph.yaml). The graph has
  16 nodes, 4 leaves and 50 edges, and was transcribed from the EAU 2026
  diagnostic recommendations.
* **Evidence.** Four specialists, called *eligibility*, *history*, *radiology*
  and *biochemical*, each read the case and may request the sections they own.
  For example, the history specialist can request the previous notes to recover
  the grade of an earlier biopsy. Reveals happen only through MCP.
* **Decision.** [`agentic/policy_graph.py`](src/chimera_agent_baseline/agentic/policy_graph.py)
  walks the graph. Readings the graph needs that are not raw fields, such as how
  far the imaging sits above the recorded grade or how strongly the PSA,
  PSA density and PSA velocity disagree with the MRI, are derived variables
  declared in the YAML.
* **Confidence.** `briefs.confidence_rule` reads three rungs in order: a
  named absent fact, then a conflict between findings, then a grey-zone
  reading. It returns at the first hit, so a strong present finding is never
  overruled by a weak missing one.
* **Variable weights.** The specialists emit them, and each specialist is
  critiqued once against its own clinical weight rules.

### Task 2: management strategy

* **Decision** comes from [`resources/task2_graph.yaml`](resources/task2_graph.yaml)
  (9 nodes, 4 leaves, 22 edges). The graph reads the biopsy grade first, then
  stage, PSA and imaging. Grade group 2 gets a dedicated surveillance ladder
  with explicit preconditions, and a missing fact never qualifies a man for
  surveillance. A severe-symptoms switch applies only while the cancer
  question is quiescent.
* **Pathology reveals are sensitivity-gated.** The pathology report is fetched
  only when cribriform or intraductal morphology could change the answer.
  The evaluator rewards revealing as little as possible, so this matters.
* **Confidence** comes from `briefs2.confidence_rule`. **Variable weights**
  come from the deterministic reference policy
  ([`policy.py`](src/chimera_agent_baseline/policy.py) →
  [`task2.py`](src/chimera_agent_baseline/task2.py)), which weights each variable
  for the role it plays on the branch that decided the case.
* **Next step.** The rationale closes on a per-branch next step from
  `task2_policy.yaml`. For example, high-risk disease is staged before a
  modality is chosen.

### Task 3: time to biochemical recurrence

No language model is loaded. The three relevant documents are revealed over
MCP, and a **two-stage Cox model** predicts risk:

1. **Stage A** turns the 960-dimensional prostatectomy-slide embedding (a
   platform-supplied input) into a risk score.
2. **Stage B** combines that score with 11 report-visible clinical features
   (surgical ISUP grade, tertiary pattern, pT stage, margins, extraprostatic
   extension, seminal vesicle invasion, lymphovascular invasion, nodal status,
   PI-RADS, PSA density and biopsy ISUP). All coefficients are constrained
   non-negative, so each feature can only raise risk, which matches its
   clinical sign.

A restricted-mean survival time is decoded into months, and a deterministic
rationale states the model's reading without claiming per-feature attribution.
The fitted head is [`resources/task3_head.json`](resources/task3_head.json).

### Rationale grounding (Tasks 1 and 2)

The evaluator's judge sees only the clinical-data sections, not the structured
prompt. So the evidence packet has two tiers:

* `observed`: judge-visible values. Only these may be quoted.
* `background`: context such as age, examination or IPSS. The model may use it
  but may not quote it.

The validator enforces a few more rules. It rejects any number or claim not
anchored in visible text. It rejects any statement that something is "missing"
unless that absence can be shown. It enforces a tone that matches the
confidence label: a `clear` note may not hedge, and an `uncertain` note must
name what is missing. A malformed reply is retried, and a hard failure falls
back to a code-rendered rationale that passes the same validator.

---

## Repository structure

```
.
├── Dockerfile                  # Two-stage build: compiles llama-server (CUDA), then a slim Python runtime
├── .dockerignore               # Deny-all allow-list: exactly the files below enter the image
├── requirements.txt            # Pinned runtime dependencies
├── inference.py                # Container entry point: one case per invocation (Grand Challenge sockets)
├── configs/
│   ├── config.yaml             # Default runtime config (Hydra/OmegaConf)
│   └── experiment/             # Model/endpoint overrides used during development
├── templates/prompts/          # Case prompt template
├── resources/
│   ├── task1_graph.yaml        # Task 1 decision graph  ← the Task 1 policy
│   ├── task2_graph.yaml        # Task 2 decision graph  ← the Task 2 policy
│   ├── agentic_task1.yaml      # Thresholds read by the Task 1/2 agents and confidence rules
│   ├── task1_policy.yaml       # Reference policy: rationale prose, per-branch next steps
│   ├── task2_policy.yaml       # Reference policy: Task 2 variable weights, next steps
│   └── task3_head.json         # Fitted Task 3 survival head
└── src/chimera_agent_baseline/
    ├── run.py                  # Agent runner: starts the MCP server, dispatches cases per task
    ├── mcp_server.py           # Official-interface MCP tool server over the case's clinical record
    ├── agentic/                # ── Tasks 1 & 2 ──
    │   ├── orchestrate.py      #   Task 1 case loop: specialists → graph → confidence → rationale
    │   ├── orchestrate2.py     #   Task 2 case loop
    │   ├── policy_graph.py     #   Graph loader, validator and walker (missing-variable handling)
    │   ├── briefs.py           #   Task 1 specialist prompts, weight rules, confidence rule
    │   ├── briefs2.py          #   Task 2 specialist prompts, confidence rule
    │   ├── agents.py           #   LLM call + KEY: value reply parsing (robust to small models)
    │   ├── rationale.py        #   Two-tier evidence packet and rationale validator
    │   └── consultant.py       #   LLM hand-off for graph nodes that request one (none do in this release)
    ├── agent/
    │   ├── rationale.py        #   Shared rationale rules: anchoring, grade vocabulary, next steps
    │   └── form_fill.py        #   Reveal-sequence building, prior-grade reader, claim guards
    ├── task1.py, task2.py,     # Deterministic reference policies (Task 2 weights and next-step
    │   policy.py               #   prose are read from here at runtime)
    ├── tools/
    │   ├── definitions.py      #   MCP tool sets per task
    │   ├── task3_features.py   #   Task 3 clinical feature extraction from reports
    │   ├── task3_head.py       #   Two-stage Cox / RMST head
    │   └── task3_recurrence.py #   Task 3 prediction + rationale assembly
    ├── models/                 # OpenAI-compatible client (used against llama-server) and vLLM backend
    ├── output/schema.py        # Challenge output schema (unchanged from the official baseline)
    └── deterministic.py, features.py, case_loader.py, utils.py
```

Where to look first:

* **The policies:** start with the two graph YAMLs. Every node has an `asks`
  question, `edges` with guard expressions, its EAU `source`, and prose notes
  explaining *why* each edge exists.
* **One case end to end:** `run_case` in `agentic/orchestrate.py` (Task 1) or
  `agentic/orchestrate2.py` (Task 2).

---

## Building and running

The container is built and run the way Grand Challenge runs it. Each
invocation processes **one case**: `/input` holds that case's three socket
files directly (structured prompt, clinical data, neural representations), and
`/output` receives the prediction and reasoning JSON for the detected task.

```bash
docker build -t chimera-agent .

docker run --rm --gpus all --network none \
  -v /path/to/case:/input:ro \
  -v /path/to/model:/opt/ml/model:ro \
  -v /path/to/output:/output \
  chimera-agent
```

**Model weights are not in this repository.** Tasks 1 and 2 expect exactly one
GGUF file under `/opt/ml/model`. We used `gemma-4-E2B-it-Q4_K_M.gguf`, a 4-bit
quantisation of [Gemma 4 E2B-it](https://huggingface.co/google/gemma-4-E2B-it),
which is subject to its own licence terms. `CHIMERA_GGUF_MODEL` selects a file
by name when there are several. Task 3 needs no model file.

Useful environment variables (defaults in `inference.py`):
`CHIMERA_INPUT_PATH`, `CHIMERA_OUTPUT_PATH`, `CHIMERA_MODEL_PATH`,
`CHIMERA_GGUF_MODEL`, `CHIMERA_N_GPU_LAYERS`, `CHIMERA_LLAMA_PORT`.

Inside the container, generation runs at temperature 0, with a 1,024-token cap
per call and a 60-second request timeout.

---

## Results

Measured on the released labelled training cases (91 for Task 1, 72 for
Task 2, 75 for Task 3) with the official evaluator, the rationale judge on, and
every clinical section read through MCP:

| | Ranking score | Decisions |
|---|---:|---:|
| Task 1 | 0.806 | 78 / 91 |
| Task 2 | 0.850 | 68 / 72 |
| Task 3 (C-index, nested 5-fold CV) | 0.866 ± 0.078 | — |

Read these numbers with three caveats:

* The Task 1/2 figures come from the 2026-09-11 build, our last full
  end-to-end measurement. The final image adds reporting-only changes to
  Task 2 (variable weights and next-step prose) that were checked offline but
  not re-measured with the judge.
* The labelled cases were also used to design the policies, so these are
  development estimates, not held-out ones.
* The Task 3 head was fitted on its 75 cases, so the nested CV figure is the
  honest generalisation estimate.

**This is research software for a benchmark.** It has not been clinically
validated and must not be used for patient care.

---

## Provenance

The repository is derived from the official
[CHIMERA-Agent baseline](https://github.com/DIAGNijmegen/chimera-agent-baseline)
(MCP server, output schema, container layout).

The files under `src/`, `configs/`, `templates/` and `resources/`, plus
`inference.py`, `Dockerfile` and `requirements.txt`, match the application
layer of our final image (built 2026-09-14, archive sha256 `11cab87b16e8…`) byte for byte, with one
exception. Before publication we **removed case identifiers and quoted
reference annotations from code comments and from the prose-only notes in the
policy YAMLs**. That change was checked mechanically:

* Every Python file's syntax tree, with docstrings stripped, is identical to
  the image's.
* Every YAML file parses to the same structure, and the only changed values
  are prose notes that no code reads.

## Licence

Code: [Apache License 2.0](LICENSE).

**EAU guideline content is not covered by that licence.** The policy files
contain structured decision rules and short excerpts derived from the EAU
Guidelines on Prostate Cancer (2026 edition). Under the challenge's
[guideline-use terms](https://chimera-agent.grand-challenge.org/eau-guideline-use/),
that content is provided only for non-commercial research within CHIMERA-Agent.
See [NOTICE](NOTICE).
