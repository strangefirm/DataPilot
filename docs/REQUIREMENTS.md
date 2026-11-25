# DataPilot: Self-Correcting Data Analyst — Requirements Analysis

Track 3 Agentic AI · AutoGen · Intermediate · Streamlit chat

## 1. Problem

A business owner asks a question in plain English about a sales CSV/Excel file
("Which category dropped the most last month, and which region caused it?").
Ordinary chatbots guess numbers. DataPilot **calculates** them: a team of agents
plans, writes Python, runs it in a sandbox, fixes its own errors, gets reviewed,
and returns a plain-English answer with charts and a re-runnable `analysis.py`.

**Users:** business owners, sales managers, operations heads (non-programmers).

**Why it is agentic:** the path is decided at run time. Code error → Coder
rewrites. Reviewer unsatisfied → work goes back. Loop ends on `APPROVED` or a
message limit.

## 2. Functional requirements

| ID | Requirement | Source |
|----|-------------|--------|
| F1 | Upload any sales CSV and ask a question in plain English | 3.2 |
| F2 | **Planner** reads question + data schema, writes a short step-by-step plan | 3.2 |
| F3 | **Coder** writes pandas + matplotlib code for the plan; rewrites on error | 3.2 |
| F4 | **Executor** runs code safely in a sandbox folder or Docker; returns output/errors | 3.2, 3.8 |
| F5 | **Reviewer** checks: answers the question? numbers sensible? charts labelled? → `APPROVED` or feedback | 3.2 |
| F6 | **User Proxy** (human) approves the final answer or asks a follow-up | 3.2, 3.4 |
| F7 | Watch the full agent conversation live in the app, including errors and fixes | 3.2 |
| F8 | Output: short written answer, charts as PNG, final working `analysis.py` | 3.2, 3.6 |
| F9 | Final answer format: ANSWER / WHERE IT CAME FROM / FILES / SUGGESTED NEXT QUESTION | 3.6 B |
| F10 | Follow-up question continues the same session | 3.8 |

## 3. Non-functional / technical requirements

| ID | Requirement | Source |
|----|-------------|--------|
| T1 | AutoGen `AssistantAgent` for Planner, Coder, Reviewer with clear system messages | 3.4 |
| T2 | `CodeExecutorAgent` with Docker or local command-line executor | 3.4 |
| T3 | `RoundRobinGroupChat` team (stretch: `SelectorGroupChat`) | 3.4 |
| T4 | Termination: `TextMentionTermination("APPROVED") \| MaxMessageTermination(12)` | 3.4, 3.2 |
| T5 | Human-in-the-loop via `UserProxyAgent` | 3.4 |
| T6 | Numbers come from pandas code, never from the LLM | 3.4 |
| T7 | Streamlit chat interface | header |
| T8 | LLM: Groq or DeepSeek (OpenAI-compatible), switchable via config | user choice |

## 4. Done checklist → how we verify it

| # | Checklist item (3.8) | Verification |
|---|----------------------|--------------|
| D1 | At least one run shows agents fixing a code error on their own | Trace contains `execution_error` followed by a successful run; eval `self_correction` |
| D2 | Code runs only inside the sandbox folder or Docker | Executor `work_dir` = per-run sandbox; guardrail blocks paths outside; unit tests |
| D3 | Stops on APPROVED or after 12 messages | Termination condition + unit test with a fake model |
| D4 | Re-running saved `analysis.py` gives exactly the same numbers | Eval `reproducibility`: re-execute and diff stdout |
| D5 | Every chart has a title, axis labels and units | Static AST check of matplotlib calls + Reviewer rubric; eval `chart_labels` |
| D6 | Human approval works; follow-up continues the same session | Streamlit approve/follow-up flow; team state kept in session |

## 5. Stretch goals
- `SelectorGroupChat` so the LLM picks the next speaker (config switch)
- Excel files with multiple sheets
- Export answer + charts as a one-page PDF report

## 6. Added by this build: guardrails, observability, evals, reasoning

**Guardrails**
- Input: file type/size/row limits, schema sniffing, question length, prompt-injection screening of the question and of column names/cell text.
- Code: AST-based policy before execution — blocks `os.system`, `subprocess`, sockets/network, `eval/exec`, file access outside the sandbox, deleting files, installing packages.
- Runtime: per-execution timeout, sandbox working directory per run, max code size.
- Output: reviewer rubric plus a deterministic check that the numbers quoted in the final answer appear in the executor's output (stops the LLM from making numbers up).
- Cost: message cap (12), token budget per run, retry limit.

**Observability**
- A structured JSONL trace per run: every agent message, the code, execution results, guardrail decisions, latency, token usage.
- Per-run metrics: messages, self-corrections, review rounds, tokens, duration, outcome.
- A run-history view in Streamlit.

**Evals**
- Golden questions on a synthetic dataset whose ground truth is computed in pandas when the data is generated.
- Metrics: answer correctness (numeric tolerance), self-correction occurrence, reproducibility, chart-label compliance, guardrail violations, termination within limit.
- Offline unit tests with a scripted fake model (no API cost) plus a live eval runner.

**Reasoning**
- The Planner produces an explicit numbered plan with its assumptions (e.g. revenue formula, what "last month" means).
- The Reviewer gives reasons with each verdict.
- DeepSeek-reasoner / Groq reasoning models: reasoning content captured in the trace and shown in a collapsible panel in the UI.

## 7. Architecture

```
Streamlit UI ──► Session (per user) ──► DataPilot Team (AutoGen)
   │                                     Planner → Coder → Executor → Reviewer
   │                                        ▲        │ error          │ feedback
   │                                        └────────┴────────────────┘
   │                                     Termination: APPROVED | 12 msgs
   ├─ Guardrails (input, code AST policy, output number grounding)
   ├─ Sandbox: runs/<run_id>/  (data copy, analysis.py, *.png)
   ├─ Observability: runs/<run_id>/trace.jsonl + metrics.json
   └─ Human approval / follow-up (UserProxy) ──► same team, same session
```

## 8. Build steps
1. Requirements, scaffold, config, LLM client factory, sample data + ground truth
2. Sandbox executor + code/input guardrails (+ tests)
3. Agents, prompts, team, termination; CLI run end-to-end
4. Observability: tracing, metrics, token usage
5. Streamlit app: upload, live conversation, answer, charts, files, approval, follow-up
6. Evals: golden set, offline fake-model tests, live eval runner, report
7. Stretch: SelectorGroupChat, Excel multi-sheet, PDF export; README
