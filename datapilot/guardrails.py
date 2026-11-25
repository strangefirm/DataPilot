"""Guardrails: input checks, code safety policy, chart label checks, answer grounding.

Every check returns a GuardrailResult so callers can log the decision (observability)
and feed violations back to the agents (self-correction) instead of crashing.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- results


@dataclass
class GuardrailResult:
    name: str
    ok: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"guardrail": self.name, "ok": self.ok, "violations": self.violations, "warnings": self.warnings}

    def message(self) -> str:
        lines = [f"GUARDRAIL BLOCKED ({self.name}):"] + [f"- {v}" for v in self.violations]
        return "\n".join(lines)


# --------------------------------------------------------------------------- input

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_QUESTION_CHARS = 1000
MIN_QUESTION_CHARS = 5

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+(instructions|prompts?|rules)",
    r"disregard\s+(the\s+)?(system|previous|above)",
    r"\bsystem\s+prompt\b",
    r"you\s+are\s+now\s+",
    r"\bjailbreak\b",
    r"\b(api[_\s-]?key|secret[_\s-]?key|password|credentials?)\b",
    r"\brm\s+-rf\b",
    r"\b(os\.system|subprocess|__import__|eval\(|exec\()",
    r"(reveal|print|show|leak)\s+(your|the)\s+(instructions|prompt|env|environment)",
    r"\bAPPROVED\b",  # the user must not be able to force termination
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def check_upload(filename: str, size_bytes: int, max_mb: int) -> GuardrailResult:
    v: list[str] = []
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        v.append(f"File type '{ext or 'none'}' not allowed. Use CSV or Excel ({', '.join(sorted(ALLOWED_EXTENSIONS))}).")
    if size_bytes <= 0:
        v.append("File is empty.")
    if size_bytes > max_mb * 1024 * 1024:
        v.append(f"File is {size_bytes / 1e6:.1f} MB; the limit is {max_mb} MB.")
    return GuardrailResult("upload", not v, v)


def check_question(question: str) -> GuardrailResult:
    q = (question or "").strip()
    v: list[str] = []
    if len(q) < MIN_QUESTION_CHARS:
        v.append("Question is too short. Ask a full business question.")
    if len(q) > MAX_QUESTION_CHARS:
        v.append(f"Question is {len(q)} characters; the limit is {MAX_QUESTION_CHARS}.")
    m = _INJECTION_RE.search(q)
    if m:
        v.append(f"Question contains a disallowed instruction pattern: '{m.group(0)}'.")
    return GuardrailResult("question", not v, v)


def scan_data_for_injection(columns: list[str], sample_text: list[str]) -> GuardrailResult:
    """Data is untrusted: flag instruction-like text in headers/cells (warn, don't block)."""
    w = [f"Column name looks like an instruction: {c!r}" for c in columns if _INJECTION_RE.search(str(c))]
    w += [f"Cell text looks like an instruction: {t[:80]!r}" for t in sample_text if _INJECTION_RE.search(str(t))]
    return GuardrailResult("data_injection", True, [], w)


# --------------------------------------------------------------------------- code policy

MAX_CODE_CHARS = 20_000
ALLOWED_LANGUAGES = {"python", "py", "python3"}

ALLOWED_IMPORTS = {
    "pandas", "numpy", "matplotlib", "seaborn", "math", "statistics", "datetime", "json",
    "collections", "itertools", "functools", "re", "decimal", "fractions", "warnings",
    "textwrap", "string", "calendar", "typing", "dataclasses", "pathlib", "csv", "openpyxl",
}
BLOCKED_CALLS = {"eval", "exec", "compile", "__import__", "globals", "locals", "vars", "breakpoint",
                 "input", "memoryview", "setattr", "delattr"}
BLOCKED_ATTRS = {"system", "popen", "spawn", "fork", "kill", "remove", "unlink", "rmdir", "rmtree",
                 "chmod", "chown", "symlink", "rename", "replace_file", "to_pickle", "read_pickle",
                 "to_sql", "read_sql", "read_html", "read_clipboard", "to_clipboard", "urlopen"}
_URL_RE = re.compile(r"^\s*(https?|ftp|file|s3|gs)://", re.IGNORECASE)
_ABS_PATH_RE = re.compile(r"^\s*([A-Za-z]:[\\/]|[\\/]{1,2}\w|~)")


def _root_module(name: str) -> str:
    return name.split(".")[0]


def check_code(code: str, language: str = "python") -> GuardrailResult:
    """Static AST policy. Runs before any code reaches the executor."""
    v: list[str] = []
    if language.lower() not in ALLOWED_LANGUAGES:
        return GuardrailResult("code_policy", False, [f"Only Python code may run; got '{language}'."])
    if len(code) > MAX_CODE_CHARS:
        v.append(f"Code is {len(code)} characters; the limit is {MAX_CODE_CHARS}.")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        # Not a policy violation: let it run so the Coder sees the real SyntaxError and fixes it.
        return GuardrailResult("code_policy", True, [], [f"SyntaxError: {e.msg} (line {e.lineno})"])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if _root_module(a.name) not in ALLOWED_IMPORTS:
                    v.append(f"Import of '{a.name}' is not allowed (line {node.lineno}).")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or _root_module(mod) not in ALLOWED_IMPORTS:
                v.append(f"Import from '{mod or '.'}' is not allowed (line {node.lineno}).")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in BLOCKED_CALLS:
                v.append(f"Call to '{fn.id}()' is not allowed (line {node.lineno}).")
            if isinstance(fn, ast.Attribute) and fn.attr in BLOCKED_ATTRS:
                v.append(f"Call to '.{fn.attr}()' is not allowed (line {node.lineno}).")
            if isinstance(fn, ast.Name) and fn.id == "open":
                mode = _open_mode(node)
                if mode and any(c in mode for c in "wax+"):
                    target = _first_str_arg(node)
                    if not (target and target.lower().endswith((".py", ".txt", ".json", ".csv", ".md"))):
                        v.append(f"open() for writing is only allowed for result files (line {node.lineno}).")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in {"__name__", "__init__"}:
                v.append(f"Access to dunder attribute '{node.attr}' is not allowed (line {node.lineno}).")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if _URL_RE.match(s):
                v.append(f"Network/remote path '{s[:60]}' is not allowed (line {node.lineno}).")
            elif _ABS_PATH_RE.match(s) and len(s) < 260 and "\n" not in s:
                v.append(f"Absolute path '{s[:60]}' is not allowed; use files in the working folder (line {node.lineno}).")
            elif re.search(r"(^|[\\/])\.\.([\\/]|$)", s):
                v.append(f"Parent-directory path '{s[:60]}' is not allowed (line {node.lineno}).")
    return GuardrailResult("code_policy", not v, sorted(set(v)))


def _open_mode(call: ast.Call) -> str | None:
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
        return str(call.args[1].value)
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    return "r"


def _first_str_arg(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


# --------------------------------------------------------------------------- charts

_UNIT_HINT = re.compile(r"(rs\.?|₹|inr|lakh|\bL\b|crore|units?|%|percent|count|qty|quantity|\(.+\)|\[.+\])", re.IGNORECASE)


def check_chart_meta(charts: list[dict]) -> GuardrailResult:
    """Verify every saved chart has a title, axis labels and units.

    `charts` comes from the sandbox savefig hook: [{"file", "axes": [{"title","xlabel","ylabel"}], "suptitle"}].
    """
    v: list[str] = []
    for c in charts:
        f = c.get("file", "?")
        axes = c.get("axes") or []
        if not axes:
            v.append(f"{f}: chart has no axes.")
        for i, ax in enumerate(axes):
            tag = f"{f}" + (f" (subplot {i + 1})" if len(axes) > 1 else "")
            if not (ax.get("title") or c.get("suptitle")):
                v.append(f"{tag}: missing title.")
            if ax.get("is_pie"):
                continue
            if not ax.get("xlabel"):
                v.append(f"{tag}: missing x-axis label.")
            if not ax.get("ylabel"):
                v.append(f"{tag}: missing y-axis label.")
            labels = f"{ax.get('xlabel', '')} {ax.get('ylabel', '')}"
            if ax.get("xlabel") and ax.get("ylabel") and not _UNIT_HINT.search(labels):
                v.append(f"{tag}: axis labels have no units (e.g. 'Revenue (Rs)', 'Units').")
    return GuardrailResult("chart_labels", not v, v)


# --------------------------------------------------------------------------- answer grounding

_NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _variants(x: float) -> set[float]:
    """Display forms a number may take in an answer (lakh, crore, thousands, %, rounding)."""
    return {round(x, 1), round(x / 1e5, 1), round(x / 1e7, 2), round(x / 1e3, 1), round(x * 100, 1), round(x)}


def check_answer_grounding(answer: str, execution_outputs: list[str], *, ignore_below: float = 13) -> GuardrailResult:
    """Every number in the final answer must be traceable to executor output.

    Small whole numbers (counts like '2 products', month numbers) and years are ignored.
    Returns warnings for ungrounded numbers; blocks if more than half are ungrounded.
    """
    source = set()
    for out in execution_outputs:
        for n in _numbers(out):
            source |= _variants(n)
    def _trivial(n: float) -> bool:  # small counts ("2 products"), month numbers, years
        return n == int(n) and (abs(n) < ignore_below or 2000 <= n <= 2100)

    claimed = [n for n in _numbers(answer) if not _trivial(n)]
    ungrounded = [n for n in claimed if not any(abs(round(n, 1) - s) <= max(0.051, abs(s) * 0.005) for s in source)]
    msgs = [f"Number {n:g} in the answer does not appear in any code output." for n in ungrounded]
    blocked = bool(claimed) and len(ungrounded) > len(claimed) / 2
    return GuardrailResult("answer_grounding", not blocked, msgs if blocked else [], [] if blocked else msgs)
