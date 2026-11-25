"""Load an uploaded table and describe its schema for the Planner (the LLM never sees all rows)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .guardrails import GuardrailResult, scan_data_for_injection

MAX_ROWS = 2_000_000
MAX_SHEETS = 10


@dataclass
class DatasetInfo:
    filename: str  # file name inside the sandbox
    rows: int  # total across sheets
    columns: list[str]  # columns of the first (or only) table
    schema_text: str  # compact description passed to agents
    injection_check: GuardrailResult
    sheets: list[str] | None = None  # Excel sheet names (None for CSV)


def load_tables(path: Path) -> dict[str, pd.DataFrame]:
    """CSV -> {"": df}; Excel -> {sheet_name: df} for every non-empty sheet (up to MAX_SHEETS)."""
    if path.suffix.lower() == ".csv":
        return {"": pd.read_csv(path)}
    sheets = pd.read_excel(path, sheet_name=None)
    sheets = {str(k): v for k, v in sheets.items() if not v.dropna(how="all").empty}
    if len(sheets) > MAX_SHEETS:
        raise ValueError(f"The workbook has {len(sheets)} non-empty sheets; the limit is {MAX_SHEETS}.")
    return sheets


def load_table(path: Path) -> pd.DataFrame:
    return next(iter(load_tables(path).values()))


def _describe_frame(df: pd.DataFrame) -> list[str]:
    lines = ["Columns:"]
    for col in df.columns:
        s = df[col]
        info = f"  - {col!s} [{s.dtype}] nulls={int(s.isna().sum())}"
        if pd.api.types.is_numeric_dtype(s):
            info += f" min={s.min()!r} max={s.max()!r}"
        else:
            uniq = s.dropna().astype(str).unique()
            shown = ", ".join(u[:40] for u in uniq[:8])
            info += f" distinct={len(uniq)} e.g. [{shown}{', ...' if len(uniq) > 8 else ''}]"
        lines.append(info)
    lines.append("First 3 rows (CSV):")
    lines.append(df.head(3).to_csv(index=False).strip())
    return lines


def describe(path: Path) -> DatasetInfo:
    tables = load_tables(path)
    if not tables or all(df.empty for df in tables.values()):
        raise ValueError("The file has no data rows.")
    total = sum(len(df) for df in tables.values())
    if total > MAX_ROWS:
        raise ValueError(f"The file has {total:,} rows; the limit is {MAX_ROWS:,}.")

    is_excel = path.suffix.lower() != ".csv"
    if not is_excel:
        df = tables[""]
        lines = [f"File: {path.name}  ({len(df):,} rows x {df.shape[1]} columns)",
                 f"Read with: pd.read_csv('{path.name}')"] + _describe_frame(df)
    else:
        lines = [f"Excel workbook: {path.name} with {len(tables)} sheet(s): {', '.join(repr(s) for s in tables)}"]
        if len(tables) > 1:
            lines.append("The sheets may need to be joined or compared; pick the sheet(s) the question needs.")
        for name, df in tables.items():
            lines += ["", f"Sheet {name!r}  ({len(df):,} rows x {df.shape[1]} columns)",
                      f"Read with: pd.read_excel('{path.name}', sheet_name={name!r})"] + _describe_frame(df)

    all_cols = [str(c) for df in tables.values() for c in df.columns]
    text_cells = [str(x) for df in tables.values() for c in df.columns if df[c].dtype == object
                  for x in df[c].dropna().unique()[:200]]
    first = next(iter(tables.values()))
    return DatasetInfo(
        filename=path.name,
        rows=total,
        columns=[str(c) for c in first.columns],
        schema_text="\n".join(lines),
        injection_check=scan_data_for_injection(all_cols + list(tables), text_cells),
        sheets=list(tables) if is_excel else None,
    )
