"""Stretch goal: Excel workbooks with multiple sheets."""

from pathlib import Path

import pandas as pd
import pytest
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock

from datapilot.dataset import describe
from datapilot.sandbox import SandboxExecutor, Workspace

DATA = Path(__file__).resolve().parent.parent / "data" / "sales_2026.csv"


@pytest.fixture
def workbook(tmp_path) -> Path:
    sales = pd.read_csv(DATA)
    targets = pd.DataFrame({"category": sorted(sales.category.unique()), "monthly_target": [900000, 2500000, 700000, 600000, 300000]})
    path = tmp_path / "sales_book.xlsx"
    with pd.ExcelWriter(path) as xw:
        sales.head(500).to_excel(xw, sheet_name="Orders", index=False)
        targets.to_excel(xw, sheet_name="Targets", index=False)
        pd.DataFrame().to_excel(xw, sheet_name="Empty", index=False)
    return path


def test_describe_lists_every_non_empty_sheet(workbook):
    info = describe(workbook)
    assert info.sheets == ["Orders", "Targets"]  # empty sheet skipped
    assert info.rows == 505
    assert "pd.read_excel('sales_book.xlsx', sheet_name='Targets')" in info.schema_text
    assert "monthly_target" in info.schema_text and "joined or compared" in info.schema_text


def test_csv_schema_has_read_hint():
    assert "Read with: pd.read_csv('sales_2026.csv')" in describe(DATA).schema_text


async def test_sandbox_can_join_sheets(workbook, tmp_path):
    ws = Workspace.create(workbook, tmp_path / "runs")
    code = ("import pandas as pd\n"
            "o = pd.read_excel('sales_book.xlsx', sheet_name='Orders')\n"
            "t = pd.read_excel('sales_book.xlsx', sheet_name='Targets')\n"
            "print('joined_rows:', len(o.merge(t, on='category')))")
    r = await SandboxExecutor(ws, timeout=60).execute_code_blocks([CodeBlock(code=code, language="python")],
                                                                   CancellationToken())
    assert r.exit_code == 0 and "joined_rows: 500" in r.output
