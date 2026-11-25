import os
from pathlib import Path

import pytest
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock

from datapilot.dataset import describe
from datapilot.sandbox import SandboxExecutor, Workspace

DATA = Path(__file__).resolve().parent.parent / "data" / "sales_2026.csv"


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(DATA, tmp_path)


async def run(ex, code, lang="python"):
    return await ex.execute_code_blocks([CodeBlock(code=code, language=lang)], CancellationToken())


async def test_success_saves_analysis_py(ws):
    ex = SandboxExecutor(ws, timeout=60)
    r = await run(ex, "import pandas as pd\ndf = pd.read_csv('sales_2026.csv')\nprint('ROWS', len(df))")
    assert r.exit_code == 0 and "ROWS 12480" in r.output
    assert ws.analysis_file.exists()


async def test_error_is_returned_for_self_correction(ws):
    ex = SandboxExecutor(ws, timeout=60)
    r = await run(ex, "import pandas as pd\ndf = pd.read_csv('sales_2026.csv')\nprint(df['revenue'].sum())")
    assert r.exit_code != 0 and "KeyError" in r.output and "revenue" in r.output
    assert not ws.analysis_file.exists()  # only working code becomes analysis.py


async def test_traceback_is_compact_and_hides_host_paths(ws):
    ex = SandboxExecutor(ws, timeout=60)
    r = await run(ex, "import pandas as pd\ndf = pd.read_csv('sales_2026.csv')\nprint(df.groupby('category')['revenue'].sum())")
    assert 'File "step_01.py", line 3' in r.output and "KeyError" in r.output
    assert "site-packages" not in r.output and "[library frames hidden]" in r.output
    assert str(Path.home()) not in r.output


async def test_blocked_code_never_runs(ws):
    ex = SandboxExecutor(ws, timeout=60)
    r = await run(ex, "import os\nos.system('echo pwned > pwned.txt')")
    assert r.exit_code == 1 and "GUARDRAIL BLOCKED" in r.output
    assert not (ws.root / "pwned.txt").exists()
    assert ex.history[0].guardrail["ok"] is False


async def test_shell_blocks_rejected(ws):
    ex = SandboxExecutor(ws, timeout=60)
    r = await run(ex, "echo hi", lang="bash")
    assert r.exit_code == 1 and "Only Python" in r.output


async def test_timeout(ws):
    ex = SandboxExecutor(ws, timeout=2)
    r = await run(ex, "while True:\n    pass")
    assert r.exit_code == 124 and "Timeout" in r.output


async def test_secrets_not_visible_in_sandbox(ws, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-test-secret")
    ex = SandboxExecutor(ws, timeout=60)
    # pathlib/json are allowed; reading env needs os, which is blocked — so we check the env builder directly too.
    assert "GROQ_API_KEY" not in ex._env()
    assert ex._env()["PYTHONHASHSEED"] == "0"


async def test_chart_label_check_feeds_back(ws):
    ex = SandboxExecutor(ws, timeout=60)
    unlabeled = "import matplotlib.pyplot as plt\nplt.bar(['a','b'],[1,2])\nplt.savefig('bad.png')"
    r = await run(ex, unlabeled)
    assert r.exit_code == 0 and "CHART CHECK FAILED" in r.output
    labeled = ("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.bar(['a','b'],[1,2])\n"
               "ax.set_title('Revenue by region')\nax.set_xlabel('Region')\nax.set_ylabel('Revenue (Rs)')\n"
               "fig.savefig('bad.png')")
    r = await run(ex, labeled)
    assert r.exit_code == 0 and "CHART CHECK FAILED" not in r.output
    assert (ws.root / "bad.png").exists()


async def test_reproducible_numbers(ws):
    code = ("import pandas as pd\ndf = pd.read_csv('sales_2026.csv')\n"
            "df['revenue'] = df.units*df.unit_price*(1-df.discount)\nprint(round(df.revenue.sum(),2))")
    ex = SandboxExecutor(ws, timeout=60)
    a, b = await run(ex, code), await run(ex, code)
    assert a.exit_code == 0 and a.output == b.output


async def test_works_with_autogen_code_executor_agent(ws):
    from autogen_agentchat.agents import CodeExecutorAgent
    from autogen_agentchat.messages import TextMessage

    agent = CodeExecutorAgent("executor", code_executor=SandboxExecutor(ws, timeout=60))
    msg = TextMessage(content="```python\nprint(6*7)\n```", source="coder")
    resp = await agent.on_messages([msg], CancellationToken())
    assert "42" in resp.chat_message.content


def test_workspace_isolation(ws, tmp_path):
    assert ws.root.parent == tmp_path.resolve()
    assert (ws.root / "sales_2026.csv").exists()


def test_describe_schema():
    info = describe(DATA)
    assert info.rows == 12480
    assert info.columns == ["order_date", "region", "category", "product", "units", "unit_price", "discount"]
    assert "Electronics" in info.schema_text and info.injection_check.warnings == []
