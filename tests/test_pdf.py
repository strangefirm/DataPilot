"""Stretch goal: one-page PDF report."""

import re

from datapilot.pdf_report import build_pdf
from tests import fakes
from tests.test_team import QUESTION, make_session


async def test_pdf_is_one_page_with_answer_and_charts(tmp_path):
    s = make_session(tmp_path, "routed", [fakes.PLAN], [fakes.good_code()], [fakes.APPROVE])
    o = await s.run(QUESTION)
    pdf = build_pdf(question=QUESTION, answer=o.answer, charts=o.charts, dataset="sales_2026.csv",
                    run_id=s.workspace.run_id, metrics={"messages": o.messages, "executions": 1,
                                                        "self_corrections": 0, "total_tokens": 1234})
    await s.close()
    assert pdf.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page[^s]", pdf)) == 1          # exactly one page
    assert len(re.findall(rb"/Subtype\s*/Image", pdf)) >= 2  # both charts (PNG alpha adds mask objects)
    (tmp_path / "report.pdf").write_bytes(pdf)


def test_pdf_handles_long_answers_and_special_characters():
    answer = "ANSWER\n" + ("Revenue <fell> & rose; " * 400) + "\n\nWHERE IT CAME FROM\nSouth & North"
    pdf = build_pdf(question="Q & A <test>", answer=answer, charts=[], dataset="x.csv", run_id="r1")
    assert len(re.findall(rb"/Type\s*/Page[^s]", pdf)) == 1  # shrinks to fit instead of spilling over
