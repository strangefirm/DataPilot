"""Stretch goal: export an answer + charts as a one-page PDF report (reportlab)."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepInFrame, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BRAND = colors.HexColor("#8B1A1A")
SECTIONS = ("ANSWER", "WHERE IT CAME FROM", "FILES", "SUGGESTED NEXT QUESTION")


def _sections(answer: str) -> dict[str, str]:
    out, current = {}, None
    for line in answer.splitlines():
        if line.strip().upper() in SECTIONS:
            current = line.strip().upper()
            out[current] = ""
        elif current:
            out[current] += line + "\n"
    return {k: v.strip() for k, v in out.items()} or {"ANSWER": answer.strip()}


def _image(path: Path, max_w: float, max_h: float) -> Image:
    w, h = ImageReader(str(path)).getSize()
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def build_pdf(*, question: str, answer: str, charts: list[Path], dataset: str, run_id: str,
              metrics: dict | None = None, status: str = "approved") -> bytes:
    buf = BytesIO()
    page_w, page_h = A4
    margin = 1.6 * cm
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=margin,
                            bottomMargin=margin, title="DataPilot report", author="DataPilot")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=BRAND, alignment=0, fontSize=18, spaceAfter=2)
    meta = ParagraphStyle("meta", parent=ss["Normal"], fontSize=8, textColor=colors.grey)
    label = ParagraphStyle("label", parent=ss["Normal"], fontSize=8, textColor=BRAND, spaceBefore=6,
                           fontName="Helvetica-Bold")
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=10, leading=13)
    p = lambda text, style=body: Paragraph(escape(text).replace("\n", "<br/>"), style)

    sec = _sections(answer)
    story = [
        Paragraph("DataPilot report", h1),
        p(f"{dataset} · generated {datetime.now():%d %b %Y %H:%M} · run {run_id} · status: {status}", meta),
        Spacer(1, 6),
        p("QUESTION", label), p(question),
        p("ANSWER", label), p(sec.get("ANSWER", "(no approved answer)")),
    ]
    if sec.get("WHERE IT CAME FROM"):
        story += [p("WHERE IT CAME FROM", label), p(sec["WHERE IT CAME FROM"])]

    charts = [c for c in charts if Path(c).exists()][:4]
    if charts:
        avail_w = page_w - 2 * margin
        per_row = 1 if len(charts) == 1 else 2
        cell_w = avail_w / per_row - 0.3 * cm
        cell_h = 9 * cm if len(charts) <= 2 else 6 * cm
        imgs = [_image(Path(c), cell_w, cell_h) for c in charts]
        rows = [imgs[i:i + per_row] for i in range(0, len(imgs), per_row)]
        if len(rows[-1]) < per_row:
            rows[-1] += [""] * (per_row - len(rows[-1]))
        caps = [[p(Path(c).name, meta) for c in charts[i:i + per_row]] for i in range(0, len(charts), per_row)]
        grid = []
        for r, c in zip(rows, caps):
            grid += [r, c + [""] * (per_row - len(c))]
        t = Table(grid, colWidths=[avail_w / per_row] * per_row)
        t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story += [p("CHARTS", label), t]

    if sec.get("SUGGESTED NEXT QUESTION"):
        story += [p("SUGGESTED NEXT QUESTION", label), p(sec["SUGGESTED NEXT QUESTION"])]
    if metrics:
        n = lambda k, word: f"{metrics.get(k, 0)} {word}{'' if metrics.get(k, 0) == 1 else 's'}"
        story += [Spacer(1, 8), p(
            f"How it was computed: {n('messages', 'agent message')}, {n('executions', 'code run')}, "
            f"{n('self_corrections', 'self-correction')}, {metrics.get('total_tokens', 0):,} tokens. "
            "All numbers come from executed pandas code (analysis.py), not from the language model.", meta)]

    # Guarantee one page: shrink everything to fit the frame if needed.
    frame = KeepInFrame(page_w - 2 * margin, page_h - 2 * margin, story, mode="shrink")
    doc.build([frame])
    return buf.getvalue()
