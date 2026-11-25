import json
from pathlib import Path

from evals import run as eval_run
from evals.golden import build_cases
from evals.scoring import entities_found, number_found

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sales_2026.csv"
GT = json.loads((ROOT / "data" / "ground_truth.json").read_text())


def test_golden_matches_independent_ground_truth():
    cases = {c.id: c for c in build_cases(DATA)}
    assert len(cases) == 10
    spec = cases["spec_drop"]
    a = GT["aug_vs_sep"]
    assert spec.entities == [[a["biggest_drop_category"]], [a["top_region"]]]
    assert round(spec.numbers[0], 2) == a["category_aug"] and round(spec.numbers[1], 2) == a["category_sep"]
    assert round(spec.numbers[2][0], 2) == a["category_drop_pct"] and round(spec.numbers[2][1], 2) == a["category_drop"]
    assert round(cases["total_revenue"].numbers[0], 2) == GT["total_revenue"]


def test_number_matching_display_forms():
    assert number_found(2_620_000.5, "fell from Rs 26.2 L")             # lakh
    assert number_found(2_620_000.5, "Rs 26,20,000")                    # raw, Indian commas
    assert number_found(31.76, "down 31.8%")                            # rounded percent
    assert number_found(24_300_000, "Rs 2.43 crore")                    # crore
    assert not number_found(2_620_000.5, "fell from Rs 44.4 L")
    assert not number_found(31.76, "down 35.0%")


def test_entity_matching():
    assert entities_found([["October", "Oct"]], "Revenue peaked in Oct.") == [True]
    assert entities_found([["South"]], "Southern stores") == [False]  # whole words only


async def test_offline_eval_end_to_end(tmp_path):
    rc = await eval_run.main(["--offline", "--out", str(tmp_path)])
    assert rc == 0
    out = next(tmp_path.iterdir())
    r = json.loads((out / "results.jsonl").read_text().splitlines()[0])
    assert r["passed"] and r["approved"] and r["self_corrected"] and r["reproducible"]
    assert r["entities_ok"] and r["numbers_ok"] and r["charts_ok"] and r["within_limit"] and r["grounded"]
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "D1 self-correction seen: True" in report and "D4 analysis.py reproducible: 1.0" in report
