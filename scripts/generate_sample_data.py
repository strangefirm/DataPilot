"""Generate a realistic sales_2026.csv plus ground-truth answers for evals.

Schema: order_date,region,category,product,units,unit_price,discount
Revenue = units * unit_price * (1 - discount)

Planted signal: Electronics revenue falls Aug -> Sep 2026, driven mostly by the
South region, specifically Earbuds X2 and Smartwatch Lite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED = 2026

CATALOG = {
    "Electronics": {"Earbuds X2": 1999, "Smartwatch Lite": 3499, "Power Bank 20K": 1499,
                    "Bluetooth Speaker Mini": 2499, "USB-C Charger": 799},
    "Grocery": {"Millet Mix 1kg": 249, "Basmati Rice 5kg": 649, "Cold Pressed Oil 1L": 299,
                "Masala Tea 500g": 199},
    "Apparel": {"Cotton Kurta": 899, "Denim Jeans": 1299, "Running Shoes": 2199, "Sports T-Shirt": 499},
    "Home & Kitchen": {"Non-stick Pan": 1199, "Steel Bottle": 399, "Mixer Grinder": 3299,
                       "Bedsheet Set": 999},
    "Personal Care": {"Herbal Shampoo": 299, "Face Wash": 199, "Trimmer Pro": 1499},
}
REGIONS = ["North", "South", "East", "West"]
REGION_W = np.array([0.27, 0.28, 0.22, 0.23])
CATEGORY_W = {"Electronics": 0.30, "Grocery": 0.24, "Apparel": 0.18, "Home & Kitchen": 0.16, "Personal Care": 0.12}
UNITS_MAX = {"Electronics": 5, "Grocery": 20, "Apparel": 5, "Home & Kitchen": 4, "Personal Care": 8}
# Earbuds X2 and Smartwatch Lite are the bestsellers.
PRODUCT_W = {"Electronics": [0.35, 0.30, 0.13, 0.12, 0.10]}
DISCOUNTS = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
DISCOUNT_W = np.array([0.45, 0.20, 0.20, 0.10, 0.05])


def demand_multiplier(month: int, region: str, category: str, product: str) -> float:
    """Probability multiplier for an order line; encodes the planted September drop."""
    m = 1.0
    if month in (10, 11):  # festive season lift
        m *= 1.15
    if month == 9 and category == "Electronics":
        if region == "South" and product in ("Earbuds X2", "Smartwatch Lite"):
            m *= 0.15
        elif region == "South":
            m *= 0.90
        else:
            m *= 0.96
    return m


def generate(n_rows: int = 12_480) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2026-01-01", "2026-10-31", freq="D")
    rows: list[tuple] = []
    # Oversample candidates, then accept by demand multiplier, until we have n_rows.
    while len(rows) < n_rows:
        d = dates[rng.integers(len(dates))]
        region = REGIONS[rng.choice(4, p=REGION_W)]
        cats = list(CATEGORY_W)
        category = cats[rng.choice(len(cats), p=np.array(list(CATEGORY_W.values())))]
        names = list(CATALOG[category])
        product = names[rng.choice(len(names), p=PRODUCT_W.get(category))]
        if rng.random() > demand_multiplier(d.month, region, category, product) / 1.15:
            continue
        units = int(rng.integers(1, UNITS_MAX[category] + 1))
        price = CATALOG[category][product]
        discount = float(rng.choice(DISCOUNTS, p=DISCOUNT_W))
        rows.append((d.strftime("%Y-%m-%d"), region, category, product, units, price, discount))
    df = pd.DataFrame(rows, columns=["order_date", "region", "category", "product", "units", "unit_price", "discount"])
    return df.sort_values(["order_date", "region", "category", "product"], kind="stable").reset_index(drop=True)


def ground_truth(df: pd.DataFrame) -> dict:
    d = df.assign(revenue=df.units * df.unit_price * (1 - df.discount),
                  month=pd.to_datetime(df.order_date).dt.month)
    aug, sep = d[d.month == 8], d[d.month == 9]
    cat = pd.DataFrame({"aug": aug.groupby("category").revenue.sum(),
                        "sep": sep.groupby("category").revenue.sum()}).fillna(0)
    cat["drop"] = cat.aug - cat.sep
    worst = cat["drop"].idxmax()
    reg = pd.DataFrame({"aug": aug[aug.category == worst].groupby("region").revenue.sum(),
                        "sep": sep[sep.category == worst].groupby("region").revenue.sum()}).fillna(0)
    reg["drop"] = reg.aug - reg.sep
    top_region = reg["drop"].idxmax()
    ra, rs = aug[(aug.category == worst) & (aug.region == top_region)], sep[(sep.category == worst) & (sep.region == top_region)]
    prod = pd.DataFrame({"aug": ra.groupby("product").revenue.sum(), "sep": rs.groupby("product").revenue.sum()}).fillna(0)
    prod["drop"] = prod.aug - prod.sep
    r2 = lambda x: round(float(x), 2)
    return {
        "rows": int(len(df)),
        "total_revenue": r2(d.revenue.sum()),
        "monthly_revenue": {int(k): r2(v) for k, v in d.groupby("month").revenue.sum().items()},
        "category_revenue": {k: r2(v) for k, v in d.groupby("category").revenue.sum().items()},
        "region_revenue": {k: r2(v) for k, v in d.groupby("region").revenue.sum().items()},
        "aug_vs_sep": {
            "biggest_drop_category": worst,
            "category_aug": r2(cat.loc[worst, "aug"]),
            "category_sep": r2(cat.loc[worst, "sep"]),
            "category_drop": r2(cat.loc[worst, "drop"]),
            "category_drop_pct": r2(cat.loc[worst, "drop"] / cat.loc[worst, "aug"] * 100),
            "all_category_drops": {k: r2(v) for k, v in cat["drop"].items()},
            "top_region": top_region,
            "top_region_drop": r2(reg.loc[top_region, "drop"]),
            "top_region_share_pct": r2(reg.loc[top_region, "drop"] / cat.loc[worst, "drop"] * 100),
            "region_drops": {k: r2(v) for k, v in reg["drop"].items()},
            "top_products": [p for p in prod["drop"].sort_values(ascending=False).index[:2]],
            "product_drops": {k: r2(v) for k, v in prod["drop"].sort_values(ascending=False).items()},
        },
    }


def main() -> int:
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(out / "sales_2026.csv", index=False)
    gt = ground_truth(df)
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    a = gt["aug_vs_sep"]
    lakh = lambda x: f"Rs {x / 1e5:.1f} L"
    print(f"Wrote {gt['rows']} rows -> {out / 'sales_2026.csv'}")
    print(f"Biggest Aug->Sep drop: {a['biggest_drop_category']} {lakh(a['category_aug'])} -> "
          f"{lakh(a['category_sep'])} (down {a['category_drop_pct']:.1f}%)")
    print(f"Top region: {a['top_region']} {lakh(a['top_region_drop'])} of {lakh(a['category_drop'])} "
          f"({a['top_region_share_pct']:.0f}%); products: {', '.join(a['top_products'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
