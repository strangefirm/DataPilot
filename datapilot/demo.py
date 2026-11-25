"""Scripted agent replies: offline tests + the UI demo mode (no API calls). Code really runs in the sandbox.

Only valid for the bundled sample dataset (sales_2026.csv) and the spec's sample question.
"""

from autogen_ext.models.replay import ReplayChatCompletionClient

from datapilot.team import ModelClients

PLAN = """ASSUMPTIONS:
- revenue = units * unit_price * (1 - discount)
PLAN:
1) Compute revenue  2) Group by month + category  3) Find biggest Aug->Sep drop  4) Split by region and product
OUTPUTS:
- Printed numbers: category Aug/Sep revenue, drop %, top region drop and share
- Charts: category_trend.png, region_breakdown.png"""

PLAN_OK = "PLAN OK - Coder, continue."

BAD_CODE = """Computing revenue by category.
```python
import pandas as pd
df = pd.read_csv('sales_2026.csv')
print(df.groupby('category')['revenue'].sum())
```"""


def good_code(labels: bool = True) -> str:
    y = "ax.set_ylabel('Revenue (Rs)')" if labels else ""
    x = "ax.set_xlabel('Month')" if labels else ""
    return f"""Fixed: revenue column is now created before grouping.
```python
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('sales_2026.csv')
df['revenue'] = df.units * df.unit_price * (1 - df.discount)
df['month'] = pd.to_datetime(df.order_date).dt.month
m = df[df.month.isin([8, 9])].pivot_table(index='category', columns='month', values='revenue', aggfunc='sum')
m['drop'] = m[8] - m[9]
cat = m['drop'].idxmax()
print(f"biggest_drop_category: {{cat}}")
print(f"category_aug_revenue: {{m.loc[cat, 8]:.2f}}")
print(f"category_sep_revenue: {{m.loc[cat, 9]:.2f}}")
print(f"category_drop_pct: {{m.loc[cat, 'drop'] / m.loc[cat, 8] * 100:.2f}}")
e = df[(df.category == cat) & df.month.isin([8, 9])]
r = e.pivot_table(index='region', columns='month', values='revenue', aggfunc='sum')
r['drop'] = r[8] - r[9]
top = r['drop'].idxmax()
print(f"top_region: {{top}}")
print(f"top_region_drop: {{r.loc[top, 'drop']:.2f}}")
print(f"top_region_share_pct: {{r.loc[top, 'drop'] / m.loc[cat, 'drop'] * 100:.2f}}")
fig, ax = plt.subplots()
df[df.category == cat].groupby('month').revenue.sum().plot(ax=ax)
ax.set_title(cat + ' revenue by month')
{x}
{y}
fig.savefig('category_trend.png', dpi=80)
plt.close(fig)
fig, ax = plt.subplots()
r['drop'].sort_values().plot.bar(ax=ax)
ax.set_title('Aug to Sep revenue drop by region')
ax.set_xlabel('Region')
ax.set_ylabel('Revenue drop (Rs)')
fig.savefig('region_breakdown.png', dpi=80)
plt.close(fig)
```"""


REVISE_ERROR = "REVISE:\n- KeyError 'revenue': create the revenue column before grouping."
REVISE_CHART = "REVISE:\n- category_trend.png has no axis labels. Add labels and rupee units."

# Numbers match data/ground_truth.json (Electronics 26.2 L -> 17.9 L, down 31.8%; South 4.9 L, 59%).
APPROVE = """APPROVED
ANSWER
Electronics had the biggest drop: revenue fell from Rs 26.2 L in August to Rs 17.9 L in September (down 31.8%).

WHERE IT CAME FROM
South region caused Rs 4.9 L of the drop (59.0%).

FILES
category_trend.png | region_breakdown.png | analysis.py

SUGGESTED NEXT QUESTION
Did discounts on Earbuds X2 change between August and September?"""

APPROVE_MADE_UP = APPROVE.replace("26.2", "44.4").replace("17.9", "12.1").replace("31.8", "72.7")


def clients(planner, coder, reviewer) -> ModelClients:
    p = ReplayChatCompletionClient(planner)
    return ModelClients(planner=p, coder=ReplayChatCompletionClient(coder),
                        reviewer=ReplayChatCompletionClient(reviewer), selector=p)



SAMPLE_QUESTION = ("Which category had the biggest revenue drop in September vs August, "
                   "and which region caused most of it?")


def demo_clients() -> ModelClients:
    """The spec 3.6 conversation: KeyError -> fix -> unlabelled chart -> Reviewer feedback -> APPROVED."""
    return clients([PLAN], [BAD_CODE, good_code(labels=False), good_code()], [REVISE_CHART, APPROVE])
