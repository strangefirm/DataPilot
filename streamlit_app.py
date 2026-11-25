import streamlit as st

from datapilot.app_state import init_state

st.set_page_config(page_title="DataPilot", page_icon=":material/insights:", layout="wide")
init_state()

page = st.navigation(
    [
        st.Page("app_pages/analyst.py", title="Analyst", icon=":material/query_stats:", default=True),
        st.Page("app_pages/history.py", title="Run history", icon=":material/monitoring:"),
    ],
    position="top",
)
page.run()
