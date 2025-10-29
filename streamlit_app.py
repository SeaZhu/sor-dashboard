from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="SOR Organizational Performance Dashboard",
    page_icon="📊",
    layout="wide",
)

navigation = st.navigation(
    {
        "": [
            st.Page(page="pages/01_Overview.py", title="Overview", icon="📊"),
            st.Page(page="pages/02_Evaluation.py", title="Evaluation", icon="🧮"),
            st.Page(page="pages/03_PCA.py", title="PCA", icon="🧭"),
            st.Page(page="pages/04_Validation.py", title="Validation", icon="✅"),
            st.Page(page="pages/05_Conclusion.py", title="Conclusion", icon="📝"),
        ]
    }
)

navigation.run()
