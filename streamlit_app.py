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
            st.Page("pages/01_Overview.py", "Overview", icon="📊"),
            st.Page("pages/02_Evaluation.py", "Evaluation", icon="🧮"),
            st.Page("pages/03_PCA.py", "PCA", icon="🧭"),
            st.Page("pages/04_Validation.py", "Validation", icon="✅"),
            st.Page("pages/05_Conclusion.py", "Conclusion", icon="📝"),
        ]
    }
)

navigation.run()
