import streamlit as st


def apply_custom_style():
    # disable top menu
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                display: none !important;
                visibility: hidden !important;
                position: absolute !important;
                width: 0 !important;
                height: 0 !important;
                overflow: hidden !important;
                opacity: 0 !important;
                pointer-events: none !important;
                clip: rect(0, 0, 0, 0) !important;
                margin: -1px !important;
                padding: 0 !important;
                border: 0 !important;
            }
                
        </style>
        """,
        unsafe_allow_html=True,
    )

    # include bootstrap icons
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
        """,
        unsafe_allow_html=True,
    )
