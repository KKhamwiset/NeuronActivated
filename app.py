import streamlit as st
import pandas as pd
import numpy as np
from components.sidebar import create_sidebar
from pages.ml_implementation import ML_implement_viewset
from pages.ml_preparations import ML_prepare_viewset
from pages.neuron_implementation import neuron_implement_viewset
from pages.neuron_preparations import neuron_prepare_viewset
from pages.reference import Ref_viewset

st.set_page_config(
    page_title="NeuronActivated",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.markdown(
    """
<style>
    .stButton>button {
        width: 100%;
    }
    .card {
        transition: transform 0.3s;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    .feature-icon {
        font-size: 24px;
        margin-right: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


class PageHandler:
    def __init__(self):
        pass

    def show_ml_implementation(self):
        show = ML_implement_viewset()
        show.app()

    def show_ml_preparations(self):
        show = ML_prepare_viewset()
        show.tabs_manager()

    def show_neuron_implementation(self):
        show = neuron_implement_viewset()
        show.app()

    def show_neuron_preparations(self):
        show = neuron_prepare_viewset()
        show.app()

    def show_reference(self):
        show = Ref_viewset()
        show.app()


selected = create_sidebar("Home")
page_handler = PageHandler()

if "show_home" not in st.session_state or selected == "Home":
    st.session_state.show_home = True
else:
    st.session_state.show_home = False


if st.session_state.show_home:
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
        """,
        unsafe_allow_html=True,
    )

    # Main header with Bootstrap classes
    st.markdown(
        """
    <div class="container mt-4">
        <div class="row">
            <div class="col-md-12">
                <div class="text-center p-4 bg-light rounded shadow-sm">
                    <h1 class="display-4 fw-bold text-primary">
                        <span class="me-2">🧠</span>NeuronActivated
                    </h1>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="card mb-4 shadow-sm h-100">
            <div class="card-body">
                <h5 class="card-title">
                    <span class="feature-icon">🤖</span>Machine Learning
                </h5>
                <p class="card-text">Explore various machine learning algorithms and implementations with our step-by-step guides.</p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        ml_col1, ml_col2 = st.columns(2)
        with ml_col1:
            if st.button("Implementation", key="ml_impl"):
                st.session_state.show_home = False
                st.session_state.current_page = "ML Implementation"
                st.rerun()
        with ml_col2:
            if st.button("Preparations", key="ml_prep"):
                st.session_state.show_home = False
                st.session_state.current_page = "ML Preparations"
                st.rerun()

    with col2:
        st.markdown(
            """
        <div class="card mb-4 shadow-sm h-100">
            <div class="card-body">
                <h5 class="card-title">
                    <span class="feature-icon">🔮</span>Neural Networks
                </h5>
                <p class="card-text">Dive into neural network architectures and learn how to build your own models from scratch.</p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        nn_col1, nn_col2 = st.columns(2)
        with nn_col1:
            if st.button(
                "Implementation",
                key="nn_impl",
            ):
                st.session_state.show_home = False
                st.session_state.current_page = "Neuron Implementation"
                st.rerun()
        with nn_col2:
            if st.button("Preparations", key="nn_prep"):
                st.session_state.show_home = False
                st.session_state.current_page = "Neuron Preparations"
                st.rerun()

    st.markdown(
        """
    <div class="container">
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="alert alert-info shadow-sm" role="alert">
                    <h4 class="alert-heading"><span class="me-2">💡</span>Getting Started</h4>
                    <p>Select a topic from the sidebar to begin exploring. Each section contains code examples, explanations, and interactive demos.</p>
                    <hr>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="container">
        <div class="card shadow-sm">
            <div class="card-header">
                <h5 class="mb-0"><span class="me-2">✨</span>Featured Content</h5>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Footer with Bootstrap
    st.markdown(
        """
    <div class="container">
        <div class="row mt-5">
            <div class="col-md-12">
                <div class="text-center text-muted small p-3">
                    <p>NeuronActivated © 2025 | Created with Streamlit</p>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


if selected == "ML Implementation":
    st.session_state.show_home = False
    page_handler.show_ml_implementation()

elif selected == "ML Preparations":
    st.session_state.show_home = False
    page_handler.show_ml_preparations()

elif selected == "Neuron Implementation":
    st.session_state.show_home = False
    page_handler.show_neuron_implementation()

elif selected == "Neuron Preparations":
    st.session_state.show_home = False
    page_handler.show_neuron_preparations()
elif selected == "Reference":
    st.session_state.show_home = False
    page_handler.show_reference()
