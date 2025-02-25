import streamlit as st
import pandas as pd
import numpy as np
from components.sidebar import create_sidebar
from pages.ml_implementation import ML_implement_viewset
from pages.ml_preparations import ML_prepare_viewset
from pages.neuron_implementation import neuron_implement_viewset
from pages.neuron_preparations import neuron_prepare_viewset
from pages.reference import Ref_viewset

# Set up the page
st.set_page_config(page_title="NeuronActivated", layout="wide", initial_sidebar_state="expanded")

# Sidebar selection
selected = create_sidebar("Home")

# Main title
st.title("Welcome to NeuronActivated")
st.write("Your home for machine learning and neural network projects")

# Handling navigation
if selected == "ML Implementation":
    show = ML_implement_viewset()
    show.app()

elif selected == "ML Preparations":
    show = ML_prepare_viewset()
    show.app()

elif selected == "Neuron Implementation":
    show = neuron_implement_viewset()
    show.app()

elif selected == "Neuron Preparations":
    show = neuron_prepare_viewset()
    show.app()

elif selected == "Reference":
    show = Ref_viewset()
    show.app()
