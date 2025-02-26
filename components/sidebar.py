import streamlit as st
from streamlit_option_menu import option_menu
from components.main_style import apply_custom_style

def create_sidebar(current_page="Home"):
    apply_custom_style()
    
    # Initialize options and icons
    options = ["Home","---","ML Preparations", "ML Implementation","Neuron Preparations",
               "Neuron Implementation","---","Reference"]
    icons = ["house","","bounding-box", "option","radar","rss","---","book"]
    

    if 'sidebar_selection' not in st.session_state:
        st.session_state.sidebar_selection = current_page
    
    page_indices = {
        "Home": 0,
        "ML Preparations": 2, 
        "ML Implementation": 3,
        "Neuron Preparations": 4,
        "Neuron Implementation": 5,
        "Reference": 7
    }
    
    default_index = page_indices.get(current_page, 0)
    
    with st.sidebar:
        selected = option_menu( 
            menu_title="NeuronActivated",
            options=options,
            icons=icons,
            menu_icon="robot",
            default_index=default_index,
            orientation="vertical",
            styles={
                "container": {"padding": "5px", "background-color": "#262730"},
                "icon": {"color": "orange", "font-size": "25px"}, 
                "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "#FF5757"},
             
            }
        )
        
        st.markdown("""
            <style>
                .footer {
                    margin-left: 2rem;
                }
            </style>
            <div class="footer">
                <i class="bi bi-c-circle"></i>
                &nbsp;Created by <b>KKhamwiset</b>
            </div>
        """, unsafe_allow_html=True)
        
    
    if selected != "---":
        st.session_state.sidebar_selection = selected
    return selected
