import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


# class SVM(nn.Module):
#     def __init__(self, input_dim):
#         super(SVM, self).__init__()
#         self.linear = nn.Linear(input_dim, 2) 
        
#     def forward(self, x):
#         return self.linear(x)


# @st.cache_resource  
# def load_svm_model():
#     input_dim = 14
#     model = SVM(input_dim)
    
#     model_path = "exported_models/svm_model.pt"
#     checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint)
#     model.eval()
    
#     return model

class ML_implement_viewset:
    def __init__(self):
        # self.model = load_svm_model()
        pass
            
    def app(self):
        st.session_state.current_page = "ML Implementation"
        
        st.header("Machine Learning Implementation")
        
        st.subheader("Make Predictions")