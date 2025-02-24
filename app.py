import streamlit as st

# Basic app title
st.title("Hello Streamlit")

# Simple text
st.write("This is a very basic Streamlit app example")

# Add a slider
value = st.slider("Select a value", 0, 100, 50)
st.write(f"You selected: {value}")

# Add a button
if st.button("Click Me"):
    st.success("Button was clicked!")

# Add a text input
name = st.text_input("Enter your name")
if name:
    st.write(f"Hello, {name}!")

# Display some data
import pandas as pd
import numpy as np

# Create a simple dataframe
data = pd.DataFrame({
    'Column 1': [1, 2, 3, 4],
    'Column 2': [10, 20, 30, 40]
})

# Display the dataframe
st.subheader("Sample Data")
st.dataframe(data)

# Simple chart
st.subheader("Simple Chart")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C'])
st.line_chart(chart_data)