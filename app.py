import streamlit as st
import pandas as pd
import numpy as np
from components.card import card_component, Counter, form_component    

# Basic app title
st.title("NeuronMap")

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

# Simple card
st.subheader("Simple Card")
st.title("Streamlit Components Demo")
    

st.header("Card Components")
card_component("Info Card", "This is a simple card component", "#e6f7ff")
card_component("Warning Card", "This shows a warning message", "#fff7e6")
card_component("Success Card", "Operation completed successfully", "#f6ffed")

st.header("Counter Components")
counter1 = Counter("counter1")
counter1.render()

counter2 = Counter("counter2", 10)
counter2.render()


st.header("Form Component")
result = form_component("Contact Form", ["Name", "Email", "Message"])

if result:
    st.success("Form submitted!")
    st.write(result)