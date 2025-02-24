import streamlit as st

def card_component(title, content, background_color="#ffffff"):
    with st.container():
        st.markdown(
            f"""
            <div style="
                padding: 20px;
                border-radius: 10px;
                background-color: {background_color};
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                <h3 style="margin-bottom: 10px; color: black;">{title}</h3>
                <p style="color: black;">{content}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

class Counter:
    def __init__(self, key, initial_value=0):
        self.key = key
        if key not in st.session_state:
            st.session_state[key] = initial_value
    
    def increment(self):
        st.session_state[self.key] += 1
    
    def decrement(self):
        st.session_state[self.key] -= 1
    
    def render(self):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.button("➖", key=f"dec_{self.key}", on_click=self.decrement)
        with col2:
            st.write(f"### {st.session_state[self.key]}")
        with col3:
            st.button("➕", key=f"inc_{self.key}", on_click=self.increment)


def form_component(form_title, fields):
    with st.form(key=form_title.lower().replace(" ", "_")):
        st.subheader(form_title)
        field_values = {}
        for field in fields:
            field_id = field.lower().replace(" ", "_")
            field_values[field_id] = st.text_input(field)
        
        # Submit button
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            return field_values
        return None