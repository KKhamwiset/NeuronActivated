import streamlit as st


class neuron_prepare_viewset:
    def __init__(self):
        pass

    def app(self):
        st.title("Neuron Preparation Implementation")
        st.session_state.current_page = "Neuron Preparations"
        st.title("Machine Learning Preparations")
        st.markdown("---")
        menu = st.tabs(
            [
                "🌐การเตรียมข้อมูล",
                "🗳️ขั้นตอนการเทรน Model CNN",
            ]
        )
        with menu[0]:
            self.dataset_preparation()
        with menu[1]:
            pass
    def dataset_preparation(self):
        st.header("🌟 การเตรียมข้อมูล")
        st.markdown("---")
        st.warning("เนื่องจากข้อมูลมีขนาดใหญ่จึงไม่สามารถอัพขึ้น GitHub ได้") 
        st.subheader("🖥️Code สำหรับการเตรียมข้อมูล")

            

