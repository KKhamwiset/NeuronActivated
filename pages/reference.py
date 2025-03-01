import streamlit as st


class Ref_viewset:
    def __init__(self):
        self.content = []

    def load_file(self, path):
        with open(path, mode="r") as f:
            line = f.readlines()
            self.content = line

    def app(self):
        st.title("📚 Project References")
        st.header("Thanks to these sources that made this project happen!")
        self.load_file("ref.txt")
        for i in self.content:
            context = i.split()
            if len(context) > 1 and context[1] in ["header", "end_header"]:
                header_name = context[-1]
                if header_name != "end_header":
                    st.subheader(header_name.replace("_", "'s "))
                else:
                    st.markdown("---")
            else:
                sub_content = i.split()
                show = sub_content[-1].replace("=", " ")
                link = sub_content[0]
                st.markdown(f"""[{show}]({link})""")
