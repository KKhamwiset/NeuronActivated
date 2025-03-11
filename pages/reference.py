import streamlit as st


class Ref_viewset:
    def __init__(self):
        self.content = []
        # Dictionary mapping header names to icons
        self.icons = {
            "Streamlit_guide": "📊",
            "Dataset": "🗂️",
            "Emoji": "😊",
            "Model_guide": "🤖",
            "Styling": "🎨",
        }

    def load_file(self, path):
        with open(path, mode="r") as f:
            line = f.readlines()
            self.content = line

    def app(self):
        # Custom CSS - dark theme compatible
        st.markdown(
            """
        <style>
        .reference-card {
            background-color: rgba(70, 70, 90, 0.2);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #4CAF50;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .reference-link {
            display: block;
            padding: 10px;
            margin: 5px 0;
            background-color: rgba(70, 70, 90, 0.1);
            border-radius: 5px;
            transition: transform 0.2s;
        }
        .reference-link:hover {
            transform: translateX(10px);
            background-color: rgba(76, 175, 80, 0.2);
        }
        .reference-link a {
            color: #4CAF50;
            text-decoration: none;
        }
        h1 {
            color: #4CAF50;
        }
        h3 {
            color: #4CAF50;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        st.title("📚 Project References")
        st.markdown(
            "<h2 style='text-align: center; margin-bottom: 30px;'>Thanks to these sources that made this project happen!</h2>",
            unsafe_allow_html=True,
        )

        # Load and process content
        self.load_file("ref.txt")
        current_header = None
        header_content = []

        for i in self.content:
            context = i.split()
            if len(context) > 1 and context[1] in ["header", "end_header"]:
                if context[1] == "end_header" and current_header and header_content:
                    self.display_header_section(current_header, header_content)
                    header_content = []
                    current_header = None
                elif context[1] == "header":
                    current_header = context[-1]
            elif current_header:
                header_content.append(i)

    def display_header_section(self, header, content_list):
        icon = self.icons.get(header, "🔗")

        st.markdown("---")
        st.subheader(f"{icon} {header.replace('_', ' ')}")

        # Display content items
        for content in content_list:
            parts = content.split()
            if len(parts) >= 2:
                link = parts[0]
                title = parts[-1].replace("=", " ")
                st.markdown(
                    f"""
                <div class='reference-link'>
                    <a href="{link}" target="_blank">{title} <span style='float:right'>↗️</span></a>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)
