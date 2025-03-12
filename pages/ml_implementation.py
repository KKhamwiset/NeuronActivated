import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import pickle
import joblib
import time


@st.cache_resource
def load_model(model_path, scaler_path=None):
    try:
        try:
            model = joblib.load(model_path)
        except Exception as e1:
            st.warning(f"Could not load model with joblib: {str(e1)}")
            with open(model_path, "rb") as f:
                model = pickle.load(f)

        scaler = None
        if scaler_path:
            if not os.path.exists(scaler_path):
                st.error(f"Scaler file not found at: {scaler_path}")
            else:
                try:
                    scaler = joblib.load(scaler_path)
                except Exception as e2:
                    st.warning(f"Could not load scaler with joblib: {str(e2)}")
                    with open(scaler_path, "rb") as f:
                        scaler = pickle.load(f)

        return model, scaler

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


class ML_implement_viewset:
    def __init__(self):
        # Load models with their respective scalers
        self.svm_model, self.svm_scaler = load_model(
            "exported_models/svm/svm_income_model.pkl",
            "exported_models/svm/svm_income_scaler.pkl",
        )
        self.rf_model, self.rf_scaler = load_model(
            "exported_models/rf/rf_income_model.pkl",
            "exported_models/rf/rf_income_scaler.pkl",
        )

        if self.svm_model is None or self.rf_model is None:
            st.error(
                "Failed to load one or more models. Please check the file paths and formats."
            )

    def app(self):
        st.session_state.current_page = "ML Implementation"

        st.title("💰 Income Prediction Models")

        main_tab1, main_tab2 = st.tabs(["📝 Make a Prediction", "ℹ️ About the Models"])

        with main_tab1:
            if self.svm_model is None and self.rf_model is None:
                st.error("⚠️ Models could not be loaded. Unable to make predictions.")
                return

            st.header("Enter Your Information")
            st.write(
                "Fill in the form below and select a model to predict if your income exceeds $50K/year."
            )

            input_data = self._collect_user_input()

            st.markdown("---")

            model_tab1, model_tab2 = st.tabs(
                ["Support Vector Machine", "Random Forest"]
            )

            with model_tab1:
                if self.svm_model is not None:
                    self._predict_with_model(input_data, "svm")
                else:
                    st.error("SVM model is not available")

            with model_tab2:
                if self.rf_model is not None:
                    self._predict_with_model(input_data, "rf")
                else:
                    st.error("Random Forest model is not available")

        with main_tab2:
            # About the models tab content
            self._display_model_info()

    def _collect_user_input(self):
        """Collect and prepare user input from form"""
        with st.expander("📋 How to Use These Models"):
            st.write("1. Fill in all fields in the form below")
            st.write(
                "2. Navigate between model tabs to view predictions from different models"
            )
            st.write(
                "3. View prediction results and probability analysis for each model"
            )

        with st.container():
            st.subheader("Personal & Professional Details")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### 👤 Personal Information")
                age = st.number_input(
                    "Age",
                    min_value=18,
                    max_value=100,
                    value=30,
                    help="Your current age in years",
                )
                race = st.selectbox(
                    "Race",
                    sorted(
                        [
                            "White",
                            "Asian-Pac-Islander",
                            "Amer-Indian-Eskimo",
                            "Other",
                            "Black",
                        ]
                    ),
                    help="Census racial category",
                )
                gender = st.radio(
                    "Gender",
                    ["Male", "Female"],
                    help="Gender as recorded in census data",
                    horizontal=True,
                )

                with st.container():
                    st.markdown("##### 👪 Family Status")
                    marital_status = st.selectbox(
                        "Marital Status",
                        sorted(
                            [
                                "Married-civ-spouse",
                                "Divorced",
                                "Never-married",
                                "Separated",
                                "Widowed",
                                "Married-spouse-absent",
                                "Married-AF-spouse",
                            ]
                        ),
                        help="Current marital status",
                    )
                    relationship = st.selectbox(
                        "Relationship",
                        sorted(
                            [
                                "Wife",
                                "Own-child",
                                "Husband",
                                "Not-in-family",
                                "Other-relative",
                                "Unmarried",
                            ]
                        ),
                        help="Relationship status in household",
                    )

                native_country = st.selectbox(
                    "Native Country",
                    sorted(
                        [
                            "United-States",
                            "Cambodia",
                            "England",
                            "Puerto-Rico",
                            "Canada",
                            "Germany",
                            "Outlying-US(Guam-USVI-etc)",
                            "India",
                            "Japan",
                            "Greece",
                            "South",
                            "China",
                            "Cuba",
                            "Iran",
                            "Honduras",
                            "Philippines",
                            "Italy",
                            "Poland",
                            "Jamaica",
                            "Vietnam",
                            "Mexico",
                            "Portugal",
                            "Ireland",
                            "France",
                            "Dominican-Republic",
                            "Laos",
                            "Ecuador",
                            "Taiwan",
                            "Haiti",
                            "Columbia",
                            "Hungary",
                            "Guatemala",
                            "Nicaragua",
                            "Scotland",
                            "Thailand",
                            "Yugoslavia",
                            "El-Salvador",
                            "Trinadad&Tobago",
                            "Peru",
                            "Hong",
                            "Holand-Netherlands",
                        ]
                    ),
                    help="Country of origin",
                )

            with col2:
                st.markdown("##### 🎓 Education")
                education = st.selectbox(
                    "Education Level",
                    sorted(
                        [
                            "Bachelors",
                            "Some-college",
                            "11th",
                            "HS-grad",
                            "Prof-school",
                            "Assoc-acdm",
                            "Assoc-voc",
                            "9th",
                            "7th-8th",
                            "12th",
                            "Masters",
                            "Doctorate",
                            "10th",
                            "1st-4th",
                            "5th-6th",
                            "Preschool",
                        ]
                    ),
                    help="Highest level of education completed",
                )
                education_num = st.number_input(
                    "Years of Education",
                    min_value=1,
                    max_value=20,
                    value=13,
                    help="Years of education completed",
                )

                with st.container():
                    st.markdown("##### 💼 Employment")
                    workclass = st.selectbox(
                        "Work Class",
                        sorted(
                            [
                                "Private",
                                "Self-emp-not-inc",
                                "Self-emp-inc",
                                "Federal-gov",
                                "Local-gov",
                                "State-gov",
                                "Without-pay",
                                "Never-worked",
                            ]
                        ),
                        help="Type of employer",
                    )
                    occupation = st.selectbox(
                        "Occupation",
                        sorted(
                            [
                                "Tech-support",
                                "Craft-repair",
                                "Other-service",
                                "Sales",
                                "Exec-managerial",
                                "Prof-specialty",
                                "Handlers-cleaners",
                                "Machine-op-inspct",
                                "Adm-clerical",
                                "Farming-fishing",
                                "Transport-moving",
                                "Priv-house-serv",
                                "Protective-serv",
                                "Armed-Forces",
                            ]
                        ),
                        help="Type of work you do",
                    )
                    hours_per_week = st.number_input(
                        "Hours per week",
                        min_value=1,
                        max_value=168,
                        value=40,
                        help="Average hours worked per week",
                    )

                st.markdown("##### 💰 Financial Factors")
                fnlwgt = st.number_input(
                    "Final Weight",
                    min_value=0,
                    value=200000,
                    help="Census weighting factor (you can leave as default)",
                )

                cap_gain_col, cap_loss_col = st.columns(2)

                with cap_gain_col:
                    capital_gain = st.number_input(
                        "Capital Gain ($)",
                        min_value=0,
                        max_value=100000,
                        value=0,
                        help="Income from investment sources",
                    )

                with cap_loss_col:
                    capital_loss = st.number_input(
                        "Capital Loss ($)",
                        min_value=0,
                        max_value=10000,
                        value=0,
                        help="Losses from investment sources",
                    )

            return pd.DataFrame(
                [
                    {
                        "age": age,
                        "workclass": workclass,
                        "fnlwgt": fnlwgt,
                        "education": education,
                        "education-num": education_num,
                        "occupation": occupation,
                        "marital-status": marital_status,
                        "relationship": relationship,
                        "race": race,
                        "sex": gender,
                        "capital-gain": capital_gain,
                        "capital-loss": capital_loss,
                        "hours-per-week": hours_per_week,
                        "native-country": native_country,
                    }
                ]
            )

    def _predict_with_model(self, input_data, model_type="svm"):
        if model_type == "svm":
            model = self.svm_model
            scaler = self.svm_scaler
            model_name = "Support Vector Machine"
        else:
            model = self.rf_model
            scaler = self.rf_scaler
            model_name = "Random Forest"

        button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
        with button_col2:
            submit_button = st.button(
                f"🔮 Predict with {model_name}",
                key=f"predict_{model_type}",
                use_container_width=True,
            )


        if submit_button and model is not None:

            with st.spinner(f"Processing your data with {model_name}..."):
                progress_bar = st.progress(0)
                for i in range(100):

                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                processed_data = self._preprocess_data(input_data, scaler)

                progress_bar.empty()

            debug_col1, debug_col2 = st.columns(2)
            with debug_col1:
                with st.expander("🔍 Show Input Features"):
                    st.write("Features before processing:")
                    st.dataframe(input_data, use_container_width=True)

            with debug_col2:
                with st.expander("🔍 Show Processed Features"):
                    st.write("Features after processing:")
                    st.dataframe(processed_data, use_container_width=True)

            # Display results
            st.markdown("---")
            self._display_results(processed_data, model, model_name)

    def _preprocess_data(self, input_data, scaler):
        """Preprocess input data for model prediction"""
        features_categorical = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        df = input_data.copy()

        #
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            df = df[scaler.feature_names_in_]


        for col in features_categorical:
            if col in df.columns:
                df[col] = self._map_categorical_feature(df[col])

        df = df.astype(int)

        if scaler is not None:
            scaled_data = scaler.transform(df)
            df = pd.DataFrame(scaled_data, columns=df.columns)

        return df

    def _map_categorical_feature(self, feature_series):
        unique_values = feature_series.unique()
        mapping = dict(zip(unique_values, range(1, len(unique_values) + 1)))
        return feature_series.map(mapping)

    def _display_results(self, processed_data, model, model_name):
        st.header(f"🔮 {model_name} Prediction Results")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(processed_data)

            results_container = st.container()

            with results_container:
                res_col1, res_col2 = st.columns([2, 3])

                with res_col1:
                    if probability[0][1] > probability[0][0]:
                        st.success("##### Income prediction: >$50K")
                        st.metric(
                            label="Probability of >$50K",
                            value=f"{probability[0][1]:.1%}",
                        )
                    else:
                        st.info("##### Income prediction: ≤$50K")
                        st.metric(
                            label="Probability of ≤$50K",
                            value=f"{probability[0][0]:.1%}",
                        )

                with res_col2:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    labels = ["≤$50K", ">$50K"]
                    colors = ["#3498db", "#2ecc71"]

                    bars = ax.bar(labels, probability[0], color=colors, width=0.6)

                    for bar, prob in zip(bars, probability[0]):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.02,
                            f"{prob:.1%}",
                            ha="center",
                            va="bottom",
                            fontweight="bold",
                        )

                    ax.set_ylabel("Probability", fontsize=12)
                    ax.set_title(
                        f"{model_name} Income Prediction Probability", fontsize=14
                    )
                    ax.set_ylim(0, 1.1)

                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    ax.yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: "{:.0%}".format(x))
                    )

                    plt.tight_layout()
                    st.pyplot(fig)

            st.subheader("What Does This Mean?")

            with st.expander("See Interpretation", expanded=True):
                if probability[0][1] > probability[0][0]:
                    st.write(f"""
                    Based on the information you provided, our {model_name} model predicts that your income profile 
                    matches patterns typically associated with earnings **above $50,000 per year**.
                    
                    Key factors that likely influenced this prediction include:
                    - Your education level and years of education
                    - Your occupation and work class
                    - Your weekly working hours
                    """)
                else:
                    st.write(f"""
                    Based on the information you provided, our {model_name} model predicts that your income profile 
                    matches patterns typically associated with earnings **at or below $50,000 per year**.
                    
                    Key factors that likely influenced this prediction include:
                    - Your education level and years of education
                    - Your occupation and work class
                    - Your weekly working hours
                    """)

                st.info(
                    "Note: This prediction is based solely on statistical patterns in historical census data and should not be used for financial planning."
                )

        else:
            prediction = model.predict(processed_data)
            if prediction[0] == 1:
                st.success("Income prediction: >$50K")
            else:
                st.info("Income prediction: ≤$50K")

    def _display_model_info(self):
        st.header("About These Models")

        svm_tab, rf_tab = st.tabs(["Support Vector Machine", "Random Forest"])

        with svm_tab:
            stat1, stat2, stat3 = st.columns(3)
            with stat1:
                st.metric(label="Accuracy", value="81%")
            with stat2:
                st.metric(label="Precision", value="74%")
            with stat3:
                st.metric(label="Recall", value="65%")

            st.subheader("SVM Model Details")
            st.write("""
                SVM หรือ Support Vector Machine ถือเป็นวิธีคลาสสิคที่น่าเรียนรู้มากทีเดียว 
            เพราะไอเดียจากโมเดลนี้ก็เป็นหนึ่งในรากฐานสำคัญที่ทำให้เข้าใจโมเดลใหญ่ๆในปัจจุบันมากขึ้น 
            อีกทั้งหนึ่งในผู้ที่คิดค้นวิธี Vladimir Naumovich Vapnik ซึ่งเป็นชาวรัสเซีย 
            และยังเป็นนักวิทยาศาสตร์ที่ยิ่งใหญ่ในวงการ Machine Learning อีกด้วย
            """)
            
            st.markdown("---")
            
            st.write("""
            หลักการพื้นฐานของ SVM คือการหาเส้นแบ่ง (Hyperplane) ที่ดีที่สุดในการแยกข้อมูลออกเป็นสองกลุ่ม โดยเส้นแบ่งที่ดีที่สุดคือเส้นที่มีระยะห่างจากจุดข้อมูลที่ใกล้ที่สุดของแต่ละกลุ่มมากที่สุด 
            หรือที่เรียกว่ามี "margin" กว้างที่สุด
            จุดข้อมูลที่อยู่ใกล้กับเส้นแบ่งมากที่สุดเหล่านี้เรียกว่า "Support Vectors" ซึ่งเป็นที่มาของชื่ออัลกอริทึม นี่คือความฉลาดของ SVM ที่ใช้เพียงบางจุดข้อมูลที่สำคัญในการกำหนดเส้นแบ่ง ไม่ใช่ทุกจุดข้อมูล
            """)
            
            st.markdown("---")
            
            image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "svm", "svm_data.png")
            img = Image.open(image_path)
            st.image(img, caption="SVM Data Visualization", use_container_width=True)

        with rf_tab:
            stat1, stat2, stat3 = st.columns(3)
            with stat1:
                st.metric(label="Accuracy", value="84%")
            with stat2:
                st.metric(label="Precision", value="77%")
            with stat3:
                st.metric(label="Recall", value="68%")

            st.subheader("Random Forest Model Details")
            st.write("""
            Random forest เป็นหนึ่งในกลุ่มของโมเดลที่เรียกว่า Ensemble learning ที่มีหลักการคือการเทรนโมเดลที่เหมือนกันหลายๆ ครั้ง (หลาย Instance) บนข้อมูลชุดเดียวกัน
            โดยแต่ละครั้งของการเทรนจะเลือกส่วนของข้อมูลที่เทรนไม่เหมือนกัน แล้วเอาการตัดสินใจของโมเดลเหล่านั้นมาโหวตกันว่า Class ไหนถูกเลือกมากที่สุด
            """)
            
            st.write("""
            กระบวนการทำงานหลักๆ มีดังนี้:
                1. การสุ่มตัวอย่าง (Bootstrap Sampling): แต่ละต้นไม้จะถูกสร้างจากตัวอย่างที่สุ่มมาจากชุดข้อมูลเดิมด้วยวิธีการสุ่มแบบทดแทน (sampling with replacement) ทำให้บางตัวอย่างอาจถูกเลือกซ้ำ และบางตัวอย่างอาจไม่ถูกเลือกเลย
                2. การสุ่มคุณลักษณะ (Feature Randomness): ในแต่ละโหนดของต้นไม้ อัลกอริทึมจะไม่พิจารณาคุณลักษณะทั้งหมด แต่จะสุ่มเลือกเพียงบางส่วน ซึ่งช่วยเพิ่มความหลากหลายให้กับต้นไม้แต่ละต้น
                3. การรวมผลลัพธ์ (Aggregation): สำหรับการจำแนกประเภท ผลลัพธ์สุดท้ายจะได้จากการโหวตเสียงข้างมากของทุกต้นไม้ ส่วนการถดถอย จะใช้ค่าเฉลี่ยของผลลัพธ์จากทุกต้นไม้
            """)
            
            image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "rf", "rf_img.png")
            img = Image.open(image_path)
            st.image(img, caption="RandomForest Data Visualization", use_container_width=True)
            
        st.subheader("Features Used")
        

        feature_col1, feature_col2 = st.columns(2)

        with feature_col1:
            st.markdown("**Demographics**")
            st.markdown("- Age\n- Gender\n- Race\n- Native country")

            st.markdown("**Education**")
            st.markdown("- Education level\n- Years of education")

        with feature_col2:
            st.markdown("**Employment**")
            st.markdown("- Occupation\n- Work class\n- Hours worked per week")

            st.markdown("**Financial & Personal**")
            st.markdown(
                "- Capital gain\n- Capital loss\n- Marital status\n- Relationship"
            )

        st.warning(
            "These models are for educational purposes only and should not be used for financial decisions."
        )