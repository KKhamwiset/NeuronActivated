import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pickle
import joblib
import time


@st.cache_resource
def load_svm_model():
    try:
        # Try to load model with joblib first (more robust for sklearn objects)
        model_path = "exported_models/svm/svm_income_model.pkl"
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            st.info(f"Current working directory: {os.getcwd()}")
            st.info(
                f"Files in exported_models/svm: {os.listdir('exported_models/svm')}"
            )
            return None, None

        try:
            model = joblib.load(model_path)
        except Exception as e1:
            st.warning(f"Could not load with joblib: {str(e1)}")
            # Fall back to pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)

        # Try to load scaler
        scaler_path = "exported_models/svm/svm_income_scaler.pkl"
        if not os.path.exists(scaler_path):
            st.error(f"Scaler file not found at: {scaler_path}")
            return model, None

        try:
            scaler = joblib.load(scaler_path)
        except Exception as e2:
            st.warning(f"Could not load scaler with joblib: {str(e2)}")
            # Fall back to pickle
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        return model, scaler

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


class ML_implement_viewset:
    def __init__(self):
        self.model, self.scaler = load_svm_model()
        if self.model is None:
            st.error("Failed to load model. Please check the file paths and formats.")

    def app(self):
        st.session_state.current_page = "ML Implementation"

        # Sidebar for app navigation and info
        with st.sidebar:
            st.markdown("---")
            st.subheader("Option for Implementation")
            st.title("Income Predictor")
            st.markdown("---")
            st.info(
                "This app predicts if your income exceeds $50K based on your profile information."
            )

            # Add some information about the model
            with st.expander("About the Model"):
                st.write(
                    "This is a Support Vector Machine (SVM) model trained on the UCI Adult Census Income dataset."
                )
                st.write(
                    "Features include demographics, education, employment, and finances."
                )
                st.write("Model Accuracy: ~83%")

            # Add a disclaimer
            st.markdown("---")
            st.caption("⚠️ For educational purposes only. Not for financial decisions.")

        # Main content
        st.title("💰 Income Prediction Model")

        if self.model is None:
            st.error("⚠️ Model could not be loaded. Unable to make predictions.")
            return

        # Create tabs for better organization
        tab1, tab2 = st.tabs(["📝 Make a Prediction", "ℹ️ About the Model"])

        with tab1:
            # Header with description
            st.header("Enter Your Information")
            st.write(
                "Fill in the form below and click 'Predict Income' to see if your income exceeds $50K/year."
            )

            # Instructions in an expander
            with st.expander("📋 How to Use This Model"):
                st.write("1. Fill in all fields in the form below")
                st.write("2. Click the 'Predict Income' button at the bottom")
                st.write("3. View your prediction results and probability analysis")

            # Create a container for the form
            with st.container():
                st.subheader("Personal & Professional Details")

                # Create two columns for input fields
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### 👤 Personal Information")
                    age = st.slider(
                        "Age",
                        min_value=18,
                        max_value=90,
                        value=30,
                        help="Your current age in years",
                    )
                    race = st.selectbox(
                        "Race",
                        [
                            "White",
                            "Asian-Pac-Islander",
                            "Amer-Indian-Eskimo",
                            "Other",
                            "Black",
                        ],
                        help="Census racial category",
                    )
                    gender = st.radio(
                        "Gender",
                        ["Male", "Female"],
                        help="Gender as recorded in census data",
                        horizontal=True,
                    )

                    # Use a container for related fields
                    with st.container():
                        st.markdown("##### 👪 Family Status")
                        marital_status = st.selectbox(
                            "Marital Status",
                            [
                                "Married-civ-spouse",
                                "Divorced",
                                "Never-married",
                                "Separated",
                                "Widowed",
                                "Married-spouse-absent",
                                "Married-AF-spouse",
                            ],
                            help="Current marital status",
                        )
                        relationship = st.selectbox(
                            "Relationship",
                            [
                                "Wife",
                                "Own-child",
                                "Husband",
                                "Not-in-family",
                                "Other-relative",
                                "Unmarried",
                            ],
                            help="Relationship status in household",
                        )

                    native_country = st.selectbox(
                        "Native Country",
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
                        ],
                        help="Country of origin",
                    )

                with col2:
                    st.markdown("##### 🎓 Education")
                    education = st.selectbox(
                        "Education Level",
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
                        ],
                        help="Highest level of education completed",
                    )
                    education_num = st.slider(
                        "Years of Education",
                        min_value=1,
                        max_value=16,
                        value=13,
                        help="Years of education completed",
                    )

                    # Work container
                    with st.container():
                        st.markdown("##### 💼 Employment")
                        workclass = st.selectbox(
                            "Work Class",
                            [
                                "Private",
                                "Self-emp-not-inc",
                                "Self-emp-inc",
                                "Federal-gov",
                                "Local-gov",
                                "State-gov",
                                "Without-pay",
                                "Never-worked",
                            ],
                            help="Type of employer",
                        )
                        occupation = st.selectbox(
                            "Occupation",
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
                            ],
                            help="Type of work you do",
                        )
                        hours_per_week = st.slider(
                            "Hours per week",
                            min_value=1,
                            max_value=100,
                            value=40,
                            help="Average hours worked per week",
                        )

                    # Financial section
                    st.markdown("##### 💰 Financial Factors")
                    fnlwgt = st.number_input(
                        "Final Weight",
                        min_value=0,
                        value=200000,
                        help="Census weighting factor (you can leave as default)",
                    )

                    # Use metrics to make capital gains/losses more visually appealing
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

                # Preparation of data structure
                input_data = pd.DataFrame(
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

                # Add a divider
                st.markdown("---")

                # Centered button
                button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
                with button_col2:
                    submit_button = st.button(
                        "🔮 Predict Income", use_container_width=True
                    )

                # Results section
                if submit_button and self.model is not None:
                    # Show progress
                    with st.spinner("Processing your data..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            # Simulate computation
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)

                        # Make actual prediction
                        processed_data = self._preprocess_data(input_data)

                        # Remove progress after completion
                        progress_bar.empty()

                    # Debug options with expanders
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
                    self._display_results(processed_data)

        with tab2:
            # About the model tab content
            st.header("About This Model")

            # Create three columns for stats
            stat1, stat2, stat3 = st.columns(3)
            with stat1:
                st.metric(label="Accuracy", value="83%", delta="2.5%")
            with stat2:
                st.metric(label="Precision", value="74%", delta="1.7%")
            with stat3:
                st.metric(label="Recall", value="65%", delta="-0.5%")

            # Model information
            st.subheader("Model Details")
            st.write("""
            This application uses a Support Vector Machine (SVM) model trained on the UCI Adult Census Income dataset. 
            The model predicts whether an individual's income exceeds $50,000 per year based on census data.
            """)

            # Feature groups
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

            # Disclaimer
            st.warning(
                "This is for educational purposes only and should not be used for financial decisions."
            )

    def _preprocess_data(self, input_data):
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

        # Ensure the same column order as training data
        df = df[self.scaler.feature_names_in_]

        # Process categorical features
        for col in features_categorical:
            if col in df.columns:
                df[col] = self._map_categorical_feature(df[col])

        # Convert to int
        df = df.astype(int)

        # Scale the data
        if self.scaler is not None:
            scaled_data = self.scaler.transform(df)
            df = pd.DataFrame(scaled_data, columns=df.columns)

        return df

    def _map_categorical_feature(self, feature_series):
        """Map categorical features to integers (1 to n)"""
        unique_values = feature_series.unique()
        mapping = dict(zip(unique_values, range(1, len(unique_values) + 1)))
        return feature_series.map(mapping)

    def _display_results(self, processed_data):
        st.header("🔮 Prediction Results")

        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(processed_data)

            # Create a container for results
            results_container = st.container()

            with results_container:
                # Use columns for result display
                res_col1, res_col2 = st.columns([2, 3])

                with res_col1:
                    # Prominent display of result
                    if probability[0][1] > probability[0][0]:
                        st.success("##### Income prediction: >$50K")
                        # Use metrics to show probability
                        st.metric(
                            label="Probability of >$50K",
                            value=f"{probability[0][1]:.1%}",
                        )
                    else:
                        st.info("##### Income prediction: ≤$50K")
                        # Use metrics to show probability
                        st.metric(
                            label="Probability of ≤$50K",
                            value=f"{probability[0][0]:.1%}",
                        )

                with res_col2:
                    # Enhanced visualization
                    fig, ax = plt.subplots(figsize=(8, 4))
                    labels = ["≤$50K", ">$50K"]
                    colors = ["#3498db", "#2ecc71"]

                    # Create bars with better styling
                    bars = ax.bar(labels, probability[0], color=colors, width=0.6)

                    # Add value labels on top of bars
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

                    # Improve chart styling
                    ax.set_ylabel("Probability", fontsize=12)
                    ax.set_title("Income Prediction Probability", fontsize=14)
                    ax.set_ylim(0, 1.1)  # Add space for labels

                    # Remove spines
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    # Format y-axis as percentage
                    ax.yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: "{:.0%}".format(x))
                    )

                    plt.tight_layout()
                    st.pyplot(fig)

            # Add interpretation
            st.subheader("What Does This Mean?")

            # Use an expander for detailed explanation
            with st.expander("See Interpretation", expanded=True):
                if probability[0][1] > probability[0][0]:
                    st.write("""
                    Based on the information you provided, our model predicts that your income profile 
                    matches patterns typically associated with earnings **above $50,000 per year**.
                    
                    Key factors that likely influenced this prediction include:
                    - Your education level and years of education
                    - Your occupation and work class
                    - Your weekly working hours
                    """)
                else:
                    st.write("""
                    Based on the information you provided, our model predicts that your income profile 
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
            # For models without probability output
            prediction = self.model.predict(processed_data)
            if prediction[0] == 1:
                st.success("Income prediction: >$50K")
            else:
                st.info("Income prediction: ≤$50K")
