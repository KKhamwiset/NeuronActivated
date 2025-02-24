import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Simple page config
st.set_page_config(page_title="Simple ML App")

# Title
st.title("Simple ML Model App")
st.header("Claude cooked , I just want example :(")
# Function to load data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Tabs instead of sidebar
tab1, tab2, tab3 = st.tabs(["Data", "Train", "Predict"])

with tab1:
    st.header("Upload Data")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload dataset (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Save to session state
            st.session_state['df'] = df
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Basic stats
            st.subheader("Summary")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
            # Show missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                st.warning(f"Dataset has {missing.sum()} missing values")

with tab2:
    st.header("Train Model")
    
    if 'df' not in st.session_state:
        st.info("Please upload a dataset first")
    else:
        df = st.session_state['df']
        
        # Feature and target selection
        all_columns = df.columns.tolist()
        target_col = st.selectbox("Select target column", all_columns)
        feature_cols = st.multiselect("Select feature columns", 
                                    [col for col in all_columns if col != target_col])
        
        if feature_cols and target_col:
            # Simple model params
            test_size = st.slider("Test size %", 10, 50, 20) / 100
            n_trees = st.slider("Number of trees", 10, 200, 100)
            
            if st.button("Train Model"):
                with st.spinner("Training..."):
                    # Prepare data
                    X = df[feature_cols]
                    y = df[target_col]
                    
                    # Handle categorical features
                    for col in X.select_dtypes(include=['object']).columns:
                        X[col] = pd.Categorical(X[col]).codes
                    
                    # Handle missing values simply
                    X = X.fillna(X.mean())
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Train model
                    model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Save to session state
                    st.session_state['model'] = model
                    st.session_state['feature_cols'] = feature_cols
                    st.session_state['target_col'] = target_col
                    
                    # Performance
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Results
                    st.success(f"Model trained! Accuracy: {accuracy:.2f}")
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("Feature Importance")
                        importances = dict(zip(feature_cols, model.feature_importances_))
                        sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
                        
                        fig, ax = plt.subplots()
                        ax.bar(sorted_imp.keys(), sorted_imp.values())
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)

with tab3:
    st.header("Make Predictions")
    
    if 'model' not in st.session_state:
        st.info("Please train a model first")
    else:
        model = st.session_state['model']
        feature_cols = st.session_state['feature_cols']
        
        # Input form
        st.subheader("Enter values")
        
        # Simple input method
        input_data = {}
        for feature in feature_cols:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)
        
        if st.button("Predict"):
            # Create input dataframe
            input_df = pd.DataFrame([input_data])
            
            # Predict
            prediction = model.predict(input_df)[0]
            
            # Show result
            st.success(f"Prediction: {prediction}")
            
            # Show probabilities if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[0]
                proba_df = pd.DataFrame({
                    'Class': model.classes_,
                    'Probability': proba
                })
                st.dataframe(proba_df)

# Simple footer
st.markdown("---")
st.caption("Simple ML App")