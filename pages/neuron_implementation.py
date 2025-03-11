import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import io
import os


class neuron_implement_viewset:
    def __init__(self):
        self.model = None
        self.class_names = [
            "apple",
            "avocado",
            "banana",
            "cherry",
            "kiwi",
            "mango",
            "orange",
            "pineapple",
            "strawberries",
            "watermelon",
        ]
        self.img_height = 224
        self.img_width = 224

    def load_cnn_model(self):
        try:
            model_path = "exported_models/fruit_model.pkl"
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                return True
            else:
                st.error(f"Model not found at {model_path}. Please check the path.")
                return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

    def preprocess_image(self, image):
        image = image.resize((self.img_width, self.img_height))

        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        img_array = img_array / 255.0

        return img_array

    def predict_fruit(self, image):
        if self.model is None:
            if not self.load_cnn_model():
                return None

        processed_img = self.preprocess_image(image)

        predictions = self.model.predict(processed_img)

        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])

        return {
            "class": self.class_names[predicted_class_index],
            "confidence": confidence,
        }

    def display_prediction(self, prediction, image):
        """Display the prediction results"""
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Prediction Results")
            if prediction:
                st.success(f"Predicted fruit: {prediction['class']}")
                st.progress(prediction["confidence"])
                st.write(f"Confidence: {prediction['confidence']:.2f}")

                # Provide some interpretation
                if prediction["confidence"] > 0.8:
                    st.write("✅ High confidence prediction")
                elif prediction["confidence"] > 0.5:
                    st.write("⚠️ Moderate confidence prediction")
                else:
                    st.write("❌ Low confidence prediction")
            else:
                st.error("Could not make a prediction. Please try again.")

    def app(self):
        st.title("Fruit Identification with CNN (MobileNetV2 base model)")
        st.write("Upload an image of a fruit")
        st.info(f"Currently supported fruits: {', '.join(self.class_names)}")
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            if st.button("Predict Fruit"):
                with st.spinner("Analyzing image..."):
                    prediction = self.predict_fruit(image)
                    self.display_prediction(prediction, image)

        with st.expander("How it works"):
            st.write("""
            1. Upload an image of one of the supported fruits
            2. Click the 'Predict Fruit' button
            3. The CNN model will analyze the image and predict what fruit it is
            4. Results will show the predicted fruit and confidence level
            """)

        with st.expander("About the model"):
            st.write("""
            This application uses a Convolutional Neural Network (CNN) trained on images of various fruits.
            The model was trained to recognize patterns and features specific to different fruits.
            
            Model Architecture:
            - Input shape: 224x224x3 (RGB images)
            - Multiple convolutional layers for feature extraction
            - Fully connected layers for classification
            """)
