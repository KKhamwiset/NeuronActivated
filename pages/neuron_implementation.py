import streamlit as st
import numpy as np
import tensorflow as tf
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    Activation,
    BatchNormalization,
    Input,
    GlobalAveragePooling2D,
)
from keras.applications import MobileNetV2
from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
from PIL import Image
import io
import os


@st.cache_resource
def create_model():
    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    baseModel.trainable = False
    model = Sequential([
        baseModel,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        Dense(10, activation="softmax"),
    ])
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )
    return model


# Create a cached function for model loading
@st.cache_resource
def load_cached_model(model_path, weights_path, img_height, img_width):
    """Cache the model loading to avoid reloading on each rerun"""
    try:
        # First, try to load the full model
        if os.path.exists(model_path):
            try:
                # Try loading the complete model
                model = tf.keras.models.load_model(model_path)
                print(f"Successfully loaded entire model from {model_path}")
                
                # Warmup prediction to initialize the model completely
                dummy_input = np.zeros((1, img_height, img_width, 3))
                _ = model.predict(dummy_input)
                print("Model initialized with warmup prediction")
                
                return model, True, f"Successfully loaded entire model from {model_path}"
                
            except Exception as e:
                # Handle Conv1 layer error specifically
                if "Conv1" in str(e):
                    print(f"Error loading full model: {e}. Attempting to load weights only.")
                    
                    # Try loading from architecture+weights instead
                    if os.path.exists(weights_path):
                        model = create_model()
                        model.load_weights(weights_path)
                        print(f"Successfully loaded weights from {weights_path}")
                        
                        # Warmup prediction
                        dummy_input = np.zeros((1, img_height, img_width, 3))
                        _ = model.predict(dummy_input)
                        print("Model initialized with warmup prediction")
                        
                        return model, True, f"Successfully loaded weights from {weights_path}"
                    else:
                        return None, False, f"Weights file not found at {weights_path}"
                else:
                    # If it's a different error, raise it
                    raise e
        else:
            print(f"Model not found at {model_path}")
            
            # Try loading weights only
            if os.path.exists(weights_path):
                model = create_model()
                model.load_weights(weights_path)
                print(f"Loaded weights-only from {weights_path}")
                
                # Warmup prediction
                dummy_input = np.zeros((1, img_height, img_width, 3))
                _ = model.predict(dummy_input)
                
                return model, True, f"Successfully loaded weights-only from {weights_path}"
            else:
                return None, False, "Neither model nor weights found. Please check the paths."

    except Exception as e:
        error_message = f"Error loading model: {e}"
        print(error_message)
        return None, False, error_message


class neuron_implement_viewset:
    def __init__(self):
        # Initialize model parameters
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
        
        # Load the cached model
        model_path = "exported_models/fruit_model.keras"
        weights_path = "exported_models/fruit_model_weights.h5"
        
        # Load model using cached function
        self.model, self.model_loaded, message = load_cached_model(
            model_path, weights_path, self.img_height, self.img_width
        )
        
        # Display appropriate message based on loading result
        if self.model_loaded:
            st.success(message)
        else:
            st.error(message)

    def preprocess_image(self, image):
        """Preprocess an image for model prediction"""
        # Resize to expected dimensions
        image = image.resize((self.img_width, self.img_height))
        
        # Convert to array and add batch dimension
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        
        # Scale pixel values to 0-1 range
        img_array = img_array / 255.0
        
        return img_array

    def predict_fruit(self, image):
        """Predict the fruit class from an image"""
        # Check if model is loaded
        if self.model is None:
            if not self.model_loaded:
                st.error("Failed to load the model. Cannot make predictions.")
                return None
        
        # Preprocess the image and get predictions
        processed_img = self.preprocess_image(image)
        predictions = self.model.predict(processed_img)
        
        # Get the highest confidence prediction
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
        """Main Streamlit application"""
        st.title("Fruit Identification with CNN (MobileNetV2 base model)")
        st.write("Upload an image of a fruit")
        st.info(f"Currently supported fruits: {', '.join(self.class_names)}")
        
        # Check if model is loaded before accepting uploads
        if not self.model_loaded:
            st.error("Model could not be loaded. Please check the server logs.")
            return
            
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
            This application uses a Convolutional Neural Network (CNN) based on MobileNetV2 
            architecture trained on images of various fruits. The model was trained to recognize 
            patterns and features specific to different fruits.
            
            Model Architecture:
            - Base: MobileNetV2 (pretrained on ImageNet)
            - Global Average Pooling
            - Dense layers with regularization
            - 10-class softmax output layer
            """)