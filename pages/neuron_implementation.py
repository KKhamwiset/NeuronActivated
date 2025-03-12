import streamlit as st
import numpy as np
import tensorflow as tf
from keras.layers import (
    Dropout,
    Dense,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from keras.applications import MobileNetV2
from keras.models import Sequential
from PIL import Image
import os


@st.cache_resource
def create_model():
    baseModel = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    baseModel.trainable = False
    model = Sequential(
        [
            baseModel,
            GlobalAveragePooling2D(),
            Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            ),
            BatchNormalization(),
            Dropout(0.3),
            Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            ),
            BatchNormalization(),
            Dropout(0.3),
            Dense(
                32,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            ),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )
    return model


@st.cache_resource
def load_cached_model(model_path, weights_path, img_height, img_width):
    try:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                print(f"Successfully loaded entire model from {model_path}")
                return (
                    model,
                    True,
                    f"Successfully loaded entire model from {model_path}",
                )

            except Exception as e:
                if "Conv1" in str(e):
                    print(
                        f"Error loading full model: {e}. Attempting to load weights only."
                    )
                    if os.path.exists(weights_path):
                        model = create_model()
                        model.load_weights(weights_path)
                        print(f"Successfully loaded weights from {weights_path}")

                        return (
                            model,
                            True,
                            f"Successfully loaded weights from {weights_path}",
                        )
                    else:
                        return None, False, f"Weights file not found at {weights_path}"
                else:
                    raise e
        else:
            print(f"Model not found at {model_path}")

            if os.path.exists(weights_path):
                model = create_model()
                model.load_weights(weights_path)
                print(f"Loaded weights-only from {weights_path}")


                dummy_input = np.zeros((1, img_height, img_width, 3))
                _ = model.predict(dummy_input)

                return (
                    model,
                    True,
                    f"Successfully loaded weights-only from {weights_path}",
                )
            else:
                return (
                    None,
                    False,
                    "Neither model nor weights found. Please check the paths.",
                )

    except Exception as e:
        error_message = f"Error loading model: {e}"
        print(error_message)
        return None, False, error_message


class neuron_implement_viewset:
    def __init__(self):
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

        model_path = "exported_models/fruit_model.keras"
        weights_path = "exported_models/fruit_model_weights.h5"

        self.model, self.model_loaded, message = load_cached_model(
            model_path, weights_path, self.img_height, self.img_width
        )

        if self.model_loaded:
            st.success(message)
        else:
            st.error(message)

    def preprocess_image(self, image):
        image = image.resize((self.img_width, self.img_height))

        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        img_array = img_array / 255.0

        return img_array

    def predict_fruit(self, image):
        if self.model is None:
            if not self.model_loaded:
                st.error("Failed to load the model. Cannot make predictions.")
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
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Prediction Results")
            if prediction:
                st.success(f"Predicted fruit: {prediction['class'].capitalize()}")
                st.progress(prediction["confidence"])
                st.write(f"Confidence: {prediction['confidence']:.2f}")
                if prediction["confidence"] > 0.8:
                    st.write("✅ High confidence prediction")
                elif prediction["confidence"] > 0.5:
                    st.write("⚠️ Moderate confidence prediction")
                else:
                    st.write("❌ Low confidence prediction")
            else:
                st.error("Could not make a prediction. Please try again.")
    def display_model(self):
        st.header("About CNN and MobileNetV2 base model")
        
        cnn_tab, transfer_tab = st.tabs(["😶‍🌫️Convolutional Neural Network (CNN)", "↔️Transfer Learning"])
        
        with cnn_tab:
            st.subheader("😶‍🌫️ Convolutional Neural Network (CNN)")
            
            st.write("""
            Convolutional Neural Network หรือ CNN เป็นหนึ่งในโมเดลที่ได้รับความนิยมมากที่สุดที่ใช้ในงานประมวลผลภาษาธรรมชาติ ข้อได้เปรียบที่สำคัญที่สุดของโมเดลนี้คือสามารถตรวจจับลักษณะเด่นที่สำคัญได้โดยอัตโนมัติด้วยตัวเอง นอกจากนี้ CNN ยังพิสูจน์แล้วว่ามีประสิทธิภาพในการคำนวณด้วย สามารถประมวลผลบนเครื่องคอมพิวเตอร์ทั่วไปได้ และมีความพิเศษในการใช้การทำงานแบบคอนโวลูชันและพูลลิ่ง
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**หลักการทำงาน**")
                st.write("""
                คำว่า "คอนโวลูชัน" แทนฟังก์ชันทางคณิตศาสตร์ของการรวมชุดข้อมูลสองชุดเข้าด้วยกัน CNN รักษาคุณลักษณะความไม่เป็นเชิงเส้น (nonlinearity) ซึ่งเป็นสิ่งที่ควรมีในเครือข่ายประสาทเทียมที่มีประสิทธิภาพ 
                
                การพูลลิ่ง (Pooling) ใช้เพื่อลดมิติโดยลดจำนวนพารามิเตอร์ และช่วยลดเวลาที่ใช้ในการประมวลผล 
                
                CNN ถูกฝึกโดยใช้การเรียนรู้แบบย้อนกลับ (backpropagation) ร่วมกับ gradient descent
                """)
            
            with col2:
                st.write("**องค์ประกอบสำคัญ**")
                st.write("""
                โมเดล CNN มีสองส่วนหลัก ได้แก่:
                
                1. **การสกัดลักษณะเด่น (Feature Extraction)** และการจัดหมวดหมู่ตามลักษณะเหล่านั้น
                
                2. **เลเยอร์คอนโวลูชัน (Convolution Layers)** ซึ่งทำหน้าที่เป็นแรงขับเคลื่อนหลักของโมเดล CNN
                """)
                image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "cnn", "cnn_img.jpg")
                img = Image.open(image_path)
                st.image(img, caption="สถาปัตยกรรมพื้นฐานของ CNN", use_container_width=True)
        
        with transfer_tab:
            st.subheader("↔️ Transfer Learning")
            st.write("""เนื่องจากผมสังเกตว่าการเทรน Model CNN โดยข้อมูลเป็นรูปภาพนั้นใช้เวลามากจนเกินไปเลยได้หาข้อมูลวิธีการ
                     และได้พบกับ Transfer Learning
                     """)
            st.write("""
            Transfer Learning คือเทคนิคในวงการ Machine Learning ที่นำความรู้หรือข้อมูลที่ได้จากการแก้ปัญหาหนึ่งมาประยุกต์ใช้กับอีกปัญหาหนึ่งที่มีความเกี่ยวข้องกัน โดยเฉพาะในงานด้าน Deep Learning ที่ต้องการข้อมูลจำนวนมากและทรัพยากรในการประมวลผลสูง
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**หลักการทำงาน**")
                st.write("""
                แนวคิดพื้นฐานคือการนำโมเดลที่ผ่านการเทรนมาแล้วบนชุดข้อมูลขนาดใหญ่ (เช่น ImageNet) มาปรับใช้กับงานใหม่ที่มีข้อมูลน้อยกว่า ซึ่งช่วยลดเวลาในการเทรนและทรัพยากรที่ต้องใช้

                วิธีการทำ Transfer Learning ทั่วไปคือ:
                1. นำโมเดลที่เทรนมาแล้ว (pre-trained model) เช่น MobileNetV2, ResNet, VGG มาใช้
                2. ตัดเอาเฉพาะส่วนที่ใช้สกัดคุณลักษณะ (feature extraction)
                3. เพิ่มเลเยอร์ใหม่สำหรับการจำแนกประเภทตามงานของเรา
                4. เทรนเฉพาะเลเยอร์ใหม่ที่เพิ่มเข้าไป โดยไม่ปรับค่าน้ำหนักในส่วนของโมเดลเดิม
                """)
            
            with col2:
                st.write("**ประโยชน์ของ Transfer Learning**")
                st.write("""
                1. **ประหยัดเวลาและทรัพยากร** - ไม่จำเป็นต้องเทรนโมเดลตั้งแต่เริ่มต้น
                2. **ใช้งานได้กับข้อมูลน้อย** - สามารถเรียนรู้และให้ผลลัพธ์ที่ดีแม้มีข้อมูลฝึกอบรมจำนวนไม่มาก
                3. **ประสิทธิภาพสูง** - นำความรู้จากโมเดลที่เทรนด้วยข้อมูลขนาดใหญ่มาช่วยเพิ่มประสิทธิภาพ
                4. **ลดปัญหา Overfitting** - เนื่องจากโมเดลฐานได้เรียนรู้คุณลักษณะทั่วไปมาแล้ว
                """)
                image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "cnn", "transfer_learning.jpg")
                img = Image.open(image_path)
                st.image(img, caption="แนวคิดของ Transfer Learning", use_container_width=True)

                
    def app(self):
        pred,about = st.tabs(["Make Prediction","About the CNN Model"])
        with pred:
            st.title("Fruit Identification with CNN (MobileNetV2 base model)")
            st.write("Upload an image of a fruit")
            st.info(f"Currently supported fruits: {', '.join(self.class_names)}")
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
        with about:
            self.display_model()
        st.markdown("---")
