import streamlit as st
import pandas as pd


class neuron_prepare_viewset:
    def __init__(self):
        pass

    def app(self):
        st.session_state.current_page = "Neuron Preparations"
        st.title("Neural Network Preparations")
        st.markdown("---")
        menu = st.tabs(
            [
                "🌐การเตรียมข้อมูล",
                "🗳️ขั้นตอนการเทรน Model CNN",
                "🧠การโหลด CNN Model ไปใช้งาน",
            ]
        )
        with menu[0]:
            self.dataset_preparation()
        with menu[1]:
            self.cnn_model_training()
            st.markdown("""---""")
        with menu[2]:
            self.load_model()

    def dataset_preparation(self):
        st.header("🔍 Data Preparation")
        st.markdown(
            """
            สำหรับขั้นตอนการเตรียมข้อมูลในการเทรน CNN models ผมได้ค้นหา datasets ใน [<span style='color:blue; text-decoration:none'>Kaggle</span>](https://www.kaggle.com/) และได้เลือกข้อมูล
            "[<span style='color:orange; text-decoration:none; font-weight:bold'>Fruit Classification dataset</span>](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class/data)"
            ซึ่งเป็นข้อมูลรูปภาพของผลไม้ 10 ชนิด เพื่อนำมาทำ Image Classification ด้วย CNN
            """,
            unsafe_allow_html=True,
        )
        with st.expander("📊 รายละเอียดชุดข้อมูล (Dataset Details)"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**แหล่งที่มา:** Kaggle Dataset")
                st.write("**จำนวนตัวอย่าง:** 4,323 รูปภาพ")
            with col2:
                st.write("**จำนวนคลาส:** 10 ชนิดผลไม้")
                st.write("**ขนาดรูปภาพ:** หลากหลายขนาด (จะถูกปรับให้เป็นขนาดเดียวกัน)")

        st.header("🤖 Model Selection")
        st.markdown(
            "ผมจะใช้โมเดล **Convolutional Neural Network (CNN)** ซึ่งเป็นโมเดลที่เหมาะกับงาน Image Classification"
        )

        st.header("📋 รายละเอียด Features Fruit Classification")

        fruit_data = {
            "ชื่อผลไม้": [
                "Apple (แอปเปิล)",
                "Banana (กล้วย)",
                "Orange (ส้ม)",
                "Strawberry (สตรอเบอร์รี่)",
                "Watermelon (แตงโม)",
                "Pineapple (สับปะรด)",
                "Mango (มะม่วง)",
                "Grape (องุ่น)",
                "Kiwi (กีวี่)",
                "Peach (พีช)",
            ],
            "จำนวนรูป": [
                "~430 รูป",
                "~430 รูป",
                "~430 รูป",
                "~430 รูป",
                "~430 รูป",
                "~430 รูป",
                "~430 รูป",
                "~430 รูป",
                "~430 รูป",
                "~430 รูป",
            ],
            "รูปแบบไฟล์": [
                "JPG/JPEG",
                "JPG/JPEG",
                "JPG/JPEG",
                "JPG/JPEG",
                "JPG/JPEG",
                "JPG/JPEG",
                "JPG/JPEG",
                "JPG/JPEG",
                "JPG/JPEG",
                "JPG/JPEG",
            ],
        }

        df_fruits = pd.DataFrame(fruit_data)
        st.dataframe(df_fruits, use_container_width=True, height=400)

        st.info("""
        **หมายเหตุ:** 
        * ข้อมูลชุดนี้แบ่งเป็น folder train และ test
        * แต่ละ folder แบ่งตามชนิดของผลไม้ (Class)
        * รูปภาพมีความหลากหลายทั้งขนาด, สี, และพื้นหลัง
        * โมเดล CNN จะถูกเทรนให้สามารถจำแนกผลไม้เหล่านี้ได้อย่างแม่นยำ
        """)

        st.header("⚙️ ขั้นตอนการเตรียมข้อมูล (Data Preparation Process)")

        code_tabs = st.tabs(
            ["1️⃣ การนำเข้า Libraries", "2️⃣ การสำรวจข้อมูล", "3️⃣ การเตรียมข้อมูลรูปภาพ"]
        )
        st.markdown(
            """
        <style>
            .stTabs [data-baseweb="tab"] {
                font-size: 28px; 
                font-weight: bold;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 1.5rem;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )
        with code_tabs[0]:
            st.code(
                """
                import numpy as np
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                import os
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, MaxPooling2D, Activation, Flatten, Dropout, Conv2D
                from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
                """
            )

        with code_tabs[1]:
            st.code("""
                # ค้นหา Path ของข้อมูล
                dataset_dir = "../data/fruit/MY_data"
                train_path = os.path.join(dataset_dir, "train")
                test_path = os.path.join(dataset_dir, "test")

                # ตรวจสอบจำนวน Class ทั้งหมด
                classes = os.listdir(train_path)
                print(f"จำนวน Class ทั้งหมด: {len(classes)}")
                print(f"รายชื่อ Class: {classes}")

                # ตรวจสอบจำนวนรูปภาพในแต่ละ Class (folder)
                for cls in classes:
                    cls_path = os.path.join(train_path, cls)
                    num_images = len(os.listdir(cls_path))
                    print(f"Class {cls}: {num_images} รูปภาพ")

                # แสดงตัวอย่างรูปภาพ
                img = load_img(train_path + "/Apple/img_01.jpeg")
                plt.imshow(img)
                plt.axis("on")
                plt.show()

                # ตรวจสอบขนาดรูปภาพ
                img = img_to_array(img)
                print(f"ขนาดรูปภาพ: {img.shape}")
                """)

        with code_tabs[2]:
            st.code("""
                # สร้าง Data Generator สำหรับปรับแต่งรูปภาพ (Data Augmentation)
                train_datagen = ImageDataGenerator(
                    rescale=1./255,         # ปรับค่าพิกเซลให้อยู่ในช่วง 0-1
                    shear_range=0.3,        # การบิดภาพ
                    horizontal_flip=True,    # การพลิกภาพแนวนอน
                    vertical_flip=False,     # ไม่พลิกภาพแนวตั้ง
                    zoom_range=0.3          # การซูมภาพ
                )

                test_datagen = ImageDataGenerator(rescale=1./255)  # สำหรับชุดทดสอบเราจะเพียงปรับค่าพิกเซลเท่านั้น

                # สร้าง Data Generator สำหรับโหลดรูปภาพจาก directory
                train_generator = train_datagen.flow_from_directory(
                    train_path,
                    target_size=(img.shape[0], img.shape[1]),  # ปรับขนาดรูปภาพให้เท่ากันทั้งหมด
                    batch_size=32,
                    color_mode='rgb',
                    class_mode='categorical'  # กำหนดรูปแบบของ Class เป็น one-hot encoding
                )

                test_generator = test_datagen.flow_from_directory(
                    test_path,
                    target_size=(img.shape[0], img.shape[1]),
                    batch_size=32,
                    color_mode='rgb',
                    class_mode='categorical'
                )

                # ตรวจสอบข้อมูลที่ได้จาก generator
                print(f"จำนวน batch ในชุดข้อมูลฝึก: {len(train_generator)}")
                print(f"จำนวน batch ในชุดข้อมูลทดสอบ: {len(test_generator)}")
                print(f"รูปแบบข้อมูล X: {train_generator.next()[0].shape}")
                print(f"รูปแบบข้อมูล y: {train_generator.next()[1].shape}")
                """)

        second_code_tabs = st.tabs(
            [
                "4️⃣ การแสดงตัวอย่างรูปภาพหลังการ Augmentation",
                "5️⃣ การเตรียมข้อมูลสำหรับการเทรน",
                "6️⃣ การกำหนดโครงสร้างโมเดล CNN",
            ]
        )

        with second_code_tabs[0]:
            st.code("""
                # แสดงตัวอย่างรูปภาพที่ผ่านการ augment จำนวน 5 รูป
                fig, axes = plt.subplots(1, 5, figsize=(20, 5))
                for i, (x_batch, y_batch) in enumerate(train_generator):
                    if i >= 5:
                        break
                    axes[i].imshow(x_batch[0])
                    axes[i].set_title(f"Sample {i+1}")
                    axes[i].axis('off')
                plt.tight_layout()
                plt.show()
            """)

        with second_code_tabs[1]:
            st.code("""
                # ตรวจสอบคลาสที่มีในชุดข้อมูล
                class_indices = train_generator.class_indices
                print("Class Indices:", class_indices)

                # สลับ key กับ value เพื่อให้สามารถแปลงตัวเลขกลับเป็นชื่อผลไม้ได้
                class_names = {v: k for k, v in class_indices.items()}
                print("Class Names:", class_names)

                # ตรวจสอบจำนวนตัวอย่างในแต่ละคลาส
                class_counts = np.bincount(train_generator.classes)
                for class_id, count in enumerate(class_counts):
                    print(f"Class {class_names[class_id]}: {count} samples")
            """)

        with second_code_tabs[2]:
            st.code("""
                # กำหนดโครงสร้างโมเดล CNN
                model = Sequential([
                    # ชั้นที่ 1: Convolutional Layer
                    Conv2D(128, 3, activation="relu", input_shape=(img.shape[0], img.shape[1], 3)),
                    MaxPooling2D(2, 2),
                    
                    # ชั้นที่ 2: Convolutional Layer
                    Conv2D(64, 3, activation="relu"),
                    
                    # ชั้นที่ 3: Convolutional Layer
                    Conv2D(32, 3, activation="relu"),
                    MaxPooling2D(2, 2),
                    
                    # Dropout เพื่อป้องกัน overfitting
                    Dropout(0.5),
                    
                    # แปลงข้อมูลให้เป็น 1 มิติ
                    Flatten(),
                    
                    # Fully connected layers
                    Dense(256, activation="relu"),
                    Dense(16, activation="relu"),
                    
                    # Output layer (10 classes)
                    Dense(10, activation="softmax")
                ])

                # แสดงสรุปโครงสร้างโมเดล
                model.summary()

                # คอมไพล์โมเดล
                model.compile(
                    loss='categorical_crossentropy',
                    optimizer='SGD',
                    metrics=['accuracy']
                )
            """)

        st.markdown("---")
        st.subheader("🎉🎉🎉 เรียบร้อยครับสำหรับขั้นตอนการเตรียมพร้อมข้อมูลรูปภาพ")

    def cnn_model_training(self):
        st.header("🌟 การเทรน Model CNN")
        st.markdown("---")
        st.subheader("🪛Code สำหรับการเทรน Model CNN")
        st.markdown("---")
        st.subheader("🖥️การเทรนโมเดล")
        st.code("""
                # กำหนดจำนวน epoch ในการเทรน
                epochs = 30

                # เทรนโมเดล
                history = model.fit(
                    train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    verbose=1
                )

                # บันทึกโมเดล
                model.save('../exported_models/cnn/fruit_classifier_model.h5')
                
                # Output ตัวอย่างการเทรน
                # Epoch 1/30
                # 72/72 [==============================] - 111s 2s/step - loss: 2.2900 - accuracy: 0.1156 - val_loss: 2.2474 - val_accuracy: 0.1668
                # Epoch 2/30
                # 72/72 [==============================] - 115s 2s/step - loss: 2.2109 - accuracy: 0.1678 - val_loss: 2.1626 - val_accuracy: 0.2000
                # ...
                # Epoch 30/30
                # 72/72 [==============================] - 98s 1s/step - loss: 0.2845 - accuracy: 0.9124 - val_loss: 0.3574 - val_accuracy: 0.8932
        """)

        st.subheader("🖥️การแสดงผลการเทรน")
        st.code("""
                # แสดงกราฟประสิทธิภาพของโมเดล
                plt.figure(figsize=(12, 4))
                
                # กราฟแสดง Accuracy
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'], label='Train Accuracy')
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.title('Model Accuracy')
                
                # กราฟแสดง Loss
                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.title('Model Loss')
                
                plt.tight_layout()
                plt.show()
        """)

        st.subheader("🖥️การประเมินประสิทธิภาพโมเดล")
        st.code("""
                # ประเมินโมเดลกับชุดข้อมูลทดสอบ
                test_loss, test_acc = model.evaluate(test_generator)
                print(f"ความแม่นยำของโมเดลบนชุดข้อมูลทดสอบ: {test_acc:.4f}")
                print(f"ค่า loss ของโมเดลบนชุดข้อมูลทดสอบ: {test_loss:.4f}")
                
                # ทำนายผลบนชุดข้อมูลทดสอบ
                predictions = model.predict(test_generator)
                predicted_classes = np.argmax(predictions, axis=1)
                
                # คำนวณ confusion matrix
                from sklearn.metrics import confusion_matrix, classification_report
                
                # ใช้ generator.classes เพื่อเข้าถึง label ที่แท้จริง
                true_classes = test_generator.classes[:len(predicted_classes)]
                
                # สร้าง confusion matrix
                cm = confusion_matrix(true_classes, predicted_classes)
                
                # แสดง confusion matrix ด้วย seaborn
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=list(class_names.values()),
                           yticklabels=list(class_names.values()))
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                plt.show()
                
                # แสดงรายงานการจำแนกประเภท
                print("Classification Report:")
                print(classification_report(true_classes, predicted_classes, 
                                          target_names=list(class_names.values())))
        """)

        st.subheader("🎉🎉🎉 เรียบร้อยครับสำหรับขั้นตอนการเทรน Model CNN")

    def load_model(self):
        st.header("🌟 การโหลด Model CNN ไปใช้งาน")
        st.markdown("---")
        st.subheader("🖥️Code สำหรับการโหลด Model CNN")
        st.code("""
                import tensorflow as tf
                import numpy as np
                from tensorflow.keras.preprocessing.image import load_img, img_to_array
                import os
                
                def load_cnn_model(model_path):
                    try:
                        model = tf.keras.models.load_model(model_path)
                        return model
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
                        return None
                
                # โหลดโมเดล
                model = load_cnn_model('../exported_models/cnn/fruit_classifier_model.h5')
                
                # ฟังก์ชันสำหรับทำนายรูปภาพ
                def predict_image(model, img_path, target_size=(224, 224)):
                    # โหลดรูปภาพและปรับขนาด
                    img = load_img(img_path, target_size=target_size)
                    
                    # แปลงรูปภาพเป็น array และปรับค่าพิกเซล
                    img_array = img_to_array(img) / 255.0
                    
                    # ขยายมิติเพื่อให้เข้ากับรูปแบบ input ของโมเดล
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # ทำนาย
                    prediction = model.predict(img_array)
                    
                    # แปลงผลลัพธ์เป็นคลาส
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction) * 100
                    
                    return predicted_class, confidence, prediction[0]
                """)

        st.markdown("---")
        st.subheader("ตัวอย่างการใช้งานโมเดล")
        st.code("""
                class Neuron_implement_viewset:
                    def __init__(self):
                        # โหลดโมเดล
                        self.cnn_model = load_cnn_model("exported_models/cnn/fruit_classifier_model.h5")
                        
                        # กำหนดชื่อคลาส
                        self.class_names = {
                            0: "Apple", 1: "Banana", 2: "Grape", 3: "Kiwi", 4: "Mango",
                            5: "Orange", 6: "Peach", 7: "Pineapple", 8: "Strawberry", 9: "Watermelon"
                        }
                    
                    def predict(self, image_path):
                        # ทำนายรูปภาพ
                        predicted_class, confidence, all_probs = predict_image(
                            self.cnn_model, image_path, target_size=(224, 224)
                        )
                        
                        # แปลงคลาสเป็นชื่อผลไม้
                        fruit_name = self.class_names[predicted_class]
                        
                        # สร้างผลลัพธ์
                        result = {
                            "fruit_name": fruit_name,
                            "confidence": f"{confidence:.2f}%",
                            "all_probabilities": {self.class_names[i]: f"{prob*100:.2f}%" for i, prob in enumerate(all_probs)}
                        }
                        
                        return result
                """)

        st.subheader("🎉🎉🎉 เรียบร้อยครับสำหรับขั้นตอนการโหลด Model CNN ไปใช้งาน")
        st.markdown("---")
