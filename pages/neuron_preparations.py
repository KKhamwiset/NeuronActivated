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
        with menu[2]:
            self.load_model()
        st.markdown("---")

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
        * โมเดล CNN จะถูกเทรนให้สามารถจำแนกผลไม้
        """)

        st.header("⚙️ ขั้นตอนการเตรียมข้อมูล (Data Preparation Process) และตั้งค่า Model")

        code_tabs = st.tabs(
            ["1️⃣ การนำเข้า Libraries", "2️⃣ การเตรียมข้อมูลรูปภาพ", "3️⃣ การตั้งค่า Model"]
        )
        with code_tabs[0]:
            st.write("Import library สำหรับการเทรน Model และจัดการข้อมูล")
            st.code("""
                    import numpy as np
                    import matplotlib.pyplot as plt
                    import os
                    import tensorflow as tf
                    from keras.layers import (
                        Dropout,
                        Dense,
                        BatchNormalization,
                        GlobalAveragePooling2D,
                    )
                    from keras.applications import MobileNetV2
                    from keras.models import Sequential
                    from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
                    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
                    """)
        with code_tabs[1]:
            st.write(
                "ทำการตั้ง path สำหรับ data ในการ train,test,validate และ show รูปตัวอย่าง"
            )
            st.code("""
                    dataset_dir = "../data/fruit/MY_data"
                    train_path = os.path.join(dataset_dir, "train")
                    test_path = os.path.join(dataset_dir, "test")
                    img = load_img(train_path + "/Apple/img_01.jpeg")
                    plt.imshow(img)
                    plt.axis("on")
                    plt.show()
                    img = img_to_array(img)
                    img.shape
                    """)
            st.write(
                "ทำการกำหนด scale รูปภาพเพื่อให้ model สามารถเรียนรู้ label ในสภาพที่ผิดปกติ (corrupted data)"
            )
            st.code("""
                    train_datagen = ImageDataGenerator(
                        rescale=1.0 / 255,
                        rotation_range=30,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        horizontal_flip=True,
                        zoom_range=0.2,
                        shear_range=0.2,
                        validation_split=0.2,
                    )
                    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
                    """)
            st.markdown("---")
            st.write(
                "ทำการสร้างชุดข้อมูลในการเทรนโดยแบ่ง 20% ของข้อมูลการเทรนสำหรับ validation จาก path ที่กำหนดไว้"
            )
            st.info("""
                    * Keras จะทำการแยก features แต่ละอันโดยอัตโนมัติตามชื่อโฟลเดอร์
                    * มีฟังก์ชัน Visualization สำหรับ Debug missmatch label ที่เกิดขึ้นระหว่างการเทรน
                    * กำหนด batch_size (ขนาดข้อมูลที่แบ่งไปเทรนในแต่ละ neuron ไว้ที่ 32)
                    """)
            st.code("""
                train_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=(224, 224),
                batch_size=32,
                color_mode="rgb",
                class_mode="categorical",
                subset="training",
            )
            test_generator = test_datagen.flow_from_directory(
                test_path,
                target_size=(224, 224),
                batch_size=32,
                color_mode="rgb",
                class_mode="categorical",
            )
            val_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=(224, 224),
                batch_size=32,
                color_mode="rgb",
                class_mode="categorical",
                subset="validation",
            )
            def visualize_dataset_samples(generator, num_samples=4, class_names=None):

                if class_names is None:
                    class_names = list(generator.class_indices.keys())
                
                x_batch, y_batch = next(generator)
                
                n_classes = len(class_names)
                fig, axes = plt.subplots(n_classes, num_samples, figsize=(num_samples * 3, n_classes * 3))
                
                for i, class_name in enumerate(class_names):
                    class_idx = generator.class_indices[class_name]
                    class_samples = []
                    
                    for j in range(len(y_batch)):
                        if np.argmax(y_batch[j]) == class_idx:
                            class_samples.append(x_batch[j])
                            if len(class_samples) >= num_samples:
                                break
                    
                    while len(class_samples) < num_samples:
                        x_batch, y_batch = next(generator)
                        for j in range(len(y_batch)):
                            if np.argmax(y_batch[j]) == class_idx:
                                class_samples.append(x_batch[j])
                                if len(class_samples) >= num_samples:
                                    break
                    
                    # Plot the samples
                    for j in range(num_samples):
                        ax = axes[i, j] if n_classes > 1 else axes[j]
                        ax.imshow(class_samples[j])
                        ax.set_title(class_name)
                        ax.axis('off')
                
                plt.tight_layout()
                plt.suptitle("Random Samples from Each Class", y=1.02)
                plt.show()
                visualize_dataset_samples(train_generator)
                """)
        with code_tabs[2]:
            st.write(
                "ทำการตั้งค่า Model โดยที่ผมจะใช้ base model เป็น MobileNetV2 เพื่อความรวดเร็วในการเทรน"
            )
            st.code("""
                    baseModel = MobileNetV2(
                        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
                        )
                    baseModel.trainable = False #ไม่ให้เปลี่ยนค่า weight ของ MobileNetV2 ระหว่าง train
                    """)
            st.markdown("---")
            st.write(
                "ตั้งค่า Model สำหรับ Train โดยใช้ Pooling จาก Keras,เพิ่ม regularizer และ กำหนด lr \nเพื่อไม่ให้ model overfigting"
            )
            st.code("""
                    model = Sequential(
                    [
                        baseModel,
                        GlobalAveragePooling2D(),
                        Dense(
                            128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
                        ),
                        BatchNormalization(),
                        Dropout(0.3),
                        Dense(
                            64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
                        ),
                        BatchNormalization(),
                        Dropout(0.3),
                        Dense(
                            32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
                        ),
                        Dense(10, activation="softmax"),
                    ]
                    )
                    model.summary()
                    model.compile(
                        loss="categorical_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        metrics=["accuracy"],
                    )
                    """)
            st.subheader("🎉🎉🎉 เรียบร้อยครับสำหรับขั้นตอนการเตรียมข้อมูลและตั้งค่า Model")
            st.markdown("---")

    def cnn_model_training(self):
        st.header("⚙️ ขั้นตอนวิธีการเทรน Model CNN (MobileNetV2 base model)")
        st.write(
            "ผมจะใช้ EarlyStopping กับ ReduceLROnPlateau เพื่อหยุดการเทรนในกรณีที่ Model แย่ลง"
        )
        st.code("""
                    early_stopping = EarlyStopping(
                        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
                    )

                    reduce_lr = ReduceLROnPlateau(
                        monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1
                    )
                    """)
        st.write("ทำการเทรน Model")
        st.code("""
                    history = model.fit(
                    train_generator,
                    epochs=30,
                    validation_data=val_generator,
                    callbacks=[early_stopping, reduce_lr],
                    )""")

        st.write("Save ตัว model และ weight และ evalute model")
        st.code("""
                    model.save("../exported_models/fruit_model.keras")
                    model.save_weights("../exported_models/fruit_model_weights.h5")
                    model.summary()
                    test_loss, test_acc = model.evaluate(test_generator)
                    print(f"Test accuracy: {test_acc:.4f}")
                    """)
        st.subheader("🎉🎉🎉 เรียบร้อยครับสำหรับขั้นตอนการเทรน Model")

    def load_model(self):
        st.header("⚙️ ขั้นตอนวิธีการโหลด Model CNN (MobileNetV2 base model)")
        st.info("""
                    * เนื่องจากผมไม่สามารถโหลดทั้ง Model ได้จึงได้ทำการสร้าง architecture ของ model ไว้แล้วโหลด weight แทน
                    """)
        code_tabs_cnn = st.tabs(
            ["1️⃣ การสร้าง Architectire บนหน้าเว็ป Streamlit", "2️⃣ การโหลด Model"]
        )
        with code_tabs_cnn[0]:
            st.write("ทำการสร้าง Architecture ของ Model ขึ้นมา")
            st.code("""
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
                        return model""")
            st.warning("""
                            * ค่าที่ตั้งด้านบนต้องเหมือนกับที่ตั้งไว้ตอนเทรน
                            """)
        with code_tabs_cnn[1]:
            st.write(
                "ในการโหลด model ทั้ง model ผมเจอปัญหา Conv1 error จึงทำการใส่ fallback เพื่อ load weight แทน"
            )
            st.code("""
                        def load_cached_model(model_path, weights_path, img_height, img_width):
                            try:
                                if os.path.exists(model_path):
                                    try:
                                        model = tf.keras.models.load_model(model_path) #ลองโหลดทั้งโมเดล
                                        print(f"Successfully loaded entire model from {model_path}")
                                        return model, True, f"Successfully loaded entire model from {model_path}"
                                        
                                    except Exception as e:
                                        if "Conv1" in str(e):
                                            print(f"Error loading full model: {e}. Attempting to load weights only.")
                                            if os.path.exists(weights_path):
                                                model = create_model() #ถ้าไม่สำเร็จ สร้าง architecture แล้วค่อย load weight ของโมเดลที่เทรนมา
                                                model.load_weights(weights_path)
                                                print(f"Successfully loaded weights from {weights_path}")
                                    
                                                return model, True, f"Successfully loaded weights from {weights_path}"
                                            else:
                                                return None, False, f"Weights file not found at {weights_path}"
                                        else:
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
                         """)
            st.subheader("🎉🎉🎉 เรียบร้อยครับสำหรับขั้นตอนการ save และ load model")
