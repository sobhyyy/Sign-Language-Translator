# def train_model():
#     global model
#     data_dir = r'F:\Last_Semester\Final_Project\Model\archive\asl_alphabet_train'

#     print("Using data_dir:", data_dir)
#     if not os.path.exists(data_dir):
#         raise FileNotFoundError(f"Folder not found: {data_dir}")

#     contents = os.listdir(data_dir)
#     print("Contents of data_dir:", contents)

#     train_folder = os.path.join(data_dir, 'asl_alphabet_train')
#     actual_class_path = train_folder if os.path.exists(train_folder) else data_dir

#     print("Final class path:", actual_class_path)
#     class_dirs = sorted(glob.glob(os.path.join(actual_class_path, '*')))
#     print(f"Found {len(class_dirs)} class folders (expect ~29)")

#     if len(class_dirs) < 20:
#         raise ValueError("Too few classes found — wrong path!")

#     X, y_onehot = load_asl_data(actual_class_path, max_per_class=500)

#     X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    
#     print(f"Training with {num_classes} classes (model output shape should be (None, {num_classes}))")

#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=30,
#         batch_size=64,
#         verbose=1,
#         callbacks=[
#             tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
#             tf.keras.callbacks.ModelCheckpoint(
#                 "best_trained_model.keras",
#                 save_best_only=True,
#                 monitor='val_accuracy',
#                 mode='max'
#             )
#         ]
#     )

#     model.save("trained_gesture_model.keras")
#     print("Training complete!")






# from model import create_cnn_lstm_model
# print("Import successful!")
# model = create_cnn_lstm_model()
# print("Model created!")




# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# layers = tf.keras.layers
# models = tf.keras.models
# utils = tf.keras.utils

# # ==========================================================
# # 1️⃣ Model Definition
# # ==========================================================

# def create_cnn_lstm_model(num_classes=29, timesteps=23, features=63):
#     """
#     CNN-LSTM model for landmark sequence input.
#     Input shape: (batch, 23, 63)
#     """

#     inputs = tf.keras.Input(shape=(timesteps, features))

#     # CNN layers
#     x = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same')(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

#     x = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

#     # LSTM layer
#     x = tf.keras.layers.Bidirectional(
#         tf.keras.layers.LSTM(256, return_sequences=False)
#     )(x)

#     # Dense layers
#     x = tf.keras.layers.Dense(256, activation='relu')(x)
#     x = tf.keras.layers.Dropout(0.5)(x)

#     outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

#     model = tf.keras.Model(inputs, outputs)

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     return model


# # ==========================================================
# # 2️⃣ Data Preparation
# # ==========================================================

# # Normalize landmarks
# X = X / np.max(np.abs(X))

# # Convert labels to one-hot safely
# y = tf.keras.utils.to_categorical(y, num_classes=29)

# # Train / Validation split
# X_train, X_val, y_train, y_val = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=np.argmax(y, axis=1)
# )

# # ==========================================================
# # 3️⃣ Create Model
# # ==========================================================

# model = create_cnn_lstm_model()

# model.summary()

# # ==========================================================
# # 4️⃣ Training
# # ==========================================================

# callbacks = [
#     tf.keras.callbacks.EarlyStopping(
#         monitor='val_loss',
#         patience=8,
#         restore_best_weights=True
#     ),
#     tf.keras.callbacks.ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=4,
#         verbose=1
#     )
# ]

# history = model.fit(
#     X_train,
#     y_train,
#     validation_data=(X_val, y_val),
#     epochs=50,
#     batch_size=32,
#     callbacks=callbacks
# )

# # ==========================================================
# # 5️⃣ Evaluation
# # ==========================================================

# loss, accuracy = model.evaluate(X_val, y_val)
# print("Validation Accuracy:", accuracy)

# # Detailed classification report
# y_pred = model.predict(X_val)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true = np.argmax(y_val, axis=1)

# print(classification_report(y_true, y_pred_classes))

# # ==========================================================
# # 6️⃣ Save Model
# # ==========================================================

# model.save("sign_language_model.keras")

# print("Model saved successfully!")
# model 



# import cv2
# import numpy as np
# import mediapipe as mp
# import time
# import os

# # Legacy MediaPipe imports
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# # Hard-coded hand connections
# HAND_CONNECTIONS = [
#     (0, 1), (1, 2), (2, 3), (3, 4),
#     (0, 5), (5, 6), (6, 7), (7, 8),
#     (5, 9), (9, 10), (10, 11), (11, 12),
#     (9, 13), (13, 14), (14, 15), (15, 16),
#     (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
# ]

# def setup_hand_tracker():
#     """
#     Sets up the legacy MediaPipe Hands tracker.
#     Returns: hands object
#     """
#     hands = mp_hands.Hands(
#         static_image_mode=False,
#         max_num_hands=1,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )
#     return hands

# def process_frame(frame, hands):
#     """
#     Processes one frame with legacy MediaPipe Hands.
#     Returns: (annotated_frame, landmarks_this_frame)
#     """
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(rgb_frame)

#     landmarks_this_frame = None
#     annotated_frame = frame.copy()

#     if results.multi_hand_landmarks:
#         print("Hand detected! Drawing landmarks...")
#         print("Detected hands:", len(results.multi_hand_landmarks))

#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 annotated_frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )

#             # Extract 21 landmarks × 3 = 63 features
#             landmarks = []
#             for lm in hand_landmarks.landmark:
#                 landmarks.extend([lm.x, lm.y, lm.z])

#             landmarks_this_frame = np.array(landmarks)

#     return annotated_frame, landmarks_this_frame
# handtracking


# import cv2
# import numpy as np
# import sys
# from model import create_cnn_lstm_model
# from hand_tracking import setup_hand_tracker, process_frame

# if __name__ == '__main__':
#     # Create model
#     model = create_cnn_lstm_model(num_classes=29, timesteps=23, features=63)
#     print("Model created with landmark input shape (None, 23, 63)")
#     model.summary()

#     # Setup legacy MediaPipe tracker
#     hands = setup_hand_tracker()

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         sys.exit()

#     frame_buffer = []

#     print("Webcam started with legacy MediaPipe hand tracking. Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture frame.")
#             break

#         # Process frame (detection + drawing)
#         annotated_frame, landmarks_this_frame = process_frame(frame, hands)

#         cv2.imshow('frame with hand tracking', annotated_frame)

#         if landmarks_this_frame is not None:
#             frame_buffer.append(landmarks_this_frame)

#         if len(frame_buffer) >= 23:
#             input_array = np.array(frame_buffer[-23:])
#             input_data = np.expand_dims(input_array, axis=0)

#             print("Landmark input shape:", input_data.shape)

#             prediction = model.predict(input_data, verbose=0)[0]
#             pred_idx = np.argmax(prediction)
#             confidence = prediction[pred_idx]

#             text = "Predicting..."
#             color = (50, 50, 50)

#             if confidence > 0.5:
#                 # Placeholder - later replace with real label decoding
#                 text = f"Class {pred_idx} ({confidence:.2f})"
#                 print(text)
#                 color = (0, 255, 0)

#             cv2.putText(
#                 annotated_frame,
#                 text,
#                 (50, 100),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1.2,
#                 color,
#                 3,
#                 cv2.LINE_AA
#             )

#             cv2.imshow('frame+prediction', annotated_frame)

#             frame_buffer = []

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print("Webcam closed.") 
# main

# import os

# print(sorted(os.listdir('F:\Last_Semester\Final_Project\Model\RGB ArSL dataset')))


# from gtts import gTTS
# import os

# text = "مرحبا كيف حالك"
# tts = gTTS(text=text, lang='ar')
# tts.save("arabic_voice.mp3")

# # Play it
# os.system("start arabic_voice.mp3")  # Windows


import pyttsx3

engine = pyttsx3.init()
engine.say("مرحبا")
engine.runAndWait()

voices = engine.getProperty('voices')
for voice in voices:
    print(voice.id)