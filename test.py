import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import threading

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "asl_landmark_cnn_lstm_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

# Must match training order
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['del', 'nothing', 'space']

confidence_threshold = 0.8
required_stable_frames = 15
timesteps = 23

# -------------------- TEXT TO SPEECH (FIXED) --------------------
def speak_async(text):

    def run():
        engine = pyttsx3.init()   # Reinitialize every time
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    threading.Thread(target=run, daemon=True).start()

# -------------------- MEDIAPIPE --------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------- WEBCAM --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# -------------------- STATE VARIABLES --------------------
sentence = ""
stable_prediction = ""
stable_count = 0
last_added_letter = ""
current_prediction = ""
current_confidence = 0.0

print("Press 'q' to quit | 's' to speak | 'c' to clear")


window_name = "ASL Real-Time Recognition"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1280, 720))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_class = None
    pred_conf = 0.0

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = np.array([
                coord
                for landmark in hand_landmarks.landmark
                for coord in (landmark.x, landmark.y, landmark.z)
            ])

            # Mirror correction
            if results.multi_handedness:
                handedness_label = results.multi_handedness[idx].classification[0].label
                if handedness_label == "Right":
                    lm[0::3] = 1.0 - lm[0::3]

            # Create LSTM input
            X_seq = np.repeat(lm[np.newaxis, :], timesteps, axis=0)
            X_seq = X_seq[np.newaxis, :, :]

            max_val = np.max(np.abs(X_seq))
            if max_val != 0:
                X_seq = X_seq / max_val

            pred_probs = model.predict(X_seq, verbose=0)
            pred_index = np.argmax(pred_probs)
            pred_conf = pred_probs[0][pred_index]

            current_prediction = classes[pred_index]
            current_confidence = pred_conf

            if pred_conf >= confidence_threshold:
                predicted_class = classes[pred_index]
    else:
        current_prediction = ""
        current_confidence = 0.0

    # -------------------- STABILITY + RELEASE LOGIC --------------------
    if predicted_class and pred_conf >= confidence_threshold:

        if predicted_class != stable_prediction:
            stable_prediction = predicted_class
            stable_count = 1
        else:
            stable_count += 1

        if stable_count >= required_stable_frames:

            # Add letter ONLY if different from last added
            if predicted_class.lower() == "space":
                sentence += " "
                last_added_letter = ""

            elif predicted_class.lower() == "del":
                sentence = sentence[:-1]
                last_added_letter = ""

            elif predicted_class != last_added_letter:
                sentence += predicted_class
                last_added_letter = predicted_class

            stable_count = 0

    else:
        # IMPORTANT: Reset when hand disappears OR confidence drops
        stable_prediction = ""
        stable_count = 0
        last_added_letter = ""   # <-- THIS IS THE FIX

    # -------------------- UI --------------------
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (40, 40, 40), -1)

    cv2.putText(
        frame,
        f"Sentence: {sentence}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    if current_prediction:
        cv2.putText(
            frame,
            f"Prediction: {current_prediction} ({current_confidence:.2f})",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        "English Sign Language Recognition (CNN-LSTM)",
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        if sentence.strip():
            speak_async(sentence.strip())
    elif key == ord('c'):  
        sentence = ""
        last_added_letter = ""
        print("Sentence cleared")

cap.release()
cv2.destroyAllWindows()