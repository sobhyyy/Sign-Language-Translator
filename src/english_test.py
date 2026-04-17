import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import threading

# -------------------- PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "asl_landmark_cnn_lstm_model.keras")

# -------------------- MODEL --------------------
model = tf.keras.models.load_model(MODEL_PATH)

# Must match training order exactly
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]

confidence_threshold = 0.8
required_stable_frames = 15
release_frames_required = 5
timesteps = 23

# -------------------- TEXT TO SPEECH --------------------
def speak_async(text):
    if not text.strip():
        return

    def run():
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
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

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

window_name = "AI Sign Translator - English"
cv2.namedWindow(window_name)

# -------------------- STATE --------------------
sentence = ""
current_prediction = ""
current_confidence = 0.0
sequence_buffer = []

state = "IDLE"
candidate_sign = None
candidate_count = 0
release_count = 0

def prepare_sequence_input(buffer):
    x_seq = np.array(buffer, dtype=np.float32)

    max_val = np.max(np.abs(x_seq))
    if max_val != 0:
        x_seq = x_seq / max_val

    return x_seq[np.newaxis, :, :]

def apply_prediction_to_sentence(predicted_class, current_sentence):
    if predicted_class == "space":
        return current_sentence + " "
    if predicted_class == "del":
        return current_sentence[:-1]
    return current_sentence + predicted_class

print("English mode | Press 's' to speak | 'c' to clear | 'q' to quit")

# -------------------- LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
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
            ], dtype=np.float32)

            if results.multi_handedness:
                handedness_label = results.multi_handedness[idx].classification[0].label
                if handedness_label == "Right":
                    lm[0::3] = 1.0 - lm[0::3]

            sequence_buffer.append(lm)

            if len(sequence_buffer) > timesteps:
                sequence_buffer.pop(0)

            if len(sequence_buffer) == timesteps:
                x_seq = prepare_sequence_input(sequence_buffer)

                pred_probs = model.predict(x_seq, verbose=0)
                pred_index = int(np.argmax(pred_probs))
                pred_conf = float(pred_probs[0][pred_index])

                if pred_index < len(classes):
                    current_prediction = classes[pred_index]
                    current_confidence = pred_conf

                    if pred_conf >= confidence_threshold:
                        predicted_class = classes[pred_index]
            else:
                current_prediction = ""
                current_confidence = 0.0
    else:
        current_prediction = ""
        current_confidence = 0.0
        sequence_buffer.clear()

    # -------------------- STABILITY + RELEASE LOGIC --------------------
    if predicted_class is None:
        if state == "WAIT_RELEASE":
            release_count += 1
            if release_count >= release_frames_required:
                state = "IDLE"
                candidate_sign = None
                candidate_count = 0
                release_count = 0
        else:
            state = "IDLE"
            candidate_sign = None
            candidate_count = 0
            release_count = 0

    else:
        if state == "IDLE":
            candidate_sign = predicted_class
            candidate_count = 1
            release_count = 0
            state = "DETECTING"

        elif state == "DETECTING":
            if predicted_class == candidate_sign:
                candidate_count += 1
                if candidate_count >= required_stable_frames:
                    sentence = apply_prediction_to_sentence(predicted_class, sentence)
                    state = "WAIT_RELEASE"
                    release_count = 0
            else:
                candidate_sign = predicted_class
                candidate_count = 1

        elif state == "WAIT_RELEASE":
            if predicted_class != candidate_sign:
                release_count += 1
                if release_count >= release_frames_required:
                    candidate_sign = predicted_class
                    candidate_count = 1
                    release_count = 0
                    state = "DETECTING"
            else:
                release_count = 0

    # -------------------- UI --------------------
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (40, 40, 40), -1)

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
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        "English Sign Language | CNN-LSTM",
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s"):
        if sentence.strip():
            speak_async(sentence.strip())
    elif key == ord("c"):
        sentence = ""
        current_prediction = ""
        current_confidence = 0.0
        state = "IDLE"
        candidate_sign = None
        candidate_count = 0
        release_count = 0
        sequence_buffer.clear()
        print("Sentence cleared")

cap.release()
cv2.destroyAllWindows()