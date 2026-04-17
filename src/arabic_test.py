import os
import cv2
import time
import uuid
import queue
import threading
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pygame

from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display
from gtts import gTTS

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# -------------------- PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "arsl_landmark_cnn_lstm_model.keras")
FONT_PATH = os.path.join(PROJECT_ROOT, "assets", "arial.ttf")
AUDIO_DIR = os.path.join(PROJECT_ROOT, "generated_audio")

# -------------------- MODEL --------------------
model = tf.keras.models.load_model(MODEL_PATH)

# Must match training order exactly
classes = [
    "Ain", "Al", "Alef", "Beh", "Dad", "Dal", "Feh", "Ghain", "Hah", "Heh",
    "Jeem", "Kaf", "Khah", "Laa", "Lam", "Meem", "Noon", "Qaf", "Reh", "Sad",
    "Seen", "Sheen", "Tah", "Teh", "Teh_Marbuta", "Thal", "Theh", "Waw", "Yeh",
    "Zah", "Zain", "del", "space"
]

arabic_map = {
    "Alef": "ا", "Beh": "ب", "Teh": "ت", "Theh": "ث", "Jeem": "ج",
    "Hah": "ح", "Khah": "خ", "Dal": "د", "Thal": "ذ", "Reh": "ر",
    "Zain": "ز", "Seen": "س", "Sheen": "ش", "Sad": "ص", "Dad": "ض",
    "Tah": "ط", "Zah": "ظ", "Ain": "ع", "Ghain": "غ", "Feh": "ف",
    "Qaf": "ق", "Kaf": "ك", "Lam": "ل", "Meem": "م", "Noon": "ن",
    "Heh": "ه", "Waw": "و", "Yeh": "ي", "Laa": "لا", "Al": "ال",
    "Teh_Marbuta": "ة"
}

confidence_threshold = 0.8
required_stable_frames = 15
timesteps = 23

# -------------------- ARABIC RENDER --------------------
font_large = ImageFont.truetype(FONT_PATH, 40)
font_small = ImageFont.truetype(FONT_PATH, 28)

def draw_arabic(frame, text, right_margin, y, font):
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = bbox[2] - bbox[0]

    x_position = frame.shape[1] - text_width - right_margin
    draw.text((x_position, y), bidi_text, font=font, fill=(255, 255, 255))

    return np.array(img_pil)

# -------------------- ARABIC SPEECH --------------------
print("Initializing Arabic MP3 speech...")

os.makedirs(AUDIO_DIR, exist_ok=True)

pygame.mixer.init()

speech_queue = queue.Queue()
stop_speech_worker = False

def cleanup_old_mp3_files(folder, keep_last=5):
    try:
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".mp3")
        ]
        files.sort(key=os.path.getmtime, reverse=True)

        for old_file in files[keep_last:]:
            try:
                os.remove(old_file)
            except Exception:
                pass
    except Exception:
        pass

def speech_worker():
    while not stop_speech_worker:
        try:
            text = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if not text.strip():
            continue

        try:
            print("Generating Arabic MP3:", text)

            filename = f"speech_{uuid.uuid4().hex}.mp3"
            mp3_path = os.path.join(AUDIO_DIR, filename)

            tts = gTTS(text=text, lang="ar", slow=False)
            tts.save(mp3_path)

            print("Playing:", mp3_path)

            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                time.sleep(0.2)

            pygame.mixer.music.load(mp3_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            cleanup_old_mp3_files(AUDIO_DIR, keep_last=5)

        except Exception as e:
            print("Speech generation/playback failed:", e)

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak_async(text):
    if not text.strip():
        return

    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            break

    speech_queue.put(text)

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

window_name = "AI Sign Translator - Arabic"
cv2.namedWindow(window_name)

# -------------------- STATE --------------------
sentence = ""
stable_prediction = ""
stable_count = 0
last_added_letter = ""
current_prediction = ""
current_confidence = 0.0

print("Arabic mode | Press 's' to speak | 'c' to clear | 'q' to quit")

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

            x_seq = np.repeat(lm[np.newaxis, :], timesteps, axis=0)
            x_seq = x_seq[np.newaxis, :, :]

            max_val = np.max(np.abs(x_seq))
            if max_val != 0:
                x_seq = x_seq / max_val

            pred_probs = model.predict(x_seq, verbose=0)
            pred_index = int(np.argmax(pred_probs))
            pred_conf = float(pred_probs[0][pred_index])

            if pred_index >= len(classes):
                continue

            label_name = classes[pred_index]
            current_confidence = pred_conf

            if label_name in arabic_map:
                current_prediction = arabic_map[label_name]
            else:
                current_prediction = label_name

            if pred_conf >= confidence_threshold:
                predicted_class = label_name
    else:
        current_prediction = ""
        current_confidence = 0.0

    # -------------------- STABILITY LOGIC --------------------
    if predicted_class:
        if predicted_class != stable_prediction:
            stable_prediction = predicted_class
            stable_count = 1
        else:
            stable_count += 1

        if stable_count >= required_stable_frames:
            if predicted_class == "space":
                sentence += " "
                last_added_letter = ""

            elif predicted_class == "del":
                sentence = sentence[:-1]
                last_added_letter = ""

            else:
                letter_to_add = arabic_map.get(predicted_class, predicted_class)

                if letter_to_add != last_added_letter:
                    sentence += letter_to_add
                    last_added_letter = letter_to_add

            stable_count = 0
    else:
        stable_prediction = ""
        stable_count = 0

    # -------------------- UI --------------------
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (40, 40, 40), -1)

    if sentence:
        frame = draw_arabic(frame, sentence, 40, 30, font_large)

    if current_prediction:
        frame = draw_arabic(frame, current_prediction, 50, 75, font_large)

        cv2.putText(
            frame,
            f"{current_confidence:.2f}",
            (120, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        "Arabic Sign Language | CNN-LSTM",
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
        last_added_letter = ""
        stable_prediction = ""
        stable_count = 0
        current_prediction = ""
        current_confidence = 0.0

# -------------------- CLEANUP --------------------
stop_speech_worker = True

try:
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    pygame.mixer.quit()
except Exception:
    pass

cap.release()
cv2.destroyAllWindows()