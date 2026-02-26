import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import threading
import pyttsx3
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display
from gtts import gTTS

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "arsl_landmark_cnn_lstm_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

# EXACT TRAINING ORDER (DO NOT CHANGE)
classes = [
'Ain','Al','Alef','Beh','Dad','Dal','Feh','Ghain','Hah','Heh',
'Jeem','Kaf','Khah','Laa','Lam','Meem','Noon','Qaf','Reh','Sad',
'Seen','Sheen','Tah','Teh','Teh_Marbuta','Thal','Theh','Waw','Yeh',
'Zah','Zain','del','nothing','space'
]

# English label -> Arabic display mapping
arabic_map = {
    'Alef':'ا','Beh':'ب','Teh':'ت','Theh':'ث','Jeem':'ج',
    'Hah':'ح','Khah':'خ','Dal':'د','Thal':'ذ','Reh':'ر',
    'Zain':'ز','Seen':'س','Sheen':'ش','Sad':'ص','Dad':'ض',
    'Tah':'ط','Zah':'ظ','Ain':'ع','Ghain':'غ','Feh':'ف',
    'Qaf':'ق','Kaf':'ك','Lam':'ل','Meem':'م','Noon':'ن',
    'Heh':'ه','Waw':'و','Yeh':'ي','Laa':'لا','Al':'ال',
    'Teh_Marbuta':'ة'
}

confidence_threshold = 0.8
required_stable_frames = 15
timesteps = 23

# -------------------- ARABIC RENDER --------------------
font_path = "arial.ttf"
font_large = ImageFont.truetype(font_path, 40)
font_small = ImageFont.truetype(font_path, 28)

def draw_arabic(frame, text, right_margin, y, font):
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    # Get text width
    bbox = draw.textbbox((0,0), bidi_text, font=font)
    text_width = bbox[2] - bbox[0]

    # Align text from RIGHT side
    x_position = frame.shape[1] - text_width - right_margin

    draw.text((x_position, y), bidi_text, font=font, fill=(255,255,255))

    return np.array(img_pil)

# -------------------- SPEECH --------------------
print("Initializing speech engine...")

try:
    engine = pyttsx3.init()
    print("Engine initialized successfully.")
except Exception as e:
    print("Engine initialization failed:", e)
    engine = None

if engine:
    engine.setProperty('rate', 150)

    voices = engine.getProperty('voices')
    print("Available voices:")

    for v in voices:
        print(" -", v.name)

    arabic_voice_found = False

    for voice in voices:
        if "Arabic" in voice.name or "ar_" in voice.id.lower():
            engine.setProperty('voice', voice.id)
            arabic_voice_found = True
            print(f"Arabic voice selected: {voice.name}")
            break

    if not arabic_voice_found:
        print("No Arabic voice found. Using default voice.")

def speak_async(text):
    if engine is None:
        print("Speech engine not available.")
        return

    print("Speaking:", text)

    def run():
        engine.stop()
        engine.say(text)
        engine.runAndWait()

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

window_name = "Arabic Sign Language Recognition"
cv2.namedWindow(window_name)

# -------------------- STATE --------------------
sentence = ""
stable_prediction = ""
stable_count = 0
last_added_letter = ""
current_prediction = ""
current_confidence = 0.0

print("Press 's' to speak | 'c' to clear | 'q' to quit")

# -------------------- LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_class = None
    pred_conf = 0.0

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = np.array([coord for landmark in hand_landmarks.landmark
                           for coord in (landmark.x, landmark.y, landmark.z)])

            if results.multi_handedness:
                label = results.multi_handedness[idx].classification[0].label
                if label == "Right":
                    lm[0::3] = 1.0 - lm[0::3]

            X_seq = np.repeat(lm[np.newaxis,:], timesteps, axis=0)
            X_seq = X_seq[np.newaxis,:,:]

            max_val = np.max(np.abs(X_seq))
            if max_val != 0:
                X_seq = X_seq / max_val

            pred_probs = model.predict(X_seq, verbose=0)
            pred_index = np.argmax(pred_probs)
            pred_conf = pred_probs[0][pred_index]

            if pred_index >= len(classes):
                continue

            label_name = classes[pred_index]   # English folder name
            current_confidence = pred_conf

            # Map for display only
            if label_name in arabic_map:
                display_letter = arabic_map[label_name]
            else:
                display_letter = label_name

            current_prediction = display_letter

            # IMPORTANT: keep predicted_class as ENGLISH label
            if pred_conf >= confidence_threshold:
                predicted_class = label_name
    else:
        current_prediction = ""
        current_confidence = 0.0

    # Stability + repeat fix
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
                if predicted_class in arabic_map:
                    letter_to_add = arabic_map[predicted_class]
                else:
                    letter_to_add = predicted_class
                    
                if letter_to_add != last_added_letter:
                    sentence += letter_to_add
                    last_added_letter = letter_to_add

            stable_count = 0
    else:
        stable_prediction = ""
        stable_count = 0

    # -------------------- UI --------------------
    cv2.rectangle(frame,(0,0),(frame.shape[1],120),(40,40,40),-1)

    if sentence:
        frame = draw_arabic(frame, sentence, 40, 30, font_large)
        
    if current_prediction:
        frame = draw_arabic(frame, current_prediction, 50, 75, font_large)
        
        cv2.putText(frame,
                    f"{current_confidence:.2f}",
                    (120,85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

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

cap.release()
cv2.destroyAllWindows()