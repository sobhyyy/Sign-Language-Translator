import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def extract_landmarks(image):
    """
    Extracts 21 hand landmarks from an image.
    Returns 63-length numpy array (x, y, z for each landmark) or None if no hand detected.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords)
    return None