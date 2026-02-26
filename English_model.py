# main.py

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import create_cnn_lstm_model
from hand_tracking import extract_landmarks

# ------------------- CONFIG -------------------

dataset_paths = [
    r"F:\Last_Semester\Final_Project\Model\archive\asl_alphabet_train\asl_alphabet_train" 
]

english_classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N',
           'O','P','Q','R','S','T','U','V','W','X','Y','Z',
           'del','nothing','space']
timesteps = 23
max_images_per_class = None   # Set a number like 2000 if you want to limit per class
print_every = 100

data = []
labels = []

# ------------------- LOAD DATA -------------------

print("Classes:", english_classes)

for dataset_path in dataset_paths:
    print(f"\nLoading dataset: {dataset_path}")

    for idx, cls in enumerate(english_classes):
        cls_path = os.path.join(dataset_path, cls)

        if not os.path.exists(cls_path):
            print(f"Class folder not found: {cls_path}")
            continue

        files = os.listdir(cls_path)

        if max_images_per_class:
            files = files[:max_images_per_class]

        print(f"\nProcessing class '{cls}' ({len(files)} images)")
        processed_count = 0

        for i, file in enumerate(files):
            img_path = os.path.join(cls_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (480, 480))
            lm = extract_landmarks(img)

            if lm is not None:
                data.append(lm)
                labels.append(idx)
                processed_count += 1

            if (i + 1) % print_every == 0:
                print(f"  Processed {i+1}/{len(files)} images")

        print(f"Finished class '{cls}' — Landmarks collected: {processed_count}")

# ------------------- CHECK DATA -------------------

if len(data) == 0:
    raise ValueError("No landmarks detected! Check dataset path or MediaPipe detection.")

X = np.array(data)
y = np.array(labels)

print("\nTotal samples:", X.shape[0])
print("Feature shape:", X.shape)

# ------------------- SHUFFLE DATA -------------------

shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# ------------------- CREATE SEQUENCES -------------------

# Repeat static landmarks to fake sequence for LSTM
X_seq = np.repeat(X[:, np.newaxis, :], timesteps, axis=1)

# Normalize
X_seq = X_seq / np.max(np.abs(X_seq))

# One-hot encode labels
y = tf.keras.utils.to_categorical(y, num_classes=len(english_classes))

print("Sequence shape:", X_seq.shape)
print("Labels shape:", y.shape)

# ------------------- TRAIN / VALIDATION SPLIT -------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_seq,
    y,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)

# ------------------- CREATE MODEL -------------------

model = create_cnn_lstm_model(num_classes=len(english_classes))
model.summary()

# ------------------- CALLBACKS -------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4
    )
]

# ------------------- TRAIN -------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# ------------------- EVALUATE -------------------

loss, acc = model.evaluate(X_val, y_val)
print("\nValidation accuracy:", acc)

# ------------------- SAVE MODEL -------------------

os.makedirs("models", exist_ok=True)
model.save("asl_landmark_cnn_lstm_model.keras")

print("\nModel saved successfully!")