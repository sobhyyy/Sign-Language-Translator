import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import create_cnn_lstm_model
from hand_tracking import extract_landmarks

# ------------------- CONFIG -------------------

dataset_path = r"F:\Last_Semester\Final_Project\Model\RGB ArSL dataset"

timesteps = 23
max_images_per_class = None
print_every = 200

data = []
labels = []



# ------------------- AUTO LOAD CLASSES -------------------

arabic_classes = sorted([
    cls for cls in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, cls))
])
print("Detected Arabic Classes:", arabic_classes)

print("Number of classes:", len(arabic_classes))

# Save class order (VERY IMPORTANT for test.py)
os.makedirs("models", exist_ok=True)
np.save("models/arabic_classes.npy", arabic_classes)

# ------------------- LOAD DATA -------------------

for idx, cls in enumerate(arabic_classes):

    cls_path = os.path.join(dataset_path, cls)

    if not os.path.isdir(cls_path):
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
            print(f"  Processed {i+1}/{len(files)}")

    print(f"Finished '{cls}' — Landmarks collected: {processed_count}")

# ------------------- CHECK DATA -------------------

if len(data) == 0:
    raise ValueError("No landmarks detected! Check dataset path or MediaPipe.")

X = np.array(data)
y = np.array(labels)

print("\nTotal samples:", X.shape[0])
print("Feature shape:", X.shape)

# ------------------- SHUFFLE -------------------

shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# ------------------- CREATE SEQUENCES -------------------

X_seq = np.repeat(X[:, np.newaxis, :], timesteps, axis=1)

# Normalize safely
max_val = np.max(np.abs(X_seq))
if max_val != 0:
    X_seq = X_seq / max_val

# One-hot encode
y = tf.keras.utils.to_categorical(y, num_classes=len(arabic_classes))

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

model = create_cnn_lstm_model(num_classes=len(arabic_classes))
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

model.save("models/arsl_landmark_cnn_lstm_model.keras")

print("\nArabic model saved successfully!")