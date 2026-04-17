import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc
)
from src import model
from src import hand_tracking
from model import create_cnn_lstm_model
from hand_tracking import extract_landmarks

import matplotlib.pyplot as plt
import seaborn as sns


#  CONFIG 

dataset_path = r"F:\Last_Semester\Final_Project\Sign-Language-Translator\Datasets\RGB ArSL dataset"

timesteps = 23
max_images_per_class = None
print_every = 200

output_dir = "models/evaluation"
os.makedirs(output_dir, exist_ok=True)

data = []
labels = []


# LOAD CLASSES 

arabic_classes = sorted([
    cls for cls in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, cls))
])

arabic_classes = [c for c in arabic_classes if c.lower() != "nothing"]

print("Detected Arabic Classes:", arabic_classes)
print("Number of classes:", len(arabic_classes))

os.makedirs("models", exist_ok=True)
np.save("models/arabic_classes.npy", arabic_classes)


# LOAD DATA 

for idx, cls in enumerate(arabic_classes):

    cls_path = os.path.join(dataset_path, cls)

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

        lm = extract_landmarks(img)

        if lm is not None:
            data.append(lm)
            labels.append(idx)
            processed_count += 1

        if (i + 1) % print_every == 0:
            print(f"  Processed {i+1}/{len(files)}")

    print(f"Finished '{cls}' — Landmarks collected: {processed_count}")


# CHECK DATA 

if len(data) == 0:
    raise ValueError("No landmarks detected!")

X = np.array(data)
y = np.array(labels)

print("\nTotal samples:", X.shape[0])
print("Feature shape:", X.shape)


# SHUFFLE 

shuffle_idx = np.random.permutation(len(X))

X = X[shuffle_idx]
y = y[shuffle_idx]


# CREATE SEQUENCES

X_seq = np.repeat(X[:, np.newaxis, :], timesteps, axis=1)

max_val = np.max(np.abs(X_seq))

if max_val != 0:
    X_seq = X_seq / max_val


# ONE HOT

y_onehot = tf.keras.utils.to_categorical(y, num_classes=len(arabic_classes))

print("Sequence shape:", X_seq.shape)
print("Labels shape:", y_onehot.shape)


# TRAIN SPLIT

X_train, X_val, y_train, y_val = train_test_split(

    X_seq,
    y_onehot,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y_onehot, axis=1)
)


# CREATE MODEL

model = create_cnn_lstm_model(num_classes=len(arabic_classes))

model.summary()


# CALLBACKS

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


# TRAIN

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=50,
    batch_size=32,

    callbacks=callbacks
)


# EVALUATE

loss, acc = model.evaluate(X_val, y_val)

print("\nValidation accuracy:", acc)
print("Validation loss:", loss)


# PREDICTIONS

y_pred_prob = model.predict(X_val)

y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_val, axis=1)


# CONFUSION MATRIX

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(16,14))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=arabic_classes,
    yticklabels=arabic_classes
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()

conf_matrix_path = os.path.join(output_dir, "Arabic - confusion_matrix.png")

plt.savefig(conf_matrix_path)

print("Confusion matrix saved:", conf_matrix_path)

plt.show()


# CLASSIFICATION REPORT

report = classification_report(

    y_true,
    y_pred,
    target_names=arabic_classes,
    digits=4
)

print("\nClassification Report\n")
print(report)

with open(os.path.join(output_dir,"Arabic - classification_report.txt"),"w") as f:
    f.write(report)



# PRECISION / RECALL / F1

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"\nWeighted Precision: {precision:.4f}")
print(f"Weighted Recall:    {recall:.4f}")
print(f"Weighted F1-score:  {f1:.4f}")


# ROC + AUC

n_classes = len(arabic_classes)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_pred_prob[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])


# MACRO AUC

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

macro_auc = auc(all_fpr, mean_tpr)

print("\nMacro AUC:", macro_auc)


# ROC CURVE

plt.figure(figsize=(10,8))

plt.plot(
    all_fpr,
    mean_tpr,
    label=f"Macro ROC (AUC = {macro_auc:.3f})",
    linewidth=3
)

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Multi-Class ROC Curve")

plt.legend(loc="lower right")

plt.tight_layout()

roc_path = os.path.join(output_dir,"Arabic - roc_curve.png")

plt.savefig(roc_path)

print("ROC curve saved:", roc_path)

plt.show()


# SAVE MODEL

model.save("models/arsl_landmark_cnn_lstm_model.keras")

print("\nArabic model saved successfully!")