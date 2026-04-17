import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src import model
from src import hand_tracking

from model import create_cnn_lstm_model
from hand_tracking import extract_landmarks

#  CONFIG 

dataset_paths = [
    r"F:\Last_Semester\Final_Project\Sign-Language-Translator\Datasets\asl_alphabet_train"
]

english_classes = [
'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
'O','P','Q','R','S','T','U','V','W','X','Y','Z',
'del','space'
]

timesteps = 23
print_every = 100
max_images_per_class = None

data = []
labels = []

#  OUTPUT DIR 

output_dir = "models/evaluation"
os.makedirs(output_dir, exist_ok=True)

#  LOAD DATA 

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

        print(f"\nProcessing '{cls}' ({len(files)} images)")

        processed_count = 0

        for i, file in enumerate(files):

            img_path = os.path.join(cls_path, file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (480,480))

            lm = extract_landmarks(img)

            if lm is not None:
                data.append(lm)
                labels.append(idx)
                processed_count += 1

            if (i+1) % print_every == 0:
                print(f"  Processed {i+1}/{len(files)}")

        print(f"Finished '{cls}' — Landmarks collected: {processed_count}")

#  CHECK DATA 

if len(data) == 0:
    raise ValueError("No landmarks detected!")

X = np.array(data)
y = np.array(labels)

print("\nTotal samples:", X.shape[0])
print("Feature shape:", X.shape)

#  SHUFFLE 

shuffle_idx = np.random.permutation(len(X))

X = X[shuffle_idx]
y = y[shuffle_idx]

#  CREATE SEQUENCES 

X_seq = np.repeat(X[:, np.newaxis, :], timesteps, axis=1)

# Normalize
X_seq = X_seq / np.max(np.abs(X_seq))

# One hot
y_cat = tf.keras.utils.to_categorical(y, num_classes=len(english_classes))

print("Sequence shape:", X_seq.shape)
print("Labels shape:", y_cat.shape)

#  TRAIN / VALIDATION SPLIT 

X_train, X_val, y_train, y_val = train_test_split(

    X_seq,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y_cat, axis=1)
)

#  CREATE MODEL 

model = create_cnn_lstm_model(num_classes=len(english_classes))

model.summary()

#  CALLBACKS 

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

#  TRAIN 

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=50,
    batch_size=32,

    callbacks=callbacks
)

#  EVALUATE 

loss, acc = model.evaluate(X_val, y_val)

print("\nValidation accuracy:", acc)

#  PREDICTIONS 

y_pred_prob = model.predict(X_val)

y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_val, axis=1)

#  CONFUSION MATRIX 

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14,12))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=english_classes,
    yticklabels=english_classes
)

plt.title("English Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()

plt.savefig(os.path.join(output_dir,"confusion_matrix.png"))

plt.show()

#  CLASSIFICATION REPORT 

report = classification_report(
    y_true,
    y_pred,
    target_names=english_classes,
    digits=4
)

print("\nClassification Report\n")
print(report)

with open(os.path.join(output_dir,"classification_report.txt"),"w") as f:
    f.write(report)

# ROC + AUC 

n_classes = len(english_classes)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y_val[:,i], y_pred_prob[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Macro average

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

macro_auc = auc(all_fpr, mean_tpr)

print("Macro AUC:", macro_auc)

#  ROC PLOT 

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

plt.savefig(os.path.join(output_dir,"roc_curve.png"))

plt.show()

#  SAVE MODEL 

os.makedirs("models", exist_ok=True)

model.save("models/test_asl_landmark_cnn_lstm_model.keras")

print("\nModel saved successfully!")