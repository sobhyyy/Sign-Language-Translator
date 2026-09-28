"""
Train the existing CNN + BiLSTM architecture on REAL temporal sequences.

Input:
    .npz produced by build_sequence_dataset.py

Expected X shape:
    (samples, 23, 63)

Unlike the old training scripts, this file never repeats one landmark frame
23 times. Every timestep comes from a different video frame.

Example:
    python src/train_temporal_model.py ^
        --data data\temporal\asl_subset.npz ^
        --model-out models\v2_asl_temporal.keras ^
        --epochs 30 ^
        --batch-size 16
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold


def load_data(path):
    data = np.load(path, allow_pickle=False)

    required = {"X", "y", "groups", "class_names"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Dataset is missing fields: {sorted(missing)}")

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    groups = data["groups"].astype(str)
    class_names = data["class_names"].astype(str).tolist()

    if X.ndim != 3 or X.shape[-1] != 63:
        raise ValueError(f"Expected X=(N,T,63), got {X.shape}")

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y and groups must contain the same number of samples.")

    return X, y, groups, class_names


def build_model(num_classes, timesteps, features=63):
    from src.model import create_cnn_lstm_model

    return create_cnn_lstm_model(
        num_classes=num_classes,
        timesteps=timesteps,
        features=features,
    )


def make_splits(X, y, groups, random_state=42):
    """
    Create an 80/20-ish train/validation split while keeping groups together.

    StratifiedGroupKFold is used because windows from the same video must not
    appear in both training and validation. This avoids temporal-window leakage.
    """
    unique_groups = np.unique(groups)

    if len(unique_groups) < 5:
        raise ValueError(
            "Need at least 5 unique videos/groups for the temporal split. "
            "Collect more videos before training."
        )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    train_idx, val_idx = next(splitter.split(X, y, groups))

    return train_idx, val_idx


def print_distribution(name, y, class_names):
    counts = np.bincount(y, minlength=len(class_names))
    print(f"\n{name} distribution:")
    for idx, count in enumerate(counts):
        print(f"  {class_names[idx]}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-out", default="models/v2_temporal.keras")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X, y, groups, class_names = load_data(args.data)

    print("TensorFlow:", tf.__version__)
    print("Devices:", tf.config.list_physical_devices())
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print(f"Samples: {len(X)}")
    print(f"Sequence shape: {X.shape}")
    print(f"Classes: {len(class_names)}")
    print(f"Unique video groups: {len(np.unique(groups))}")

    if X.shape[1] < 2:
        raise ValueError("Temporal dimension must contain multiple frames.")

    train_idx, val_idx = make_splits(X, y, groups, args.random_state)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print_distribution("Training", y_train, class_names)
    print_distribution("Validation", y_val, class_names)

    y_train_cat = tf.keras.utils.to_categorical(
        y_train,
        num_classes=len(class_names),
    )
    y_val_cat = tf.keras.utils.to_categorical(
        y_val,
        num_classes=len(class_names),
    )

    model = build_model(
        num_classes=len(class_names),
        timesteps=X.shape[1],
        features=X.shape[2],
    )

    output_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print("\nStarting temporal training...")
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
    )

    best_model = tf.keras.models.load_model(output_path)

    loss, accuracy = best_model.evaluate(
        X_val,
        y_val_cat,
        verbose=0,
    )

    probabilities = best_model.predict(X_val, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print("\n==============================")
    print("TEMPORAL MODEL RESULT")
    print("==============================")
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")

    report = classification_report(
        y_val,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    print("Confusion matrix:")
    print(confusion_matrix(y_val, predictions))

    evaluation_dir = output_path.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    with open(evaluation_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    with open(evaluation_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tensorflow_version": tf.__version__,
                "samples": int(len(X)),
                "train_samples": int(len(train_idx)),
                "validation_samples": int(len(val_idx)),
                "sequence_shape": list(X.shape[1:]),
                "num_classes": len(class_names),
                "class_names": class_names,
                "unique_groups": int(len(np.unique(groups))),
                "batch_size": args.batch_size,
                "epochs_requested": args.epochs,
                "best_validation_accuracy": float(accuracy),
                "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
            },
            f,
            indent=2,
        )

    print(f"\nBest model saved to: {output_path}")
    print(f"Evaluation saved to: {evaluation_dir}")


if __name__ == "__main__":
    main()
