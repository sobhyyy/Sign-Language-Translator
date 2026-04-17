import os

print("Starting script...")

import tensorflow as tf

print("TensorFlow imported successfully.")

# -------------------- PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARABIC_KERAS_PATH = os.path.join(BASE_DIR, "models", "arsl_landmark_cnn_lstm_model.keras")
ENGLISH_KERAS_PATH = os.path.join(BASE_DIR, "models", "asl_landmark_cnn_lstm_model.keras")

ARABIC_TFLITE_PATH = os.path.join(BASE_DIR, "models", "arsl_model.tflite")
ENGLISH_TFLITE_PATH = os.path.join(BASE_DIR, "models", "asl_model.tflite")


def convert_keras_to_tflite(keras_path, tflite_path):
    print(f"\nLoading model: {keras_path}")
    model = tf.keras.models.load_model(keras_path)

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Fix for CNN-LSTM / BiLSTM models
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.experimental_enable_resource_variables = True

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model to: {tflite_path}")


if __name__ == "__main__":
    convert_keras_to_tflite(ARABIC_KERAS_PATH, ARABIC_TFLITE_PATH)
    convert_keras_to_tflite(ENGLISH_KERAS_PATH, ENGLISH_TFLITE_PATH)
    print("\nDone.")