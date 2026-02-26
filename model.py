import tensorflow as tf

def create_cnn_lstm_model(num_classes=29, timesteps=23, features=63):
    """
    CNN-LSTM model for landmark sequence input.
    Input shape: (batch, timesteps, features) = (None, 23, 63)
    """
    inputs = tf.keras.Input(shape=(timesteps, features))

    # CNN layers
    x = tf.keras.layers.Conv1D(64, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(128, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # LSTM layer
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256))(x)

    # Dense layers
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model