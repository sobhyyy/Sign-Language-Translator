import tensorflow as tf

def create_cnn_lstm_model(num_classes=29, timesteps=23, features=63):
    """
    Optimized CNN-LSTM model for landmark sequence input.
    Input shape: (batch, timesteps, features) = (None, 23, 63)
    """
    # Input
    inputs = tf.keras.Input(shape=(timesteps, features), name="landmark_input")

    # CNN layers
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # LSTM layer (bidirectional)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=False)
    )(x)

    # Dense layers
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Landmark_CNN_LSTM")

    # Optimizer with mixed precision support
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model