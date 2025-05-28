import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Input

# Configuration
INPUT_SHAPE = (64, 51)
NUM_CLASSES = 8
METHODS = ["FedAvg", "FedMA", "FedPA"]

# CNN Model Creator
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# LSTM Model Creator
def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),

        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Save initial models for each method
for method in METHODS:
    cnn_model = create_cnn_model(INPUT_SHAPE, NUM_CLASSES)
    cnn_model.save(f"global_model_cnn_{method.lower()}.keras")
    print(f"? Saved CNN model for {method}")

    lstm_model = create_lstm_model(INPUT_SHAPE, NUM_CLASSES)
    lstm_model.save(f"global_model_lstm_{method.lower()}.keras")
    print(f"? Saved LSTM model for {method}")