import socket
import pickle
import numpy as np
from tensorflow.keras.models import load_model, clone_model
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# Configuration
CLIENT_ID = 104
PORT = 5009
NUM_ROUNDS = 5
NUM_CLASSES = 8
ALPHA = 0.1
METHODS = ["FedAvg", "FedMA", "FedPA"]

# Load data
X_train = np.load(f"X_client{CLIENT_ID}.npy")
y_train = np.load(f"y_client{CLIENT_ID}.npy")
X_test = np.load(f"X_test_client{CLIENT_ID}.npy")
y_test = np.load(f"y_test_client{CLIENT_ID}.npy")

if y_train.ndim > 1:
    y_train_labels = np.argmax(y_train, axis=1)
else:
    y_train_labels = y_train

def get_dirichlet_split_indices(y_labels, num_rounds, num_classes, alpha=0.5):
    indices = np.arange(len(y_labels))
    class_indices = [np.where(y_labels == i)[0] for i in range(num_classes)]
    round_indices = [[] for _ in range(num_rounds)]

    for c, class_idx in enumerate(class_indices):
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet(alpha=[alpha]*num_rounds)
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        class_split = np.split(class_idx, proportions)

        for r in range(num_rounds):
            round_indices[r].extend(class_split[r])
    return round_indices

round_indices = get_dirichlet_split_indices(y_train_labels, NUM_ROUNDS, NUM_CLASSES, alpha=ALPHA)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(("127.0.0.1", PORT))

def send_data(conn, data):
    packet = pickle.dumps(data)
    conn.sendall(len(packet).to_bytes(4, 'big'))
    conn.sendall(packet)

# Store personal models for personalization phase
personal_models = {}

# Federated Training Rounds
for rnd in range(NUM_ROUNDS):
    print(f"\n?? Client {CLIENT_ID} - Round {rnd + 1}")

    idx = round_indices[rnd]
    X_batch = X_train[idx]
    y_batch = y_train[idx]

    label_counts = Counter(np.argmax(y_batch, axis=1) if y_batch.ndim > 1 else y_batch)
    for cls in range(NUM_CLASSES):
        print(f"   Activity {cls}: {label_counts.get(cls, 0)} samples")

    updated_weights = {}

    for method in METHODS:
        model = load_model(f"global_model_cnn_{method.lower()}.keras")

        if rnd == 0:
            # Save copy of initial model for personalization
            personal_models[method] = clone_model(model)
            personal_models[method].set_weights(model.get_weights())
            personal_models[method].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_batch, y_batch, epochs=5, batch_size=32, verbose=0)

        y_pred = model.predict(X_test)
        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        print(f"? {method} - Round {rnd + 1} Accuracy: {acc * 100:.2f}%")

        updated_weights[method] = model.get_weights()

    send_data(client_socket, {
        "weights": updated_weights,
        "data_size": len(X_batch),
        "client_id": CLIENT_ID
    })

    if client_socket.recv(1024).decode() != "NEXT_ROUND":
        print("? Unexpected signal. Exiting.")
        break

client_socket.close()
print(f"\n? Client {CLIENT_ID} training completed.")

print("\n?? Evaluating final global models (post-rounds) on test data...")

for method in METHODS:
    print(f"\n?? Evaluating final global model for method: {method}")
    model = load_model(f"global_model_cnn_{method.lower()}.keras")

    y_pred = model.predict(X_test)
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    
    print(f"?? Final Global Accuracy on Client {CLIENT_ID} Test Data ({method}): {acc * 100:.2f}%")
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))