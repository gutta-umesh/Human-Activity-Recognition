import socket
import pickle
import numpy as np
import threading
from tensorflow.keras.models import load_model, save_model
from sklearn.metrics import accuracy_score, confusion_matrix

# Configuration
METHODS = ["FedAvg", "FedMA", "FedPA"]
NUM_CLIENTS = 8
NUM_ROUNDS = 5
PORT = 5005

X_test = np.load("X_test_central.npy")
y_test = np.load("y_test_central.npy")

# Evaluation
def evaluate_model(model):
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true_labels, y_pred_labels)
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print("Confusion Matrix:\n", cm)
    return acc

# Aggregation methods
def fedavg_aggregate(weights, sizes):
    total = sum(sizes)
    return [sum((sizes[i] / total) * np.array(w) for i, w in enumerate(layer))
            for layer in zip(*weights)]

def fedma_aggregate(weights):
    return [np.mean(np.array(layer), axis=0) for layer in zip(*weights)]

def fedpa_aggregate(weights, global_weights):
    lam = 0.5
    return [lam * gw + (1 - lam) * np.mean(np.array(layer), axis=0)
            for layer, gw in zip(zip(*weights), global_weights)]

# Communication functions
def receive_data(conn):
    size = int.from_bytes(conn.recv(4), 'big')
    data = b''
    while len(data) < size:
        packet = conn.recv(4096)
        data += packet
    return pickle.loads(data)

def send_signal(conn, msg):
    conn.sendall(msg.encode())

# Handle each client asynchronously
def handle_client(conn, addr):
    print(f"?? Connected from {addr}")
    for rnd in range(NUM_ROUNDS):
        print(f"? Receiving from {addr} - Round {rnd + 1}")
        data = receive_data(conn)

        for method in METHODS:
            model_path = f"global_model_lstm_{method.lower()}.keras"
            model = load_model(model_path)

            if method == "FedAvg":
                new_weights = fedavg_aggregate([data["weights"][method], model.get_weights()],
                                               [data["data_size"], 1])  # 1 = dummy server size
            elif method == "FedMA":
                new_weights = fedma_aggregate([data["weights"][method], model.get_weights()])
            elif method == "FedPA":
                new_weights = fedpa_aggregate([data["weights"][method]], model.get_weights())

            model.set_weights(new_weights)
            save_model(model, model_path)

            print(f"? {method} - Aggregated asynchronously from client {data['client_id']}")

        send_signal(conn, "NEXT_ROUND")

    conn.close()
    print(f"?? Connection closed for {addr}")
    for method in METHODS:
        model_path = f"global_model_lstm_{method.lower()}.keras"
        model = load_model(model_path)
        acc = evaluate_model(model)
        print(f"? {method} Accuracy: {acc * 100:.2f}%")

# Accept all clients (multithreaded)
def accept_clients():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", PORT))
    server.listen(NUM_CLIENTS)
    print(f"?? Server listening on port {PORT}")
    for _ in range(NUM_CLIENTS):
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr)).start()

# Start server
accept_clients()
