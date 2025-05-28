import socket
import pickle
import numpy as np
import threading
from tensorflow.keras.models import load_model, save_model
from sklearn.metrics import accuracy_score, confusion_matrix

METHODS = ["FedAvg", "FedMA", "FedPA"]
NUM_CLIENTS = 8
NUM_ROUNDS = 5
PORT = 5009

X_test = np.load("X_test_central.npy")
y_test = np.load("y_test_central.npy")

def evaluate_model(model):
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true_labels, y_pred_labels)
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print("Confusion Matrix:\n", cm)
    return acc

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

client_data = {m: [] for m in METHODS}
client_sizes = {m: [] for m in METHODS}
aggregation_lock = threading.Lock()
round_complete = threading.Event()

def receive_data(conn):
    size = int.from_bytes(conn.recv(4), 'big')
    data = b''
    while len(data) < size:
        packet = conn.recv(4096)
        data += packet
    return pickle.loads(data)

def send_signal(conn, msg):
    conn.sendall(msg.encode())

def handle_client(conn, addr):
    for rnd in range(NUM_ROUNDS):
        print(f"Waiting from {addr} - Round {rnd + 1}")
        data = receive_data(conn)
        with aggregation_lock:
            for method in METHODS:
                client_data[method].append(data["weights"][method])
                client_sizes[method].append(data["data_size"])
        round_complete.wait()
        send_signal(conn, "NEXT_ROUND")
    conn.close()

def accept_clients():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", PORT))
    server.listen(NUM_CLIENTS)
    print("Server listening on port", PORT)
    for _ in range(NUM_CLIENTS):
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr)).start()

accept_thread = threading.Thread(target=accept_clients)
accept_thread.start()
accept_thread.join()

# Run Rounds
for rnd in range(NUM_ROUNDS):
    print(f"\n?? Aggregation Round {rnd + 1}")
    while any(len(client_data[m]) < NUM_CLIENTS for m in METHODS):
        pass
    for method in METHODS:
        model = load_model(f"global_model_cnn_{method.lower()}.keras")
        if method == "FedAvg":
            new_weights = fedavg_aggregate(client_data[method], client_sizes[method])
        elif method == "FedMA":
            new_weights = fedma_aggregate(client_data[method])
        elif method == "FedPA":
            new_weights = fedpa_aggregate(client_data[method], model.get_weights())
        model.set_weights(new_weights)
        save_model(model, f"global_model_cnn_{method.lower()}.keras")
        acc = evaluate_model(model)
        print(f"? {method} Accuracy: {acc*100:.2f}%")
        client_data[method].clear()
        client_sizes[method].clear()
    round_complete.set()
    round_complete.clear()

print("? Training Complete!")