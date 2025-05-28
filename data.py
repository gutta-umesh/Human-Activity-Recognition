import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# Configuration
DATA_FOLDER = "/nfsshare/uma/HAR using FL/dataset"
SELECTED_CLIENTS = [101, 102, 103, 104, 105, 106, 107, 108]
SELECTED_ACTIVITIES = [1, 2, 3, 4, 12, 13, 16, 17]
NUM_CLASSES = len(SELECTED_ACTIVITIES)
WINDOW_SIZE = 64
STEP_SIZE = 32
TEST_RATIO = 0.2
NUM_ROUNDS = 5

# Map labels to 0-based class indices
label_map = {act: idx for idx, act in enumerate(sorted(SELECTED_ACTIVITIES))}

# Define column names
columns = ['timestamp', 'activityID', 'heart_rate'] + \
          [f'hand_{i}' for i in range(1, 18)] + \
          [f'chest_{i}' for i in range(1, 18)] + \
          [f'ankle_{i}' for i in range(1, 18)]

# Sliding window function
def sliding_window(X, y, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    X_windows, y_windows = [], []
    for i in range(0, len(X) - window_size + 1, step_size):
        X_windows.append(X[i:i + window_size])
        y_windows.append(y[i + window_size // 2])
    return np.array(X_windows), np.array(y_windows)

# Load and organize client data
client_data = {}

for cid in SELECTED_CLIENTS:
    path = os.path.join(DATA_FOLDER, f"subject{cid}.dat")
    print(f"?? Reading Subject {cid}")

    df = pd.read_csv(path, sep=" ", header=None, names=columns).fillna(0)
    df = df[df['activityID'].isin(SELECTED_ACTIVITIES)]

    client_data[cid] = {}
    for act in SELECTED_ACTIVITIES:
        act_df = df[df['activityID'] == act]
        client_data[cid][act] = act_df
        print(f"  ? Client {cid}, Activity {act}: {len(act_df)} samples")

# Process each client individually
central_test_data = []

for cid in SELECTED_CLIENTS:
    print(f"\n?? Processing Client {cid}")
    all_train, all_test = [], []

    for act in SELECTED_ACTIVITIES:
        df = client_data[cid][act]
        X = df.drop(columns=['timestamp', 'activityID', 'heart_rate']).values
        y = np.array([label_map[act]] * len(df))

        if len(df) < WINDOW_SIZE:
            print(f"  ?? Skipping Activity {act} for Client {cid} due to insufficient samples")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_RATIO, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        all_train.append((X_train, y_train))
        all_test.append((X_test, y_test))

    if not all_train or not all_test:
        print(f"  ? Skipping Client {cid} due to no valid activity data.")
        continue

    # Merge and scale
    X_train_all = np.concatenate([x for x, _ in all_train], axis=0)
    y_train_all = np.concatenate([y for _, y in all_train], axis=0)

    X_test_all = np.concatenate([x for x, _ in all_test], axis=0)
    y_test_all = np.concatenate([y for _, y in all_test], axis=0)

    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(X_train_all)
    X_test_all = scaler.transform(X_test_all)

    # Apply sliding window
    X_train_win, y_train_win = sliding_window(X_train_all, y_train_all)
    X_test_win, y_test_win = sliding_window(X_test_all, y_test_all)

    # One-hot encode labels
    y_train_onehot = to_categorical(y_train_win, num_classes=NUM_CLASSES)
    y_test_onehot = to_categorical(y_test_win, num_classes=NUM_CLASSES)

    # Save processed data
    np.save(f"X_client{cid}.npy", X_train_win)
    np.save(f"y_client{cid}.npy", y_train_onehot)
    np.save(f"X_test_client{cid}.npy", X_test_win)
    np.save(f"y_test_client{cid}.npy", y_test_onehot)

    central_test_data.append((X_test_win, y_test_onehot))

    print(f"? Client {cid}: Train {X_train_win.shape}, Test {X_test_win.shape}")

# Combine test data for central server
X_test_central = np.vstack([x for x, _ in central_test_data])
y_test_central = np.vstack([y for _, y in central_test_data])

np.save("X_test_central.npy", X_test_central)
np.save("y_test_central.npy", y_test_central)
print(f"\n?? Central Test Shape: {X_test_central.shape}, {y_test_central.shape}")
