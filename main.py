import os
import warnings
from datetime import datetime
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import psutil
import tensorflow as tf
from GAFGenerator import GAFGenerator
from MTFGenerator import MTFGenerator
from RPGenerator import RPGenerator

SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]
CHEST_FREQ = 700
WRIST_FREQ = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
FEATURES = ['chest_acc_x', 'chest_acc_y', 'chest_acc_z', 'chest_ecg', 'chest_eda', 'chest_emg', 'chest_resp',
            'chest_temp', 'wrist_acc_x', 'wrist_acc_y', 'wrist_acc_z', 'wrist_bvp', 'wrist_eda', 'wrist_temp']
MULTICLASS_LABELS = {"Baseline": 1, "Stress": 2, "Amusement": 3, "Meditation": 4}


def resample_frequency(data, freq_in, freq_out):
    """Resample data using Linear Interpolation."""
    n_in = len(data)
    n_out = int(n_in * freq_out / freq_in)
    data_resampled = np.interp(np.linspace(0, n_in, n_out), np.arange(n_in), data)
    return data_resampled


def resample_labels(labels, frequency):
    """Convert labels to binary or multiclass."""
    labels = resample_frequency(labels, CHEST_FREQ, frequency)
    labels_df = pd.DataFrame(labels, columns=["label"])
    labels_df = labels_df.fillna(7)
    labels_df["label"] = labels_df["label"].astype(int)
    return labels_df


def convert_labels(data, binary=False):
    """Keep only labels baseline and stress from the dataset."""
    data = data.drop(data[(data["label"] == 5) | (data["label"] == 6) | (data["label"] == 7)].index)
    if binary:
        data = data.drop(data[(data["label"] == 3) | (data["label"] == 4) | (data["label"] == 0)].index)
        #data["label"] = data["label"].replace([0, 3, 4], 1)
    return data


def data_normalizer(df):
    """Normalize data from 1 to -1 using MinMaxScaler."""
    features_to_ignore = ["timestamp", "label"]
    df_saved = df[features_to_ignore]
    scaler = MinMaxScaler(feature_range=(-0.99, 0.99))
    df = df.drop(columns=features_to_ignore)
    df_normalized = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    df_normalized = pd.concat([df_normalized, df_saved], axis=1)
    return df_normalized


def hampel_filter(data, window_size):
    """Filter"""
    columns = data.columns
    rolling = data[columns].rolling(window=window_size, center=True)
    z_score = abs(data[columns] - rolling.mean()) / rolling.std()
    z_score = z_score.fillna(-float('inf'))
    data[columns] = data[columns].mask(z_score > 3)
    data = data.interpolate(method="nearest").bfill().ffill()
    return data


def process_wrist_data(data, frequency):
    """Load data from the Empatica E wristband."""
    wrist_acc_x = data["signal"]["wrist"]["ACC"][:, 0]
    wrist_acc_y = data["signal"]["wrist"]["ACC"][:, 1]
    wrist_acc_z = data["signal"]["wrist"]["ACC"][:, 2]
    wrist_bvp = data["signal"]["wrist"]["BVP"].flatten()
    wrist_eda = data["signal"]["wrist"]["EDA"].flatten()
    wrist_temp = data["signal"]["wrist"]["TEMP"].flatten()

    wrist_acc_x = resample_frequency(wrist_acc_x, WRIST_FREQ["ACC"], frequency)
    wrist_acc_y = resample_frequency(wrist_acc_y, WRIST_FREQ["ACC"], frequency)
    wrist_acc_z = resample_frequency(wrist_acc_z, WRIST_FREQ["ACC"], frequency)
    wrist_bvp = resample_frequency(wrist_bvp, WRIST_FREQ["BVP"], frequency)
    wrist_eda = resample_frequency(wrist_eda, WRIST_FREQ["EDA"], frequency)
    wrist_temp = resample_frequency(wrist_temp, WRIST_FREQ["TEMP"], frequency)

    df = pd.DataFrame({
        "wrist_acc_x": wrist_acc_x,
        "wrist_acc_y": wrist_acc_y,
        "wrist_acc_z": wrist_acc_z,
        "wrist_bvp": wrist_bvp,
        "wrist_eda": wrist_eda,
        "wrist_temp": wrist_temp,
    })

    return df


def process_chest_data(data, frequency):
    """Load data from the RespiBAN chestband."""
    chest_acc_x = data["signal"]["chest"]["ACC"][:, 0]
    chest_acc_y = data["signal"]["chest"]["ACC"][:, 1]
    chest_acc_z = data["signal"]["chest"]["ACC"][:, 2]
    chest_ecg = data["signal"]["chest"]["ECG"].flatten()
    chest_emg = data["signal"]["chest"]["EMG"].flatten()
    chest_eda = data["signal"]["chest"]["EDA"].flatten()
    chest_temp = data["signal"]["chest"]["Temp"].flatten()
    chest_resp = data["signal"]["chest"]["Resp"].flatten()

    chest_acc_x = resample_frequency(chest_acc_x, CHEST_FREQ, frequency)
    chest_acc_y = resample_frequency(chest_acc_y, CHEST_FREQ, frequency)
    chest_acc_z = resample_frequency(chest_acc_z, CHEST_FREQ, frequency)
    chest_ecg = resample_frequency(chest_ecg, CHEST_FREQ, frequency)
    chest_emg = resample_frequency(chest_emg, CHEST_FREQ, frequency)
    chest_eda = resample_frequency(chest_eda, CHEST_FREQ, frequency)
    chest_temp = resample_frequency(chest_temp, CHEST_FREQ, frequency)
    chest_resp = resample_frequency(chest_resp, CHEST_FREQ, frequency)
    timestamp = np.linspace(0, len(chest_acc_x) / frequency, len(chest_acc_x))

    df = pd.DataFrame({
        "chest_acc_x": chest_acc_x,
        "chest_acc_y": chest_acc_y,
        "chest_acc_z": chest_acc_z,
        "chest_ecg": chest_ecg,
        "chest_emg": chest_emg,
        "chest_eda": chest_eda,
        "chest_temp": chest_temp,
        "chest_resp": chest_resp,
        "timestamp": timestamp
    })

    return df


def load_subject(selected_subject, DATASET_FOLDER):
    """Load a subject's data"""
    file_path = f"{DATASET_FOLDER}/{selected_subject}/{selected_subject}.pkl"
    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")
    return data


def merge_all_subjects(frequency, PREPROCESS_OUTPUT_FOLDER):
    """Merge all subjects into one DataFrame"""
    df_all = pd.DataFrame()
    for subject in SUBJECTS:
        df = pd.read_csv(f"{PREPROCESS_OUTPUT_FOLDER}/resampling/{frequency}/WESAD_{subject}_{frequency}.csv")
        df_all = pd.concat([df_all, df], axis=0, ignore_index=True)
        del df
    df_all.to_csv(f"{PREPROCESS_OUTPUT_FOLDER}/resampling/{frequency}/WESAD_all_{frequency}.csv", index=False)


# load all subjects
def process_subjects(frequency, binary=False, PREPROCESS_OUTPUT_FOLDER="data/WESAD/Prepocessed_Subjects",
                     DATASET_FOLDER="data/WESAD", WINDOW_SEC=60):
    """Process all subjects"""
    for subject in SUBJECTS:
        print("Processing subject " + str(subject))
        data = load_subject(subject, DATASET_FOLDER)
        # load wrist and chest data
        wrist_df = process_wrist_data(data, frequency)
        chest_df = process_chest_data(data, frequency)
        df = pd.concat([wrist_df, chest_df], axis=1)
        # Filter
        df = hampel_filter(df, window_size=frequency * WINDOW_SEC)
        # Labels
        labels_df = resample_labels(data["label"], frequency)
        df = pd.concat([df, labels_df], axis=1)
        df = convert_labels(df, binary=binary)
        # Normalize data
        df = data_normalizer(df)
        os.makedirs(f"{PREPROCESS_OUTPUT_FOLDER}/resampling/{frequency}", exist_ok=True)
        df.to_csv(f"{PREPROCESS_OUTPUT_FOLDER}/resampling/{frequency}/WESAD_{subject}_{frequency}.csv", index=False)
    merge_all_subjects(frequency, PREPROCESS_OUTPUT_FOLDER)


def load_all_subject_data(frequency, subjects=SUBJECTS, PREPROCESS_OUTPUT_FOLDER="data/WESAD/Prepocessed_Subjects"):
    """Load all subjects"""
    all_dict = {}
    for subject in subjects:
        df = pd.read_csv(f"{PREPROCESS_OUTPUT_FOLDER}/resampling/{frequency}/WESAD_{subject}_{frequency}.csv")
        all_dict[subject] = df
    return all_dict


def plot_accuracy(history, name, path):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model accuracy {name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"{path}/{name}_accuracy.png")
    plt.show()


def plot_loss(history, name, path):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model loss {name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"{path}/{name}_loss.png")
    plt.show()


def get_model(input_dim, optimizer, learning_rate, n_classes=3, add_noise=False, stdev=0.1, seed=42, channels=14):
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    model = tf.keras.Sequential()

    if add_noise:
        model.add(tf.keras.layers.GaussianNoise(stddev=stdev))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=None,
                                     input_shape=(input_dim, input_dim, channels), data_format='channels_last'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=None))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    if optimizer == 'AMSGrad':
        op = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
    elif optimizer == 'Adam':
        op = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False)
    else:
        op = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(op, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model


def start_training(train_generator, val_generator, name="", n_classes=3, channels=14, PATIENTE=20, EPOCHS=100, IMAGE_SIZE=128,
                   OPTIMIZER='AMSGrad', LEARNING_RATE=0.0012, MODEL_OUTPUT_FOLDER="data/WESAD/Models",
                   RESAMPLE_FREQ=30):
    # Start time
    global device
    start_time = time.time()

    # Create a TensorBoard callback
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                     histogram_freq=1,
                                                     profile_batch='1,500')  # Profiling from batch no. 1 to 500

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=PATIENTE,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restore model weights from the epoch with the best value
    )

    model = get_model(IMAGE_SIZE, OPTIMIZER, LEARNING_RATE, n_classes=n_classes, add_noise=False, stdev=0.1, seed=42,
                      channels=channels)
    history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS,
                        callbacks=[early_stopping, tboard_callback])

    # End time
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_step = total_time / (EPOCHS * len(train_generator))
    memory_usage = psutil.virtual_memory().used / (1024.0 ** 3)
    cpu_usage = psutil.cpu_percent()

    # Check if TensorFlow is using a GPU
    cuda = f"Is TensorFlow built with CUDA (GPU)? {tf.test.is_built_with_cuda()}"
    device_avaliable = f"Is a GPU available? {tf.test.is_gpu_available()}"

    # If a GPU is available, this will print the device name
    if tf.test.is_gpu_available():
        device = f"GPU device name: {tf.test.gpu_device_name()}"

    path = f"{MODEL_OUTPUT_FOLDER}/{RESAMPLE_FREQ}/{name}_model"
    os.makedirs(path, exist_ok=True)

    plot_accuracy(history, name, path)
    plot_loss(history, name, path)
    # Model save
    model.save(f"{path}/{name}.keras")

    # Save metrics to a file
    with open(f"{path}/metrics.txt", "w") as f:
        f.write(str(cuda))
        f.write(str(device_avaliable))
        f.write(str(device))
        f.write(f"Total training time: {total_time} seconds\n")
        f.write(f"Average time per step: {avg_time_per_step} seconds\n")
        f.write(f"Average memory usage: {memory_usage} GB\n")
        f.write(f"Average CPU usage: {cpu_usage} %\n")


def main(frequency, dataset_path, window_sec):
    # Define constants and variables
    # Folder where the dastaset is
    DATASET_FOLDER = dataset_path
    # Output frequency after resampling
    RESAMPLE_FREQ = frequency
    # Folder to store the resampled subject
    PREPROCESS_OUTPUT_FOLDER = f"{DATASET_FOLDER}/Prepocessed_Subjects"
    # Model output Folder
    MODEL_OUTPUT_FOLDER = f"{DATASET_FOLDER}/Models"
    # Windows Length - Resampling
    WINDOW_SEC = window_sec
    # Window Size (1 minute of data at Frequency)
    WINDOW_SIZE = WINDOW_SEC * RESAMPLE_FREQ
    # Time step (1 second of data at Frequency)
    # To overlap windows change it to a value lower than WINDOW_SIZE
    TIME_STEP = 1 * RESAMPLE_FREQ
    # DEFAULT Hyperparameters
    # Set to true if you want to treat the problem as binary classification Baseline
    # VS Stress
    BINARY_CLASS = True
    # Seed
    SEED = 42
    # Batch size
    BATCH_SIZE = 31
    # Image size
    IMAGE_SIZE = 128
    # Learning Rate
    LEARNING_RATE = 0.0012
    # Optimizer
    OPTIMIZER = 'AMSGrad'
    # EPOCHS
    EPOCHS = 100
    # EARLY STOPPING PATIENTE
    PATIENTE = 5
    # MTF Bins
    N_BINS = 3
    # Reurrence Plot Threshold
    THRESHOLD = 0.5
    process_subjects(frequency=RESAMPLE_FREQ, binary=BINARY_CLASS, PREPROCESS_OUTPUT_FOLDER=PREPROCESS_OUTPUT_FOLDER,
                     DATASET_FOLDER=DATASET_FOLDER, WINDOW_SEC=WINDOW_SEC)

    train_subjects, validation_subjects = train_test_split(SUBJECTS, test_size=3 / 15, random_state=SEED)
    print(f'Train subjects: {" ".join(train_subjects)}')
    print(f'Validation subjects: {" ".join(validation_subjects)}')

    all_subject_data = load_all_subject_data(RESAMPLE_FREQ, subjects=SUBJECTS)
    if BINARY_CLASS:
        labels = [1, 2]

    train_generator = GAFGenerator(train_subjects, all_subject_data, FEATURES, labels, batch_size=BATCH_SIZE,
                                   image_size=IMAGE_SIZE, method='difference', window_size=WINDOW_SIZE,
                                   time_step=TIME_STEP, shuffle=True, seed=SEED)
    val_generator = GAFGenerator(validation_subjects, all_subject_data, FEATURES, labels, batch_size=BATCH_SIZE,
                                 image_size=IMAGE_SIZE, method='difference', window_size=WINDOW_SIZE,
                                 time_step=TIME_STEP, shuffle=True, seed=SEED)

    start_training(train_generator, val_generator, "GAF difference", n_classes=3, channels=14, PATIENTE=PATIENTE,
                   EPOCHS=EPOCHS, IMAGE_SIZE=IMAGE_SIZE, OPTIMIZER=OPTIMIZER, LEARNING_RATE=LEARNING_RATE,
                   MODEL_OUTPUT_FOLDER=MODEL_OUTPUT_FOLDER, RESAMPLE_FREQ=RESAMPLE_FREQ)

    train_generator = GAFGenerator(train_subjects, all_subject_data, FEATURES, labels, batch_size=BATCH_SIZE,
                                   image_size=IMAGE_SIZE, method='summation', window_size=WINDOW_SIZE,
                                   time_step=TIME_STEP, shuffle=True, seed=SEED)
    val_generator = GAFGenerator(validation_subjects, all_subject_data, FEATURES, labels, batch_size=BATCH_SIZE,
                                 image_size=IMAGE_SIZE, method='summation', window_size=WINDOW_SIZE,
                                 time_step=TIME_STEP, shuffle=True, seed=SEED)

    start_training(train_generator, val_generator, "GAF summation", n_classes=3, channels=14, PATIENTE=PATIENTE,
                   EPOCHS=EPOCHS, IMAGE_SIZE=IMAGE_SIZE, OPTIMIZER=OPTIMIZER, LEARNING_RATE=LEARNING_RATE,
                   MODEL_OUTPUT_FOLDER=MODEL_OUTPUT_FOLDER, RESAMPLE_FREQ=RESAMPLE_FREQ)

    warnings.filterwarnings('ignore', category=UserWarning, module='pyts.preprocessing')
    train_generator = MTFGenerator(train_subjects, all_subject_data, FEATURES, labels, batch_size=BATCH_SIZE,
                                   image_size=IMAGE_SIZE, window_size=WINDOW_SIZE, time_step=TIME_STEP, n_bins=N_BINS,
                                   shuffle=True, seed=SEED)
    val_generator = MTFGenerator(validation_subjects, all_subject_data, FEATURES, labels, batch_size=BATCH_SIZE,
                                 image_size=IMAGE_SIZE, window_size=WINDOW_SIZE, time_step=TIME_STEP, n_bins=N_BINS,
                                 shuffle=True, seed=SEED)

    start_training(train_generator, val_generator, "MTF", n_classes=3, channels=14, PATIENTE=PATIENTE, EPOCHS=EPOCHS,
                   IMAGE_SIZE=IMAGE_SIZE, OPTIMIZER=OPTIMIZER, LEARNING_RATE=LEARNING_RATE,
                   MODEL_OUTPUT_FOLDER=MODEL_OUTPUT_FOLDER, RESAMPLE_FREQ=RESAMPLE_FREQ)

    train_generator = RPGenerator(train_subjects, all_subject_data, FEATURES, labels, batch_size=BATCH_SIZE,
                                  image_size=IMAGE_SIZE, threshold=THRESHOLD, window_size=WINDOW_SIZE,
                                  time_step=TIME_STEP, shuffle=True, seed=SEED)
    val_generator = RPGenerator(validation_subjects, all_subject_data, FEATURES, labels, batch_size=BATCH_SIZE,
                                image_size=IMAGE_SIZE, threshold=THRESHOLD, window_size=WINDOW_SIZE,
                                time_step=TIME_STEP, shuffle=True, seed=SEED)

    start_training(train_generator, val_generator, "Recurrence Plot", n_classes=3, channels=14, PATIENTE=PATIENTE,
                     EPOCHS=EPOCHS, IMAGE_SIZE=IMAGE_SIZE, OPTIMIZER=OPTIMIZER, LEARNING_RATE=LEARNING_RATE,
                     MODEL_OUTPUT_FOLDER=MODEL_OUTPUT_FOLDER, RESAMPLE_FREQ=RESAMPLE_FREQ)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=int, default=30, help="Frequency of the preprocessing")
    parser.add_argument("--dataset", type=str, default="data/WESAD", help="Path to the dataset")
    parser.add_argument("--sec", type=int, default=60, help="Window in seconds")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        os.makedirs(args.dataset)
        print("No dataset provided, downloading WESAD dataset")
        # Download the dataset
        os.system("wget -v https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download -O data/WESAD.zip")
        os.system("unzip data/WESAD.zip -d data")
        os.system("rm data/WESAD.zip")


    if args.freq and args.dataset and args.sec:
        main(args.freq, args.dataset, args.sec)
    else:
        print("Please provide the frequency and the path to the dataset")
