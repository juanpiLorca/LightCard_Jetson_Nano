import socket
from threading import Thread, Lock
import pickle
import time
import numpy as np
import os


class JetsonClient:
    def __init__(self, HOST, PORT, test_directory):
        self.socket = socket.socket()
        self.socket.connect((HOST, PORT))

        self.test_directory = test_directory  # Directorio con archivos .npy

        # Lock para controlar flujo de mensajes
        self.lock = Lock()
        self.process_files()

    def process_files(self):
        test_files = [file for file in os.listdir(self.test_directory) if file.endswith('.npy')]
        for file in test_files:
            file_name = file.split('/')[-1]

            # Notificar al servidor sobre el nombre del archivo actual
            self.send_file_name(file_name)
            print(f"Filename sent: {file_name}")
            with open("/../results/metrics.txt", 'a') as f:
                f.write(f"Filename sent: {file_name}\n")

            # Cargar datos del archivo actual
            data = np.load(os.path.join(self.test_directory, file), allow_pickle=True)

            real_labels = data[:, -2]
            other_predictions = data[:, -1]
            data = data[:, :-2]  # Eliminar las dos últimas columnas

            # Iniciar recolección de predicciones
            self.predictions = []
            self.times = []

            Thread(target=self.receive_predictions, daemon=True).start()
            self.send_data(data)

            # Calcular métricas al finalizar archivo
            self.calculate_metrics(real_labels, other_predictions)

    def send_file_name(self, file_name):
        self.lock.acquire()
        print(file_name)
        self.socket.send(pickle.dumps({"file_name": file_name}))
        self.lock.release()

    def send_data(self, data):
        for row in data:
            self.lock.acquire()
            row_dict = {f"col_{i}": val for i, val in enumerate(row)}
            serialized_data = pickle.dumps(row_dict)
            self.start_time = time.time()
            self.socket.send(serialized_data)

    def receive_predictions(self):
        while True:
            data = self.socket.recv(4096)
            self.end_time = time.time()
            if not data:
                break

            prediction = pickle.loads(data)
            self.predictions.append(prediction)
            self.times.append(self.end_time - self.start_time)

            self.lock.release()

    def calculate_metrics(self, real_labels, other_predictions):
        txt_path = "/../results/metrics.txt"
        # Calcular accuracy
        prediction_accuracy = sum(1 for p, r in zip(self.predictions, real_labels) if p == r) / len(real_labels)
        other_accuracy = sum(1 for p, o in zip(self.predictions, other_predictions) if p == o) / len(other_predictions)
        mean_time = np.mean(self.times)
        print(f"Prediction accuracy: {prediction_accuracy * 100:.2f}%")
        print(f"Other system accuracy: {other_accuracy * 100:.2f}%")
        print("Mean time: ", mean_time)

        with open(txt_path, 'a') as f:
            f.write(f"Prediction accuracy: {prediction_accuracy * 100:.2f}%\n")
            f.write(f"Other system accuracy: {other_accuracy * 100:.2f}%\n")
            f.write("Mean time: " + str(mean_time) + "\n")
