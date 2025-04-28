import csv
import socket
import pickle
import time
import pandas as pd
import numpy as np  


class LightCardServer:

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = None
        self.conn = None
        self.addr = None

        self.models = []
        self.model_idx = 0
        self.current_model = None

        # Data storage
        self.fieldnames = ["pred_time", "pred"]
        self.csv_file = "/results/LC_times_E1.{}_C{}.csv"

    def start(self): 
        # Initialize the server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        print(f"Server listening in {self.host}:{self.port}")
        self.conn, self.addr = self.server_socket.accept()
        self.conn.settimeout(10)
        print(f"Connection made: {self.addr}")

    def close(self):
        if self.conn:
            self.conn.close()
        if self.server_socket:
            self.server_socket.close()
        print("Connection closed.")

    def load_models(self, model_paths: list):
        # Load the decision tree model
        for path in model_paths:
            with open(path, 'rb') as file:
                self.models.append(pickle.load(file))
                print(f"Loaded: {path}")

    def process_data(self):

        try: 
            raw_length = self.conn.recv(4)
            if len(raw_length) < 4:
                print("Incomplete data length received.")
                return None
            data_length = int.from_bytes(raw_length, 'big')
            if data_length == 0:
                return None

            # Receive the data
            data = b""
            while len(data) < data_length:
                packet = self.conn.recv(data_length - len(data))
                if not packet:
                    print("Client disconnected.")
                    return None
                data += packet

            # Deserialize the data
            row = pickle.loads(data)

            # Convert to DataFrame and then to NumPy array
            df_row = pd.DataFrame([row])
            return df_row.to_numpy()

        except (socket.timeout, ConnectionResetError, EOFError, pickle.UnpicklingError) as e:
            print(f"Error receiving data: {e}")
            return None
    

    def handle_client(self, num_test, num_rows_in_test):
        while self.model_idx < len(self.models):
            
            # Load the current model   
            self.current_model = self.models[self.model_idx]
            print(f"Processing model {self.model_idx + 1} of {len(self.models)}")

            i = 0
            while i < num_test:

                j = 0
                print(f"Processing test {i + 1} of {num_test}")
                csv_file = self.csv_file.format(self.model_idx + 1, i + 1)
                with open(csv_file, "w") as file:
                    csv_writer = csv.DictWriter(file, self.fieldnames)
                    csv_writer.writeheader()


                while j < num_rows_in_test:

                    # Process the data
                    row = self.process_data()
                    if row is None:
                        break
                        
                    start_time = time.time()
                    prediction = self.current_model.predict(row)
                    end_time = time.time()

                    pred_value = prediction[0] if hasattr(prediction, '__getitem__') else prediction

                    # Serialize the prediction: Only one value!
                    serialized_data = pickle.dumps(pred_value)
                    self.conn.sendall(len(serialized_data).to_bytes(4, 'big'))
                    self.conn.sendall(serialized_data)

                    time_pred = end_time - start_time
                    data = [time_pred, prediction[0]]
                    self.write_data_file(data, csv_file, self.fieldnames)
                    
                    j += 1

                print(f"Finished processing test {i + 1} of {num_test}")
                print(f"Total predictions: {j}")
                i += 1

            self.model_idx += 1
            print(f"Finished processing model {self.model_idx} of {len(self.models)}")

        print("Finished processing all models.")

    def write_data_file(self, data, csv_file, fieldnames):
        with open(csv_file, "a") as file:
            csv_writer = csv.DictWriter(file, fieldnames)
            info = {
                "pred_time": data[0],
                "pred": data[1]
            }
            csv_writer.writerow(info)



def LightCardLocal():

    models = []
    tests = []
    model_paths = "models/model_F1.{}__results.pkl_class{}.pkl"
    test_paths = "test_sets/F1.{}__results.pkl_class{}_test.npy"

    LC_times = []
    LC_times_path = "/results/LC_times_E1.{}_C{}.csv"

    for i in range(1, 4): 
        for j in range(1, 4):
            models.append(model_paths.format(i, j))
            tests.append(test_paths.format(i, j))
            LC_times.append(LC_times_path.format(i, j))

    i = 0 
    N = len(models)
    fieldnames = ["pred_time", "pred"]
    while i < N: 

        csv_file = LC_times[i]
        with open(csv_file, "w") as file: 
            csv_writer = csv.DictWriter(file, fieldnames)
            csv_writer.writeheader()

        model = models[i]
        with open(model, 'rb') as archivo:
            current_model = pickle.load(archivo)

        test = tests[i]
        data = np.load(test)
        data = data[:, :-2]
        print('Amount of rows:', len(data))

        total_predictions = 0
        for row in data:
            row = row.reshape(1,-1)

            start_time = time.time()
            prediction = current_model.predict(row)
            end_time = time.time()

            total_predictions += 1

            data = [end_time - start_time, prediction[0]]
            write_data_file(data, csv_file, fieldnames)

        print('Number of predictions made:', total_predictions)

        i += 1 

def write_data_file(data, csv_file, fieldnames): 
    with open(csv_file, "a") as file: 
        csv_writer = csv.DictWriter(file, fieldnames)
        info = {
            "pred_time": data[0],
            "pred": data[1]
        }
        csv_writer.writerow(info)
    


