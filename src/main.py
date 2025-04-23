import pickle
import time
import csv
import numpy as np


def write_data_file(data, csv_file, fieldnames): 
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


if __name__ == "__main__":
    LightCardLocal()

