import numpy as np
import pickle


def LightCardLocal():

    model_path = "models/model_F1.1__results.pkl_class1.pkl"                
    with open(model_path, 'rb') as archivo:
        current_model = pickle.load(archivo)

    archivo = np.load('test_sets/F1.1__results.pkl_class1_test.npy')
    data = archivo[:, :-2]  

    print('cantidad de filas:', len(data))

    total_predictions = 0
    for row in data:
        row2 = row.reshape(1,-1)
        prediction = current_model.predict(row2)

        print('Tamano de la predicción:', prediction.shape)
        print('predicción:', prediction)

        total_predictions += 1
        
    print('predicciones hechas:', total_predictions)


if __name__ == "__main__":
    LightCardLocal()


