from params import *
from LightCard import LightCardServer, LightCardLocal


def main():

    if LOCAL_TEST:
        LightCardLocal()

    else:
        # Initialize the server
        server = LightCardServer(IP_HOST, PORT_NUM)
        # Start the server
        server.start()

        # Load models
        models = []
        model_paths = "models/model_F1.{}__results.pkl_class{}.pkl"
        for i in range(1, NUM_MODELS): 
            for j in range(1, NUM_CLASSES):
                models.append(model_paths.format(i, j))
        # Create a list of model paths
        server.load_models(model_paths)

        # Handle client connections
        server.handle_client(NUM_CLASSES, NUM_ROWS_IN_TEST)

        # Close the server
        server.close()



if __name__ == "__main__":
    main()

