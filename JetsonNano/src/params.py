import os 

IP_HOST = os.getenv("IP_HOST", "192.168.0.126")
PORT_NUM = int(os.getenv("PORT_NAME", 7632))

NUM_MODELS = int(os.getenv("NUM_MODELS", 4))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 3))

NUM_ROWS_IN_TEST = int(os.getenv("NUM_ROWS_IN_TEST", 117177))

LOCAL_TEST = int(os.getenv("LOCAL_TEST", 0))