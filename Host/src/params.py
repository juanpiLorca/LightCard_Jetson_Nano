import os

IP_HOST = os.getenv("IP_HOST", "100.69.70.72")
PORT = int(os.getenv("PORT", 7632))

NUM_EXP = int(os.getenv("NUM_EXP", 1))
NUM_TEST = int(os.getenv("NUM_TEST", 1))
NUM_CLASS = int(os.getenv("NUM_CLASS", 1))