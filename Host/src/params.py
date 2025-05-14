import os

IP_HOST = os.getenv("IP_HOST", "10.8.33.129")
PORT = int(os.getenv("PORT", 7632))

NUM_EXP = int(os.getenv("NUM_EXP", 2))
NUM_TEST = int(os.getenv("NUM_TEST", 4))
NUM_CLASS = int(os.getenv("NUM_CLASS", 1))