import os

IP_HOST = os.getenv("IP_HOST", "192.168.0.126")
PORT = int(os.getenv("PORT", 7632))
PATH_TEST_SPLITS = os.getenv("PATH_TEST_SPLITS", "test_sets")