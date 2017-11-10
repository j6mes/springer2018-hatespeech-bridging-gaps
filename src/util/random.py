import os


def get_seed():
    os.getenv("RANDOM_SEED",1234)