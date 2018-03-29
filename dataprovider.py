from enum import Enum


class Optimizer(Enum):

    ADAM = 1
    RMSPROP = 2
    SGD = 3

class Parameters:

    def __init__(self, data_dict):
        pass