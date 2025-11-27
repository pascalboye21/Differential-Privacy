from pandas import DataFrame
import numpy as np


class Sta:
    func = list

    def __init__(self, func):
        self.func = func

    def cal(self, x: DataFrame):
        result = np.array([element(x.values) for element in self.func])
        return result
