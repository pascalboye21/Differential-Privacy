from pandas import DataFrame
import numpy as np


class Sta:
    func = list

    def __init__(self, func):
        self.func = func

    def cal(self, x: DataFrame):
        """Evaluation of each statistic in self.func of dataframe x.

        Args:
            x (DataFrame): dataframe

        Returns:
            list: Results of each statistic on dataframe x.
        """
        result = np.array([element(x.values) for element in self.func])
        return result

