import numpy as np
import pandas as pd
import time
from st import Sta
from tqdm import tqdm
import itertools


class Exponential_Mechanism:
    statistics = list
    weights = list
    alpha = float
    x = pd.DataFrame
    statistics_list = np.array

    def __init__(
        self,
        statistics: list,
        x: pd.DataFrame,
        weights: list,
        alpha: float,
    ):
        self.statistics = statistics
        self.x = x
        self.weights = weights
        self.alpha = alpha
        self.statistics_list = Sta(self.statistics).cal(x=self.x)

    def query_function(self, z: pd.DataFrame):
        return np.max(
            np.abs(
                np.multiply(
                    self.weights,
                    self.statistics_list - Sta(self.statistics).cal(x=z),
                )
            )
        )

    def sensitivity_Delta_S(self):
        maximum = 0
        n = len(self.x)
        print("Calculating sensitivity:")
        for l in tqdm(range(n - 1)):
            element = self.x.values[l]
            subset1 = list(self.x.values)
            del subset1[l]
            for i in range(n - 1):
                subset2 = subset1.copy()
                subset2[i] = element
                # check = np.array(subset1) - np.array(subset2)
                res = np.max(
                    np.abs(
                        np.multiply(
                            self.weights,
                            Sta(self.statistics).cal(x=pd.DataFrame(subset1))
                            - Sta(self.statistics).cal(x=pd.DataFrame(subset2)),
                        ),
                    )
                )
                if res >= maximum:
                    maximum = res
        return maximum

    def starting_point(self, k: int):
        df = self.x.copy().sample(n=k)
        for column in self.x:
            if self.x[column].dtypes == "float64":
                df[column] = np.random.uniform(
                    low=np.min(self.x[column].values),
                    high=np.max(self.x[column].values),
                    size=k,
                )
            else:
                df[column] = np.random.choice(
                    list(set(self.x[column].values)),
                    size=k,
                )
        return df

    def metropol_haste(
        self,
        k: int,
        proposal_distribution: callable,
        stats: list,
        sigma: list,
        iterations: int = 10_000,
        sample_size: int = 10,
        sensitivity: float = None,
        stepsize: int = 1,
    ):
        stop = False
        counter = 0
        counter_2 = 0
        min = 1

        result_query = []

        # z = self.x.sample(n=k)
        starting_point = self.starting_point(k)
        z = starting_point.copy()

        result_x = Sta(stats).cal(x=self.x)
        result_z = []
        result_difference = []
        result_z.append(Sta(stats).cal(x=z))
        result_difference.append(np.abs(result_x - result_z[-1]))

        query_function_value_old = self.query_function(z)
        result_query.append(query_function_value_old)

        if sensitivity == None:
            sensitivity = self.sensitivity_Delta_S()

        epsilon = self.alpha / (2 * sensitivity)

        query_function_values_over_time = 0

        for i in range(iterations):
            t = time.perf_counter_ns()
            y = proposal_distribution(z, self.x, sigma=sigma, stepsize=stepsize)

            query_function_values_over_time += query_function_value_old

            query_function_value = self.query_function(y)
            diff = query_function_value_old - query_function_value

            if diff >= 0:
                acceptance_ratio = 1
            else:
                acceptance_ratio = np.exp(epsilon * diff)

            if np.random.uniform() <= acceptance_ratio:
                if acceptance_ratio <= min:
                    min = acceptance_ratio
                if acceptance_ratio < 1:
                    counter_2 += 1
                z = y.copy()
                query_function_value_old = query_function_value.copy()
                counter += 1
                print(
                    f"  {i+1:9d}/{iterations}          {round((time.perf_counter_ns() - t)*1e-09, 2):00.2f} s"
                    + "              accepted    "
                    + f"{acceptance_ratio:0000.3f}"
                    + "   "
                    + f"{query_function_values_over_time / (i+1):0000000.5f}"
                    + "   "
                    + f"{query_function_value_old:0000000.5f}"
                )
            else:
                print(
                    f"  {i+1:9d}/{iterations}          {round((time.perf_counter_ns() - t)*1e-09, 2):00.2f} s"
                    + "          not accepted    "
                    + f"{acceptance_ratio:0000.3f}"
                    + "   "
                    + f"{query_function_values_over_time / (i+1):0000000.5f}"
                    + "   "
                    + f"{query_function_value_old:0000000.5f}"
                )
            result_z.append(Sta(stats).cal(x=z))
            result_difference.append(np.abs(result_x - result_z[-1]))
            result_query.append(query_function_value_old)
            if stop:
                l = i
                print("Counter accepted:" + str(counter) + " / " + str(l))
                return (
                    counter_2,
                    min,
                    z,
                    counter,
                    l,
                    result_difference,
                    result_x,
                    result_z,
                    result_query,
                    starting_point,
                )
        print("Counter accepted:" + str(counter) + " / " + str(iterations))
        return (
            counter_2,
            min,
            z,
            counter,
            iterations,
            result_difference,
            result_x,
            result_z,
            result_query,
            starting_point,
        )
