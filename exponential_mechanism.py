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
        """Initialization

        Args:
            statistics (list): Statistics for Metropolitan Hasting.
            x (pd.DataFrame): Original dataframe.
            weights (list): Weights for statistics.
            alpha (float): Alpha for alpha-differential-privacy.
        """
        self.statistics = statistics
        self.x = x
        self.weights = weights
        self.alpha = alpha
        self.statistics_list = Sta(self.statistics).cal(x=self.x)

    def query_function(self, z: pd.DataFrame):
        """Query function for expoential mechanism

        Args:
            z (pd.DataFrame): synthetic dataframe

        Returns:
            float: Query function Value of synthetic dataframe
        """
        return np.max(
            np.abs(
                np.multiply(
                    self.weights,
                    self.statistics_list - Sta(self.statistics).cal(x=z),
                )
            )
        )

    def starting_point(self, k: int):
        """Creating random dataframe within the range of the original dataframe

        Args:
            k (int): Number of rows of the synthetic dataframe

        Returns:
            Dataframe: Synthetic dataframe within the range of the original dataframe
        """
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
        sensitivity: float,
        iterations: int = 10_000,
        stepsize: int = 1,
    ):
        """Metropolis-Hastings algorithm to obtain a synthetic dataframe with the same properties as the original dataframe in terms of self.statistics.

        Args:
            k (int): Number of rows in the synthetic dataframe.
            proposal_distribution (callable): Function that receives and modifies a dataframe.
            stats (list): Statistics comparing the synthetic and original dataframes. For illustrative purposes only, not used for Metropolitan Hasting.
            sigma (list): Variance or standard derivation for proposal distribution.
            sensitivity (float): Sensitivity of the original dataframe.            
            iterations (int, optional): Number of iterations. Defaults to 100_000.
            stepsize (int, optional): Number of rows that change with each iteration of the proposal distribution. Defaults to 1.

        Returns:
            tuple: Synthetic dataframe and other data for making plots.
        """
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


