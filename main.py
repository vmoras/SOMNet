from math import exp, pi, sqrt
from numpy import ndarray
from sklearn.datasets import load_digits
from numpy.linalg import norm
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class SOM:
    def __init__(self, neighbor_function, data, width, length,
                 initial_learning_rate=0.01):

        self.neighbor_function = neighbor_function
        self.data = self.normalize(data)
        self.num_iterations = 0
        self.max_iterations = 5000
        self.initial_learning_rate = initial_learning_rate
        self.initial_radio = min(width, length) / 2
        self.learning_rate = 0
        self.radio = 0
        self.grid = self.set_grid(width, length)

    @staticmethod
    def normalize(data) -> ndarray[any, int]:
        """
        Normalizes the data between 0 and 1, so each feature will have
        the same impact in the distance calculation.
        Returns the same input data but all the values between 0 and 1
        """
        scaler = MinMaxScaler()
        scaler.fit(data)
        return scaler.transform(data)

    def set_grid(self, width: int, length: int) -> ndarray[any, float]:
        """
        Returns a numpy array in 2D like a grid with a specified width and
        length with a neuron in each cell, each cell will have the same
        dimension as the input data.
        """
        dimension = self.data.shape[1]
        return np.random.random_sample((width, length, dimension))

    def train(self) -> None:
        """
        TODO: write a description
        """
        change = 50000
        while change > 0:
            self.update_learning_rate()
            self.update_radio()
            for input_vector in self.data:
                winner_index = self.get_BMU(input_vector)
                change = self.update_weights(winner_index, input_vector)
            self.num_iterations += 1
            change -= 1

    def classify(self):
        """
        TODO
        """
        pass

    def plot(self):
        """
        TODO
        """
        pass

    def get_BMU(self, input_vector: ndarray[any, float]) -> ndarray[any, int]:
        """
        Returns the index in the grid of the BMU (best matching unit) for a given input vector.
        It uses the euclidean distance to get the BMU
        """
        min_distance = float("inf")
        winner_neuron = None

        for row in range(len(self.grid)):
            for column in range(row):
                distance = norm(input_vector - self.grid[row, column])
                if distance < min_distance:
                    min_distance = distance
                    winner_neuron = [row, column]

        return np.array(winner_neuron)

    def update_weights(self, winner_index: ndarray[any, int], input_vector: ndarray[any, float]) -> bool:
        """
        Change the winner's neighbors' weights (if it is inside its radio)
        using a neighborhood function and learning rate.
        Returns True if one or more neurons changed their weights,
        or returns False if no neuron changed its weight
        """
        change = False
        for row in range(len(self.grid)):
            for column in range(row):
                radio = self.radio
                neuron_index = np.array([row, column])
                distance = norm(winner_index - neuron_index)

                if distance <= radio:
                    self.grid[row, column] = self.update_neighbor_weight(winner_index,
                                                                         neuron_index,
                                                                         input_vector)
                    change = True

        return change

    def update_neighbor_weight(self, winner_index: ndarray[any, int], neighbor_index: ndarray[any, int],
                               input_vector: ndarray[any, float]) -> ndarray[any, int]:
        """
        Returns the updated value of a neuron given a winner and an input vector.
        It uses the neighborhood function and a learning rate
        """
        learning = self.learning_rate
        influence = self.get_neighbor_function(winner_index, neighbor_index)

        row = neighbor_index[0]
        column = neighbor_index[1]
        neighbor_weight = self.grid[row, column]

        return neighbor_weight + (learning * influence * (input_vector - neighbor_weight))

    def get_neighbor_function(self, winner_index: ndarray[any, int], neighbor_index: ndarray[any, int]) -> float:
        """
        Based on a neighbor function and the distance between a winner neuron, and its neighbor
        returns the value. The idea is to make the neighbor's weight more similar to the
        winner's weights
        """
        sigma = self.radio
        winner_weight = self.grid[winner_index[0], winner_index[1]]
        neighbor_weight = self.grid[neighbor_index[0], neighbor_index[1]]
        dist = norm(winner_weight - neighbor_weight)

        # Use the gaussian function
        if self.neighbor_function == "gaussian":
            return exp((- dist ** 2) / (2 * (sigma ** 2)))

        # Use the mexican function
        a = 2 / (sqrt(3 * sigma) * pow(pi, 1/4))
        b = 1 - pow((dist / sigma), 2)
        return a * b * exp((- dist ** 2) / (2 * (sigma ** 2)))

    def update_radio(self) -> None:
        """
        Updates the radio in which the neighbor's weight will change.
        Based on  and the number of iterations done. Used to make the
        neighbourhood function narrower
        """
        radio_0 = self.initial_radio
        i = self.num_iterations
        T = self.max_iterations

        self.radio = radio_0 * exp(-i / T)

    def update_learning_rate(self) -> None:
        """
        Updates the learning rate based on  and the number of iterations.
        Used to decrease the learning
        """
        lambda_0 = self.initial_learning_rate
        i = self.num_iterations
        T = self.max_iterations

        self.learning_rate = lambda_0 * exp(-i / T)


def main():
    digits = load_digits(return_X_y=True)

    model = SOM(neighbor_function="gaussian", data=digits[0], width=4, length=4)
    model.train()
    model.classify()
    model.plot()


main()
