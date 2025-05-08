import numpy as np
import pandas as pd

class DataReader:
    def __init__(self, base_path):
        self.base_path = base_path

    def get_data(self):
        file_path = self.base_path + "/sudoku.csv"
        data = pd.read_csv(file_path)
        quizzes = data['quizzes']
        solutions = data['solutions']

        x = []
        y = []

        for quiz, sol in zip(quizzes, solutions):
            board = np.array([int(c) for c in quiz]).reshape(9, 9)
            solution = np.array([int(c) for c in sol]).reshape(9, 9)

            x.append(board)
            y.append(solution)

        x = np.array(x).reshape(-1, 9, 9, 1) / 9.0  # normalize inputs
        y = np.array(y)

        # One-hot encode the labels
        y_encoded = np.zeros((y.shape[0], 9, 9, 9))
        for i in range(y.shape[0]):
            for r in range(9):
                for c in range(9):
                    y_encoded[i, r, c, y[i, r, c] - 1] = 1

        return x, y_encoded
