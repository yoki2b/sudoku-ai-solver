import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

tf.config.run_functions_eagerly(True)

# ----- Data Reader -----
class DataReader:
    def __init__(self, base_path):
        self.base_path = base_path

    def get_data(self):
        df = pd.read_csv(f"{self.base_path}/sudoku.csv")
        x = []
        y = []
        for quiz, solution in zip(df['quizzes'], df['solutions']):
            x.append([int(c) for c in quiz])
            y.append([int(c) for c in solution])

        x = np.array(x).reshape(-1, 9, 9, 1)
        y = np.array(y).reshape(-1, 9, 9)

        y_encoded = np.zeros((y.shape[0], 9, 9, 10))
        for i in range(y.shape[0]):
            for j in range(9):
                for k in range(9):
                    val = y[i, j, k]
                    if val > 0:
                        y_encoded[i, j, k, val - 1] = 1

        return x, y_encoded

BASE_PATH = "C:/Users/Bayan/Documents/sudoku_ai_final"
reader = DataReader(BASE_PATH)
x_train, y_train = reader.get_data()

model = load_model("model_ep40.h5")

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint("model_ep45.h5", save_best_only=False, verbose=1)

model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    callbacks=[checkpoint]
)
