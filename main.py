import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization, Flatten, Dense, Reshape
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("sudoku.csv")
print("Columns in the CSV:", df.columns)
print("Number of puzzles:", len(df))

# Convert string to 9x9 grid
def string_to_grid(s):
    return np.array(list(s), dtype='int').reshape((9, 9))

# Preprocess data
puzzles = np.array([string_to_grid(p) for p in df["quizzes"]])
solutions = np.array([string_to_grid(s) for s in df["solutions"]])

# One-hot encode solutions
solutions_cat = to_categorical(solutions, num_classes=10)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(puzzles, solutions_cat, test_size=0.2, random_state=42)

# Expand input shape to (9, 9, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Build the model
model = Sequential()
model.add(Conv2D(64, (3, 3), padding="same", input_shape=(9, 9, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(810))  # 9x9x10 = 810
model.add(Reshape((9, 9, 10)))
model.add(Activation("softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train for 5 epochs
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Save the model
model.save("model_ep5.h5")
