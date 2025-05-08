import numpy as np
from keras.models import load_model
from data_prep import DataReader

BASE_PATH = "C:/Users/Bayan/Documents/sudoku_ai_final"

model = load_model("model_ep45.h5")

reader = DataReader(BASE_PATH)
x, y = reader.get_data()

index = 0
sample = x[index]
solution = y[index].argmax(axis=-1)

prepared = np.expand_dims(sample, axis=0) 
prediction = model.predict(prepared)

predicted_grid = prediction.reshape(9, 9, 10).argmax(axis=-1)

print("Original puzzle:")
print(sample.reshape(9, 9).astype(int))

print("\nModel prediction:")
print(predicted_grid)

print("\nExpected solution:")
print(solution.astype(int))
