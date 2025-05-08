# Sudoku AI Solver 🧠🧩

This project implements a deep learning model that solves Sudoku puzzles automatically using a Convolutional Neural Network (CNN).  
It is inspired by the paper **"A Deep Learning Approach to Solve Sudoku Puzzle"** by Vamsi et al. (2021).

---

## 📌 Project Overview

- 🔍 **Goal:** Predict the correct numbers for a partially-filled 9x9 Sudoku grid.
- 🧠 **Model:** CNN with 3 convolutional layers + dense layers + softmax classification for each cell.
- 📊 **Output:** A complete Sudoku grid filled with predicted values.

---

## 🛠️ Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy, Pandas
- Jupyter / VS Code

---

## 📁 Files Included

| File                | Description                                 |
|---------------------|---------------------------------------------|
| `main.py`           | Builds and trains the initial CNN model     |
| `continue_training.py` | Continues training from a saved model     |
| `data_prep.py`      | Preprocesses the Sudoku dataset             |
| `predict_test.py`   | Predicts and visualizes Sudoku solutions    |

---

## 📦 Dataset & Model

- 🔗 **Dataset:** [Sudoku CSV dataset](https://www.kaggle.com/datasets/bryanpark/sudoku) (not included in this repo due to size).
- 💾 **Model Weights:** Final model (`model_ep45.h5`) trained for 50 epochs is available separately.

> 📁 For access to the full dataset and trained model, please contact the team or check the shared Google Drive link (if available).

---

## 👩‍💻 Contributors

- Bayan 
- Asma
- Tasneem
- Hind

---

## 📌 Notes

> This project is part of our coursework for the Artificial Intelligence course at [Your University Name Here].  
> Special thanks to our supervisor and everyone who contributed!

---

