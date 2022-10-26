import numpy as np
import pandas as pd


def create_gram_matrix(listX: np.ndarray):
    maxRows = len(listX)
    gram = np.zeros(shape=(maxRows, maxRows))
    for row in range(maxRows):
        for column in range(maxRows):
            gram[row][column] = np.dot(listX[row], listX[column])
    return gram


if __name__ == "__main__":
    # read the datasets for training and resting
    df = pd.read_csv("train.csv")
    dt = pd.read_csv("test.csv")
    a = [df['X'], df['y']]
    X = np.transpose(np.array(a))
    print(X)
    print(len(X), X.size)
    print(create_gram_matrix(X))
