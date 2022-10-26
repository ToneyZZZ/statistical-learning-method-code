'''
这是感知机, 最早在1958年被Rosenblatt提出,
是世界上最早, 最简单的的统计学习模型。

这或许都不算人类的一小步, 却是我的一大步。
——TZ 于 2022.10.14
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint


class perceptron:
    def __init__(self):
        self.learningStep = 1
        self.maxIteration = 5000

    def train(self, listX: np.ndarray, listY: np.ndarray):
        self.w = np.transpose(np.zeros(2))
        self.b = 0
        count = 0
        print(listY.size)
        for time in range(self.maxIteration):
            # 选择一个样本
            index = randint(0, listY.size - 1)
            x = listX[index]
            y = listY[index]
            if y * (np.dot(self.w, x) + self.b) <= 0:
                self.b += self.learningStep * y
                self.w += self.learningStep * y * x
                count += 1
            if count >= listY.size * 2:
                break

    def test(self, listX: np.ndarray, listY: np.ndarray):
        times = listY.size
        for i in range(times):
            if listY[i] * (np.dot(self.w, listX[i]) + self.b) <= 0:
                print(f'第{str(i)}组数据检测失败！')
                break
            else:
                print(f'第{i}组数据检测成功！')


def show_diagram(df: pd.DataFrame, title: str, color=["", "royalblue", "crimson"]):
    # Show the data in a diagram
    colors = [color[df["underLine"][i]] for i in range(len(df))]
    plt.title(title)
    plt.scatter(x=df['X'], y=df['y'], c=colors)
    if p.w[0] != 0:
        testLine = [(-p.b - p.w[1] * i) / p.w[0] for i in range(20)]
        plt.plot(testLine)
    plt.show()


if __name__ == "__main__":
    # read the datasets for training and resting
    df = pd.read_csv("train.csv")
    dt = pd.read_csv("test.csv")

    # Training process
    transX = [df['X'], df['y']]
    X = np.transpose(np.array(transX))
    y = np.array(df['underLine'])
    p = perceptron()
    p.train(X, y)
    print(p.w, p.b)

    # Testing process
    transTestX = [dt['X'], dt['y']]
    testX = np.transpose(np.array(transTestX))
    testY = np.array(dt['underLine'])
    p.test(testX, testY)
    show_diagram(df, "Train data")
