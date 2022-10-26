'''
感知机的对偶形式实现
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint, uniform
from list_summoner import create_data


class perceptron:
    def __init__(self):
        self.learningStep = 0.5
        self.maxIteration = 2000

    def train(self, listX: np.ndarray, listY: np.ndarray):
        # 初始值设定
        count = 0                                   # 误分类个数, 用于判断停止学习过程
        gram = create_gram_matrix(listX)            # Gram矩阵, 存放所有散点点乘结果
        length = listY.size                         # 储存散点个数, 用于之后循环
        self.a = np.transpose(np.zeros(length))     # 对偶形式的参数
        self.b = 0                                  # 偏置值
        # 迭代学习过程
        for time in range(self.maxIteration):
            # 选择一个样本
            index = randint(0, length - 1)
            y = listY[index]
            # 辨别误分类点
            temp_wx = sum([self.a[j] * listY[j] * gram[j][index] for j in range(length)])
            if y * (temp_wx + self.b) <= 0:
                self.b += self.learningStep * y
                self.a[index] += self.learningStep
                count += 1
            if count >= length * 3:
                break
        # 计算最后得到的w
        self.w = np.zeros(2)
        for index in range(length):
            self.w += self.a[index] * listY[index] * listX[index]
        print("w: ", self.w, "\nb: ", self.b)

    def test(self, listX: np.ndarray, listY: np.ndarray):
        times = listY.size
        for i in range(times):
            if listY[i] * (np.dot(self.w, listX[i]) + self.b) <= 0:
                print(f'第{str(i + 1)}组数据检测失败！')
                break
            else:
                print(f'第{i + 1}组数据检测成功！')
        else:
            print("模型无误!")


def show_diagram(df: pd.DataFrame, title: str, rangeX: list,
                 rangeY: list, color=["", "royalblue", "crimson"]):
    # 展示散点图和最终得到的划线
    def chosenPoint(x) -> bool:
        # 判断是否属于可画线的点
        return (-p.b - p.w[0] * x) / p.w[1] >= rangeY[0] - 5 \
            and (-p.b - p.w[0] * x) / p.w[1] <= rangeY[1] + 5
    colors = [color[df["underLine"][i]] for i in range(len(df))]
    plt.title(title)
    plt.scatter(x=df['X'], y=df['y'], c=colors)
    # 判定画线坐标点
    if p.w[1] != 0:
        testX = np.array([])
        testY = np.array([])
        MAXMARK = True
        LINE = False
        for x in range(rangeX[0], rangeX[1] + 1):
            if chosenPoint(x):
                if MAXMARK:
                    testX = np.append(testX, [x - 1])
                    testY = np.append(testY, [(-p.b - p.w[0] * (x - 1)) / p.w[1]])
                    MAXMARK = False
                    LINE = True
            if LINE and (not chosenPoint(x)):
                testX = np.append(testX, [x])
                testY = np.append(testY, [(-p.b - p.w[0] * x) / p.w[1]])
                break
        x = rangeX[1]
        if chosenPoint(x) and len(testX) == 1:
            testX = np.append(testX, [x + 1])
            testY = np.append(testY, [(-p.b - p.w[0] * (x + 1)) / p.w[1]])
        print(testX, testY)
        plt.plot(testX, testY)
    plt.show()


def create_gram_matrix(listX: np.ndarray):
    # Creating a matrix saving dot product of every xi
    # for i = 1, 2, 3, ..., n
    maxRows = len(listX)
    gram = np.zeros(shape=(maxRows, maxRows))
    for row in range(maxRows):
        for column in range(maxRows):
            gram[row][column] = np.dot(listX[row], listX[column])
    return gram


if __name__ == "__main__":
    # 初始化数据以及读取数据
    setLine = [round(uniform(-5, 5), 2), uniform(-5, 5)]
    rangeX = [0, 40]
    rangeY = [-20, 20]
    create_data(trainLen=50, testLen=10, X=rangeX, Y=rangeY, div=setLine)
    df = pd.read_csv("train.csv")
    dt = pd.read_csv("test.csv")

    # Training process
    transX = [df['X'], df['y']]
    X = np.transpose(np.array(transX))
    y = np.array(df['underLine'])
    p = perceptron()
    p.train(X, y)
    print("具体修正值a:", p.a)

    # Testing process
    transTestX = [dt['X'], dt['y']]
    testX = np.transpose(np.array(transTestX))
    testY = np.array(dt['underLine'])
    p.test(testX, testY)
    show_diagram(df, "Train data", rangeX=rangeX, rangeY=rangeY)
