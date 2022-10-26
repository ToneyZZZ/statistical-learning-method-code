'''
This is a programme for:
creating [train lists] and [test lists] that are required in perceptron.
'''
import pandas as pd
from random import uniform


def create_list(length=20, rangeX=[0, 10], rangeY=[0, 10], div=[1, 0],
                noise=0.5, forTest=False, scatter=False):
    # creating a list with given length and range, only for percentron
    data = []
    for i in range(length):
        # devision: y = dvs[0]x + dvs[1]
        # no data will be created within dcs[2] up and down of the line
        x = form_rand(rangeX[0], rangeX[1], 4)
        tempRange = (div[0] * x + div[1] - noise, div[0] * x + div[1] + noise)
        # recreate a y in cases
        if scatter:
            y = form_rand(rangeY[0], rangeY[1], 4)
            while (y > tempRange[0] and y < tempRange[1]):
                y = form_rand(rangeY[0], rangeY[1], 4)
        else:
            y = form_rand(tempRange[0], tempRange[1], 4)
        # recording data and using color to divide
        if y > div[0] * x + div[1]:
            data.append([x, y, 1])
        else:
            data.append([x, y, -1])
    df = pd.DataFrame(data, columns=['X', 'y', "underLine"])
    if forTest:
        df.to_csv("test.csv", index=False)
    else:
        df.to_csv("train.csv", index=False)


def form_rand(min: float, max: float, num: int):
    # return a random number between [min] and [max] with [num] demical places
    return round(uniform(min, max), num)


def create_data(trainLen=50, testLen=10, X=[0, 20], Y=[0, 20], div=[1, 0], noise=2.5, sct=True):
    # Summonning a group of data
    create_list(length=trainLen, rangeX=X, rangeY=Y, noise=noise, div=div, scatter=sct)
    create_list(length=testLen, rangeX=X, rangeY=Y, noise=noise, div=div, scatter=sct, forTest=True)


# 测试数据！
'''
if __name__ == "__main__":
    create_list(length=50, rangeX=[0, 20], rangeY=[0, 20], noise=2.5, scatter=True)
    create_list(length=5, rangeX=[0, 20], rangeY=[0, 20], noise=2.5, scatter=True, forTest=True)
'''
