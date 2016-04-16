# -*- coding: utf-8 -*

#-------------------------------------------------------------------------------
# Name:        Backpropagation
# Purpose:     Machine Learning and Image Recognition
# Author:      yuma
# Created:     11/08/2015
#memo
#リストやクラス型オブジェクトは代入時に参照渡しとなることを利用して、前の層と後ろの層の入出力を手に入れて誤差を計算している。
#n-分割交差検定
#n分割した標本群をnパターンつくり、それぞれのパターンで検定して、計n回検定する
#そうして得られたn回の結果を平均して1つの推定を得る
#このプログラムでは、１つのパターンの検定を行うのみとする
#-------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, csv, math, random

def formatList(value, dim, inputDim):
    """
    formatList(要素を初期化する値、自分の層の次元数、自分の前の層の次元数)、リストを初期化する関数
    inputDimが0の値をとるときには１次元の処理を、0以外の値をとるときは２次元の処理を行い
    作成したリストを返り値とする
    """
    if inputDim == 0.0:
        return [value for i in range(dim)]
    else:
        return [[value for i in range(inputDim)] for j in range(dim)]

def inputFile(inputName, sortingObject):
    """
    inputFile(読み込むファイル名、ファイルの中身と置換する内容を含むリスト)
    外部ファイル名inputnameと処理時に置き換える教師データリストsortingObjectを引数に持ち
    読み込んだファイルをstring型の要素のリストとして取得する。教師データ部分以外はfloat型に直し
    教師データ部分は１つのリストを代入させるように例外処理を行う
    このときに作成したinputData, targetDataを返り値とする。
    """
    f = open(inputName)
    dataReader = csv.reader(f)
    rawData = []
    for row in dataReader:
        rawData.append(row)
    inputData = []
    targetData = []
    random.shuffle(rawData)#csvから取得したリストをシャッフル
    for i  in range(len(rawData)):
        tempData = []
        for j in range(len(rawData[0])):
            if rawData[i][j] == sortingObject[0]:
                targetData.append(sortingObject[1])
            elif rawData[i][j] == sortingObject[2]:
                targetData.append(sortingObject[3])
            elif rawData[i][j] == sortingObject[4]:
                targetData.append(sortingObject[5])
            else:
                tempData.append(float(rawData[i][j]))
        inputData.append(tempData)
    return [inputData, targetData]

class InputLayer:
    def __init__(self, dim):
        self.dim = dim
        self.data = [0.0 for i in range(self.dim)]

    def forward(self):
        pass

    def backward(self):
        pass

    def updateWeight(self, alha):
        pass

class NeuroLayer:
    def __init__(self, dim, preLayer, rand_choice, bias, randA, randB):
        self.dim = dim
        self.preLayer = preLayer
        self.data = [0.0 for i in range(self.dim)]
        self.rand_choice = rand_choice
        self.weight = [[rand_choice(randA, randB) for i in range(self.preLayer.dim)] for j in range(self.dim)]
        self.bias = [bias for i in range(self.dim)]
        self.nextLayer = None
        self.preLayer.nextLayer = self
        self.diff = [0.0 for i in range(self.preLayer.dim)]
        self.diffWeight = [[0.0 for i in range(self.preLayer.dim)] for j in range(self.dim)]
        self.diffBias = [0.0 for i in range(self.dim)]

    def forward(self):
        temp = [0.0 for i in range(self.dim)]
        for m in range(self.dim):
            for p in range(self.preLayer.dim):
                temp[m] += self.preLayer.data[p] * self.weight[m][p]
            self.data[m] = temp[m] + self.bias[m]

    def backward(self):
        for m in range(self.dim):
            self.diffBias[m] += self.nextLayer.diff[m] * 1
        for p in range(self.preLayer.dim):
            for m in range(self.dim):
                self.diffWeight[m][p] += self.nextLayer.diff[m] * self.preLayer.data[p]
                self.diff[p] = self.nextLayer.diff[m] * self.weight[m][p]

    def updateWeight(self, alha):
        for m in range(self.dim):
            self.bias[m] -= alha * self.diffBias[m]
            for p in range(self.preLayer.dim):
                self.weight[m][p] -= alha * self.diffWeight[m][p]
        self.diffBias = [0.0 for i in range(self.dim)]
        self.diffWeight = [[0.0 for i in range(self.preLayer.dim)] for j in range(self.dim)]

class ActionLayer:
    def __init__(self, preLayer):
        self.preLayer = preLayer
        self.dim = self.preLayer.dim
        self.data = [0.0 for i in range(self.preLayer.dim)]
        self.nextLayer = None
        self.preLayer.nextLayer = self
        self.diff = [0.0 for i in range(self.preLayer.dim)]
        self.diffTemp =[0.0 for i in range(self.dim)]

    def activation(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def deactivation(self, y):
        return y * (1 - y)

    def forward(self):
        for m in range(self.dim):
            self.data[m] = self.activation(self.preLayer.data[m])

    def backward(self):
        for m in range(self.dim):
            self.diff[m] = self.nextLayer.diff[m] * self.deactivation(self.data[m])

    def updateWeight(self, alha):
        pass

class ErrorLayer:
    def __init__(self,preLayer):
        self.preLayer = preLayer
        self.dim = self.preLayer.dim
        self.data = 0.0
        self.target = [0.0 for i in range(self.dim)]
        self.diff = [0.0 for i in range(self.preLayer.dim)]
        self.preLayer.nextLayer = self
        self.result = [0.0 for i in range(self.dim)]

    def forward(self):
        for m in range(self.dim):
            self.data += (self.preLayer.data[m] - self.target[m]) ** 2
            if self.preLayer.data[m] > 0.5:
                self.result[m] = 1.0
            else:
                self.result[m] = 0.0

    def backward(self):
        for p in range(self.preLayer.dim):
            self.diff[p] = 2 * (self.preLayer.data[p] - self.target[p])

    def updateWeight(self, alha):
        self.data = 0.0


def main():
    start_time = time.clock()
    #input_file ファイルから読み込んだデータリストに関する値の入力
    fileName = "../data/iris.csv"#読み込むファイルの名前と拡張子
    sortingObject = ["Iris-setosa",[1, 0, 0], "Iris-versicolor",[0, 1, 0], "Iris-virginica",[0, 0, 1]]   #csvで書き換える要素群
    foldSelect = 4  #n分割したリストの中から任意の1つをtestDataに使う。5=>1~4 (foldSelect < nFoldCV)
    nFoldCV = 5     #入力したcsvファイルをn分割して学習データと検定データにわける(ex)n=5=>training4,test1

    #input_datalist 実際に外部データを読み込みリストの作成
    inputData, inputTarget = inputFile(fileName, sortingObject)#ファイルの読み込みをして、数値データをリストとして入手
    oneFold = int(len(inputData) / nFoldCV)#分割したリスト１つに含まれる要素数 150/5=30
    trainingData    = inputData[:oneFold * foldSelect]
    trainingTarget  = inputTarget[:oneFold * foldSelect]
    testData        = inputData[oneFold * foldSelect:]
    testTarget      = inputTarget[oneFold * foldSelect:]

    #input_data BPに必要なデータを各変数に代入する
    alpha = 0.008    #学習係数
    bias  = 0.9     #NeuronLayerのバイアスの大きさ
    iteration = 1000       #学習の実行回数
    hiddenDim = 12         #中間層の次元
    randA = -0.3
    randB = 0.3
    randHidden = random.uniform
    randOutput = random.uniform

    inputLayer = InputLayer(len(trainingData[0]))
    hiddenLayer = NeuroLayer(hiddenDim, inputLayer, randHidden, bias, randA, randB)
    hiddenActionLayer = ActionLayer(hiddenLayer)
    outputLayer = NeuroLayer(len(trainingTarget[0]), hiddenActionLayer, randOutput, bias, randA, randB)
    outputActionLayer = ActionLayer(outputLayer)
    errorLayer = ErrorLayer(outputActionLayer)

    neuralNetwork = [inputLayer, hiddenLayer, hiddenActionLayer, outputLayer, outputActionLayer, errorLayer]

    for itr in range(iteration):
        for (d, t) in zip(trainingData, trainingTarget):
            inputLayer.data = d
            errorLayer.target = t
            for layer in neuralNetwork:
                layer.forward()
            for layer in reversed(neuralNetwork):
                layer.backward()
        print("error", errorLayer.data)
        for layer in neuralNetwork:
            layer.updateWeight(alpha)
    #test
    correct = 0
    for (d, t) in zip(testData, testTarget):
        inputLayer.data = d
        errorLayer.target = t
        for layer in neuralNetwork:
            layer.forward()
        if errorLayer.result == t:
                correct += 1
    print("正解率は", int((correct / len(testData)) * 100), "%")

    #culuculate_time 計算時間計測
    elapsed_time = time.clock() - start_time
    print("経過時間",elapsed_time)

if __name__ == '__main__':
    main()
