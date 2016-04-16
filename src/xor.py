# -*-coding:utf-8-*-
#-------------------------------------------------------------------------------
# Name:        BP
# Author:      yuma
# Created:     26/05/2015
# Copyright:   (c) yuma 2015
#-------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math, random, time

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

    def forward(self):
        for m in range(self.dim):
            self.data += (self.preLayer.data[m] - self.target[m]) ** 2

    def backward(self):
        for p in range(self.preLayer.dim):
            self.diff[p] = 2 * (self.preLayer.data[p] - self.target[p])

    def updateWeight(self, alha):
        self.data = 0.0

def main():
    start_time = time.clock()
    alha = 0.7
    iteration = 10000
    bias = 0.6
    randA = -0.5
    randB = 0.5
    randHidden = random.uniform
    randOutput = random.uniform
    trainingData = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    trainingTarget = [[1.0], [0.0], [1.0], [0.0]]

    inputLayer = InputLayer(len(trainingData[0]))
    hiddenLayer = NeuroLayer(5, inputLayer, randHidden, bias, randA, randB)
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
            #print("output", outputActionLayer.data)
        #print("error", errorLayer.data)
        for layer in neuralNetwork:
            layer.updateWeight(alha)

    #culuculate_time 経過時間測定
    elapsed_time = time.clock() - start_time
    print("経過時間",elapsed_time)

if __name__ == '__main__':
    main()
