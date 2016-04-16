# -*-coding:utf-8-*-
#-------------------------------------------------------------------------------
# Name:        BP_numpy
# Author:      yuma
# Created:     29/05/2015
#-------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, numpy as np

class InputLayer:
    def __init__(self, dim):
        self.dim = dim
        self.data = np.zeros((1, self.dim))

    def forward(self):
        pass

    def backward(self):
        pass

    def updateWeight(self, alha):
        pass

class NeuroLayer:
    def __init__(self, dim, preLayer, bias, randMax, randMin):
        self.dim = dim
        self.preLayer = preLayer
        self.data = np.zeros((1, self.dim))
        self.weight = np.random.rand(self.dim, self.preLayer.dim) * (randMax - randMin) - randMax
        self.bias = np.zeros((1, self.dim))
        self.bias.fill(bias)
        self.nextLayer = None
        self.preLayer.nextLayer = self
        self.diff = np.zeros((1, self.preLayer.dim))
        self.diffWeight = np.zeros((self.dim, self.preLayer.dim))
        self.diffBias = np.zeros((1, self.dim))

    def forward(self):
        temp = np.dot(self.preLayer.data, self.weight.T)
        self.data = temp + self.bias

    def backward(self):
        self.diffWeight += np.dot(self.nextLayer.diff.T, self.preLayer.data) #片方を転地して考える
        self.diffBias += self.nextLayer.diff * 1
        self.diff = np.dot(self.nextLayer.diff, self.weight)

    def updateWeight(self, alha):
        self.bias   -= self.diffBias * alha
        self.weight -= self.diffWeight * alha
        self.diffBias = np.zeros((1, self.dim))
        self.diffWeight = np.zeros((self.dim, self.preLayer.dim))

class ActionLayer:
    def __init__(self, preLayer):
        self.preLayer = preLayer
        self.dim = self.preLayer.dim
        self.data = np.zeros((1, self.preLayer.dim))
        self.nextLayer = None
        self.preLayer.nextLayer = self
        self.diff = np.zeros((1, self.preLayer.dim))

    def activation(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def deactivation(self, y):
        return y * (1 - y)

    def forward(self):
        self.data = (np.ones(self.dim) / (np.ones(self.dim) + np.exp(-self.preLayer.data)) )

    def backward(self):
        self.diff = self.nextLayer.diff * (self.data * (np.ones(self.dim) - self.data) )

    def updateWeight(self, alha):
        pass

class ErrorLayer:
    def __init__(self,preLayer):
        self.preLayer = preLayer
        self.dim = self.preLayer.dim
        self.data = 0.0
        self.target = np.zeros((1, self.dim))
        self.diff = np.zeros((1, self.preLayer.dim))
        self.preLayer.nextLayer = self

    def forward(self):
        self.data += np.power(self.preLayer.data - self.target, 2)  #numpy! n**2

    def backward(self):
        self.diff = 2 * (self.preLayer.data - self.target)

    def updateWeight(self, alha):
        self.data = 0.0

def main():
    start_time = time.clock()
    alha = 0.7
    iteration = 10000
    bias = 0.6
    randMax = 0.3
    randMin = -0.3
    trainingData = np.array([[[0.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[1.0, 0.0]]])
    trainingTarget = np.array([[[1.0]], [[0.0]], [[1.0]], [[0.0]]])

    inputLayer = InputLayer(len(trainingData[0, 0]))
    hiddenLayer = NeuroLayer(5, inputLayer, bias, randMax, randMin)
    hiddenActionLayer = ActionLayer(hiddenLayer)
    outputLayer = NeuroLayer(len(trainingTarget[0, 0]), hiddenActionLayer, bias, randMax, randMin)
    outputActionLayer = ActionLayer(outputLayer)
    errorLayer = ErrorLayer(outputActionLayer)

    neuralNetwork = np.array([inputLayer, hiddenLayer, hiddenActionLayer, outputLayer, outputActionLayer, errorLayer])

    for itr in range(iteration):
        for (d, t) in zip(trainingData, trainingTarget):
            inputLayer.data = d
            errorLayer.target = t
            for layer in neuralNetwork:
                layer.forward()
            for layer in reversed(neuralNetwork):
                layer.backward()
            #print("output", outputActionLayer.data)#debug
        #print("error", errorLayer.data)#debug
        for layer in neuralNetwork:
            layer.updateWeight(alha)

    #culuculate_time 計算時間計測
    elapsed_time = time.clock() - start_time
    print("経過時間",elapsed_time)

if __name__ == '__main__':
    main()
