#-------------------------------------------------------------------------------
# Name:        Iris_BP_Numpy
# Author:      yuma
# Created:     02/06/2015
#-------------------------------------------------------------------------------

import time, numpy as np

def inputFile(inputName):

    rawData = np.loadtxt(inputName, delimiter=",")
    np.random.shuffle(rawData)
    inputData = rawData[:, :4]
    targetData = rawData[:, 4:7]
    return [inputData, targetData]

class InputLayer:
    def __init__(self, dim):
        self.dim = dim
        self.data = np.zeros((1, self.dim)) #numpy!

    def forward(self):
        pass

    def backward(self):
        pass

    def updateWeight(self, alpha):
        pass

class NeuroLayer:
    def __init__(self, dim, preLayer, bias, randMax, randMin):
        self.dim = dim
        self.preLayer = preLayer
        self.data = np.zeros((1, self.dim)) #numpy!
        self.weight = np.random.rand(self.dim, self.preLayer.dim) * (randMax - randMin) - randMax #numpy!
        self.bias = np.zeros((1, self.dim)) #numpy!
        self.bias.fill(bias) #numpy!
        self.nextLayer = None
        self.preLayer.nextLayer = self
        self.diff = np.zeros((1, self.preLayer.dim)) #numpy!
        self.diffWeight = np.zeros((self.dim, self.preLayer.dim)) #numpy!
        self.diffBias = np.zeros((1, self.dim)) #numpy!

    def forward(self):
        temp = np.dot(self.preLayer.data, self.weight.T) #numpy!
        self.data = temp + self.bias

    def backward(self):
        self.diffWeight += np.dot(self.nextLayer.diff.T, self.preLayer.data) #numpy! 片方を転地して考える
        self.diffBias += self.nextLayer.diff * 1 #numpy!
        self.diff = np.dot(self.nextLayer.diff, self.weight)

    def updateWeight(self, alpha):
        self.bias   -= self.diffBias * alpha #numpy!
        self.weight -= self.diffWeight * alpha #numpy!
        self.diffBias = np.zeros((1, self.dim)) #numpy!
        self.diffWeight = np.zeros((self.dim, self.preLayer.dim)) #numpy!

class ActionLayer:
    def __init__(self, preLayer):
        self.preLayer = preLayer
        self.dim = self.preLayer.dim
        self.data = np.zeros((1, self.preLayer.dim)) #numpy!
        self.nextLayer = None
        self.preLayer.nextLayer = self
        self.diff = np.zeros((1, self.preLayer.dim)) #numpy!

    def forward(self):
        self.data = (np.ones(self.dim) / (np.ones(self.dim) + np.exp(-self.preLayer.data)) ) #numpy!

    def backward(self):
        self.diff = self.nextLayer.diff * (self.data * (np.ones(self.dim) - self.data) ) #numpy!

    def updateWeight(self, alpha):
        pass

class ErrorLayer:
    def __init__(self,preLayer):
        self.preLayer = preLayer
        self.dim = self.preLayer.dim
        self.data = 0.0
        self.target = np.zeros((1, self.dim)) #numpy!
        self.diff = np.zeros((1, self.preLayer.dim)) #numpy!
        self.preLayer.nextLayer = self
        self.result = np.zeros((1, self.dim)) #numpy!

    def forward(self):
        dataSum = np.power(self.preLayer.data - self.target, 2)  #numpy! n**2
        self.data += dataSum.sum() #numpy!
        self.result = self.preLayer.data.copy()
        self.result[self.result > 0.5] = 1
        self.result[self.result <= 0.5] = 0

    def backward(self):
        self.diff = 2 * (self.preLayer.data - self.target) #numpy!

    def updateWeight(self, alpha):
        self.data = 0.0

def main():
    start_time = time.clock()
    #input_file
    fileName = "ChangedIris.csv"
    foldSelect = 4
    nFoldCV = 5
    inputData, inputTarget = inputFile(fileName)
    oneFold = int(len(inputData) / nFoldCV)

    trainingData    = inputData[:oneFold * foldSelect]
    trainingTarget  = inputTarget[:oneFold * foldSelect]
    testData        = inputData[oneFold * foldSelect:]
    testTarget      = inputTarget[oneFold * foldSelect:]
    #input_data
    alpha = 0.008       #学習係数
    bias  = 0.9         #NeuronLayerのバイアスの大きさ
    iteration = 1000    #学習の実行回数
    hiddenDim = 12      #中間層の次元
    randMax = -0.3
    randMin = 0.3
    #make_layer
    inputLayer = InputLayer(len(trainingData[0]) )
    hiddenLayer = NeuroLayer(hiddenDim, inputLayer, bias, randMax, randMin)
    hiddenActionLayer = ActionLayer(hiddenLayer)
    outputLayer = NeuroLayer(len(trainingTarget[0]), hiddenActionLayer, bias, randMax, randMin)
    outputActionLayer = ActionLayer(outputLayer)
    errorLayer = ErrorLayer(outputActionLayer)

    neuralNetwork = np.array([inputLayer, hiddenLayer, hiddenActionLayer, outputLayer, outputActionLayer, errorLayer])
    #training
    for itr in range(iteration):
        for (d, t) in zip(trainingData, trainingTarget):
            inputLayer.data = np.array([d])
            errorLayer.target = np.array([t])
            for layer in neuralNetwork:
                layer.forward()
            for layer in reversed(neuralNetwork):
                layer.backward()
        print("error", errorLayer.data)#debug
        for layer in neuralNetwork:
            layer.updateWeight(alpha)
    #test
    correct = 0
    for (d, t) in zip(testData, testTarget):
        inputLayer.data = np.array([d])
        errorLayer.target = np.array([t])
        for layer in neuralNetwork:
            layer.forward()
        test_result = errorLayer.result == np.array([t])
        if test_result.all():
                correct += 1
    print("正解率は", int((correct / len(testData)) * 100), "%")
    #culuculate_time 計算時間計測
    elapsed_time = time.clock() - start_time
    print("経過時間",elapsed_time)

if __name__ == '__main__':
    main()
