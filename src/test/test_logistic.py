import sys
sys.path.append('..')

import algorithm.logistic_regressive as logReg
from loader.load_mnist import load_mnist
import numpy as np

def test() :
    path = "../../dataset/mnist"
    dataSet,labels = load_mnist(path)
    trainX = []
    trainY = []
    for i in range(10):
        trainX.extend(dataSet[i*1000:i*1000+99])
        trainY.extend(labels[i*1000:i*1000+99])
    logReg.train(trainX,trainY)
    # testX,testY = load_mnist(path,'t10k')
    testX = trainX
    testY = trainY
    testX = np.hstack((np.ones((len(testX),1)),testX))
    check = 0
    total = 0
    for i in range(len(testX)):
        
        predict = logReg.classify(testX[i])
        print('label:%d,predict:%d'%(testY[i],predict))
        if predict == testY[i]:
            check += 1
        total += 1
    print("total:%d time(s),hit:%d time(s),accuracy:%f." % (total,check,check/float(total)))

def t() :
    data = np.vstack((np.random.rand(100,2) + 3,np.random.rand(100,2) + 2))
    label = np.hstack((np.zeros((1,100)),np.ones((1,100))))
    w = logReg.train(data,label)
    draw(data,label,w)

def draw(d,l,w) :
    import matplotlib.pyplot as plt

    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(len(d)):
        if l[0][i] == 1:
            xcord1.append(d[i][0])
            ycord1.append(d[i][1])
        else:
            xcord2.append(d[i][0])
            ycord2.append(d[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s = 30,c = 'red',marker = 's')
    ax.scatter(xcord2,ycord2,s = 30,c = 'green')
    x = np.arange(-3.0,5.0,0.1)
    y = (-w[0]-w[1]*x)/w[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    test()